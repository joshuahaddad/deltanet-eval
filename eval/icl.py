"""HELMET's many-shot in-context learning task on BANKING77 and CLINC150: data, demos and prompts.

Replicates HELMET's data.py `load_icl` (github.com/princeton-nlp/HELMET, commit aeadacc): balanced demos from the
train split, a fresh random integer label mapping per test item, and a label-balanced test set."""

import hashlib
import math
import random

import httpx
import numpy as np
import pyarrow.parquet as pq

from context import RESERVE, n_tokens
from paths import CACHE

SEED = 42  # HELMET's default --seed
HELMET_TEST = 500  # HELMET's max_test_samples; our test items are a prefix of its balanced 500
# name: HF repo, pinned revision, train file, test file (HELMET uses CLINC's validation split), label column,
# label count (CLINC "plus" includes the out-of-scope label), and HELMET's shots per target length
DATASETS = {
    "banking77": ("PolyAI/banking77", "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
                  "data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet", "label", 77,
                  {8192: 360, 16384: 720, 32768: 1450, 65536: 2900, 131072: 5900}),
    "clinc150": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec",
                 "plus/train-00000-of-00001.parquet", "plus/validation-00000-of-00001.parquet", "intent", 151,
                 {8192: 440, 16384: 880, 32768: 1750, 65536: 3525, 131072: 7050}),
}
# HELMET: 'Use the provided mapping from the text to label to assign a label to the text. Only output "label:
# {label}" and nothing else.' The output sentence is replaced by the schema, since answers go through Instructor.
SYSTEM = "Use the provided mapping from the text to label to assign a label to the text."
SCHEMA = {"type": "object", "properties": {"label": {
    "type": "integer", "description": "The label for the final text, from the mapping in the examples."}}}


def _split(name: str, which: int) -> list[dict]:
    repo, rev, *files, label, _, _ = DATASETS[name]
    path = CACHE / "icl" / name / files[which].replace("/", "_")
    if not path.exists():
        r = httpx.get(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{files[which]}", timeout=120,
                      follow_redirects=True)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    t = pq.read_table(path, columns=["text", label]).to_pydict()
    return [{"id": i, "text": x, "label": int(y)} for i, (x, y) in enumerate(zip(t["text"], t[label]))]


_data = {}


def load(name: str) -> tuple[list[dict], list[dict]]:
    """(train, test) rows in the dataset's own order, which HELMET's seeded sampling depends on."""
    if name not in _data:
        _data[name] = _split(name, 0), _split(name, 1)
    return _data[name]


def balance_labels(data: list[dict], shots: int, seed: int) -> list[dict]:
    """HELMET's balance_labels, line for line: rounds of one example per label, each round in random order."""
    rand = random.Random(seed)
    by_label = {}
    for x in data:  # labels in order of first appearance, as HELMET's dict comprehension does
        by_label.setdefault(x["label"], []).append(x)
    rounds = math.ceil(shots / len(by_label))
    out = [[] for _ in range(rounds)]
    for samples in by_label.values():
        idx = rand.sample(range(len(samples)), rounds % len(samples))
        while len(idx) < rounds:  # with replacement once a label runs out, as in HELMET
            idx += rand.sample(range(len(samples)), min(rounds - len(idx), len(samples)))
        for i, j in enumerate(idx):
            out[i].append(samples[j])
    for r in out:
        rand.shuffle(r)
    return [x for r in out for x in r][:shots]


def test_items(name: str, n: int) -> list[dict]:
    """The first n of HELMET's 500 test items: a seeded shuffle (datasets' Dataset.shuffle uses
    numpy.random.default_rng(seed).permutation), then label-balanced, so the first n cover n distinct labels."""
    test = load(name)[1]
    shuffled = [test[i] for i in np.random.default_rng(SEED).permutation(len(test))]
    return balance_labels(shuffled, HELMET_TEST, SEED)[:n]


def item_seed(text: str) -> int:
    return (int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) + SEED) % 2**31


def label_map(num_labels: int, seed: int) -> list[int]:
    """Original label -> the integer shown. HELMET shuffles range(num_labels) with random.seed(local_seed)."""
    m = list(range(num_labels))
    random.Random(seed).shuffle(m)
    return m


def build_prompt(name: str, item: dict, length: int, train: list[dict] | None = None) -> dict:
    """The demos for one test item at one target length, cut to whole demos that fit length - RESERVE tokens."""
    train = train if train is not None else load(name)[0]
    num_labels, shots = DATASETS[name][5], DATASETS[name][6][length]
    seed = item_seed(item["text"])
    demos, mapping = balance_labels(train, shots, seed), label_map(num_labels, seed)
    blocks = [f"{d['text']}\nlabel: {mapping[d['label']]}" for d in demos]
    # HELMET cuts the context's trailing tokens to fit; we drop whole trailing demos instead.
    # shortcut: assumes the instruction and question fit in RESERVE, true for these one-line intents.
    budget = length - RESERVE - n_tokens(SYSTEM + item["text"]) - 20
    sizes = [n_tokens(b) + 1 for b in blocks]  # "\n\n" between demos is about one token
    keep, used = 0, 0
    while keep < len(blocks) and used + sizes[keep] <= budget:
        used += sizes[keep]
        keep += 1
    context = "\n\n".join(blocks[:keep])
    while keep and n_tokens(context) > budget:  # per-block counts can drift from the joined count by a few tokens
        keep -= 1
        context = "\n\n".join(blocks[:keep])
    return {"context": context, "question": item["text"], "gold": mapping[item["label"]],
            "shots": keep, "helmet_shots": shots, "demo_ids": [d["id"] for d in demos[:keep]]}
