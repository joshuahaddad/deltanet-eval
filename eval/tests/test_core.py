import bisect
import json
import datetime as dt
import random
import re
import sys
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import context  # noqa: E402
import edgar  # noqa: E402
import icl  # noqa: E402
import llm  # noqa: E402
import runs  # noqa: E402
import scoring  # noqa: E402


@pytest.mark.parametrize("s, want", [
    ("$391,035 million", 391_035e6),
    ("391035000000", 391_035e6),
    ("$391.035 billion", 391_035e6),
    ("391,035 (in millions)", 391_035e6),
    ("USD 29,943 thousand", 29_943e3),
    ("$(1,234) million", -1_234e6),
    ("-5.2", -5.2),
    ("$6.08", 6.08),
    ("15,115,823,000 shares", 15_115_823_000),
    ("$93.7bn", 93.7e9),
    (93736000000, 93_736e6),
    (None, None),
    ("not disclosed", None),
])
def test_parse_number(s, want):
    got = scoring.parse_number(s)
    assert got == pytest.approx(want) if want is not None else got is None


@pytest.mark.parametrize("s", ["September 28, 2024", "Sept. 28, 2024", "Sep 28 2024", "2024-09-28",
                               "9/28/2024", "28 September 2024", "The fiscal year ended September 28th, 2024."])
def test_parse_date(s):
    assert scoring.parse_date(s) == dt.date(2024, 9, 28)


def test_exact_match():
    assert scoring.exact_match("$391,035 million", 391035000000, "number")
    assert not scoring.exact_match("$391.0 billion", 391035000000, "number")  # rounded is not exact
    assert not scoring.exact_match(None, 391035000000, "number")
    assert scoring.exact_match("$6.08", 6.08, "number")
    assert scoring.exact_match("September 28, 2024", "2024-09-28", "date")
    assert not scoring.exact_match("September 30, 2023", "2024-09-28", "date")
    assert scoring.exact_match("qx-48213-7.", "QX-48213-7", "code")
    assert not scoring.exact_match("QX-48213-1", "QX-48213-7", "code")


def test_plant_depths():
    text = "\n".join(f"line {i} " + "x" * (i % 50) for i in range(5000))
    paras = [(d, f"PLANT-{d}") for d in context.DEPTHS]
    out, positions = context.plant(text, paras)
    for (d, p), pos in zip(paras, positions):
        assert out[pos:pos + len(p)] == p
        assert abs(out.index(p) / len(out) - d) < 0.01
    for _, p in paras:  # original text is untouched apart from the inserts
        out = out.replace("\n\n" + p + "\n", "")
    assert out == text


def test_truncate_to_budget():
    text = "The company reported revenue growth. " * 2000
    for n in (10, 500, 4000):
        assert context.n_tokens(context.truncate(text, n)) <= n
        assert context.n_tokens(context.truncate(text, n)) >= n - 2
    assert context.truncate("short", 100) == "short"


def test_locate_and_renderings():
    text = "Cover. Total net sales | $391,035 | 2% and EPS 6.08 then September 28, 2024."
    assert scoring.locate(text, 391035000000, "number") == text.index("391,035")
    assert scoring.locate(text, 6.08, "number") == text.index("6.08")
    assert scoring.locate(text, "2024-09-28", "date") == text.index("September")
    assert scoring.locate("1,391,035,1", 391035000000, "number") is None  # no partial-number hits


def test_planted_facts_deterministic():
    a, b = context.planted_facts("AAPL-2024"), context.planted_facts("AAPL-2024")
    assert a == b and len(a) == 2 * context.PLANTS_PER_DEPTH * len(context.DEPTHS)
    assert len({f["field"] for f in a}) == len(a) and len({f["truth"] for f in a}) == len(a)
    for d in context.DEPTHS:  # each depth gets its facts, all inside its heatmap bin
        near = [f["target"] for f in a if f["kind"] == "exact" and abs(f["target"] - d) < 0.1]
        assert len(near) == context.PLANTS_PER_DEPTH and all(int(t * 5) == int(d * 5) for t in near)


def test_build_context_plants_land_at_depth():
    filing = {"ticker": "T", "fy": 2024, "truths": {},
              "text": "\n".join(f"Line {i} of the filing body." for i in range(20000))}
    for budget in (8192, 65536):
        ctx, qs = context.build_context(filing, budget)
        assert context.n_tokens(ctx) <= budget
        for q in qs:
            assert abs(q["depth"] - q["target"]) < 0.01


def test_schema_model_types():
    m = llm.schema_model({"type": "object", "properties": {
        "revenue": {"type": "number"}, "eps": {"type": ["number", "null"]},
        "segments": {"type": "array", "items": {"type": "string"}}}})
    assert m(revenue="391035", eps=None, segments=["Americas"]).revenue == 391035.0
    with pytest.raises(ValueError, match="unknown type 'float'"):
        llm.schema_model({"type": "object", "properties": {"revenue": {"type": "float"}}})


def _fake_run():
    q = lambda field, kind, source, depth, correct, **kw: {
        "field": field, "kind": kind, "source": source, "depth": depth, "question": "?", "truth": "t", "answer": "a",
        "correct": correct, "present": depth is not None} | kw
    r = lambda mode, length, fields, **kw: {"model": "M", "ticker": "AAPL", "fy": 2024, "mode": mode, "length": length,
                                           "fields": fields} | kw
    return {"id": "test-run", "preset": "smoke", "judge": "j", "status": "done", "total_cost": 0.0,
            "filings": [{"ticker": "AAPL", "fy": 2024}], "models": [{"name": "M", "model": "m/x", "base_url": "u"}],
            "results": [
                r("batched", 8192, [
                    q("a", "exact", "planted", 0.5, True, outcome="correct"),
                    q("b", "exact", "planted", 0.1, False, outcome="decoy"),
                    q("c", "exact", "real", None, False, outcome="wrong"),  # not in the context: a hallucination
                    q("e", "exact", "real", None, False, outcome="abstained"),  # not in the context, rightly null
                    q("f", "exact", "real", 0.2, True, outcome="correct"),  # known closed-book too
                    q("g", "exact", "real", 0.6, False, outcome="wrong"),
                    q("d", "tolerant", "planted", 0.5, True)], latency_s=2.0, cost=0.02),
                r("single", 8192, [q("a", "exact", "planted", 0.5, False, outcome="near_miss")], latency_s=1.0,
                  cost=0.01),
                r("closed_book", None, [q("f", "exact", "real", None, True, outcome="correct"),
                                        q("g", "exact", "real", None, False, outcome="abstained")], latency_s=0.5),
                r("batched", "full", [q("a", "exact", "planted", 0.5, None)], error="boom")]}


def test_accuracy_metrics():
    m = runs.accuracy_metrics(_fake_run()["results"])
    assert m["acc/batched/exact_planted"] == m["acc/exact_planted"] == m["acc/batched/exact_planted/8k"] == 0.5
    assert m["n/exact_planted/8k"] == 2 and (m["acc_lo/exact_planted"], m["acc_hi/exact_planted"]) == scoring.wilson(1, 2)
    assert m["acc/batched/exact_planted/d50"] == 1.0 and m["acc/batched/exact_planted/d10"] == 0.0  # pooled depth bins
    assert m["acc/single/exact_planted"] == 0.0 and m["acc/single/exact_planted/8k"] == 0.0
    assert "acc/exact_planted/d50" not in m  # only the old names are aliased
    assert m["acc/batched/tolerant_planted"] == 1.0 and m["acc/tolerant_planted"] == 1.0
    assert m["acc/batched/exact_real"] == 0.5 and m["acc/closed_book/exact_real"] == 0.5
    assert m["acc/batched/exact_real_unmemorised"] == 0.0 and m["n/batched/exact_real_unmemorised"] == 1  # f dropped
    assert m["outcome/batched/exact_planted/decoy"] == m["outcome/exact_planted/decoy"] == 0.5
    assert m["outcome/single/exact_planted/near_miss"] == 1.0 and "outcome/exact_planted/near_miss" in m
    assert m["outcome/closed_book/exact_real/abstained"] == 0.5
    assert m["halluc/exact_real"] == 0.5 and m["n/absent_exact_real"] == 2  # batched only: closed book is not counted
    assert m["calls_failed/batched"] == 1 and m["calls_failed/single"] == 0
    assert m["latency_s_median/batched"] == 2.0 and m["latency_s_median/closed_book"] == 0.5
    assert m["notional_cost_usd/batched"] == pytest.approx(0.02)
    assert m["cost_per_correct/batched"] == pytest.approx(0.02 / 3) and "cost_per_correct/single" not in m


def test_accuracy_metrics_old_run():  # results from before call modes count as batched
    old = [{k: v for k, v in r.items() if k != "mode"} for r in _fake_run()["results"][:1]]
    m = runs.accuracy_metrics(old)
    assert m["acc/exact_planted"] == 0.5 and "acc/batched/exact_real_unmemorised" not in m


def test_wilson():
    assert scoring.wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert scoring.wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert scoring.wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)


def test_agreement_kappa():
    a = [True, True, False, False, True, False, True, True, False, False]
    b = [True, False, False, False, True, False, True, True, True, False]
    r = scoring.agreement(a, b)  # po 0.8, pe 0.5
    assert r["n"] == 10 and r["agreement"] == pytest.approx(0.8) and r["kappa"] == pytest.approx(0.6)
    assert scoring.agreement([True] * 3, [True] * 3)["kappa"] is None  # no variation: kappa undefined
    assert scoring.agreement([], [])["agreement"] is None


@pytest.mark.parametrize("answer, truth, typ, want", [
    ("$391,035 million", 391035000000, "number", "correct"),
    ("$391.0 billion", 391035000000, "number", "near_miss"),  # the truth rounded to the answer's precision
    ("$6.1", 6.08, "number", "near_miss"),  # rounding, though 1.3% off
    ("$6.00", 6.08, "number", "wrong"),  # given to the cent and 1.3% off
    ("$393 billion", 391035000000, "number", "near_miss"),  # within 1%
    ("391,035", 391035000000, "number", "wrong"),  # the scale is missing
    ("$(391,035) million", 391035000000, "number", "wrong"),  # wrong sign
    ("$0 billion", 350_000_000, "number", "wrong"),
    ("September 30, 2024", "2024-09-28", "date", "near_miss"),
    ("September 1, 2024", "2024-09-28", "date", "near_miss"),  # right month and year
    ("October 3, 2024", "2024-09-28", "date", "near_miss"),  # within 7 days
    ("September 28, 2023", "2024-09-28", "date", "wrong"),
    ("QX-48213-1", "QX-48213-7", "code", "near_miss"),
    ("QX-4821-1", "QX-48213-7", "code", "near_miss"),
    ("ZK-11111-2", "QX-48213-7", "code", "wrong"),
    (None, "QX-48213-7", "code", "abstained"),
    ("  ", 6.08, "number", "abstained"),
    ("null", "2024-09-28", "date", "abstained"),
])
def test_outcome(answer, truth, typ, want):
    assert scoring.outcome(answer, truth, typ) == want


def test_find_passage():
    text = "Item 1\nIntro line.\nAs of June 30 we employed approximately 228,000 people.\n" + "More text.\n" * 400
    pos, passage = edgar.find_passage(text, [r"nothing here", edgar.TOLERANT["employees"][1][0]])
    assert pos == text.index("As of June") and passage.startswith("As of June")
    assert edgar.PASSAGE_CHARS <= len(passage) < edgar.PASSAGE_CHARS + 20 and passage == text[pos:pos + len(passage)]
    assert edgar.find_passage("Approximately 317 million people have insurance.", edgar.TOLERANT["employees"][1]) is None


def _seasonality_filing(line: int) -> dict:
    lines = [f"Line {i} of the filing body." for i in range(20000)]
    lines[line] = "Seasonality. Sales peak in the holiday quarter."
    text = "\n".join(lines)
    pos, passage = edgar.find_passage(text, edgar.TOLERANT["seasonality"][1])
    return {"ticker": "T", "fy": 2024, "text": text,
            "truths": {"seasonality": {"question": "?", "type": "text", "truth": passage, "pos": pos}}}


def test_build_context_tolerant_reference_and_depth():
    filing = _seasonality_filing(3000)
    passage = filing["truths"]["seasonality"]["truth"]
    ctx, qs = context.build_context(filing, 65536)
    q = next(q for q in qs if q["field"] == "seasonality")
    assert q["truth"] == passage and ctx[round(q["depth"] * len(ctx)):].startswith("Seasonality.")
    q = next(q for q in context.build_context(filing, 8192)[1] if q["field"] == "seasonality")
    assert q["depth"] is None and q["truth"] == ""  # cut off entirely
    # cut inside the passage: the reference is only what was sent, and under half of it counts as absent
    plant_tokens = sum(context.n_tokens(t) + 2 for _, t in context.planted_paragraphs(context.planted_facts("T-2024")))
    sent = len(context.truncate(filing["text"], 8192 - context.RESERVE - plant_tokens))
    starts = [m.start() for m in re.finditer("^", filing["text"], re.M)]
    for keep, present in ((0.8, True), (0.3, False)):
        filing = _seasonality_filing(bisect.bisect(starts, sent - int(keep * len(passage))))
        t = filing["truths"]["seasonality"]
        sent = len(context.truncate(filing["text"], 8192 - context.RESERVE - plant_tokens))  # the new line shifts tokens
        ctx, qs = context.build_context(filing, 8192)
        q = next(q for q in qs if q["field"] == "seasonality")
        assert q["truth"] == t["truth"][:sent - t["pos"]] and abs(len(q["truth"]) / len(t["truth"]) - keep) < 0.05
        assert (q["depth"] is not None) == present
        if present:
            assert ctx[round(q["depth"] * len(ctx)):].startswith("Seasonality.")


def test_judge_check(monkeypatch):
    run = _fake_run()
    tol = lambda field, correct: {"field": field, "kind": "tolerant", "source": "real", "depth": 0.2, "present": True,
                                  "question": "?", "truth": "ref", "answer": "ans", "correct": correct}
    run["results"][0]["fields"] += [tol(f"t{i}", i % 2 == 0) for i in range(10)] + [
        tol("blank", False) | {"answer": None}, tol("absent", False) | {"present": False, "depth": None}]
    sample = runs.judge_check_sample(run)
    assert len(sample) == 10 and sample == runs.judge_check_sample(run)  # graded real answers only, same every time
    monkeypatch.setitem(llm.CONFIG, "judge_check", {"name": "J2", "model": "j2"})
    monkeypatch.setattr(runs, "judge", lambda q, ref, a, cfg: (True, {"reason": "ok", "cost": 0}))
    jc = runs.judge_check(run)
    assert jc["sampled"] == 10 and jc["errors"] == 0 and jc["judge"] == "j2"
    first = [i["first"] for i in jc["items"]]
    assert jc["agreement"] == pytest.approx(sum(first) / 10) and jc["kappa"] == 0  # no better than chance


def test_log_mlflow(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    run, bm = _fake_run(), scoring.bootstrap_mean
    run["paired"] = [{"mode": "batched", "a": "A", "b": "B", "length": "8k", "exact_gap": bm([1]),
                      "tolerant_gap": bm([0]), "did": bm([1])},
                     {"mode": "single", "a": "A", "b": "B", "length": "8k", "exact_gap": bm([1]), "tolerant_gap": None,
                      "did": None},
                     {"a": "A", "b": "B", "length": "32k", "exact_gap": bm([0]), "tolerant_gap": bm([0]),
                      "did": bm([])}]  # from before modes: single
    rid = runs.log_mlflow(run)
    c = MlflowClient()
    pm = c.get_run(rid).data.metrics
    assert pm["paired/A vs B/batched/did/8k"] == 1 and "paired_lo/A vs B/batched/tolerant_gap/8k" in pm
    assert "paired/A vs B/single/exact_gap/8k" in pm and "paired/A vs B/single/did/8k" not in pm
    assert "paired/A vs B/single/tolerant_gap/32k" in pm and "paired/A vs B/single/did/32k" not in pm
    child = c.search_runs([c.get_run(rid).info.experiment_id], f"tags.mlflow.parentRunId = '{rid}'")[0]
    assert child.data.metrics["acc/exact_planted"] == 0.5 and child.data.metrics["calls_failed"] == 1
    assert child.data.metrics["acc/single/exact_planted/8k"] == 0.0
    assert {a.path for a in c.list_artifacts(child.info.run_id)} == {"answers.csv", "calls.json"}
    assert "run.json" in {a.path for a in c.list_artifacts(rid)}


FAKE_DS = ("fake/ds", "rev", "train.parquet", "test.parquet", "label", 10, {8192: 400, 32768: 1000})


def _fake_icl(monkeypatch):
    """Ten labels, long training texts so 400 shots overflow 8k; separate test texts. No network."""
    train = [{"id": i, "text": f"train {i} " + "please help me with my card " * 2, "label": i % 10} for i in range(300)]
    test = [{"id": i, "text": f"test question {i}", "label": (i * 7) % 10} for i in range(60)]
    monkeypatch.setitem(icl.DATASETS, "fake", FAKE_DS)
    monkeypatch.setitem(icl._data, "fake", (train, test))
    return train, test


def test_icl_balance_labels():
    data = [{"id": i, "text": str(i), "label": i % 5} for i in range(50)]
    out = icl.balance_labels(data, 15, seed=3)
    for r in range(3):  # each round of five holds every label once
        assert sorted(x["label"] for x in out[5 * r:5 * r + 5]) == list(range(5))
    assert out == icl.balance_labels(data, 15, seed=3) and out != icl.balance_labels(data, 15, seed=4)


def test_icl_prompt(monkeypatch):
    train, _ = _fake_icl(monkeypatch)
    items = icl.test_items("fake", 10)
    assert items == icl.test_items("fake", 10) and len({i["label"] for i in items}) == 10  # balanced, seeded
    for length in (8192, 32768):
        for item in items[:3]:
            p = icl.build_prompt("fake", item, length)
            assert p == icl.build_prompt("fake", item, length)  # deterministic
            budget = length - context.RESERVE - context.n_tokens(icl.SYSTEM + item["text"])
            assert context.n_tokens(p["context"]) <= budget
            assert p["shots"] == len(p["demo_ids"]) <= p["helmet_shots"] == FAKE_DS[6][length]
            blocks = p["context"].split("\n\n")
            assert len(blocks) == p["shots"]
            assert item["text"] not in {b.split("\nlabel: ")[0] for b in blocks}  # the test item is not a demo
            mapping = icl.label_map(10, icl.item_seed(item["text"]))
            assert sorted(mapping) == list(range(10)) and p["gold"] == mapping[item["label"]]
            shown = {}
            for i, b in zip(p["demo_ids"], blocks):  # one original label -> one integer, and back
                assert b.endswith(f"\nlabel: {mapping[train[i]['label']]}")
                shown.setdefault(train[i]["label"], set()).add(int(b.rsplit(" ", 1)[1]))
            assert all(len(v) == 1 for v in shown.values()) and len(set().union(*shown.values())) == len(shown)
    assert icl.build_prompt("fake", items[0], 8192)["shots"] < 400  # 8k is too small for every shot here
    assert icl.build_prompt("fake", items[0], 32768)["shots"] == 1000  # repeats labels' examples, as HELMET does


def test_icl_plan_model_lengths(monkeypatch):
    monkeypatch.setattr(icl, "test_items", lambda name, n: [{"id": i, "text": f"{name} {i}", "label": 0} for i in range(n)])
    tasks = runs.plan("icl")
    claude = [t for t in tasks if t[2].get("provider") == "claude-code"]
    assert len(tasks) == 2 * 50 * (2 * 3 + 1) and {t[1] for t in claude} == {32768}
    assert len(runs.plan("icl-smoke")) == 5 * len(llm.CONFIG["models"])


def _fake_icl_run():
    r = lambda model, ds, length, item, correct, **kw: {
        "model": model, "dataset": ds, "length": length, "item": item, "text": "t", "gold": 3,
        "pred": 3 if correct else 5, "correct": correct, "shots": 10} | kw
    return {"id": "icl-test", "preset": "icl-smoke", "mode": "icl", "judge": None, "status": "done", "total_cost": 0.0,
            "datasets": [{"name": "banking77"}], "models": [{"name": "M", "model": "m/x", "base_url": "u"}],
            "results": [r("M", "banking77", 32768, 1, True), r("M", "banking77", 32768, 2, False),
                        r("M", "banking77", 32768, 3, True), r("M", "banking77", 65536, 1, False),
                        r("M", "banking77", 65536, 2, None, error="boom", pred=None)]}


def test_icl_metrics():
    m = runs.icl_metrics(_fake_icl_run()["results"])
    lo, hi = scoring.wilson(2, 3)
    assert m == {"icl_acc/banking77/32k": 2 / 3, "icl_acc_lo/banking77/32k": lo, "icl_acc_hi/banking77/32k": hi,
                 "icl_n/banking77/32k": 3, "icl_acc/banking77/64k": 0.0, "icl_acc_lo/banking77/64k": 0.0,
                 "icl_acc_hi/banking77/64k": scoring.wilson(0, 1)[1], "icl_n/banking77/64k": 1}
    rows = runs.icl_rows(_fake_icl_run()["results"])
    assert rows[0] == {"dataset": "banking77", "length": 32768, "item": 1, "text": "t", "gold": 3, "pred": 3,
                       "correct": True, "shots": 10, "call_error": None} and rows[-1]["call_error"] == "boom"


def test_log_mlflow_icl(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    rid = runs.log_mlflow(_fake_icl_run())
    c = MlflowClient()
    child = c.search_runs([c.get_run(rid).info.experiment_id], f"tags.mlflow.parentRunId = '{rid}'")[0]
    assert child.data.metrics["icl_acc/banking77/32k"] == pytest.approx(2 / 3) and child.data.metrics["calls_failed"] == 1
    assert {a.path for a in c.list_artifacts(child.info.run_id)} == {"answers.csv", "calls.json"}
    assert c.get_run(rid).data.params["mode"] == "icl"


def test_decoys_placed_apart():
    facts = context.planted_facts("AAPL-2024")
    codes = [f for f in facts if f["kind"] == "exact"]
    targets = sorted(t for t, _ in context.planted_paragraphs(facts))
    assert len(targets) == 3 * len(codes)  # every code, its near and its far decoy
    assert min(b - a for a, b in zip(targets, targets[1:])) >= context.NEAR_OFFSET - 1e-9  # no two collide
    for n, f in enumerate(codes):
        d = context.DEPTHS[(n // context.PLANTS_PER_DEPTH + 2) % len(context.DEPTHS)]  # 0.1 -> 0.5, 0.7 -> 0.1, ...
        far, near = f["decoy_targets"]["far"], f["decoy_targets"]["near"]
        assert abs(far - d) < 0.04 and int(far * 5) == int(d * 5)
        assert near - f["target"] == pytest.approx(context.NEAR_OFFSET) and int(near * 5) == int(f["target"] * 5)
        assert len({f["truth"], *f["decoys"].values()}) == 3
    seed = "AAPL-2024"  # the near stream is new, so the real and far decoy codes are as before it existed
    assert codes[0]["truth"] == context.code(random.Random(seed))
    assert codes[0]["decoys"]["far"] == context.code(random.Random(seed + "-decoy"))
    filing = {"ticker": "T", "fy": 2024, "truths": {},
              "text": "\n".join(f"Line {i} of the filing body." for i in range(20000))}
    ctx, qs = context.build_context(filing, 8192)
    assert context.n_tokens(ctx) <= 8192
    planted = [q for q in qs if "decoys" in q]
    assert len(planted) == len(codes)
    for q in planted:
        for k in ("near", "far"):
            assert abs(q["decoy_depths"][k] - q["decoy_targets"][k]) < 0.01 and f": {q['decoys'][k]}. " in ctx
        assert runs.depth_label(q["decoy_depths"]["near"]) == runs.depth_label(q["depth"])
        own, near = ctx.index(f": {q['truth']}. "), ctx.index(f": {q['decoys']['near']}. ")
        assert 0 < near - own and ctx[own:near].count("Internal reference code") == 1  # adjacent: nothing between
        assert "decoy_paragraphs" not in q and "paragraph" not in q


def test_decoy_outcome():
    ds = {"near": "ZK-11111-2", "far": "VV-22222-3"}
    assert scoring.outcome("ZK-11111-2", "QX-48213-7", "code", ds) == "decoy"
    assert scoring.decoy_hit("zk-11111-2.", ds, "code") == "near"
    assert scoring.decoy_hit("VV-22222-3", ds, "code") == "far"
    assert scoring.outcome("QX-48213-7", "QX-48213-7", "code", {"near": "QX-48213-7"}) == "correct"  # correct wins
    assert scoring.outcome("QX-48213-1", "QX-48213-7", "code", ds) == "near_miss"
    assert scoring.decoy_hit("QX-48213-7", None, "code") is None


def test_stems():
    for i, (_, statement, *_, stems) in enumerate(context.PLANTS):
        assert scoring.has_stems(statement, stems) and scoring.has_stems(statement.upper(), stems)
        assert not scoring.has_stems("I don't know.", stems) and not scoring.has_stems(None, stems)
        for j, other in enumerate(context.PLANTS):  # another plant's answer is wrong here
            assert i == j or not scoring.has_stems(other[1], stems), (i, j)
    assert scoring.has_stems("Flood insurance got too expensive.", ["flood", "insur"])


def _fake_filing(ticker="T", fy=2024):
    lines = [f"Line {i} of the filing body." for i in range(20000)]
    lines[5] = "For the fiscal year ended September 28, 2024."
    return {"ticker": ticker, "fy": fy, "url": f"u/{ticker}", "accn": "a", "period_end": "2024-09-28",
            "text": "\n".join(lines),
            "truths": {"fiscal_year_end": {"question": "On what date did fiscal year 2024 end?", "type": "date",
                                           "truth": "2024-09-28"}}}


def test_plan_modes(monkeypatch):
    monkeypatch.setattr(runs, "get_filing", _fake_filing)
    monkeypatch.setitem(runs.PRESETS["full"], "filings", [["T", 2024]])
    tasks = runs.plan("full")
    full = context.n_tokens(_fake_filing()["text"])
    for cfg in llm.CONFIG["models"]:
        mine = [t for t in tasks if t[2] is cfg]
        lengths = [32768] if cfg.get("provider") == "claude-code" else [n for n in runs.PRESETS["full"]["lengths"]
                                                                       if n < full]
        assert [t for t in mine if t[3] == "closed_book"] == [(mine[0][0], None, cfg, "closed_book", None)]
        assert sorted(t[1] for t in mine if t[3] == "batched") == lengths
        singles = [t[4] for t in mine if t[3] == "single"]
        assert len(singles) == 15 * len(lengths) and all(f.startswith("planted_code_") for f in singles)  # codes only
    smoke = runs.plan("smoke")
    assert sorted((t[3], t[4]) for t in smoke if t[2] is llm.CONFIG["models"][0]) == [
        ("batched", None), ("closed_book", None), ("single", "planted_code_50b")]
    missing = []  # the estimate's path: a filing that can't be loaded is named, not fatal
    monkeypatch.setattr(runs, "get_filing", lambda t, fy: 1 / 0 if t == "X" else _fake_filing(t, fy))
    monkeypatch.setitem(runs.PRESETS["full"], "filings", [["X", 2024], ["T", 2024]])
    assert len(runs.plan("full", missing)) == len(tasks)
    assert missing == ["X FY2024: ZeroDivisionError: division by zero"]
    with pytest.raises(ZeroDivisionError):
        runs.plan("full")


def test_score_task_modes(monkeypatch):
    monkeypatch.setattr(runs, "_contexts", {})
    filing, cfg, seen = _fake_filing(), {"name": "M", "model": "m"}, []
    facts = {f["field"]: f for f in context.planted_facts("T-2024")}

    def fake_extract(cfg, doc, schema, **kw):
        seen.append((doc, sorted(schema["properties"]), kw))
        return {"fiscal_year_end": "September 28, 2024",
                "planted_code_50b": facts["planted_code_50b"]["decoys"]["near"],
                "planted_note_50b": "The collector put its fees up twice."}, {"latency_s": 1.5}
    monkeypatch.setattr(runs, "extract", fake_extract)
    fields = ["fiscal_year_end", "planted_code_50b", "planted_note_50b"]
    b = runs.score_task(filing, 8192, cfg, "batched", None, fields)
    assert b["mode"] == "batched" and b["latency_s"] == 1.5 and seen[-1][1] == sorted(fields)
    got = {q["field"]: q for q in b["fields"]}
    assert got["fiscal_year_end"]["correct"] and got["planted_code_50b"]["outcome"] == "decoy"
    assert got["planted_code_50b"]["decoy_hit"] == "near" and "decoy_hit" not in got["fiscal_year_end"]
    assert got["planted_note_50b"]["correct"]  # stems, not the judge
    s = runs.score_task(filing, 8192, cfg, "single", "planted_code_50b", fields)
    assert s["mode"] == "single" and seen[-1][1] == ["planted_code_50b"] and seen[-1][0] == seen[0][0]
    c = runs.score_task(filing, 8192, cfg, "closed_book", None, fields)
    assert c["mode"] == "closed_book" and c["length"] is None and c["context_tokens"] == 0
    assert seen[-1][0] == "" and seen[-1][1] == ["fiscal_year_end"] and seen[-1][2]["system"] == runs.CLOSED_BOOK
    assert c["fields"][0]["depth"] is None and c["fields"][0]["correct"]
    assert "answer" not in runs.context_for(filing, 8192)[1][0]  # the cached questions are not mutated
    monkeypatch.setattr(runs, "extract", lambda *a, **kw: 1 / 0)
    assert "ZeroDivisionError" in runs.score_task(filing, 8192, cfg, "single", "planted_code_50b", fields)["error"]


def test_bootstrap_mean():
    assert scoring.bootstrap_mean([1, 1, 1]) == {"est": 1.0, "lo": 1.0, "hi": 1.0, "n": 3}
    assert scoring.bootstrap_mean([])["est"] is None
    r = scoring.bootstrap_mean([1] * 30 + [0] * 70)
    assert r == scoring.bootstrap_mean([1] * 30 + [0] * 70) and r["est"] == pytest.approx(0.3)
    assert 0.19 < r["lo"] < 0.3 < r["hi"] < 0.41  # about 0.3 +- 1.96 * sqrt(0.21 / 100)


def test_paired():
    q = lambda field, correct: {"field": field, "correct": correct}
    r = lambda model, length, fields, mode="single", **kw: {"model": model, "ticker": "T", "fy": 2024,
                                                            "length": length, "mode": mode, "fields": fields} | kw
    results = []
    for length in (8192, 32768):  # batched: A gets every code, B half, both every note; single: B gets 2 codes of 10
        results += [r("A", length, [q(f"planted_code_{i}", True) for i in range(10)]
                      + [q(f"planted_note_{i}", True) for i in range(10)] + [q("revenue", False)], "batched"),
                    r("B", length, [q(f"planted_code_{i}", i < 5) for i in range(10)]
                      + [q(f"planted_note_{i}", True) for i in range(10)] + [q("revenue", True)], "batched")]
        for i in range(10):
            results += [r("A", length, [q(f"planted_code_{i}", True)]), r("B", length, [q(f"planted_code_{i}", i < 2)])]
    results += [r("C", 8192, [q("planted_code_0", True)], error="x")]
    out = runs.paired(results)
    assert [(p["mode"], p["a"], p["b"], p["length"]) for p in out] == [
        (m, "A", "B", n) for m in ("batched", "single") for n in ("8k", "32k", "all")]
    pooled = out[2]
    assert pooled["exact_gap"]["est"] == 0.5 and pooled["exact_gap"]["n"] == 20  # planted only, not revenue
    assert pooled["tolerant_gap"] == {"est": 0.0, "lo": 0.0, "hi": 0.0, "n": 20}
    assert pooled["did"]["est"] == 0.5 and pooled["did"]["lo"] < 0.5 < pooled["did"]["hi"] and pooled["did"]["n"] == 20
    single = out[-1]
    assert single["exact_gap"]["est"] == 0.8 and single["exact_gap"]["n"] == 20
    assert single["tolerant_gap"] is None and single["did"] is None


def test_hf_router_catalogue(monkeypatch):
    listing = {"data": [{"id": "Qwen/Q", "providers": [
        {"provider": "other", "pricing": {"input": 9, "output": 9}},
        {"provider": "deepinfra", "context_length": 262144, "pricing": {"input": 0.09, "output": 1.1}}]}]}
    calls = []
    monkeypatch.setattr(llm, "_catalogue", {})
    monkeypatch.setattr(llm.httpx, "get", lambda url, **kw: calls.append(url) or type("R", (), {"json": lambda s: listing})())
    cfg = {"base_url": "https://router.huggingface.co/v1", "model": "Qwen/Q:deepinfra"}
    got = llm.catalogue(cfg)
    assert got["context_length"] == 262144 and got["pricing"]["prompt"] == pytest.approx(0.09e-6)
    assert got["pricing"]["completion"] == pytest.approx(1.1e-6)
    assert llm.catalogue(cfg | {"model": "Qwen/Q:missing"}) == {} and llm.catalogue(cfg | {"model": "No/Such"}) == {}
    assert calls == ["https://router.huggingface.co/v1/models"]  # fetched once
    assert llm.catalogue({"base_url": "https://example.com/v1", "model": "x"}) == {}


def test_stems_accept_alternatives():
    from scoring import has_stems
    assert has_stems("The logs were hard to read.", ["legib|hard to read|unread"])
    assert not has_stems("They wanted new badges.", ["legib|hard to read|unread"])


def test_paired_order_follows_models_json(monkeypatch):
    monkeypatch.setattr(runs, "CONFIG", {"models": [{"name": "A"}, {"name": "B"}]})
    res = [{"model": m, "mode": "single", "ticker": "T", "fy": 2024, "length": 8192,
            "fields": [{"field": "planted_code_10a", "correct": m == "A"}]} for m in ("B", "A")]
    out = runs.paired(res)
    assert out[0]["a"] == "A" and out[0]["exact_gap"]["est"] == 1


@pytest.mark.parametrize("fails, cause, attempts", [(2, "busy", 3), (4, "busy", None), (1, "invalid", None)])
def test_call_retries_transport_failures(monkeypatch, fails, cause, attempts):
    class Busy(Exception):
        status_code = 503
    tries, waits = [], []

    def create(**kw):
        tries.append(1)
        if len(tries) <= fails:  # as Instructor raises it: the API error is the cause
            e = RuntimeError("wrapped")
            e.__cause__ = Busy() if cause == "busy" else ValueError("schema")
            raise e
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, model_extra={})
        return llm.Verdict(reason="r", correct=True), SimpleNamespace(usage=usage)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create_with_completion=create)))
    monkeypatch.setattr(llm.instructor, "from_openai", lambda *a, **kw: client)
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    monkeypatch.setenv("TEST_KEY", "k")
    cfg = {"name": "M", "base_url": "https://example.com/v1", "api_key_env": "TEST_KEY", "model": "m"}
    if attempts is None:
        with pytest.raises(RuntimeError, match="wrapped"):
            llm.call(cfg, [{"role": "user", "content": "hi"}], llm.Verdict)
        assert len(tries) == (4 if cause == "busy" else 1)  # validation failures are not retried here
    else:
        out, meta = llm.call(cfg, [{"role": "user", "content": "hi"}], llm.Verdict)
        assert out["correct"] and meta["attempts"] == attempts
    assert [round(w / b) for w, b in zip(waits, llm.BACKOFF_S)] == [1] * len(waits)  # 10s, 30s, 60s plus jitter
    assert len(waits) == min(fails, 3) * (cause == "busy")


def test_start_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS", tmp_path)
    monkeypatch.setattr(runs, "_contexts", {})
    monkeypatch.setattr(runs, "get_filing", lambda t, fy: _fake_filing(t, fy))
    monkeypatch.setattr(runs, "model_cfg", lambda name: {"name": name, "model": "m"})
    monkeypatch.setattr(runs, "log_mlflow", lambda run: "new")
    monkeypatch.setattr(runs, "threading", SimpleNamespace(  # run the retry inline
        Lock=threading.Lock, Thread=lambda target, daemon: SimpleNamespace(start=target)))
    filing, cfg = _fake_filing(), {"name": "M", "model": "m"}
    monkeypatch.setattr(runs, "extract", lambda *a, **kw: ({}, {"cost": 0.1}))
    good = runs.score_task(filing, 8192, cfg, "single", "planted_code_10a", None)
    monkeypatch.setattr(runs, "extract", lambda *a, **kw: 1 / 0)
    bad = [runs.score_task(filing, 8192, cfg, "single", "planted_code_50b", None),
           runs.score_task(filing, 32768, cfg, "batched", None, None)]
    run = {"id": "r", "preset": "full", "design": context.DESIGN, "status": "done", "results": [good, *bad],
           "mlflow_run_id": "old", "total_cost": 0.1}
    runs.save_run(run)
    monkeypatch.setattr(runs, "extract", lambda cfg, doc, schema, **kw: ({k: None for k in schema["properties"]},
                                                                         {"cost": 0.5}))
    runs.start_retry("r")
    got = json.loads((tmp_path / "r.json").read_text())
    assert got["results"][0] == good and not any(r.get("error") for r in got["results"])
    assert [(r["mode"], r["length"], r["fields"][0]["field"]) for r in got["results"][1:]] == [
        ("single", 8192, "planted_code_50b"), ("batched", 32768, bad[1]["fields"][0]["field"])]
    assert got["total_cost"] == pytest.approx(1.1) and got["mlflow_run_id"] == "new" and isinstance(got["paired"], list)
    assert got["retry"]["status"] == "done" and got["retry"]["calls"] == 2 and got["retry"]["still_failed"] == 0
    assert got["retry"]["previous_mlflow_run_id"] == "old"
    for design in (None, context.DESIGN - 1):  # old runs have no design: their contexts can't be rebuilt
        runs.save_run({k: v for k, v in run.items() if k != "design"} | ({"design": design} if design else {}))
        with pytest.raises(RuntimeError, match="design"):
            runs.start_retry("r")
    runs.save_run(run | {"status": "running"})
    with pytest.raises(RuntimeError, match="wait"):
        runs.start_retry("r")
