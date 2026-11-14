import bisect
import datetime as dt
import re
import sys
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
    return {"id": "test-run", "preset": "smoke", "judge": "j", "status": "done", "total_cost": 0.0,
            "filings": [{"ticker": "AAPL", "fy": 2024}], "models": [{"name": "M", "model": "m/x", "base_url": "u"}],
            "results": [
                {"model": "M", "ticker": "AAPL", "fy": 2024, "length": 8192, "fields": [
                    q("a", "exact", "planted", 0.5, True, outcome="correct"),
                    q("b", "exact", "planted", 0.1, False, outcome="near_miss"),
                    q("c", "exact", "real", None, False, outcome="wrong"),  # not in the context: a hallucination
                    q("e", "exact", "real", None, False, outcome="abstained"),  # not in the context, rightly null
                    q("d", "tolerant", "planted", 0.5, True)]},
                {"model": "M", "ticker": "AAPL", "fy": 2024, "length": "full", "error": "boom",
                 "fields": [q("a", "exact", "planted", 0.5, None)]}]}


def test_accuracy_metrics():
    m = runs.accuracy_metrics(_fake_run()["results"])
    lo, hi = scoring.wilson(1, 2)
    assert m == {"acc/exact_planted": 0.5, "acc/exact_planted/8k": 0.5, "acc/tolerant_planted": 1.0,
                 "acc/tolerant_planted/8k": 1.0, "n/exact_planted": 2, "n/exact_planted/8k": 2,
                 "n/tolerant_planted": 1, "n/tolerant_planted/8k": 1,
                 "acc_lo/exact_planted": lo, "acc_hi/exact_planted": hi,
                 "acc_lo/exact_planted/8k": lo, "acc_hi/exact_planted/8k": hi,
                 "acc_lo/tolerant_planted": scoring.wilson(1, 1)[0], "acc_hi/tolerant_planted": 1.0,
                 "acc_lo/tolerant_planted/8k": scoring.wilson(1, 1)[0], "acc_hi/tolerant_planted/8k": 1.0,
                 "outcome/exact_planted/correct": 0.5, "outcome/exact_planted/near_miss": 0.5,
                 "outcome/exact_planted/wrong": 0.0, "outcome/exact_planted/abstained": 0.0,
                 "halluc/exact_real": 0.5, "n/absent_exact_real": 2}


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
    plant_tokens = sum(context.n_tokens(p["paragraph"]) + 2 for p in context.planted_facts("T-2024") if "paragraph" in p)
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
    assert len(sample) == 11 and sample == runs.judge_check_sample(run)  # graded answers only, same every time
    monkeypatch.setitem(llm.CONFIG, "judge_check", {"name": "J2", "model": "j2"})
    monkeypatch.setattr(runs, "judge", lambda q, ref, a, cfg: (True, {"reason": "ok", "cost": 0}))
    jc = runs.judge_check(run)
    assert jc["sampled"] == 11 and jc["errors"] == 0 and jc["judge"] == "j2"
    first = [i["first"] for i in jc["items"]]
    assert jc["agreement"] == pytest.approx(sum(first) / 11) and jc["kappa"] == 0  # no better than chance


def test_log_mlflow(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    rid = runs.log_mlflow(_fake_run())
    c = MlflowClient()
    child = c.search_runs([c.get_run(rid).info.experiment_id], f"tags.mlflow.parentRunId = '{rid}'")[0]
    assert child.data.metrics["acc/exact_planted"] == 0.5 and child.data.metrics["calls_failed"] == 1
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
