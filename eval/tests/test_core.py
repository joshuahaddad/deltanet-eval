import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import app  # noqa: E402


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
    got = app.parse_number(s)
    assert got == pytest.approx(want) if want is not None else got is None


@pytest.mark.parametrize("s", ["September 28, 2024", "Sept. 28, 2024", "Sep 28 2024", "2024-09-28",
                               "9/28/2024", "28 September 2024", "The fiscal year ended September 28th, 2024."])
def test_parse_date(s):
    assert app.parse_date(s) == dt.date(2024, 9, 28)


def test_exact_match():
    assert app.exact_match("$391,035 million", 391035000000, "number")
    assert not app.exact_match("$391.0 billion", 391035000000, "number")  # rounded is not exact
    assert not app.exact_match(None, 391035000000, "number")
    assert app.exact_match("$6.08", 6.08, "number")
    assert app.exact_match("September 28, 2024", "2024-09-28", "date")
    assert not app.exact_match("September 30, 2023", "2024-09-28", "date")
    assert app.exact_match("qx-48213-7.", "QX-48213-7", "code")
    assert not app.exact_match("QX-48213-1", "QX-48213-7", "code")


def test_plant_depths():
    text = "\n".join(f"line {i} " + "x" * (i % 50) for i in range(5000))
    paras = [(d, f"PLANT-{d}") for d in app.DEPTHS]
    out, positions = app.plant(text, paras)
    for (d, p), pos in zip(paras, positions):
        assert out[pos:pos + len(p)] == p
        assert abs(out.index(p) / len(out) - d) < 0.01
    for _, p in paras:  # original text is untouched apart from the inserts
        out = out.replace("\n\n" + p + "\n", "")
    assert out == text


def test_truncate_to_budget():
    text = "The company reported revenue growth. " * 2000
    for n in (10, 500, 4000):
        assert app.n_tokens(app.truncate(text, n)) <= n
        assert app.n_tokens(app.truncate(text, n)) >= n - 2
    assert app.truncate("short", 100) == "short"


def test_locate_and_renderings():
    text = "Cover. Total net sales | $391,035 | 2% and EPS 6.08 then September 28, 2024."
    assert app.locate(text, 391035000000, "number") == text.index("391,035")
    assert app.locate(text, 6.08, "number") == text.index("6.08")
    assert app.locate(text, "2024-09-28", "date") == text.index("September")
    assert app.locate("1,391,035,1", 391035000000, "number") is None  # no partial-number hits


def test_planted_facts_deterministic():
    a, b = app.planted_facts("AAPL-2024"), app.planted_facts("AAPL-2024")
    assert a == b and len(a) == 2 * len(app.DEPTHS)
    assert {f["target"] for f in a} == set(app.DEPTHS)


def test_schema_model_types():
    m = app.schema_model({"type": "object", "properties": {
        "revenue": {"type": "number"}, "eps": {"type": ["number", "null"]},
        "segments": {"type": "array", "items": {"type": "string"}}}})
    assert m(revenue="391035", eps=None, segments=["Americas"]).revenue == 391035.0
    with pytest.raises(ValueError, match="unknown type 'float'"):
        app.schema_model({"type": "object", "properties": {"revenue": {"type": "float"}}})


def _fake_run():
    q = lambda field, kind, source, depth, correct: {"field": field, "kind": kind, "source": source, "depth": depth,
                                                     "question": "?", "truth": "t", "answer": "a", "correct": correct}
    return {"id": "test-run", "preset": "smoke", "judge": "j", "status": "done", "total_cost": 0.0,
            "filings": [{"ticker": "AAPL", "fy": 2024}], "models": [{"name": "M", "model": "m/x", "base_url": "u"}],
            "results": [
                {"model": "M", "ticker": "AAPL", "fy": 2024, "length": 8192, "fields": [
                    q("a", "exact", "planted", 0.5, True), q("b", "exact", "planted", 0.1, False),
                    q("c", "exact", "real", None, False),  # not in the context: not scored
                    q("d", "tolerant", "planted", 0.5, True)]},
                {"model": "M", "ticker": "AAPL", "fy": 2024, "length": "full", "error": "boom",
                 "fields": [q("a", "exact", "planted", 0.5, None)]}]}


def test_accuracy_metrics():
    m = app.accuracy_metrics(_fake_run()["results"])
    assert m == {"acc/exact_planted": 0.5, "acc/exact_planted/8k": 0.5, "acc/tolerant_planted": 1.0,
                 "acc/tolerant_planted/8k": 1.0, "n/exact_planted": 2, "n/exact_planted/8k": 2,
                 "n/tolerant_planted": 1, "n/tolerant_planted/8k": 1}


def test_log_mlflow(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'm.db'}")
    rid = app.log_mlflow(_fake_run())
    c = MlflowClient()
    child = c.search_runs([c.get_run(rid).info.experiment_id], f"tags.mlflow.parentRunId = '{rid}'")[0]
    assert child.data.metrics["acc/exact_planted"] == 0.5 and child.data.metrics["calls_failed"] == 1
    assert {a.path for a in c.list_artifacts(child.info.run_id)} == {"answers.csv", "calls.json"}
    assert "run.json" in {a.path for a in c.list_artifacts(rid)}
