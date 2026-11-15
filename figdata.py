"""The eval's numbers for the figures, straight from eval/runs/*.json through eval/analyze.py, checked against its summary.md.

Scenes: `from figdata import by, acc, planted, absent, check, short`. Nothing here is typed in by hand."""
import json
import sys
import types
from pathlib import Path
from unittest import mock

EVAL = Path(__file__).parent / "eval"
RUNS = ["20261114-193139-full", "20261114-213649-full", "20261114-202305-icl", "20261114-221727-icl"]
SUMMARY = (EVAL / "analysis" / "+".join(RUNS) / "summary.md").read_text()

# shortcut: the figure-kit image lacks the eval's API and plotting deps, which analyze.py imports but never calls for
# these numbers, so stub whichever are missing; upgrade when figure-kit can install an article's extra deps.
for name in ("matplotlib", "matplotlib.pyplot", "openai", "instructor", "tiktoken", "pyarrow", "pyarrow.parquet"):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = mock.MagicMock()
try:
    import pydantic  # noqa: F401
except ImportError:  # llm.py subclasses BaseModel at import time, so it needs a real class
    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=object, Field=mock.MagicMock(), create_model=mock.MagicMock())
sys.path.insert(0, str(EVAL))
import analyze  # noqa: E402
from analyze import acc, fmt, planted, short  # noqa: E402,F401
from runs import length_label  # noqa: E402,F401

_runs = [json.loads((EVAL / "runs" / f"{r}.json").read_text()) for r in RUNS]
FILINGS = analyze.merge([r for r in _runs if r.get("mode") != "icl"])
by = {short(m): rows for m, rows in analyze.scored(FILINGS).items()}  # short model name -> scored answer rows
# real exact facts cut from the context, batched: the outcomes, as analyze.py's hallucination section collects them
absent = {short(m["name"]): [q["outcome"] for r in FILINGS["results"] if r["model"] == m["name"] and r.get("mode") == "batched"
                             and not r.get("error") for q in r["fields"]
                             if q["kind"] == "exact" and q["source"] == "real" and q["depth"] is None and "outcome" in q]
          for m in FILINGS["models"]}


def cell(after: str, model: str, col: int) -> str:
    """Cell `col` (0 = model name) of `model`'s row in the first table after the line `after` in summary.md."""
    lines = SUMMARY[SUMMARY.index(after):].splitlines()
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if ln.startswith("|") and cells[0] == model:
            return cells[col]
    raise KeyError((after, model))


def check(ok_n: tuple[int, int], after: str, model: str, col: int) -> float:
    """Assert our (ok, n) formats to the same summary.md cell, then return the share."""
    assert fmt(*ok_n) == cell(after, model, col), (fmt(*ok_n), cell(after, model, col))
    return ok_n[0] / ok_n[1]


if __name__ == "__main__":
    print({m: len(r) for m, r in by.items()}, {m: len(a) for m, a in absent.items()})
