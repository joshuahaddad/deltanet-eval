"""The post's findings, as tables and figures: `uv run python analyze.py runs/<id>.json [...]`.

Filing runs of the same design are merged into one (a model-only rerun such as `full-30b` joins the run it
complements), so paired stats cover every model; ICL runs are merged likewise. Prints markdown, one section per
finding, and writes it with the PNGs to analysis/<run ids>/."""

import argparse
import contextlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm import CONFIG
from paths import HERE
from runs import accuracy_metrics, answer_rows, length_label, paired
from scoring import wilson

MISS = ["correct", "abstained", "decoy", "near_miss", "wrong"]
COLORS = {"correct": "#3b7d4f", "abstained": "#9e9e9e", "decoy": "#e6a157", "near_miss": "#8064a2", "wrong": "#404040"}


def short(model: str) -> str:
    return model.split(" (")[0]


def fmt(ok: int, n: int) -> str:
    if not n:
        return "·"
    lo, hi = wilson(ok, n)
    return f"{ok / n:.0%} ({ok}/{n}, {lo:.0%}–{hi:.0%})"


def acc(rows) -> tuple[int, int]:
    return sum(bool(r["correct"]) for r in rows), len(rows)


def ratio(a: tuple[int, int]) -> str:
    return f"{a[0]}/{a[1]}" if a[1] else "·"


def by_length(rows) -> list:
    return sorted({r["length"] for r in rows if r["length"] is not None}, key=lambda x: float("inf") if x == "full" else x)


def table(head: list[str], rows: list[list]) -> None:
    print("| " + " | ".join(head) + " |\n" + "|---" * len(head) + "|")
    for r in rows:
        print("| " + " | ".join(map(str, r)) + " |")
    print()


def merge(runs: list[dict]) -> dict:
    """One run from several of the same kind and design; a model in two runs keeps the first run's results."""
    designs = {r.get("design", 1) for r in runs}
    if len(designs) > 1:
        raise SystemExit(f"won't merge runs of different designs {sorted(designs)}: see README, Design choices")
    models, results = [], []
    for run in runs:
        new = [m for m in run["models"] if m["name"] not in {x["name"] for x in models}
               and any(r["model"] == m["name"] for r in run["results"])]
        models += new
        results += [r for r in run["results"] if r["model"] in {m["name"] for m in new}]
    order = [m["name"] for m in CONFIG["models"]]  # models.json's order, as in paired()
    models.sort(key=lambda m: order.index(m["name"]) if m["name"] in order else len(order))
    return {"id": "+".join(r["id"] for r in runs), "models": models, "results": results,
            "total_cost": sum(r.get("total_cost") or 0 for r in runs)}


def scored(run: dict) -> dict[str, list[dict]]:
    """Scored answers per model: failed calls, and real facts absent from the context, left out."""
    out = {}
    for m in run["models"]:
        rows = answer_rows([r for r in run["results"] if r["model"] == m["name"]])
        out[m["name"]] = [r for r in rows if not r["call_error"] and r.get("correct") is not None
                          and (r["depth"] is not None or r["mode"] == "closed_book")]
    return out


def planted(rows, mode, kind="exact"):
    return [r for r in rows if r["mode"] == mode and r["source"] == "planted" and r["kind"] == kind]


def curve(ax, models: dict, lengths: list, pick, title: str) -> None:
    """Accuracy per length with Wilson 95% bars, one line per model; pick(rows, length) selects the rows."""
    for j, (m, rows) in enumerate(models.items()):
        pts = [(i + (j - len(models) / 2) * 0.05, *acc(pick(rows, L))) for i, L in enumerate(lengths)]
        pts = [p for p in pts if p[2]]
        if not pts:
            continue
        x, ok, n = zip(*pts)
        y = [o / k for o, k in zip(ok, n)]
        lo, hi = zip(*(wilson(o, k) for o, k in zip(ok, n)))
        ax.errorbar(x, y, yerr=[[max(0, a - b) for a, b in zip(y, lo)], [max(0, b - a) for a, b in zip(y, hi)]],
                    capsize=3, marker="o", label=short(m))
    ax.set_xticks(range(len(lengths)), [length_label(x) for x in lengths])
    ax.set_xlabel("context length")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    print(f"![{name}]({name})\n")


def filings(run: dict, out: Path) -> None:
    by = scored(run)
    lengths = by_length([r for rows in by.values() for r in rows])
    cols = [length_label(x) for x in lengths]
    metrics = {m: accuracy_metrics([r for r in run["results"] if r["model"] == m]) for m in by}
    pairs = paired(run["results"])
    gap = lambda s: "·" if not s or not s["n"] else f"{s['est']:+.0%} [{s['lo']:+.0%}, {s['hi']:+.0%}]"

    print("## 1–4. Exact recall of planted codes, by call mode and length\n")
    print("Batched: every question about one filing in one call. Single: one call per planted code. "
          "Accuracy (n, Wilson 95%).\n")
    for mode in ("batched", "single"):
        print(f"**{mode}**\n")
        table(["model"] + cols + ["all"], [[short(m)] + [fmt(*acc([r for r in planted(rows, mode) if r["length"] == L]))
                                                         for L in lengths] + [fmt(*acc(planted(rows, mode)))]
                                           for m, rows in by.items()])
        print(f"Paired gap, A − B, on the same items ({mode}; bootstrap 95% over items, not filings):\n")
        ps = [p for p in pairs if p["mode"] == mode and p["exact_gap"]["n"]]
        table(["A", "B"] + [p for p in dict.fromkeys(p["length"] for p in ps)],
              [[short(a), short(b)] + [gap(next((p["exact_gap"] for p in ps if (p["a"], p["b"], p["length"]) == (a, b, L)), None))
                                       for L in dict.fromkeys(p["length"] for p in ps)]
               for a, b in dict.fromkeys((p["a"], p["b"]) for p in ps)])
    print("Batching effect (batched − single accuracy, same items):\n")
    def batching(rows, L):
        (b, nb), (s, ns) = (acc([r for r in planted(rows, mode) if r["length"] == L]) for mode in ("batched", "single"))
        return f"{b / nb - s / ns:+.0%}" if nb and ns else "·"

    table(["model"] + cols, [[short(m)] + [batching(rows, L) for L in lengths] for m, rows in by.items()])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mode in zip(axes, ("batched", "single")):
        curve(ax, by, lengths, lambda rows, L, mode=mode: [r for r in planted(rows, mode) if r["length"] == L],
              f"planted exact, {mode}")
    axes[0].set_ylabel("accuracy (Wilson 95%)")
    axes[0].legend(loc="lower left")
    save(fig, out, "exact_by_length.png")

    print("Correct codes per filing at 128k and above (a batched cell is one call, so its 15 answers move together):\n")
    long = [L for L in lengths if L != "full" and L >= 131072]
    tickers = sorted({r["ticker"] for rows in by.values() for r in rows if r["length"] in long})
    table(["model", "mode"] + [f"{t} {length_label(L)}" for t in tickers for L in long], [
        [short(m), mode] + [ratio(acc([r for r in planted(rows, mode) if r["ticker"] == t and r["length"] == L]))
                            for t in tickers for L in long]
        for m, rows in by.items() for mode in ("batched", "single") if any(r["length"] in long for r in planted(rows, mode))])
    print("Single-mode answers by outcome:\n")
    table(["model", "length"] + MISS, [[short(m), length_label(L)] + [
        sum(r.get("outcome") == o for r in planted(rows, "single") if r["length"] == L) for o in MISS]
        for m, rows in by.items() for L in lengths if any(r["length"] == L for r in planted(rows, "single"))])
    bars = [(f"{short(m)}\n{length_label(L)}", [r.get("outcome") for r in planted(rows, "single") if r["length"] == L])
            for m, rows in by.items() for L in lengths if any(r["length"] == L for r in planted(rows, "single"))]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(bars) + 2), 4.5))
    bottom = [0.0] * len(bars)
    for o in MISS:
        h = [outs.count(o) / len(outs) for _, outs in bars]
        ax.bar(range(len(bars)), h, bottom=bottom, color=COLORS[o], label=o)
        bottom = [a + b for a, b in zip(bottom, h)]
    ax.set_xticks(range(len(bars)), [b for b, _ in bars], fontsize=7)
    ax.set_ylabel("share of single-mode code answers")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title("single mode: what the answers were")
    save(fig, out, "single_outcomes.png")

    print("## 5. Gist recall (tolerant)\n")
    print("Planted notes are stem-scored, batched only; real passages are judged.\n")
    table(["model"] + [f"planted {c}" for c in cols] + ["real (judged)"], [
        [short(m)] + [fmt(*acc([r for r in planted(rows, "batched", "tolerant") if r["length"] == L])) for L in lengths]
        + [fmt(*acc([r for r in rows if r["mode"] == "batched" and r["source"] == "real" and r["kind"] == "tolerant"]))]
        for m, rows in by.items()])

    print("## 6. Hallucination: answering a real fact that was cut from the context\n")
    print("Upper bound: a near miss is often a rounded figure that was in the text.\n")
    absent = {m: [q["outcome"] for r in run["results"] if r["model"] == m and r.get("mode") == "batched"
                  and not r.get("error") for q in r["fields"]
                  if q["kind"] == "exact" and q["source"] == "real" and q["depth"] is None and "outcome" in q] for m in by}
    table(["model", "answered", "wrong", "near miss", "right (memory)", "abstained"],
          [[short(m), fmt(len(a) - a.count("abstained"), len(a))] + [a.count(o) for o in ("wrong", "near_miss", "correct", "abstained")]
           for m, a in absent.items()])
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(absent)
    for o in ("wrong", "near_miss", "correct"):
        h = [a.count(o) / len(a) if a else 0 for a in absent.values()]
        ax.bar(range(len(absent)), h, bottom=bottom, color=COLORS[o], label=o)
        bottom = [x + y for x, y in zip(bottom, h)]
    ax.set_xticks(range(len(absent)), [short(m) for m in absent], fontsize=8)
    ax.set_ylabel("share of absent facts answered anyway")
    ax.legend()
    ax.set_title("hallucination on facts not in the context")
    save(fig, out, "hallucination.png")

    print("## 7. Closed book: real facts known without the filing\n")
    k = "closed_book/exact_real"
    known = {m: (round(mm[f"acc/{k}"] * mm[f"n/{k}"]), mm[f"n/{k}"]) for m, mm in metrics.items() if f"n/{k}" in mm}
    u = "batched/exact_real_unmemorised"
    table(["model", "closed book", "open book (batched)", "open book, unmemorised facts only"],
          [[short(m), fmt(*known[m])] + [fmt(round(mm[f"acc/{x}"] * mm[f"n/{x}"]), mm[f"n/{x}"]) if f"n/{x}" in mm else "·"
                                         for x in ("batched/exact_real", u)] for m, mm in metrics.items() if m in known])
    fig, ax = plt.subplots(figsize=(7, 4))
    y = [ok / n for ok, n in known.values()]
    lo, hi = zip(*(wilson(*v) for v in known.values()))
    ax.bar(range(len(known)), y, yerr=[[a - b for a, b in zip(y, lo)], [b - a for a, b in zip(y, hi)]], capsize=4,
           color="#1f77b4")
    ax.set_xticks(range(len(known)), [short(m) for m in known], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("closed-book accuracy, real exact facts")
    ax.set_title("what each model knew without the filing")
    save(fig, out, "closed_book.png")


def icl(run: dict, out: Path) -> None:
    res = [r for r in run["results"] if not r.get("error")]
    models = [m["name"] for m in run["models"]]
    lengths = by_length(res)
    print("## 8. Many-shot ICL (HELMET BANKING77)\n")
    cell = lambda m, L: [r for r in res if r["model"] == m and r["length"] == L]
    table(["model"] + [length_label(L) for L in lengths], [[short(m)] + [fmt(*acc(cell(m, L))) for L in lengths] for m in models])
    print("Paired items, each pair of models at one length:\n")
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            for L in lengths:
                ka, kb = ({r["item"]: (r["correct"], r["pred"]) for r in cell(x, L)} for x in (a, b))
                c = [(ka[k][0], kb[k][0]) for k in ka.keys() & kb.keys()]
                if c:
                    rows.append([short(a), short(b), length_label(L), len(c), sum(ka[k][1] == kb[k][1] for k in ka.keys() & kb.keys()),
                                 sum(x == y for x, y in c), c.count((True, False)), c.count((False, True))])
    table(["A", "B", "length", "items", "same label", "same verdict", "A only right", "B only right"], rows)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    curve(ax, {m: [r for r in res if r["model"] == m] for m in models}, lengths,
          lambda rows, L: [r for r in rows if r["length"] == L], "HELMET ICL, BANKING77")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy (Wilson 95%)")
    ax.legend(loc="lower left")
    save(fig, out, "icl.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "analysis")
    args = ap.parse_args()
    loaded = [json.loads(p.read_text()) for p in args.runs]
    groups = [g for g in ([r for r in loaded if r.get("mode") != "icl"], [r for r in loaded if r.get("mode") == "icl"]) if g]
    out = args.out / "+".join(r["id"] for r in loaded)
    out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"# {' + '.join(r['id'] for r in loaded)}\n")
        for g in groups:
            (icl if g[0].get("mode") == "icl" else filings)(merge(g), out)
    (out / "summary.md").write_text(buf.getvalue())
    print(buf.getvalue() + f"\n_written to {out}_")


if __name__ == "__main__":
    main()
