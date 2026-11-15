"""Runs: presets, planning and cost estimates, scoring a call, the run and judge-check lifecycle, and MLflow."""

import datetime as dt
import itertools
import json
import os
import random
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import icl
from context import DEPTHS, DESIGN, RESERVE, build_context, n_tokens, planted_facts
from edgar import get_filing
from llm import CONFIG, catalogue, context_limit, extract, judge, model_cfg
from paths import HERE, RUNS
from scoring import OUTCOMES, agreement, bootstrap_mean, decoy_hit, exact_match, has_stems, outcome, wilson

PRESETS = {
    # every call mode once per model: closed book, batched, and a single call per planted code
    "smoke": {"filings": [["AAPL", 2024]], "lengths": [8192],
              "fields": ["fiscal_year_end", "revenue", "planted_code_50b", "planted_note_50b"]},
    # 21k to 299k tokens, so the 128k and 192k columns are real; lengths may also hold "full" (the whole filing, if
    # it fits). 192k stays inside Qwen's 262k window even with its own tokenizer
    "full": {"filings": [["AAPL", 2024], ["MSFT", 2024], ["GE", 2024], ["KO", 2024], ["F", 2024], ["PFE", 2024],
                         ["BAC", 2024], ["JPM", 2024], ["C", 2024], ["WFC", 2024]],
             "lengths": [8192, 32768, 131072, 196608], "fields": None,
             "model_lengths": {"claude-code": [32768]}},  # Claude runs on subscription quota
    # HELMET many-shot ICL: one call per test item; model_lengths limits a model (by name or provider) to some lengths
    "icl-smoke": {"mode": "icl", "datasets": ["banking77"], "lengths": [32768], "items": 5},
    "icl": {"mode": "icl", "datasets": ["banking77", "clinc150"], "lengths": [32768, 65536, 131072], "items": 50,
            "model_lengths": {"claude-code": [32768]}},  # Claude runs on subscription quota
}


def model_lengths(p: dict, cfg: dict) -> list:
    allowed = p.get("model_lengths", {})
    return allowed.get(cfg["name"], allowed.get(cfg.get("provider"), p["lengths"]))


def plan(preset: str, missing: list | None = None) -> list[tuple]:
    """Every call in a preset: (filing, budget, model, mode, field), or (dataset, length, model, test item) for ICL.
    Per filing and length a model gets one batched call and one single call per planted code, plus one
    closed-book call per filing. Skips lengths past the filing or the model's window. With a missing list, a filing
    that can't be loaded is noted there and skipped instead of raising."""
    p = PRESETS[preset]
    if p.get("mode") == "icl":
        return [(name, length, cfg, item) for name in p["datasets"] for item in icl.test_items(name, p["items"])
                for cfg in CONFIG["models"] for length in model_lengths(p, cfg) if length <= context_limit(cfg)]
    tasks = []
    for ticker, fy in p["filings"]:
        try:
            filing = get_filing(ticker, fy)
        except Exception as e:
            if missing is None:
                raise
            missing.append(f"{ticker} FY{fy}: {type(e).__name__}: {e}")
            continue
        full = n_tokens(filing["text"]) + RESERVE + 500
        # tolerant answers were at the ceiling, so single calls ask codes only; notes stay in the batched call
        planted = [q["field"] for q in planted_facts(f"{filing['ticker']}-{filing['fy']}")
                   if q["kind"] == "exact" and (not p["fields"] or q["field"] in p["fields"])]
        for cfg in CONFIG["models"]:
            budgets = [None if length == "full" else length for length in model_lengths(p, cfg)
                       if (full if length == "full" else length) <= context_limit(cfg) and (length == "full" or length < full)]
            if budgets:
                tasks.append((filing, None, cfg, "closed_book", None))
            for b in budgets:  # real questions stay batched only: single calls for them would multiply the cost
                tasks += [(filing, b, cfg, "batched", None)] + [(filing, b, cfg, "single", f) for f in planted]
    return tasks


_contexts, _contexts_lock = {}, threading.Lock()


def context_for(filing: dict, budget: int | None) -> tuple[str, list[dict], int]:
    """build_context once per filing and budget: every mode and model at one length sends the same context."""
    key = (filing["ticker"], filing["fy"], budget)
    with _contexts_lock:
        if key not in _contexts:
            ctx, qs = build_context(filing, budget)
            _contexts[key] = ctx, qs, n_tokens(ctx)
        return _contexts[key]


ICL_OVERHEAD = 300  # instruction, schema and Instructor's wrapper, beyond the demos and the question
CALL_TOKENS = {"batched": (1500, 1500), "single": (300, 100), "closed_book": (600, 400)}  # (overhead in, out)


def estimate(preset: str) -> dict:
    p, cost, unknown, tokens, calls, prompts, usd, missing = PRESETS[preset], 0.0, set(), {}, {}, {}, {}, []
    tasks = plan(preset, missing)
    jp = None if p.get("mode") == "icl" else catalogue(CONFIG["judge"]).get("pricing")
    for task in tasks:
        cfg = task[2]
        if p.get("mode") == "icl":
            name, length, _, item = task
            key = (name, length, item["id"])  # the prompt is the same for every model
            if key not in prompts:
                pr = icl.build_prompt(name, item, length)
                prompts[key] = n_tokens(pr["context"] + pr["question"]) + ICL_OVERHEAD
            sent, out = prompts[key], 30
        else:
            filing, budget, _, mode, _ = task
            over, out = CALL_TOKENS[mode]
            if mode == "closed_book":
                sent = over
            else:
                _, qs, ntok = context_for(filing, budget)
                sent = ntok + over
                if jp and mode == "batched":  # the judge grades real tolerant answers only
                    cost += sum(n_tokens(q["truth"]) + 500 for q in qs if q["kind"] == "tolerant" and q["source"] == "real"
                                and (not p["fields"] or q["field"] in p["fields"])) * float(jp["prompt"])
        tokens[cfg["name"]] = tokens.get(cfg["name"], 0) + sent
        calls[cfg["name"]] = calls.get(cfg["name"], 0) + 1
        price = catalogue(cfg).get("pricing")
        if not price:
            unknown.add(cfg["name"] + (" (Claude Code)" if cfg.get("provider") == "claude-code" else ""))
            continue
        c = sent * float(price["prompt"]) + out * float(price["completion"])
        cost, usd[cfg["name"]] = cost + c, usd.get(cfg["name"], 0) + c
    return {"preset": preset, "calls": len(tasks), "usd": round(cost, 4), "unpriced_models": sorted(unknown),
            "input_tokens": tokens, "calls_per_model": calls, "usd_per_model": {k: round(v, 4) for k, v in usd.items()},
            "missing_filings": missing}


CLOSED_BOOK = (
    "You answer questions about a company's SEC 10-K filing from memory; the filing is not shown. "
    "For amounts, give the value as written in the filing, including the scale (for example '$391,035 million' or "
    "'$6.08'). For dates, give the full date. Use null if you do not know the answer."
)


def score_task(filing: dict, budget: int | None, cfg: dict, mode: str, field: str | None,
               fields: list[str] | None) -> dict:
    ctx, qs, ntok = context_for(filing, budget)
    if mode == "closed_book":  # nothing sent, so every real question counts as absent from the context
        ctx, ntok, qs = "", 0, [q | {"depth": None} for q in qs if q["source"] == "real"]
    qs = [dict(q) for q in qs if (not fields or q["field"] in fields) and (mode != "single" or q["field"] == field)]
    res = {"model": cfg["name"], "model_id": cfg["model"], "ticker": filing["ticker"], "fy": filing["fy"],
           "mode": mode, "length": None if mode == "closed_book" else budget or "full", "context_tokens": ntok,
           "fields": qs}
    schema = {"type": "object", "properties": {q["field"]: {"type": "string", "description": q["question"]} for q in qs}}
    try:
        if mode == "closed_book":  # the default instruction says "from the document only", which would force null
            answers, meta = extract(cfg, ctx, schema, system=CLOSED_BOOK,
                                    ask=f"The filing is {filing['ticker']}'s 10-K for fiscal year {filing['fy']}. "
                                        "Answer the fields.")
        else:
            answers, meta = extract(cfg, ctx, schema)
    except Exception as e:  # one failed call should not sink the run; the error is shown on the page
        return res | {"error": f"{type(e).__name__}: {e}"}
    res |= meta
    judge_cost = 0.0
    for q in qs:
        q["answer"] = answers.get(q["field"])
        q["present"] = q["depth"] is not None
        if q["kind"] == "exact":
            q["correct"] = exact_match(q["answer"], q["truth"], q["type"])
            q["outcome"] = outcome(q["answer"], q["truth"], q["type"], q.get("decoys"))
            if q["outcome"] == "decoy":
                q["decoy_hit"] = decoy_hit(q["answer"], q["decoys"], q["type"])
        elif q["source"] == "planted":
            q["correct"] = has_stems(q["answer"], q["stems"])
        elif q["present"]:  # closed-book tolerant answers are recorded but not judged
            try:
                q["correct"], jm = judge(q["question"], q["truth"], q["answer"])
                q["judge_reason"] = jm.get("reason")
                judge_cost += jm.get("cost") or 0
            except Exception as e:
                q["correct"], q["judge_error"] = None, f"{type(e).__name__}: {e}"
    return res | {"judge_cost": judge_cost}


def score_icl(name: str, length: int, cfg: dict, item: dict) -> dict:
    """One ICL test item: the demos and the question through the extraction path, scored by exact label match."""
    pr = icl.build_prompt(name, item, length)
    res = {"model": cfg["name"], "model_id": cfg["model"], "dataset": name, "length": length, "item": item["id"],
           "text": item["text"], "gold": pr["gold"], "shots": pr["shots"],
           "context_tokens": n_tokens(pr["context"] + pr["question"]), "pred": None, "correct": None}
    try:
        answers, meta = extract(cfg, pr["context"], icl.SCHEMA, system=icl.SYSTEM, ask=pr["question"])
    except Exception as e:  # as with filings, a failed call is shown and left out of accuracy
        return res | {"error": f"{type(e).__name__}: {e}"}
    return res | meta | {"pred": answers.get("label"), "correct": answers.get("label") == pr["gold"]}


def save_run(run: dict):
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{run['id']}.json").write_text(json.dumps(run, indent=1))


def start_run(preset: str) -> str:
    for cfg in CONFIG["models"] + [CONFIG["judge"]]:
        if "api_key_env" in cfg and not os.environ.get(cfg["api_key_env"]):
            raise RuntimeError(f"Set {cfg['api_key_env']} first")
    rid = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{preset}"
    p = PRESETS[preset]
    is_icl = p.get("mode") == "icl"
    run = {"id": rid, "preset": preset, "design": DESIGN, "status": "running",
           "started": dt.datetime.now().isoformat(timespec="seconds"),
           "judge": None if is_icl else CONFIG["judge"]["model"], "models": [{k: m.get(k) for k in ("name", "model", "base_url", "provider")}
                                                         for m in CONFIG["models"]], "results": []}
    lock = threading.Lock()

    def work():
        try:
            tasks = plan(preset)
            if is_icl:
                run["mode"] = "icl"
                run["datasets"] = [{"name": n, "hf": icl.DATASETS[n][0], "revision": icl.DATASETS[n][1],
                                    "items": p["items"], "helmet_shots": {str(k): v for k, v in icl.DATASETS[n][6].items()}}
                                   for n in p["datasets"]]
            else:
                run["filings"] = list({f["url"]: {k: f[k] for k in ("ticker", "fy", "accn", "url", "period_end")}
                                       | {"full_tokens": n_tokens(f["text"])} for f, *_ in tasks}.values())
            save_run(run)
            score = (lambda t: score_icl(*t)) if is_icl else (lambda t: score_task(*t, p["fields"]))
            # claude -p runs on the local machine and quota, so it keeps a smaller pool than the API models
            with ThreadPoolExecutor(8) as pool, ThreadPoolExecutor(4) as claude_pool:
                futures = [(claude_pool if t[2].get("provider") == "claude-code" else pool).submit(score, t)
                           for t in tasks]
                for f in as_completed(futures):
                    with lock:
                        run["results"].append(f.result())
                        save_run(run)
            run["status"] = "done"
            if not is_icl:
                run["paired"] = paired(run["results"])
        except Exception as e:
            run["status"], run["error"] = "error", f"{type(e).__name__}: {e}"
        run["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        run["total_cost"] = total_cost(run)
        relog_mlflow(run)
        save_run(run)

    save_run(run)
    threading.Thread(target=work, daemon=True).start()
    return rid


def busy(run: dict) -> bool:  # each job rewrites the whole run file, so only one may run at a time
    return run["status"] == "running" or any((run.get(k) or {}).get("status") == "running" for k in ("retry", "judge_check"))


def total_cost(run: dict) -> float:
    return sum((r.get("cost") or 0) + (r.get("judge_cost") or 0) for r in run["results"])


def relog_mlflow(run: dict):
    try:
        run["mlflow_run_id"] = log_mlflow(run)
    except Exception as e:  # the run file is the record; MLflow is a copy, so a logging failure doesn't fail the run
        run["mlflow_error"] = f"{type(e).__name__}: {e}"


def start_retry(rid: str):
    """Re-run only the calls of a finished filing run that failed, replacing them in place, then recompute paired
    stats and total cost and log a new MLflow run. Only for a run made by the current planted design, because the
    contexts are rebuilt from the code."""
    run = json.loads((RUNS / f"{rid}.json").read_text())
    if run.get("mode") == "icl":
        raise RuntimeError("retry covers filing runs only")
    if busy(run):
        raise RuntimeError("wait for the run, retry or judge check to finish")
    if run.get("design") != DESIGN:
        raise RuntimeError(f"run has planted design {run.get('design')} and the code has {DESIGN}; "
                           "its contexts can't be rebuilt")
    todo = [i for i, r in enumerate(run["results"]) if r.get("error")]
    run["retry"] = {"status": "running", "calls": len(todo), "previous_mlflow_run_id": run.get("mlflow_run_id")}
    save_run(run)
    lock, fields, filings = threading.Lock(), PRESETS[run["preset"]]["fields"], {}

    def task(r):
        key = (r["ticker"], r["fy"])
        if key not in filings:
            filings[key] = get_filing(*key)
        budget = None if r["length"] in (None, "full") else r["length"]
        field = r["fields"][0]["field"] if r["mode"] == "single" else None
        return filings[key], budget, model_cfg(r["model"]), r["mode"], field

    def work():
        try:
            tasks = {i: task(run["results"][i]) for i in todo}
            with ThreadPoolExecutor(8) as pool, ThreadPoolExecutor(4) as claude_pool:
                futures = {(claude_pool if t[2].get("provider") == "claude-code" else pool).submit(
                    score_task, *t, fields): i for i, t in tasks.items()}
                for f in as_completed(futures):
                    with lock:
                        run["results"][futures[f]] = f.result()
                        save_run(run)
            run["paired"] = paired(run["results"])
            run["retry"] |= {"status": "done", "still_failed": sum(1 for i in todo if run["results"][i].get("error"))}
        except Exception as e:
            run["retry"] |= {"status": "error", "error": f"{type(e).__name__}: {e}"}
        run["retry"]["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        run["total_cost"] = total_cost(run)
        relog_mlflow(run)  # a new MLflow run; the old one stays, named in retry.previous_mlflow_run_id
        save_run(run)

    threading.Thread(target=work, daemon=True).start()


ANSWER_COLUMNS = ["ticker", "fy", "mode", "length", "field", "kind", "source", "depth", "question", "truth", "decoy",
                  "decoys", "answer", "correct", "outcome", "decoy_hit", "judge_reason", "judge_error", "call_error"]


def answer_rows(results: list[dict]) -> list[dict]:
    """One row per question asked: what the model returned, and how it was scored."""
    return [{"ticker": r["ticker"], "fy": r["fy"], "mode": r.get("mode", "batched"), "length": r["length"],
             "call_error": r.get("error")}
            | {k: q.get(k) for k in ANSWER_COLUMNS if k in q} for r in results for q in r["fields"]]


ICL_COLUMNS = ["dataset", "length", "item", "text", "gold", "pred", "correct", "shots", "call_error"]


def icl_rows(results: list[dict]) -> list[dict]:
    return [{k: r.get(k) for k in ICL_COLUMNS} | {"call_error": r.get("error")} for r in results]


def length_label(length) -> str:
    return length if length == "full" else f"{length // 1024}k"


def depth_label(depth: float) -> str:
    return f"d{DEPTHS[min(len(DEPTHS) - 1, int(depth * len(DEPTHS)))] * 100:.0f}"  # the heatmap's bins


def icl_metrics(results: list[dict]) -> dict:
    """Accuracy per dataset x length with its Wilson 95% interval and count; failed calls are left out."""
    cells = {}
    for r in results:
        if not r.get("error"):
            ok, n = cells.get((r["dataset"], r["length"]), (0, 0))
            cells[(r["dataset"], r["length"])] = (ok + bool(r["correct"]), n + 1)
    m = {}
    for (ds, length), (ok, n) in cells.items():
        k, (lo, hi) = f"{ds}/{length_label(length)}", wilson(ok, n)
        m |= {f"icl_acc/{k}": ok / n, f"icl_acc_lo/{k}": lo, f"icl_acc_hi/{k}": hi, f"icl_n/{k}": n}
    return m


def accuracy_metrics(results: list[dict]) -> dict:
    """One model's results, per call mode: accuracy with its Wilson 95% interval and count per kind x source,
    overall, per length and per depth bin pooled over lengths; exact outcome rates (decoy included) per source; the
    hallucination rate on real exact facts absent from the context; closed-book accuracy, and real exact accuracy
    without the facts the model knew closed-book; median latency, list-price cost and cost per correct answer.
    Batched cells also keep their names from before call modes existed (acc/exact_planted, outcome/exact_real/...)."""
    cells, outcomes, absent, latency, spend, right, failed = {}, {}, [0, 0], {}, {}, {}, {}
    known = {(r["ticker"], r["fy"], q["field"]) for r in results if r.get("mode") == "closed_book" and not r.get("error")
             for q in r["fields"] if q.get("correct")}
    has_closed = any(r.get("mode") == "closed_book" and not r.get("error") for r in results)
    for r in results:
        mode = r.get("mode", "batched")  # runs from before call modes were all batched
        if r.get("error"):
            failed[mode] = failed.get(mode, 0) + 1
            continue
        if r.get("latency_s") is not None:
            latency.setdefault(mode, []).append(r["latency_s"])
        spend[mode] = spend.get(mode, 0) + (r.get("notional_cost") or r.get("cost") or 0)  # list price, Claude too
        for q in r["fields"]:
            if mode == "batched" and q["kind"] == "exact" and q["source"] == "real" and q["depth"] is None \
                    and "outcome" in q:
                absent[0] += q["outcome"] != "abstained"  # answered although the fact was not sent
                absent[1] += 1
            if q.get("correct") is None or (q["depth"] is None and mode != "closed_book"):
                continue
            right[mode] = right.get(mode, 0) + bool(q["correct"])
            if "outcome" in q:
                outcomes.setdefault((mode, q["source"]), []).append(q["outcome"])
            ks = f"{q['kind']}_{q['source']}"
            keys = [ks] if mode == "closed_book" else [ks, f"{ks}/{length_label(r['length'])}", f"{ks}/{depth_label(q['depth'])}"]
            keys = [f"{mode}/{k}" for k in keys] + (keys[:2] if mode == "batched" else [])
            if has_closed and mode == "batched" and ks == "exact_real" and (r["ticker"], r["fy"], q["field"]) not in known:
                keys.append("batched/exact_real_unmemorised")
            for key in keys:
                ok, n = cells.get(key, (0, 0))
                cells[key] = (ok + bool(q["correct"]), n + 1)
    m = {}
    for k, (ok, n) in cells.items():
        lo, hi = wilson(ok, n)
        m |= {f"acc/{k}": ok / n, f"acc_lo/{k}": lo, f"acc_hi/{k}": hi, f"n/{k}": n}
    for (mode, source), outs in outcomes.items():
        for o in OUTCOMES:
            m[f"outcome/{mode}/exact_{source}/{o}"] = outs.count(o) / len(outs)
            if mode == "batched":
                m[f"outcome/exact_{source}/{o}"] = outs.count(o) / len(outs)
    if absent[1]:
        m |= {"halluc/exact_real": absent[0] / absent[1], "n/absent_exact_real": absent[1]}
    for mode in spend.keys() | failed.keys():
        m[f"calls_failed/{mode}"] = failed.get(mode, 0)
        if mode in spend:
            m[f"notional_cost_usd/{mode}"] = spend[mode]
        if right.get(mode):
            m[f"cost_per_correct/{mode}"] = spend[mode] / right[mode]
        if latency.get(mode):
            m[f"latency_s_median/{mode}"] = statistics.median(latency[mode])
    return m


def paired(results: list[dict]) -> list[dict]:
    """For every call mode and model pair, on planted items both answered: the accuracy gap A-B for planted codes
    per length and pooled, with a paired bootstrap 95% interval. Batched entries also have the tolerant gap and the
    difference in differences (exact gap - tolerant gap), pairing each code with its note from the same call;
    single calls ask codes only, so those two are None there. Items pair by ticker, fy, length, field. Paired
    entries in run files from before modes have no "mode" key: they were single."""
    ok = {}  # mode -> model -> {(ticker, fy, length, field): correct}
    for r in results:
        if r.get("mode") in ("batched", "single") and not r.get("error"):
            for q in r["fields"]:
                if q.get("correct") is not None and q["field"].startswith("planted_"):
                    ok.setdefault(r["mode"], {}).setdefault(r["model"], {})[
                        (r["ticker"], r["fy"], r["length"], q["field"])] = bool(q["correct"])
    out = []
    order = [m["name"] for m in CONFIG["models"]]  # results arrive in completion order; A is first in models.json
    for mode in ("batched", "single"):
        by = ok.get(mode, {})
        for a, b in itertools.combinations(sorted(by, key=lambda m: order.index(m) if m in order else len(order)), 2):
            shared = by[a].keys() & by[b].keys()
            for length in sorted({k[2] for k in shared}, key=lambda x: float("inf") if x == "full" else x) + [None]:
                keys = sorted((k for k in shared if length in (None, k[2])), key=str)
                diff = {k: by[a][k] - by[b][k] for k in keys}
                did = [diff[k] - diff[n] for k in keys if k[3].startswith("planted_code_")
                       for n in [k[:3] + (k[3].replace("planted_code_", "planted_note_"),)] if n in diff]
                codes = [d for k, d in diff.items() if k[3].startswith("planted_code_")]
                notes = [d for k, d in diff.items() if k[3].startswith("planted_note_")]
                out.append({"mode": mode, "a": a, "b": b, "length": "all" if length is None else length_label(length),
                            "exact_gap": bootstrap_mean(codes),
                            "tolerant_gap": bootstrap_mean(notes) if mode == "batched" else None,
                            "did": bootstrap_mean(did) if mode == "batched" else None})
    return out


def mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{HERE / 'mlflow.db'}"


def log_mlflow(run: dict) -> str:
    """Log a finished run to MLflow: a parent run holding run.json, with one child run per model holding its
    metrics, call metadata and answers.csv. MLFLOW_TRACKING_URI picks the server; unset means eval/mlflow.db."""
    import csv
    import io

    import mlflow

    uri = mlflow_uri()
    mlflow.set_tracking_uri(uri)
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "filing-recall")
    if uri.startswith("sqlite:") and mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(HERE / "mlruns").as_uri())  # not relative to the cwd
    mlflow.set_experiment(name)
    is_icl = run.get("mode") == "icl"
    with mlflow.start_run(run_name=run["id"]) as parent:
        mlflow.log_params({"preset": run["preset"], "mode": "icl", "status": run["status"],
                           "datasets": ", ".join(d["name"] for d in run.get("datasets", []))} if is_icl else
                          {"preset": run["preset"], "judge": run["judge"], "status": run["status"],
                           "filings": ", ".join(f"{f['ticker']} FY{f['fy']}" for f in run.get("filings", []))})
        mlflow.log_metric("total_cost_usd", run["total_cost"])
        mlflow.log_dict(run, "run.json")
        for pr in run.get("paired", []):  # MLflow names allow letters, digits, _ - . / space only
            pair = re.sub(r"[^\w\-. /]", "", f"{pr['a']} vs {pr['b']}").replace("/", "-")
            mode = pr.get("mode", "single")  # paired stats before modes were single calls only
            mlflow.log_metrics({f"paired{x}/{pair}/{mode}/{stat}/{pr['length']}": pr[stat][y] for stat in
                                ("exact_gap", "tolerant_gap", "did") if pr.get(stat) and pr[stat]["n"]
                                for x, y in (("", "est"), ("_lo", "lo"), ("_hi", "hi"))})
        for m in run["models"]:
            results = [r for r in run["results"] if r["model"] == m["name"]]
            with mlflow.start_run(run_name=m["name"], nested=True):
                mlflow.log_params({"model": m["model"], "endpoint": m.get("provider") or m.get("base_url")})
                calls = [{k: v for k, v in r.items() if k != "fields"} for r in results]
                mlflow.log_metrics((icl_metrics(results) if is_icl else accuracy_metrics(results)) | {
                    "calls": len(results), "calls_failed": sum(1 for r in results if r.get("error")),
                    "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in results),
                    "cost_usd": sum((r.get("cost") or 0) for r in results),
                    "notional_cost_usd": sum((r.get("notional_cost") or r.get("cost") or 0) for r in results)})
                out = io.StringIO()
                w = csv.DictWriter(out, ICL_COLUMNS if is_icl else ANSWER_COLUMNS)
                w.writeheader()
                w.writerows(icl_rows(results) if is_icl else answer_rows(results))
                mlflow.log_text(out.getvalue(), "answers.csv")
                mlflow.log_dict(calls, "calls.json")
    return parent.info.run_id


JUDGE_CHECK_N = 60  # tolerant answers re-graded per check


def judge_check_sample(run: dict) -> list[dict]:
    """A deterministic sample of the real tolerant answers the main judge actually graded (blank answers never reach
    it; planted ones are scored by stems)."""
    items = sorted(({"model": r["model"], "ticker": r["ticker"], "fy": r["fy"], "length": r["length"]} | q
                    for r in run["results"] if not r.get("error") for q in r["fields"]
                    if q["kind"] == "tolerant" and q["source"] == "real" and q.get("present")
                    and q.get("correct") is not None and q.get("answer")),
                   key=lambda q: (q["model"], q["ticker"], q["fy"], str(q["length"]), q["field"]))
    return random.Random(run["id"]).sample(items, min(JUDGE_CHECK_N, len(items)))


def judge_check(run: dict) -> dict:
    """Re-grade a sample of tolerant answers with a second judge and report how often the two judges agree."""
    cfg = CONFIG["judge_check"]

    def regrade(q):
        try:
            ok, meta = judge(q["question"], q["truth"], q["answer"], cfg)
            return {"second": ok, "second_reason": meta.get("reason"), "cost": meta.get("cost") or 0}
        except Exception as e:  # one failed call should not sink the check
            return {"error": f"{type(e).__name__}: {e}"}

    sample = judge_check_sample(run)
    with ThreadPoolExecutor(4) as pool:
        items = [{k: q[k] for k in ("model", "ticker", "fy", "length", "field", "source")} | {"first": q["correct"]} | g
                 for q, g in zip(sample, pool.map(regrade, sample))]
    done = [i for i in items if "second" in i]
    return {"status": "done", "judge": cfg["model"], "first_judge": run["judge"], "sampled": len(items),
            "errors": len(items) - len(done), "cost": sum(i.get("cost", 0) for i in items)} | agreement(
        [i["first"] for i in done], [i["second"] for i in done]) | {"items": items}


def start_judge_check(rid: str):
    run = json.loads((RUNS / f"{rid}.json").read_text())
    if run.get("mode") == "icl":
        raise RuntimeError("ICL runs have no judged answers")
    if busy(run):
        raise RuntimeError("wait for the run, retry or judge check to finish")
    if "judge_check" not in CONFIG:
        raise RuntimeError('add a "judge_check" model to models.json')
    run["judge_check"] = {"status": "running", "judge": CONFIG["judge_check"]["model"]}
    save_run(run)

    def work():
        try:
            run["judge_check"] = judge_check(run)
        except Exception as e:
            run["judge_check"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        if run.get("mlflow_run_id") and run["judge_check"]["status"] == "done":
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri())
                with mlflow.start_run(run_id=run["mlflow_run_id"]):
                    jc = run["judge_check"]
                    mlflow.log_metrics({f"judge_check/{k}": jc[k] for k in ("agreement", "kappa", "n") if jc[k] is not None})
                    mlflow.log_dict(jc, "judge_check.json")
            except Exception as e:  # as with the run itself, the file is the record
                run["judge_check"]["mlflow_error"] = f"{type(e).__name__}: {e}"
        save_run(run)

    threading.Thread(target=work, daemon=True).start()
