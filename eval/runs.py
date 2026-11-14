"""Runs: presets, planning and cost estimates, scoring a call, the run and judge-check lifecycle, and MLflow."""

import datetime as dt
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import icl
from context import RESERVE, build_context, n_tokens
from edgar import get_filing
from llm import CONFIG, catalogue, context_limit, extract, judge
from paths import HERE, RUNS
from scoring import OUTCOMES, agreement, exact_match, outcome, wilson

PRESETS = {
    "smoke": {"filings": [["AAPL", 2024]], "lengths": [8192],
              "fields": ["fiscal_year_end", "planted_code_50b", "planted_note_50b"]},
    # 45k to 208k tokens, so the 128k column is real; BAC's full length still fits a 262k window
    "full": {"filings": [["AAPL", 2024], ["MSFT", 2024], ["GE", 2024], ["KO", 2024], ["F", 2024], ["PFE", 2024],
                         ["BAC", 2024]],
             "lengths": [8192, 16384, 32768, 65536, 131072, "full"], "fields": None},
    # HELMET many-shot ICL: one call per test item; model_lengths limits a model (by name or provider) to some lengths
    "icl-smoke": {"mode": "icl", "datasets": ["banking77"], "lengths": [32768], "items": 5},
    "icl": {"mode": "icl", "datasets": ["banking77", "clinc150"], "lengths": [32768, 65536, 131072], "items": 50,
            "model_lengths": {"claude-code": [32768]}},  # Claude runs on subscription quota
}


def model_lengths(p: dict, cfg: dict) -> list:
    allowed = p.get("model_lengths", {})
    return allowed.get(cfg["name"], allowed.get(cfg.get("provider"), p["lengths"]))


def plan(preset: str) -> list[tuple[dict, int | None, dict]]:
    """Every (filing, budget, model) call in a preset, or (dataset, length, model, test item) for ICL.
    Skips lengths past the filing or the model's window."""
    p = PRESETS[preset]
    if p.get("mode") == "icl":
        return [(name, length, cfg, item) for name in p["datasets"] for item in icl.test_items(name, p["items"])
                for cfg in CONFIG["models"] for length in model_lengths(p, cfg) if length <= context_limit(cfg)]
    tasks = []
    for ticker, fy in p["filings"]:
        filing = get_filing(ticker, fy)
        full = n_tokens(filing["text"]) + RESERVE + 500
        for cfg in CONFIG["models"]:
            for length in model_lengths(p, cfg):
                if length == "full":
                    if full <= context_limit(cfg):
                        tasks.append((filing, None, cfg))
                elif length < full and length <= context_limit(cfg):
                    tasks.append((filing, length, cfg))
    return tasks


ICL_OVERHEAD = 300  # instruction, schema and Instructor's wrapper, beyond the demos and the question


def estimate(preset: str) -> dict:
    p, cost, unknown, tokens, prompts = PRESETS[preset], 0.0, set(), {}, {}
    tasks = plan(preset)
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
            ctx, qs = build_context(task[0], task[1], p["fields"])
            jp = catalogue(CONFIG["judge"]).get("pricing")
            if jp:
                cost += sum(n_tokens(q["truth"]) + 500 for q in qs if q["kind"] == "tolerant") * float(jp["prompt"])
            sent, out = n_tokens(ctx) + 1500, 1500
        tokens[cfg["name"]] = tokens.get(cfg["name"], 0) + sent
        price = catalogue(cfg).get("pricing")
        if not price:
            unknown.add(cfg["name"] + (" (Claude Code)" if cfg.get("provider") == "claude-code" else ""))
            continue
        cost += sent * float(price["prompt"]) + out * float(price["completion"])
    return {"preset": preset, "calls": len(tasks), "usd": round(cost, 4), "unpriced_models": sorted(unknown),
            "input_tokens": tokens}


def score_task(filing: dict, budget: int | None, cfg: dict, fields: list[str] | None) -> dict:
    ctx, qs = build_context(filing, budget, fields)
    res = {"model": cfg["name"], "model_id": cfg["model"], "ticker": filing["ticker"], "fy": filing["fy"],
           "length": budget or "full", "context_tokens": n_tokens(ctx), "fields": qs}
    schema = {"type": "object", "properties": {q["field"]: {"type": "string", "description": q["question"]} for q in qs}}
    try:
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
            q["outcome"] = outcome(q["answer"], q["truth"], q["type"])
        elif q["present"]:
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
    run = {"id": rid, "preset": preset, "status": "running", "started": dt.datetime.now().isoformat(timespec="seconds"),
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
                                       | {"full_tokens": n_tokens(f["text"])} for f, _, _ in tasks}.values())
            save_run(run)
            score = (lambda t: score_icl(*t)) if is_icl else (lambda t: score_task(*t, p["fields"]))
            with ThreadPoolExecutor(4) as pool:
                for r in pool.map(score, tasks):
                    with lock:
                        run["results"].append(r)
                        save_run(run)
            run["status"] = "done"
        except Exception as e:
            run["status"], run["error"] = "error", f"{type(e).__name__}: {e}"
        run["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        run["total_cost"] = sum((r.get("cost") or 0) + (r.get("judge_cost") or 0) for r in run["results"])
        try:
            run["mlflow_run_id"] = log_mlflow(run)
        except Exception as e:  # the run file is the record; MLflow is a copy, so a logging failure doesn't fail the run
            run["mlflow_error"] = f"{type(e).__name__}: {e}"
        save_run(run)

    save_run(run)
    threading.Thread(target=work, daemon=True).start()
    return rid


ANSWER_COLUMNS = ["ticker", "fy", "length", "field", "kind", "source", "depth", "question", "truth", "answer",
                  "correct", "outcome", "judge_reason", "judge_error", "call_error"]


def answer_rows(results: list[dict]) -> list[dict]:
    """One row per question asked: what the model returned, and how it was scored."""
    return [{"ticker": r["ticker"], "fy": r["fy"], "length": r["length"], "call_error": r.get("error")}
            | {k: q.get(k) for k in ANSWER_COLUMNS if k in q} for r in results for q in r["fields"]]


ICL_COLUMNS = ["dataset", "length", "item", "text", "gold", "pred", "correct", "shots", "call_error"]


def icl_rows(results: list[dict]) -> list[dict]:
    return [{k: r.get(k) for k in ICL_COLUMNS} | {"call_error": r.get("error")} for r in results]


def length_label(length) -> str:
    return length if length == "full" else f"{length // 1024}k"


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
    """The heatmap cells, collapsed: accuracy, its Wilson 95% interval and count per kind x source, overall and per
    length; exact outcome rates per source; and the hallucination rate on real exact facts absent from the context."""
    cells, outcomes, absent = {}, {}, [0, 0]
    for r in results:
        if r.get("error"):
            continue
        length = length_label(r["length"])
        for q in r["fields"]:
            if q["kind"] == "exact" and q["source"] == "real" and q["depth"] is None and "outcome" in q:
                absent[0] += q["outcome"] != "abstained"  # answered although the fact was not sent
                absent[1] += 1
            if q.get("correct") is None or q["depth"] is None:
                continue
            if "outcome" in q:
                outcomes.setdefault(q["source"], []).append(q["outcome"])
            for key in (f"{q['kind']}_{q['source']}", f"{q['kind']}_{q['source']}/{length}"):
                ok, n = cells.get(key, (0, 0))
                cells[key] = (ok + bool(q["correct"]), n + 1)
    m = {}
    for k, (ok, n) in cells.items():
        lo, hi = wilson(ok, n)
        m |= {f"acc/{k}": ok / n, f"acc_lo/{k}": lo, f"acc_hi/{k}": hi, f"n/{k}": n}
    for source, outs in outcomes.items():
        m |= {f"outcome/exact_{source}/{o}": outs.count(o) / len(outs) for o in OUTCOMES}
    if absent[1]:
        m |= {"halluc/exact_real": absent[0] / absent[1], "n/absent_exact_real": absent[1]}
    return m


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
    """A deterministic sample of the tolerant answers the main judge actually graded (blank answers never reach it)."""
    items = sorted(({"model": r["model"], "ticker": r["ticker"], "fy": r["fy"], "length": r["length"]} | q
                    for r in run["results"] if not r.get("error") for q in r["fields"]
                    if q["kind"] == "tolerant" and q.get("present") and q.get("correct") is not None and q.get("answer")),
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
    if run["status"] == "running" or (run.get("judge_check") or {}).get("status") == "running":
        raise RuntimeError("wait for the run or the current judge check to finish")
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
