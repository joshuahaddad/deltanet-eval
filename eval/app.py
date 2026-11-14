"""filing-recall: exact vs tolerant recall on long SEC 10-K filings. Run with `uv run app.py`."""

import json
import os
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from context import RESERVE, n_tokens, truncate
from edgar import get_filing, html_to_text
from llm import CONFIG, context_limit, extract, model_cfg
from paths import HERE, RUNS
from runs import PRESETS, estimate, start_judge_check, start_run

app = FastAPI(title="filing-recall")


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/config")
def config():
    return {"models": [m["name"] for m in CONFIG["models"]], "judge": CONFIG["judge"]["name"], "presets": PRESETS}


@app.get("/api/runs")
def list_runs():
    out = []
    for p in sorted(RUNS.glob("*.json"), reverse=True):
        r = json.loads(p.read_text())
        out.append({k: r.get(k) for k in ("id", "preset", "status", "started", "total_cost")})
    return out


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    path = RUNS / f"{rid}.json"
    if not re.fullmatch(r"[\w-]+", rid) or not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text())


@app.get("/api/estimate/{preset}")
def get_estimate(preset: str):
    if preset not in PRESETS:
        raise HTTPException(404)
    return estimate(preset)


@app.post("/api/runs/{preset}")
def post_run(preset: str):
    if preset not in PRESETS:
        raise HTTPException(404)
    try:
        return {"id": start_run(preset)}
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/api/runs/{rid}/judge-check")
def post_judge_check(rid: str):
    if not re.fullmatch(r"[\w-]+", rid) or not (RUNS / f"{rid}.json").exists():
        raise HTTPException(404)
    try:
        start_judge_check(rid)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"id": rid}


@app.post("/api/extract")
def post_extract(model: str = Form(...), json_schema: str = Form(...), ticker: str = Form(""), fy: str = Form(""),
                 file: UploadFile | None = File(None)):
    try:
        cfg = model_cfg(model)
        sch = json.loads(json_schema)
        if file and file.filename:
            raw = file.file.read()
            text = html_to_text(raw) if file.filename.lower().endswith((".htm", ".html")) else raw.decode("utf-8", "replace")
        else:
            text = get_filing(ticker, int(fy))["text"]
        doc = truncate(text, context_limit(cfg) - RESERVE)
        data, meta = extract(cfg, doc, sch)
        return {"data": data, "meta": meta | {"context_tokens": n_tokens(doc), "truncated": len(doc) < len(text)}}
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # the model call failed or never matched the schema
        raise HTTPException(502, f"{type(e).__name__}: {e}")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8765)))


if __name__ == "__main__":
    main()
