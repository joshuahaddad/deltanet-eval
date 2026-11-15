"""Models: models.json, JSON schema to pydantic, the OpenRouter catalogue, model calls, extraction and the judge."""

import json
import os
import random
import subprocess
import time

import httpx
import instructor
from openai import APITimeoutError, OpenAI
from pydantic import BaseModel, Field, create_model

from paths import CACHE, HERE

CONFIG = json.loads((HERE / "models.json").read_text())

SYSTEM = (
    "You extract facts from an SEC filing. Answer every field from the document only. "
    "For amounts, give the value as written, including the scale (for example '$391,035 million' or '$6.08'). "
    "For dates, give the full date. Use null if the document does not contain the answer."
)

TYPES = {"string": str, "number": float, "integer": int, "boolean": bool}


def schema_model(schema: dict, name: str = "Extraction") -> type[BaseModel]:
    """JSON schema (objects, arrays, primitives) to a pydantic model; every field is optional."""
    def typ(prop: dict, key: str):
        t = prop.get("type", "string")
        if isinstance(t, list):  # ["number", "null"]: every field is optional already
            t = next((x for x in t if x != "null"), "string")
        if t == "object":
            return schema_model(prop, key.title().replace("_", ""))
        if t == "array":
            return list[typ(prop.get("items", {}), key)]
        if t not in TYPES:
            raise ValueError(f"field {key!r}: unknown type {t!r}; use one of "
                             f"{', '.join([*TYPES, 'object', 'array'])}")
        return TYPES[t]
    fields = {k: (typ(p, k) | None, Field(None, description=p.get("description")))
              for k, p in schema.get("properties", {}).items()}
    return create_model(name, **fields)


def model_cfg(name: str) -> dict:
    cfg = next((m for m in CONFIG["models"] + [CONFIG["judge"]] if m["name"] == name), None)
    if cfg is None:
        raise ValueError(f"unknown model {name}")
    return cfg


_catalogue = {}  # base_url -> {model id: listing entry}


def catalogue(cfg: dict) -> dict:
    """The public model listing's entry (context length, pricing per token) for OpenRouter or the Hugging Face
    router, or {} for other endpoints."""
    base = cfg.get("base_url", "")
    if "openrouter.ai" not in base and "router.huggingface.co" not in base:
        return {}
    if base not in _catalogue:
        _catalogue[base] = {m["id"]: m for m in httpx.get(base + "/models", timeout=30).json()["data"]}
    if "openrouter.ai" in base:
        return _catalogue[base].get(cfg["model"], {})
    model, _, provider = cfg["model"].partition(":")  # "org/model:deepinfra" pins the provider
    p = next((p for p in _catalogue[base].get(model, {}).get("providers", []) if p["provider"] == provider), None)
    if not p and "pricing" not in cfg:
        return {}
    p = p or {}
    price = p.get("pricing") or cfg.get("pricing")  # $ per million tokens; OpenRouter's shape is $ per token
    return {"context_length": p.get("context_length")} | (
        {"pricing": {"prompt": price["input"] / 1e6, "completion": price["output"] / 1e6}} if price else {})


def context_limit(cfg: dict) -> int:
    return cfg.get("context_length") or catalogue(cfg).get("context_length") or 131072


def call_claude_code(cfg: dict, messages: list[dict], response_model: type[BaseModel]) -> tuple[dict, dict]:
    """Claude through the local `claude -p`: your subscription, or ANTHROPIC_API_KEY when it is set.
    Claude Code enforces the schema itself, so Instructor isn't involved; pydantic still validates the result."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or "Answer from the user's message only."
    prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    cmd = ["claude", "-p", "--model", cfg["model"], "--system-prompt", system, "--tools", "", "--setting-sources", "",
           "--strict-mcp-config", "--no-session-persistence", "--output-format", "json",
           "--json-schema", json.dumps(response_model.model_json_schema())]
    CACHE.mkdir(exist_ok=True)  # run outside the repo so no project CLAUDE.md is read
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900, cwd=CACHE)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        raise RuntimeError(f"claude -p failed: {(p.stderr or p.stdout).strip()[-500:]}")
    if out.get("is_error") or out.get("structured_output") is None:
        raise RuntimeError(f"claude -p: {out.get('result') or out.get('subtype')}")
    u, billed = out["usage"], bool(os.environ.get("ANTHROPIC_API_KEY"))
    return response_model.model_validate(out["structured_output"]).model_dump(), {
        "provider": "claude-code/" + next(iter(out["modelUsage"].values()))["provider"],
        "prompt_tokens": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
        "completion_tokens": u["output_tokens"],
        "cost": out["total_cost_usd"] if billed else None,  # under a subscription this is list price, not a charge
        "notional_cost": out["total_cost_usd"],
        "billing": "api key" if billed else "subscription",
    }


RETRY_STATUS = {429, 500, 502, 503, 504}
BACKOFF_S = (10, 30, 60)  # waits before each extra attempt after a transport failure


def transient(e: Exception) -> bool:
    """A busy or failing provider, worth another attempt; Instructor wraps the API error as the cause."""
    e = e.__cause__ or e
    return isinstance(e, (APITimeoutError, httpx.TimeoutException)) or getattr(e, "status_code", None) in RETRY_STATUS


def call(cfg: dict, messages: list[dict], response_model: type[BaseModel]) -> tuple[dict, dict]:
    t = time.perf_counter()
    if cfg.get("provider") == "claude-code":
        out, meta = call_claude_code(cfg, messages, response_model)
        return out, meta | {"latency_s": round(time.perf_counter() - t, 3)}
    key = os.environ.get(cfg["api_key_env"])
    if not key:
        raise RuntimeError(f"Set {cfg['api_key_env']} for {cfg['name']}")
    client = instructor.from_openai(OpenAI(base_url=cfg["base_url"], api_key=key), mode=instructor.Mode.JSON)
    extra = {"usage": {"include": True}} if "openrouter.ai" in cfg["base_url"] else {}
    for attempt, wait in enumerate((*BACKOFF_S, None), 1):  # max_retries below covers validation failures only
        t = time.perf_counter()  # latency is the attempt that answered, without the backoff
        try:
            obj, completion = client.chat.completions.create_with_completion(
                model=cfg["model"], messages=messages, response_model=response_model, max_retries=2,
                temperature=0, max_tokens=4096, extra_body=extra)
            break
        except Exception as e:
            if wait is None or not transient(e):
                raise
            time.sleep(wait * random.uniform(1, 1.2))
    latency, usage = round(time.perf_counter() - t, 3), completion.usage
    cost = (usage.model_extra or {}).get("cost") if usage else None
    price = catalogue(cfg).get("pricing")
    if cost is None and usage and price:  # the Hugging Face router reports tokens but no cost
        cost = usage.prompt_tokens * float(price["prompt"]) + usage.completion_tokens * float(price["completion"])
    return obj.model_dump(), {
        "provider": getattr(completion, "provider", None),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "cost": cost, "latency_s": latency, "attempts": attempt,
    }


def extract(cfg: dict, document: str, schema: dict, system: str = SYSTEM,
            ask: str = "Extract the fields.") -> tuple[dict, dict]:
    """The one extraction path, used by the filing eval, the ICL eval and use mode alike."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"<document>\n{document}\n</document>\n\n{ask}"}]
    return call(cfg, messages, schema_model(schema))


class Verdict(BaseModel):
    reason: str = Field(description="One sentence.")
    correct: bool


def judge(question: str, reference: str, answer, cfg: dict | None = None) -> tuple[bool, dict]:
    if not answer:
        return False, {"cost": 0}
    prompt = (f"Question: {question}\n\nReference passage:\n{reference}\n\nAnswer to grade:\n{answer}\n\n"
              "Is the answer correct according to the reference passage? Wording may differ; "
              "the substance must match and must not contradict the passage.")
    v, meta = call(cfg or CONFIG["judge"], [{"role": "user", "content": prompt}], Verdict)
    return v["correct"], meta | {"reason": v["reason"]}
