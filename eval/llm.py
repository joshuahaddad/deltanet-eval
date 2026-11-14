"""Models: models.json, JSON schema to pydantic, the OpenRouter catalogue, model calls, extraction and the judge."""

import json
import os
import subprocess

import httpx
import instructor
from openai import OpenAI
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


_catalogue = {}


def catalogue(cfg: dict) -> dict:
    """OpenRouter's public model entry (context length, pricing), or {} for other endpoints."""
    if "openrouter.ai" not in cfg.get("base_url", ""):
        return {}
    if not _catalogue:
        _catalogue.update({m["id"]: m for m in httpx.get(cfg["base_url"] + "/models", timeout=30).json()["data"]})
    return _catalogue.get(cfg["model"], {})


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


def call(cfg: dict, messages: list[dict], response_model: type[BaseModel]) -> tuple[dict, dict]:
    if cfg.get("provider") == "claude-code":
        return call_claude_code(cfg, messages, response_model)
    key = os.environ.get(cfg["api_key_env"])
    if not key:
        raise RuntimeError(f"Set {cfg['api_key_env']} for {cfg['name']}")
    client = instructor.from_openai(OpenAI(base_url=cfg["base_url"], api_key=key), mode=instructor.Mode.JSON)
    extra = {"usage": {"include": True}} if "openrouter.ai" in cfg["base_url"] else {}
    obj, completion = client.chat.completions.create_with_completion(
        model=cfg["model"], messages=messages, response_model=response_model, max_retries=2,
        temperature=0, max_tokens=4096, extra_body=extra)
    usage = completion.usage
    return obj.model_dump(), {
        "provider": getattr(completion, "provider", None),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "cost": (usage.model_extra or {}).get("cost") if usage else None,
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
