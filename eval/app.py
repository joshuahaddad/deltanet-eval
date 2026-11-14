"""filing-recall: exact vs tolerant recall on long SEC 10-K filings. Run with `uv run app.py`."""

import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import subprocess
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import instructor
import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # inline XBRL filings are XHTML

HERE = Path(__file__).parent
CACHE = HERE / ".cache"
RUNS = HERE / "runs"
CONFIG = json.loads((HERE / "models.json").read_text())
# shortcut: one neutral tokenizer for every model; each run also records the prompt tokens the API reports.
ENC = tiktoken.get_encoding("o200k_base")
RESERVE = 2048  # tokens kept free for instructions, schema and the answer
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)

SYSTEM = (
    "You extract facts from an SEC filing. Answer every field from the document only. "
    "For amounts, give the value as written, including the scale (for example '$391,035 million' or '$6.08'). "
    "For dates, give the full date. Use null if the document does not contain the answer."
)

# ---------------------------------------------------------------- EDGAR


def sec_get(url: str) -> bytes:
    path = CACHE / hashlib.sha1(url.encode()).hexdigest()
    if path.exists():
        return path.read_bytes()
    contact = os.environ.get("EDGAR_CONTACT")
    if not contact:
        raise RuntimeError("Set EDGAR_CONTACT to your email; SEC requires a contact in the User-Agent.")
    time.sleep(0.2)  # SEC allows 10 requests a second
    r = httpx.get(url, headers={"User-Agent": f"filing-recall {contact}"}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    CACHE.mkdir(exist_ok=True)
    path.write_bytes(r.content)
    return r.content


# field: (question, XBRL concepts tried in order; None means the period end date)
EXACT = {
    "revenue": ("What was total revenue (net sales) for fiscal year {fy}?",
                ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]),
    "net_income": ("What was net income for fiscal year {fy}?", ["NetIncomeLoss"]),
    "operating_income": ("What was operating income for fiscal year {fy}?", ["OperatingIncomeLoss"]),
    "eps_diluted": ("What were diluted earnings per share for fiscal year {fy}?", ["EarningsPerShareDiluted"]),
    "total_assets": ("What were total assets at the end of fiscal year {fy}?", ["Assets"]),
    "total_liabilities": ("What were total liabilities at the end of fiscal year {fy}?", ["Liabilities"]),
    "cash": ("What were cash and cash equivalents at the end of fiscal year {fy}?",
             ["CashAndCashEquivalentsAtCarryingValue"]),
    "shares_outstanding": ("How many shares of common stock were outstanding as of the cover page date "
                           "of the fiscal year {fy} annual report?", ["dei:EntityCommonStockSharesOutstanding"]),
    "fiscal_year_end": ("On what date did fiscal year {fy} end?", None),
}

# field: (question, section heading regex, next section heading regex)
_SEP = r"[\s.|:\-–—]*"
TOLERANT = {
    "business": ("What are the company's main products or services, as described for fiscal year {fy}?",
                 rf"item{_SEP}1{_SEP}business", rf"item{_SEP}1a\b"),
    "main_risks": ("What are the two main risk factors the company describes for fiscal year {fy}?",
                   rf"item{_SEP}1a{_SEP}risk\s+factors", rf"item{_SEP}1b\b"),
    "revenue_drivers": ("What does the company say drove the change in its revenue in fiscal year {fy}?",
                        rf"item{_SEP}7{_SEP}management", rf"item{_SEP}7a\b"),
}


def html_to_text(html: bytes | str) -> str:
    """Filing HTML to plain text; tables become one ' | '-joined line per row."""
    soup = BeautifulSoup(html, "lxml")
    for el in soup(["script", "style", "head", "ix:header"]):
        el.decompose()
    for el in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        el.decompose()
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                s = " ".join(td.get_text(" ").split())
                if not s:
                    continue
                if cells and (s in (")", "%", ")%") or cells[-1] in ("$", "(", "($")):
                    cells[-1] += s  # filings split "$", "1,234" and ")" into separate cells
                else:
                    cells.append(s)
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for el in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        el.append("\n")
    lines = (" ".join(line.split()) for line in soup.get_text().splitlines())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def find_section(text: str, start: str, end: str) -> tuple[int, str] | None:
    """The real section is the heading match followed by the most text (the others are the table of contents)."""
    ends = [m.start() for m in re.finditer(end, text, re.I)]
    best = None
    for m in re.finditer(start, text, re.I):
        stop = next((e for e in ends if e > m.end()), len(text))
        if best is None or stop - m.start() > best[1] - best[0]:
            best = (m.start(), stop)
    return (best[0], text[best[0]:best[1]]) if best else None


def get_filing(ticker: str, fy: int) -> dict:
    """Fetch one 10-K and its ground truth from XBRL. The fiscal year is the filing's DocumentFiscalYearFocus."""
    tickers = json.loads(sec_get("https://www.sec.gov/files/company_tickers.json"))
    cik = next((r["cik_str"] for r in tickers.values() if r["ticker"] == ticker.upper()), None)
    if cik is None:
        raise ValueError(f"unknown ticker {ticker}")
    facts = json.loads(sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))["facts"]
    filings = json.loads(sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))["filings"]
    accns = {f["accn"] for u in facts["us-gaap"].get("Assets", {}).get("units", {}).values()
             for f in u if f["form"] == "10-K" and f.get("fy") == fy and f.get("fp") == "FY"}
    # "recent" holds the last 1,000 filings; heavy filers (banks) push the 10-K into the older pages
    for page in [filings["recent"]] + [p["name"] for p in filings.get("files", [])]:
        recent = page if isinstance(page, dict) else json.loads(sec_get(f"https://data.sec.gov/submissions/{page}"))
        i = next((i for i, a in enumerate(recent["accessionNumber"]) if a in accns and recent["form"][i] == "10-K"),
                 None)
        if i is not None:
            break
    if i is None:
        raise ValueError(f"no 10-K for {ticker} fiscal year {fy}")
    accn, period_end = recent["accessionNumber"][i], recent["reportDate"][i]
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/{recent['primaryDocument'][i]}"
    text_path = CACHE / f"{ticker.upper()}-{fy}.txt"
    if not text_path.exists():
        text_path.write_text(html_to_text(sec_get(url)))
    text = text_path.read_text()

    truths = {}
    for field, (q, concepts) in EXACT.items():
        if concepts is None:
            truths[field] = {"question": q.format(fy=fy), "type": "date", "truth": period_end}
            continue
        for concept in concepts:
            ns, _, name = concept.rpartition(":")
            units = facts.get(ns or "us-gaap", {}).get(name, {}).get("units", {})
            vals = {f["val"] for u in units.values() for f in u if f["accn"] == accn and f["form"] == "10-K" and (
                ns == "dei" or (f["end"] == period_end and (
                    "start" not in f or (dt.date.fromisoformat(f["end"]) - dt.date.fromisoformat(f["start"])).days > 300)))}
            if len(vals) == 1:  # several values (e.g. share classes) make the question ambiguous, so skip
                truths[field] = {"question": q.format(fy=fy), "type": "number", "truth": vals.pop()}
                break
    for field, (q, start, end) in TOLERANT.items():
        sec = find_section(text, start, end)
        if sec:
            # shortcut: the judge sees the first 12k chars of the section; widen if drivers sit deeper in MD&A
            truths[field] = {"question": q.format(fy=fy), "type": "text", "truth": sec[1][:12000], "pos": sec[0]}
    return {"ticker": ticker.upper(), "fy": fy, "cik": cik, "accn": accn, "url": url,
            "period_end": period_end, "text": text, "truths": truths}


# ---------------------------------------------------------------- normalising


NUM = re.compile(r"(?P<neg>[-−(])?\s*(?P<num>\d+(?:\.\d+)?)\)?\s*(?:\(?\s*in\s+)?"
                 r"(?P<scale>thousands?|millions?|billions?|trillions?|bn|mm|k|m|b)?\b")
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9,
         "billion": 1e9, "trillion": 1e12}


def parse_number(s) -> float | None:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if not isinstance(s, str):
        return None
    m = NUM.search(s.lower().replace(",", "").replace("$", "").replace("usd", ""))
    if not m:
        return None
    scale = SCALE[m["scale"].rstrip("s")] if m["scale"] else 1
    return (-1 if m["neg"] else 1) * float(m["num"]) * scale


DATE_FORMATS = ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%m/%d/%Y", "%d %B %Y", "%d %b %Y")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|[A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}"
                     r"|\d{1,2} [A-Za-z]{3,9}\.? \d{4}")


def parse_date(s) -> dt.date | None:
    if not isinstance(s, str):
        return None
    for cand in [s, *DATE_RE.findall(s)]:
        c = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", cand.strip()).replace(".", "").replace(",", "")
        c = re.sub(r"\bSept\b", "Sep", c)
        for f in DATE_FORMATS:
            try:
                return dt.datetime.strptime(c, f).date()
            except ValueError:
                pass
    return None


def exact_match(answer, truth, typ: str) -> bool:
    if typ == "date":
        return parse_date(answer) == dt.date.fromisoformat(truth)
    if typ == "number":
        a = parse_number(answer)
        # exact means exact: "$391.0 billion" for 391,035 million is a miss
        return a is not None and math.isclose(a, float(truth), rel_tol=1e-9)
    return isinstance(answer, str) and answer.strip().strip(".").upper() == str(truth).upper()


def renderings(truth, typ: str) -> list[str]:
    """Ways a value is printed in a filing, used to find where it sits in the document."""
    if typ == "date":
        d = dt.date.fromisoformat(truth)
        return [f"{d:%B} {d.day}, {d.year}", d.isoformat(), f"{d.month}/{d.day}/{d.year}"]
    v = abs(float(truth))
    out = [f"{int(v / s):,}" for s in (1, 1e3, 1e6, 1e9) if v / s >= 1 and v / s == int(v / s)]
    return out + ([f"{v:.2f}"] if v != int(v) else [])


def locate(text: str, truth, typ: str) -> int | None:
    # shortcut: first printed occurrence counts as the fact's depth; short values like EPS can hit a lookalike
    hits = [m.start() for r in renderings(truth, typ)
            for m in [re.search(rf"(?<![\d,.]){re.escape(r)}(?![\d,]|\.\d)", text)] if m]
    return min(hits) if hits else None


# ---------------------------------------------------------------- planting and truncation

PLANTS = [  # (subject, tolerant statement, tolerant question); PLANTS_PER_DEPTH per depth, in order
    ("the Brookfield archive audit",
     "The records team noted that the Brookfield archive will move to cold storage because staff rarely request files older than ten years.",
     "Why is the Brookfield archive moving to cold storage?"),
    ("the Ridgeway office pilot",
     "Management said the Ridgeway office pilot ended early because employees preferred the older desk-booking system.",
     "Why did the Ridgeway office pilot end early?"),
    ("the Lantern Street lease review",
     "The facilities group did not renew the Lantern Street warehouse lease because rising flood insurance costs made it uneconomic.",
     "Why was the Lantern Street warehouse lease not renewed?"),
    ("the Marlow unit survey",
     "In an internal survey, most staff in the Marlow unit asked for longer onboarding because the new ordering software was confusing.",
     "What did Marlow unit staff ask for, and why?"),
    ("the travel committee review",
     "The travel committee replaced quarterly offsites with one annual meeting to cut travel time for field teams.",
     "What did the travel committee change, and why?"),
    ("the Ashcombe fleet study",
     "The Ashcombe depot switched its delivery vans to a four-day rota after drivers reported fatigue on long weekly routes.",
     "Why did the Ashcombe depot change its delivery rota?"),
    ("the Pellham badge upgrade",
     "Security staff at the Pellham site replaced paper visitor logs with badges because the logs were often illegible.",
     "Why were the paper visitor logs at Pellham replaced?"),
    ("the Corran Bay recycling trial",
     "The Corran Bay plant stopped its cardboard recycling trial because the local collector raised its fees twice in one year.",
     "Why did the Corran Bay recycling trial stop?"),
    ("the Whitlow mentoring scheme",
     "Engineers in the Whitlow group started pairing new hires with senior staff because early attrition had doubled.",
     "Why did the Whitlow group start pairing new hires with senior staff?"),
    ("the Ostend kiosk rollout",
     "The Ostend stores delayed their self-service kiosks because older customers found the touch screens hard to read.",
     "Why were the Ostend self-service kiosks delayed?"),
    ("the Harrowgate cafeteria review",
     "The Harrowgate cafeteria moved to local suppliers after a long delivery delay left it without fresh produce for a week.",
     "Why did the Harrowgate cafeteria move to local suppliers?"),
    ("the Fenwick help desk change",
     "The Fenwick help desk extended its hours into the evening because most overseas tickets arrived after the office closed.",
     "Why did the Fenwick help desk extend its hours?"),
    ("the Kestrel parking review",
     "The Kestrel campus turned half its parking lot into bike storage after a commuter survey showed most staff cycled in summer.",
     "Why did the Kestrel campus convert parking into bike storage?"),
    ("the Dunmore printing policy",
     "The Dunmore office removed most of its printers because staff printed fewer than ten pages a week on average.",
     "Why did the Dunmore office remove most of its printers?"),
    ("the Selby training audit",
     "The Selby plant moved safety training from video to hands-on sessions because quiz scores after the videos were poor.",
     "Why did the Selby plant change how it runs safety training?"),
]
PLANTS_PER_DEPTH = 3
PLANT_SPREAD = 0.02  # the facts at one depth sit 2% apart, all inside its 20%-wide heatmap bin


def planted_facts(seed: str) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for n, (subject, statement, question) in enumerate(PLANTS):
        d, j = DEPTHS[n // PLANTS_PER_DEPTH], n % PLANTS_PER_DEPTH
        target = round(d + (j - (PLANTS_PER_DEPTH - 1) / 2) * PLANT_SPREAD, 4)
        code = f"{rng.choice('QXZKV')}{rng.choice('QXZKV')}-{rng.randint(10000, 99999)}-{rng.randint(1, 9)}"
        tag = f"{round(d * 100)}{'abc'[j]}"
        out.append({"field": f"planted_code_{tag}", "kind": "exact", "source": "planted", "type": "code",
                    "target": target, "question": f"What is the internal reference code for {subject}?",
                    "truth": code, "paragraph": f"Internal reference code for {subject}: {code}. {statement}"})
        out.append({"field": f"planted_note_{tag}", "kind": "tolerant", "source": "planted", "type": "text",
                    "target": target, "question": question, "truth": statement})
    return out


def plant(text: str, inserts: list[tuple[float, str]]) -> tuple[str, list[int]]:
    """Insert paragraphs at line breaks so each lands at its depth fraction of the *final* text."""
    inserts = sorted(inserts)
    blocks = ["\n\n" + p + "\n" for _, p in inserts]
    total = len(text) + sum(map(len, blocks))
    cuts, before = [], 0
    for (d, _), b in zip(inserts, blocks):
        want = max(0, min(len(text), int(d * total) - before))
        nl = text.find("\n", want)
        cuts.append(len(text) if nl < 0 else nl)
        before += len(b)
    out, last, positions = [], 0, []
    for cut, b in zip(cuts, blocks):
        out.append(text[last:cut])
        positions.append(sum(map(len, out)) + 2)
        out.append(b)
        last = cut
    out.append(text[last:])
    return "".join(out), positions


def n_tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def truncate(text: str, max_tokens: int) -> str:
    toks = ENC.encode(text, disallowed_special=())
    return text if len(toks) <= max_tokens else ENC.decode(toks[:max_tokens])


def build_context(filing: dict, budget: int | None, fields: list[str] | None = None) -> tuple[str, list[dict]]:
    """Truncate, plant, and locate every question in the context that is sent. budget None means the full filing."""
    plants = planted_facts(f"{filing['ticker']}-{filing['fy']}")
    plant_tokens = sum(n_tokens(p["paragraph"]) + 2 for p in plants if "paragraph" in p)
    body = filing["text"] if budget is None else truncate(filing["text"], budget - RESERVE - plant_tokens)
    ctx, positions = plant(body, [(p["target"], p["paragraph"]) for p in plants if "paragraph" in p])
    pos_of = dict(zip(sorted(p["target"] for p in plants if "paragraph" in p), positions))
    qs = []
    for field, t in filing["truths"].items():
        kind = "tolerant" if t["type"] == "text" else "exact"
        pos = ctx.find(t["truth"][:200]) if kind == "tolerant" else locate(ctx, t["truth"], t["type"])
        qs.append({"field": field, "kind": kind, "source": "real", "type": t["type"], "question": t["question"],
                   "truth": t["truth"], "depth": None if pos is None or pos < 0 else pos / len(ctx)})
    for p in plants:
        qs.append({k: v for k, v in p.items() if k != "paragraph"} | {"depth": pos_of[p["target"]] / len(ctx)})
    if fields:
        qs = [q for q in qs if q["field"] in fields]
    return ctx, qs


# ---------------------------------------------------------------- models

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


def extract(cfg: dict, document: str, schema: dict) -> tuple[dict, dict]:
    """The one extraction path, used by eval and use mode alike."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"<document>\n{document}\n</document>\n\nExtract the fields."}]
    return call(cfg, messages, schema_model(schema))


class Verdict(BaseModel):
    reason: str = Field(description="One sentence.")
    correct: bool


def judge(question: str, reference: str, answer) -> tuple[bool, dict]:
    if not answer:
        return False, {"cost": 0}
    prompt = (f"Question: {question}\n\nReference passage:\n{reference}\n\nAnswer to grade:\n{answer}\n\n"
              "Is the answer correct according to the reference passage? Wording may differ; "
              "the substance must match and must not contradict the passage.")
    v, meta = call(CONFIG["judge"], [{"role": "user", "content": prompt}], Verdict)
    return v["correct"], meta | {"reason": v["reason"]}


# ---------------------------------------------------------------- runs

PRESETS = {
    "smoke": {"filings": [["AAPL", 2024]], "lengths": [8192],
              "fields": ["fiscal_year_end", "planted_code_50b", "planted_note_50b"]},
    # 45k to 208k tokens, so the 128k column is real; BAC's full length still fits a 262k window
    "full": {"filings": [["AAPL", 2024], ["MSFT", 2024], ["GE", 2024], ["KO", 2024], ["F", 2024], ["PFE", 2024],
                         ["BAC", 2024]],
             "lengths": [8192, 16384, 32768, 65536, 131072, "full"], "fields": None},
}


def plan(preset: str) -> list[tuple[dict, int | None, dict]]:
    """Every (filing, budget, model) call in a preset. Skips lengths past the filing or the model's window."""
    p = PRESETS[preset]
    tasks = []
    for ticker, fy in p["filings"]:
        filing = get_filing(ticker, fy)
        full = n_tokens(filing["text"]) + RESERVE + 500
        for cfg in CONFIG["models"]:
            for length in p["lengths"]:
                if length == "full":
                    if full <= context_limit(cfg):
                        tasks.append((filing, None, cfg))
                elif length < full and length <= context_limit(cfg):
                    tasks.append((filing, length, cfg))
    return tasks


def estimate(preset: str) -> dict:
    p, cost, unknown = PRESETS[preset], 0.0, set()
    for filing, budget, cfg in plan(preset):
        ctx, qs = build_context(filing, budget, p["fields"])
        jp = catalogue(CONFIG["judge"]).get("pricing")
        if jp:
            cost += sum(n_tokens(q["truth"]) + 500 for q in qs if q["kind"] == "tolerant") * float(jp["prompt"])
        price = catalogue(cfg).get("pricing")
        if not price:
            unknown.add(cfg["name"] + (" (Claude Code)" if cfg.get("provider") == "claude-code" else ""))
            continue
        cost += (n_tokens(ctx) + 1500) * float(price["prompt"]) + 1500 * float(price["completion"])
    return {"preset": preset, "calls": len(plan(preset)), "usd": round(cost, 4), "unpriced_models": sorted(unknown)}


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
        elif q["present"]:
            try:
                q["correct"], jm = judge(q["question"], q["truth"], q["answer"])
                q["judge_reason"] = jm.get("reason")
                judge_cost += jm.get("cost") or 0
            except Exception as e:
                q["correct"], q["judge_error"] = None, f"{type(e).__name__}: {e}"
        if q["kind"] == "tolerant" and q["source"] == "real":
            q["truth"] = q["truth"][:300] + "…"  # keep run files small; the passage is re-derivable from the filing
    return res | {"judge_cost": judge_cost}


def start_run(preset: str) -> str:
    for cfg in CONFIG["models"] + [CONFIG["judge"]]:
        if "api_key_env" in cfg and not os.environ.get(cfg["api_key_env"]):
            raise RuntimeError(f"Set {cfg['api_key_env']} first")
    rid = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{preset}"
    run = {"id": rid, "preset": preset, "status": "running", "started": dt.datetime.now().isoformat(timespec="seconds"),
           "judge": CONFIG["judge"]["model"], "models": [{k: m.get(k) for k in ("name", "model", "base_url", "provider")}
                                                         for m in CONFIG["models"]], "results": []}
    lock = threading.Lock()

    def save():
        RUNS.mkdir(exist_ok=True)
        (RUNS / f"{rid}.json").write_text(json.dumps(run, indent=1))

    def work():
        try:
            tasks = plan(preset)
            run["filings"] = list({f["url"]: {k: f[k] for k in ("ticker", "fy", "accn", "url", "period_end")}
                                   | {"full_tokens": n_tokens(f["text"])} for f, _, _ in tasks}.values())
            save()
            with ThreadPoolExecutor(4) as pool:
                for r in pool.map(lambda t: score_task(*t, PRESETS[preset]["fields"]), tasks):
                    with lock:
                        run["results"].append(r)
                        save()
            run["status"] = "done"
        except Exception as e:
            run["status"], run["error"] = "error", f"{type(e).__name__}: {e}"
        run["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        run["total_cost"] = sum((r.get("cost") or 0) + (r.get("judge_cost") or 0) for r in run["results"])
        try:
            run["mlflow_run_id"] = log_mlflow(run)
        except Exception as e:  # the run file is the record; MLflow is a copy, so a logging failure doesn't fail the run
            run["mlflow_error"] = f"{type(e).__name__}: {e}"
        save()

    save()
    threading.Thread(target=work, daemon=True).start()
    return rid


ANSWER_COLUMNS = ["ticker", "fy", "length", "field", "kind", "source", "depth", "question", "truth", "answer",
                  "correct", "judge_reason", "judge_error", "call_error"]


def answer_rows(results: list[dict]) -> list[dict]:
    """One row per question asked: what the model returned, and how it was scored."""
    return [{"ticker": r["ticker"], "fy": r["fy"], "length": r["length"], "call_error": r.get("error")}
            | {k: q.get(k) for k in ANSWER_COLUMNS if k in q} for r in results for q in r["fields"]]


def accuracy_metrics(results: list[dict]) -> dict:
    """The heatmap cells, collapsed: accuracy and count per kind x source, overall and per length."""
    cells = {}
    for r in results:
        if r.get("error"):
            continue
        length = r["length"] if r["length"] == "full" else f"{r['length'] // 1024}k"
        for q in r["fields"]:
            if q.get("correct") is None or q["depth"] is None:
                continue
            for key in (f"{q['kind']}_{q['source']}", f"{q['kind']}_{q['source']}/{length}"):
                ok, n = cells.get(key, (0, 0))
                cells[key] = (ok + bool(q["correct"]), n + 1)
    return {f"acc/{k}": ok / n for k, (ok, n) in cells.items()} | {f"n/{k}": n for k, (_, n) in cells.items()}


def log_mlflow(run: dict) -> str:
    """Log a finished run to MLflow: a parent run holding run.json, with one child run per model holding its
    metrics, call metadata and answers.csv. MLFLOW_TRACKING_URI picks the server; unset means eval/mlflow.db."""
    import csv
    import io

    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{HERE / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "filing-recall")
    if uri.startswith("sqlite:") and mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(HERE / "mlruns").as_uri())  # not relative to the cwd
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=run["id"]) as parent:
        mlflow.log_params({"preset": run["preset"], "judge": run["judge"], "status": run["status"],
                           "filings": ", ".join(f"{f['ticker']} FY{f['fy']}" for f in run.get("filings", []))})
        mlflow.log_metric("total_cost_usd", run["total_cost"])
        mlflow.log_dict(run, "run.json")
        for m in run["models"]:
            results = [r for r in run["results"] if r["model"] == m["name"]]
            with mlflow.start_run(run_name=m["name"], nested=True):
                mlflow.log_params({"model": m["model"], "endpoint": m.get("provider") or m.get("base_url")})
                calls = [{k: v for k, v in r.items() if k != "fields"} for r in results]
                mlflow.log_metrics(accuracy_metrics(results) | {
                    "calls": len(results), "calls_failed": sum(1 for r in results if r.get("error")),
                    "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in results),
                    "cost_usd": sum((r.get("cost") or 0) for r in results),
                    "notional_cost_usd": sum((r.get("notional_cost") or r.get("cost") or 0) for r in results)})
                out = io.StringIO()
                w = csv.DictWriter(out, ANSWER_COLUMNS)
                w.writeheader()
                w.writerows(answer_rows(results))
                mlflow.log_text(out.getvalue(), "answers.csv")
                mlflow.log_dict(calls, "calls.json")
    return parent.info.run_id


# ---------------------------------------------------------------- web

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

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
