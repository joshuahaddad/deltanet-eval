"""EDGAR: fetching and caching, HTML to text, passage finding, and each filing's ground truth."""

import datetime as dt
import hashlib
import json
import os
import re
import time
import warnings

import httpx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from paths import CACHE

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # inline XBRL filings are XHTML


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

# field: (question, anchor regexes tried in order). The judge's reference is the short passage that starts at the
# anchor's line, so each question has one findable answer; a filing where no anchor matches drops the question.
TOLERANT = {
    "employees": ("Approximately how many people did the company employ at the end of fiscal year {fy}?",
                  [r"\b(?:approximately|about) \d{1,3}(?:,\d{3})+ (?:[\w,-]+ ){0,3}?(?:employees|people)\b",
                   r"number of individuals employed"]),
    "seasonality": ("According to the company, in which part of the year are its sales or production seasonally "
                    "highest, and why?", [r"^(?:Business )?Seasonality(?: and Product Introductions)?(?:$|\. )"]),
    # total revenue only: segment lines ("Americas net sales increased") would answer a different question
    "revenue_drivers": ("What does the company say drove the change in its total revenue in fiscal year {fy}?",
                        [r"(?:^|[.–—] ?|, )(?:total|consolidated|net operating) (?:net sales|revenues?) "
                         r"(?:increased|decreased|grew|declined|were \$)",
                         r"^Revenue (?:increased|decreased) \$[\d.]+ (?:billion|million) or \d+%"]),
}
PASSAGE_CHARS = 1500  # the passage runs from the anchor's line to the end of the line this many characters on


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


def find_passage(text: str, anchors: list[str]) -> tuple[int, str] | None:
    for a in anchors:
        m = re.search(a, text, re.I | re.M)
        if m:
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", start + PASSAGE_CHARS)
            return start, text[start:end if end >= 0 else len(text)]
    return None


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
    for field, (q, anchors) in TOLERANT.items():
        found = find_passage(text, anchors)
        if found:
            truths[field] = {"question": q.format(fy=fy), "type": "text", "truth": found[1], "pos": found[0]}
    return {"ticker": ticker.upper(), "fy": fy, "cik": cik, "accn": accn, "url": url,
            "period_end": period_end, "text": text, "truths": truths}
