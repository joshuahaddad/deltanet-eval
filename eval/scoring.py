"""Scoring: number and date parsing, exact match, outcomes, locating facts, Wilson intervals and agreement."""

import datetime as dt
import math
import re


NUM = re.compile(r"(?P<neg>[-−(])?\s*(?P<num>\d+(?:\.\d+)?)\)?\s*(?:\(?\s*in\s+)?"
                 r"(?P<scale>thousands?|millions?|billions?|trillions?|bn|mm|k|m|b)?\b")
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mm": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9,
         "billion": 1e9, "trillion": 1e12}


def _num(s: str):
    return NUM.search(s.lower().replace(",", "").replace("$", "").replace("usd", ""))


def parse_number(s) -> float | None:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if not isinstance(s, str):
        return None
    m = _num(s)
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


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def near_miss(answer, truth, typ: str) -> bool:
    """Close but not exact. Numbers: the truth rounded to the precision the answer was given in ("$391.0 billion"
    for 391,035 million), or within 1%. Dates: within 7 days (52/53-week fiscal years end on a weekday near the
    month end) or the right month and year. Codes: one or two edits away."""
    if typ == "date":
        a, t = parse_date(answer), dt.date.fromisoformat(truth)
        return a is not None and (abs((a - t).days) <= 7 or (a.year, a.month) == (t.year, t.month))
    if typ == "number":
        a, t = parse_number(answer), float(truth)
        if a is None or a == 0 or (a < 0) != (t < 0):
            return False
        if abs(a - t) <= 0.01 * abs(t):
            return True
        m = _num(answer) if isinstance(answer, str) else None
        if not m:
            return False
        step = 10 ** -len(m["num"].partition(".")[2]) * (SCALE[m["scale"].rstrip("s")] if m["scale"] else 1)
        return abs(a - t) <= step / 2 * (1 + 1e-9)
    return 1 <= edit_distance(str(answer).strip().strip(".").upper(), str(truth).upper()) <= 2


OUTCOMES = ("correct", "near_miss", "wrong", "abstained")


def outcome(answer, truth, typ: str) -> str:
    if answer is None or str(answer).strip().lower() in ("", "null", "none", "n/a"):
        return "abstained"
    if exact_match(answer, truth, typ):
        return "correct"
    return "near_miss" if near_miss(answer, truth, typ) else "wrong"


def wilson(ok: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for ok successes out of n; unlike the normal interval it stays inside [0, 1]."""
    p, d = ok / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def agreement(a: list[bool], b: list[bool]) -> dict:
    """Raw agreement and Cohen's kappa between two judges' verdicts on the same answers."""
    n = len(a)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    # both judges giving one verdict to everything leaves kappa undefined
    return {"n": n, "agreement": po, "kappa": None if pe == 1 else (po - pe) / (1 - pe)}


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
