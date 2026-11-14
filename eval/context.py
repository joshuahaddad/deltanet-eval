"""The context sent: planted facts, truncation to a token budget, and where each question sits in it."""

import random

import tiktoken

from scoring import locate

# shortcut: one neutral tokenizer for every model; each run also records the prompt tokens the API reports.
ENC = tiktoken.get_encoding("o200k_base")
RESERVE = 2048  # tokens kept free for instructions, schema and the answer
DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
MIN_PASSAGE_IN_CONTEXT = 0.5  # a real tolerant question counts as present when this share of its passage was sent

PLANTS = [  # (subject, tolerant statement, tolerant question, decoy subject, answer stems); PLANTS_PER_DEPTH per
    # depth, in order. A correct tolerant answer contains every stem (lowercase substring).
    ("the Brookfield archive audit",
     "The records team noted that the Brookfield archive will move to cold storage because staff rarely request files older than ten years.",
     "Why is the Brookfield archive moving to cold storage?",
     "the Brookfield archive review", ["rarely", "request"]),
    ("the Ridgeway office pilot",
     "Management said the Ridgeway office pilot ended early because employees preferred the older desk-booking system.",
     "Why did the Ridgeway office pilot end early?",
     "the Ridgeway office trial", ["prefer", "desk"]),
    ("the Lantern Street lease review",
     "The facilities group did not renew the Lantern Street warehouse lease because rising flood insurance costs made it uneconomic.",
     "Why was the Lantern Street warehouse lease not renewed?",
     "the Lantern Street lease audit", ["flood", "insur"]),
    ("the Marlow unit survey",
     "In an internal survey, most staff in the Marlow unit asked for longer onboarding because the new ordering software was confusing.",
     "What did Marlow unit staff ask for, and why?",
     "the Marlow unit census", ["onboard", "software"]),
    ("the travel committee review",
     "The travel committee replaced quarterly offsites with one annual meeting to cut travel time for field teams.",
     "What did the travel committee change, and why?",
     "the travel committee audit", ["annual", "travel"]),
    ("the Ashcombe fleet study",
     "The Ashcombe depot switched its delivery vans to a four-day rota after drivers reported fatigue on long weekly routes.",
     "Why did the Ashcombe depot change its delivery rota?",
     "the Ashcombe fleet review", ["driver", "fatigu|tired|exhaust"]),
    ("the Pellham badge upgrade",
     "Security staff at the Pellham site replaced paper visitor logs with badges because the logs were often illegible.",
     "Why were the paper visitor logs at Pellham replaced?",
     "the Pellham badge rollout", ["log", "legib|hard to read|unread"]),
    ("the Corran Bay recycling trial",
     "The Corran Bay plant stopped its cardboard recycling trial because the local collector raised its fees twice in one year.",
     "Why did the Corran Bay recycling trial stop?",
     "the Corran Bay recycling study", ["collect", "fee|price|cost|charg"]),
    ("the Whitlow mentoring scheme",
     "Engineers in the Whitlow group started pairing new hires with senior staff because early attrition had doubled.",
     "Why did the Whitlow group start pairing new hires with senior staff?",
     "the Whitlow mentoring review", ["attrition", "doubl"]),
    ("the Ostend kiosk rollout",
     "The Ostend stores delayed their self-service kiosks because older customers found the touch screens hard to read.",
     "Why were the Ostend self-service kiosks delayed?",
     "the Ostend kiosk trial", ["older", "screen"]),
    ("the Harrowgate cafeteria review",
     "The Harrowgate cafeteria moved to local suppliers after a long delivery delay left it without fresh produce for a week.",
     "Why did the Harrowgate cafeteria move to local suppliers?",
     "the Harrowgate cafeteria audit", ["deliver", "delay"]),
    ("the Fenwick help desk change",
     "The Fenwick help desk extended its hours into the evening because most overseas tickets arrived after the office closed.",
     "Why did the Fenwick help desk extend its hours?",
     "the Fenwick help desk review", ["overseas", "ticket"]),
    ("the Kestrel parking review",
     "The Kestrel campus turned half its parking lot into bike storage after a commuter survey showed most staff cycled in summer.",
     "Why did the Kestrel campus convert parking into bike storage?",
     "the Kestrel parking survey", ["survey", "cycl"]),
    ("the Dunmore printing policy",
     "The Dunmore office removed most of its printers because staff printed fewer than ten pages a week on average.",
     "Why did the Dunmore office remove most of its printers?",
     "the Dunmore printing audit", ["page|print", "week"]),
    ("the Selby training audit",
     "The Selby plant moved safety training from video to hands-on sessions because quiz scores after the videos were poor.",
     "Why did the Selby plant change how it runs safety training?",
     "the Selby training review", ["quiz|test|score"]),
]
PLANTS_PER_DEPTH = 3
PLANT_SPREAD = 0.02  # the facts at one depth sit 2% apart, all inside its 20%-wide heatmap bin
DECOY_OFFSET = 0.01  # decoys sit between the 2%-spaced real plants, so the two never share a line
DECOY_SENTENCE = "The entry was filed for the record and needs no follow-up."


def code(rng: random.Random) -> str:
    return f"{rng.choice('QXZKV')}{rng.choice('QXZKV')}-{rng.randint(10000, 99999)}-{rng.randint(1, 9)}"


def planted_facts(seed: str) -> list[dict]:
    rng, decoy_rng = random.Random(seed), random.Random(seed + "-decoy")  # own stream keeps the real codes as before
    out = []
    for n, (subject, statement, question, decoy_subject, stems) in enumerate(PLANTS):
        i, j = n // PLANTS_PER_DEPTH, n % PLANTS_PER_DEPTH
        offset = (j - (PLANTS_PER_DEPTH - 1) / 2) * PLANT_SPREAD
        target = round(DEPTHS[i] + offset, 4)
        # the decoy sits 40 points deeper, wrapped into the 10..90 bins: a lookalike in a different part of the text
        decoy_target = round(DEPTHS[(i + 2) % len(DEPTHS)] + offset + DECOY_OFFSET, 4)
        truth, decoy = code(rng), code(decoy_rng)
        tag = f"{round(DEPTHS[i] * 100)}{'abc'[j]}"
        out.append({"field": f"planted_code_{tag}", "kind": "exact", "source": "planted", "type": "code",
                    "target": target, "question": f"What is the internal reference code for {subject}?",
                    "truth": truth, "paragraph": f"Internal reference code for {subject}: {truth}. {statement}",
                    "decoy": decoy, "decoy_target": decoy_target,
                    "decoy_paragraph": f"Internal reference code for {decoy_subject}: {decoy}. {DECOY_SENTENCE}"})
        out.append({"field": f"planted_note_{tag}", "kind": "tolerant", "source": "planted", "type": "text",
                    "target": target, "question": question, "truth": statement, "stems": stems})
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
    inserts = [(p["target"], p["paragraph"]) for p in plants if "paragraph" in p] + \
              [(p["decoy_target"], p["decoy_paragraph"]) for p in plants if "decoy_paragraph" in p]
    plant_tokens = sum(n_tokens(text) + 2 for _, text in inserts)
    body = filing["text"] if budget is None else truncate(filing["text"], budget - RESERVE - plant_tokens)
    ctx, positions = plant(body, inserts)
    pos_of = dict(zip(sorted(t for t, _ in inserts), positions))
    qs = []
    for field, t in filing["truths"].items():
        kind = "tolerant" if t["type"] == "text" else "exact"
        truth = t["truth"]
        if kind == "exact":
            pos = locate(ctx, truth, t["type"])
        else:  # body is a prefix of the filing text, so the passage is cut where the body ends
            truth = t["truth"][:max(0, len(body) - t["pos"])]
            # planting inserts only at line breaks, so the passage's first line survives intact; plants only
            # push text later, so search from the passage's own offset
            pos = ctx.find(truth.split("\n")[0][:200], t["pos"]) \
                if len(truth) >= MIN_PASSAGE_IN_CONTEXT * len(t["truth"]) else None
        qs.append({"field": field, "kind": kind, "source": "real", "type": t["type"], "question": t["question"],
                   "truth": truth, "depth": None if pos is None or pos < 0 else pos / len(ctx)})
    for p in plants:
        qs.append({k: v for k, v in p.items() if k not in ("paragraph", "decoy_paragraph")}
                  | {"depth": pos_of[p["target"]] / len(ctx)}
                  | ({"decoy_depth": pos_of[p["decoy_target"]] / len(ctx)} if "decoy" in p else {}))
    if fields:
        qs = [q for q in qs if q["field"] in fields]
    return ctx, qs
