#!/usr/bin/env python3
"""
creator_ranker.py - Rank OnlyFans-style creators by genuine fan interaction.

Ranks creators from a local JSON dataset, filtered by language, reviews and
other criteria. Creators who use AI chatbots, auto-DM/mass-message tools or
outsourced "chatter" agencies are HARD-EXCLUDED from the ranking: only
creators who personally and directly interact with their customers are
ranked.

There is no public OnlyFans API and scraping violates their ToS, so this tool
operates on a dataset you maintain yourself (see creators_sample.json for the
schema). Automation is detected two ways:
  1. Explicit flags on the profile (uses_ai_chatbot / uses_auto_dm /
     uses_agency_chatters / replies_personally).
  2. Heuristic text analysis of customer reviews: red-flag phrases such as
     "bot", "copy paste", "scripted", "agency chatter" raise a suspicion
     score, while human signals such as "voice note", "remembers me",
     "genuine conversation" lower it.

Usage examples:
    python3 creator_ranker.py creators_sample.json
    python3 creator_ranker.py creators_sample.json --language en es --min-reviews 5
    python3 creator_ranker.py creators_sample.json --min-rating 4.2 --max-price 15 --top 5
    python3 creator_ranker.py creators_sample.json --show-excluded
    python3 creator_ranker.py creators_sample.json --explain nina_fit
    python3 creator_ranker.py creators_sample.json --export ranked.csv

No third-party dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Automation / authenticity heuristics
# ---------------------------------------------------------------------------

# Phrases in customer reviews that suggest automated or outsourced chat.
BOT_RED_FLAGS = {
    r"\bbots?\b": 3.0,
    r"\bchat\s*gpt\b|\ba\.?i\.?\s*(chat|bot|repl)": 3.0,
    r"\bscripted\b": 2.5,
    r"copy[\s\-]?past(e|ed|ing)": 2.5,
    r"\bautomated\b|\bautomation\b": 2.5,
    r"mass[\s\-]?(message|dm)": 2.0,
    r"\bagenc(y|ies)\b": 2.0,
    r"\bchatters?\b": 2.0,
    r"same (message|reply|response) (every|each|to every)": 2.0,
    r"generic (replies|responses|messages)": 1.5,
    r"\btemplate(d)?\b": 1.5,
    r"(doesn'?t|never|won'?t) (actually |really )?(read|reply|respond|answer)": 1.5,
    r"not (really |actually )?her\b": 2.5,
    r"someone else (is )?(replying|texting|answering)": 2.5,
}

# Phrases that suggest a real human is personally replying.
HUMAN_SIGNALS = {
    r"voice (note|message|memo)s?": 2.0,
    r"remember(s|ed)? (me|my|our|what)": 2.0,
    r"\bgenuine(ly)?\b": 1.5,
    r"real (conversation|person|chat)s?": 2.0,
    r"personal(ly|ized|ised)? (repl|respon|message|touch)": 1.5,
    r"(replied|responds?|answers?) (to me )?(personally|herself|himself)": 2.0,
    r"knows my name": 2.0,
    r"actually (talks?|chats?|listens?|cares?|reads?)": 1.5,
    r"custom (video|content|request)s?": 1.0,
    r"live ?stream(s|ed|ing)?": 0.5,
    r"(quick|fast|prompt) (repl|respon)": 0.5,
}

SUSPICION_THRESHOLD = 4.0  # cumulative red-flag weight that triggers exclusion

# "not scripted", "never feels automated" etc. negate a red flag.
NEGATORS = re.compile(r"\b(not|never|no|isn'?t|aren'?t|wasn'?t|without|zero|don'?t feel)\s+\S*\s*$")

DEFAULT_WEIGHTS = {
    "reviews": 0.35,       # Bayesian average star rating
    "volume": 0.10,        # review count + recency
    "interaction": 0.35,   # reply rate, response time, voice notes, customs
    "activity": 0.10,      # posting cadence, last seen
    "authenticity": 0.10,  # confidence the chat is genuinely human
}

BAYESIAN_PRIOR_RATING = 3.5
BAYESIAN_PRIOR_WEIGHT = 10  # pseudo-review count pulling small samples to prior


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Creator:
    raw: dict
    username: str
    display_name: str
    languages: list[str]
    price_monthly: float
    categories: list[str]
    replies_personally: bool
    uses_ai_chatbot: bool
    uses_auto_dm: bool
    uses_agency_chatters: bool
    avg_response_hours: float
    reply_rate: float
    sends_voice_notes: bool
    does_live_streams: bool
    custom_content: bool
    posts_per_week: float
    last_active_days: int
    reviews: list[dict]

    # computed
    avg_rating: float = 0.0
    bayes_rating: float = 0.0
    suspicion_score: float = 0.0
    human_signal_score: float = 0.0
    suspicion_hits: list[str] = field(default_factory=list)
    human_hits: list[str] = field(default_factory=list)
    score: float = 0.0
    breakdown: dict = field(default_factory=dict)
    exclusion_reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Creator":
        inter = d.get("interaction", {})
        return cls(
            raw=d,
            username=d["username"],
            display_name=d.get("display_name", d["username"]),
            languages=[l.lower() for l in d.get("languages", [])],
            price_monthly=float(d.get("price_monthly", 0.0)),
            categories=[c.lower() for c in d.get("categories", [])],
            replies_personally=bool(inter.get("replies_personally", False)),
            uses_ai_chatbot=bool(inter.get("uses_ai_chatbot", False)),
            uses_auto_dm=bool(inter.get("uses_auto_dm", False)),
            uses_agency_chatters=bool(inter.get("uses_agency_chatters", False)),
            avg_response_hours=float(inter.get("avg_response_hours", 48.0)),
            reply_rate=float(inter.get("reply_rate", 0.0)),
            sends_voice_notes=bool(inter.get("sends_voice_notes", False)),
            does_live_streams=bool(inter.get("does_live_streams", False)),
            custom_content=bool(inter.get("custom_content", False)),
            posts_per_week=float(d.get("posts_per_week", 0.0)),
            last_active_days=int(d.get("last_active_days", 999)),
            reviews=d.get("reviews", []),
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def analyze_reviews(creator: Creator) -> None:
    """Compute rating stats and scan review text for bot/human signals."""
    ratings = [float(r.get("rating", 0)) for r in creator.reviews if r.get("rating") is not None]
    n = len(ratings)
    creator.avg_rating = sum(ratings) / n if n else 0.0
    creator.bayes_rating = (
        (BAYESIAN_PRIOR_WEIGHT * BAYESIAN_PRIOR_RATING + sum(ratings))
        / (BAYESIAN_PRIOR_WEIGHT + n)
    )

    suspicion, human = 0.0, 0.0
    for review in creator.reviews:
        text = (review.get("text") or "").lower()
        if not text:
            continue
        weight = 1.5 if review.get("verified_purchase") else 1.0
        for pattern, w in BOT_RED_FLAGS.items():
            m = re.search(pattern, text)
            if m and not NEGATORS.search(text[max(0, m.start() - 24):m.start()]):
                suspicion += w * weight
                creator.suspicion_hits.append(_snippet(text, pattern))
        for pattern, w in HUMAN_SIGNALS.items():
            if re.search(pattern, text):
                human += w * weight
                creator.human_hits.append(_snippet(text, pattern))

    # Strong human evidence partially offsets weak/ambiguous red flags.
    creator.suspicion_score = max(0.0, suspicion - 0.4 * human)
    creator.human_signal_score = human


def _snippet(text: str, pattern: str, ctx: int = 28) -> str:
    m = re.search(pattern, text)
    if not m:
        return ""
    start, end = max(0, m.start() - ctx), min(len(text), m.end() + ctx)
    return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")


def authenticity_check(creator: Creator, allow_suspected: bool) -> bool:
    """Hard gate: only creators who personally interact with fans pass."""
    if creator.uses_ai_chatbot:
        creator.exclusion_reasons.append("uses an AI chatbot")
    if creator.uses_auto_dm:
        creator.exclusion_reasons.append("uses auto-DM / mass-message tools")
    if creator.uses_agency_chatters:
        creator.exclusion_reasons.append("chat outsourced to agency chatters")
    if not creator.replies_personally:
        creator.exclusion_reasons.append("does not personally reply to fans")
    if creator.suspicion_score >= SUSPICION_THRESHOLD and not allow_suspected:
        creator.exclusion_reasons.append(
            f"reviews suggest automated/outsourced chat "
            f"(suspicion {creator.suspicion_score:.1f} >= {SUSPICION_THRESHOLD})"
        )
    return not creator.exclusion_reasons


def compute_score(creator: Creator, weights: dict) -> None:
    """Composite 0-100 score from reviews, interaction, activity, authenticity."""
    n_reviews = len(creator.reviews)

    # 1) Review quality: Bayesian average mapped 1..5 -> 0..1
    review_q = max(0.0, min(1.0, (creator.bayes_rating - 1.0) / 4.0))

    # 2) Review volume + recency: log-scaled count, decayed by review age
    volume = min(1.0, math.log1p(n_reviews) / math.log1p(50))
    recency = _review_recency_factor(creator.reviews)
    volume_q = 0.7 * volume + 0.3 * recency

    # 3) Interaction quality
    response_q = max(0.0, 1.0 - min(creator.avg_response_hours, 72.0) / 72.0)
    interaction_q = (
        0.40 * max(0.0, min(1.0, creator.reply_rate))
        + 0.30 * response_q
        + 0.12 * (1.0 if creator.sends_voice_notes else 0.0)
        + 0.10 * (1.0 if creator.custom_content else 0.0)
        + 0.08 * (1.0 if creator.does_live_streams else 0.0)
    )

    # 4) Activity: posting cadence (saturates at 7/week) and recency of login
    cadence = min(1.0, creator.posts_per_week / 7.0)
    seen = max(0.0, 1.0 - min(creator.last_active_days, 30) / 30.0)
    activity_q = 0.6 * cadence + 0.4 * seen

    # 5) Authenticity confidence: human evidence up, residual suspicion down
    auth_q = max(0.0, min(1.0, 0.5 + creator.human_signal_score / 12.0
                          - creator.suspicion_score / (2 * SUSPICION_THRESHOLD)))

    creator.breakdown = {
        "reviews": review_q,
        "volume": volume_q,
        "interaction": interaction_q,
        "activity": activity_q,
        "authenticity": auth_q,
    }
    creator.score = 100.0 * sum(weights[k] * v for k, v in creator.breakdown.items())


def _review_recency_factor(reviews: list[dict]) -> float:
    """1.0 if the newest review is from today, decaying to 0 over ~a year."""
    newest = None
    for r in reviews:
        d = r.get("date")
        if not d:
            continue
        try:
            parsed = datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            continue
        if newest is None or parsed > newest:
            newest = parsed
    if newest is None:
        return 0.0
    age_days = max(0, (date.today() - newest).days)
    return math.exp(-age_days / 180.0)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filters(creators: list[Creator], args) -> tuple[list[Creator], list[Creator]]:
    kept, dropped = [], []
    wanted_langs = {l.lower() for l in args.language} if args.language else None
    wanted_cats = {c.lower() for c in args.category} if args.category else None

    for c in creators:
        reasons = []
        if wanted_langs and not wanted_langs & set(c.languages):
            reasons.append(f"language not in [{', '.join(sorted(wanted_langs))}]")
        if wanted_cats and not wanted_cats & set(c.categories):
            reasons.append(f"category not in [{', '.join(sorted(wanted_cats))}]")
        if len(c.reviews) < args.min_reviews:
            reasons.append(f"only {len(c.reviews)} reviews (< {args.min_reviews})")
        if c.avg_rating < args.min_rating:
            reasons.append(f"avg rating {c.avg_rating:.2f} (< {args.min_rating})")
        if args.max_price is not None and c.price_monthly > args.max_price:
            reasons.append(f"price ${c.price_monthly:.2f} (> ${args.max_price:.2f})")
        if args.max_response_hours is not None and c.avg_response_hours > args.max_response_hours:
            reasons.append(
                f"avg response {c.avg_response_hours:.1f}h (> {args.max_response_hours}h)")
        if args.strict and not (c.sends_voice_notes or c.human_signal_score >= 3.0):
            reasons.append("strict mode: no strong human-interaction evidence")

        if reasons:
            c.exclusion_reasons.extend(reasons)
            dropped.append(c)
        else:
            kept.append(c)
    return kept, dropped


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_table(creators: list[Creator]) -> str:
    headers = ["#", "Creator", "Score", "Rating", "Reviews", "Langs",
               "Reply%", "Resp(h)", "$/mo", "Human signals"]
    rows = []
    for i, c in enumerate(creators, 1):
        rows.append([
            str(i),
            f"{c.display_name} (@{c.username})",
            f"{c.score:.1f}",
            f"{c.avg_rating:.2f}",
            str(len(c.reviews)),
            ",".join(c.languages),
            f"{c.reply_rate * 100:.0f}",
            f"{c.avg_response_hours:.1f}",
            f"{c.price_monthly:.2f}",
            _human_badges(c),
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths)), sep]
    lines += [" | ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


def _human_badges(c: Creator) -> str:
    badges = []
    if c.sends_voice_notes:
        badges.append("voice")
    if c.does_live_streams:
        badges.append("live")
    if c.custom_content:
        badges.append("customs")
    if c.human_signal_score >= 3.0:
        badges.append("verified-by-reviews")
    return " ".join(badges) or "-"


def render_explain(c: Creator, weights: dict) -> str:
    lines = [
        f"Score breakdown for {c.display_name} (@{c.username}) - total {c.score:.1f}/100",
        "-" * 64,
    ]
    for key, value in c.breakdown.items():
        w = weights[key]
        lines.append(f"  {key:<13} {value:.3f} x weight {w:.2f} = {100 * value * w:5.1f} pts")
    lines.append("")
    lines.append(f"  avg rating      : {c.avg_rating:.2f} ({len(c.reviews)} reviews, "
                 f"bayesian {c.bayes_rating:.2f})")
    lines.append(f"  reply rate      : {c.reply_rate * 100:.0f}%   "
                 f"avg response: {c.avg_response_hours:.1f}h")
    lines.append(f"  bot suspicion   : {c.suspicion_score:.1f} "
                 f"(threshold {SUSPICION_THRESHOLD})")
    lines.append(f"  human signals   : {c.human_signal_score:.1f}")
    if c.human_hits:
        lines.append("  evidence of direct interaction (from reviews):")
        for hit in c.human_hits[:5]:
            lines.append(f'    - "{hit}"')
    if c.suspicion_hits:
        lines.append("  automation red flags (from reviews):")
        for hit in c.suspicion_hits[:5]:
            lines.append(f'    - "{hit}"')
    return "\n".join(lines)


def export_results(creators: list[Creator], path: str) -> None:
    fields = ["rank", "username", "display_name", "score", "avg_rating", "n_reviews",
              "languages", "reply_rate", "avg_response_hours", "price_monthly",
              "human_signal_score", "suspicion_score"]
    rows = [{
        "rank": i,
        "username": c.username,
        "display_name": c.display_name,
        "score": round(c.score, 2),
        "avg_rating": round(c.avg_rating, 2),
        "n_reviews": len(c.reviews),
        "languages": "|".join(c.languages),
        "reply_rate": c.reply_rate,
        "avg_response_hours": c.avg_response_hours,
        "price_monthly": c.price_monthly,
        "human_signal_score": round(c.human_signal_score, 2),
        "suspicion_score": round(c.suspicion_score, 2),
    } for i, c in enumerate(creators, 1)]

    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nExported {len(rows)} ranked creators to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Rank creators by genuine, direct fan interaction. "
                    "AI bots, auto-DM tools and agency chatters are excluded.")
    p.add_argument("dataset", help="path to creators JSON file")
    p.add_argument("--language", "-l", nargs="+", metavar="LANG",
                   help="only creators speaking at least one of these languages (e.g. en es)")
    p.add_argument("--category", "-c", nargs="+", metavar="CAT",
                   help="only creators in at least one of these categories")
    p.add_argument("--min-rating", type=float, default=0.0,
                   help="minimum average review rating (1-5)")
    p.add_argument("--min-reviews", type=int, default=0,
                   help="minimum number of reviews")
    p.add_argument("--max-price", type=float, default=None,
                   help="maximum monthly subscription price")
    p.add_argument("--max-response-hours", type=float, default=None,
                   help="maximum average DM response time in hours")
    p.add_argument("--strict", action="store_true",
                   help="require strong human-interaction evidence (voice notes or "
                        "review-verified personal replies)")
    p.add_argument("--allow-suspected", action="store_true",
                   help="keep creators whose reviews heuristically suggest automation "
                        "(explicitly flagged bot/agency users are always excluded)")
    p.add_argument("--top", type=int, default=None, help="show only the top N")
    p.add_argument("--sort", choices=["score", "rating", "reviews", "response", "price"],
                   default="score", help="sort key (default: score)")
    p.add_argument("--weights", metavar="K=V", nargs="+",
                   help="override score weights, e.g. reviews=0.5 interaction=0.3 "
                        f"(keys: {', '.join(DEFAULT_WEIGHTS)})")
    p.add_argument("--explain", metavar="USERNAME",
                   help="print the full score breakdown for one creator")
    p.add_argument("--show-excluded", action="store_true",
                   help="list excluded creators and why they were cut")
    p.add_argument("--export", metavar="FILE",
                   help="write ranking to a .csv or .json file")
    return p.parse_args(argv)


def resolve_weights(args) -> dict:
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        for item in args.weights:
            if "=" not in item:
                sys.exit(f"error: bad --weights entry '{item}' (expected key=value)")
            key, _, val = item.partition("=")
            if key not in weights:
                sys.exit(f"error: unknown weight '{key}' (valid: {', '.join(weights)})")
            try:
                weights[key] = float(val)
            except ValueError:
                sys.exit(f"error: weight '{key}' must be a number, got '{val}'")
        total = sum(weights.values())
        if total <= 0:
            sys.exit("error: weights must sum to a positive number")
        weights = {k: v / total for k, v in weights.items()}
    return weights


SORT_KEYS = {
    "score": lambda c: -c.score,
    "rating": lambda c: -c.avg_rating,
    "reviews": lambda c: -len(c.reviews),
    "response": lambda c: c.avg_response_hours,
    "price": lambda c: c.price_monthly,
}


def main(argv=None) -> int:
    args = parse_args(argv)
    weights = resolve_weights(args)

    try:
        with open(args.dataset, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"error: cannot read dataset '{args.dataset}': {exc}")

    records = data["creators"] if isinstance(data, dict) and "creators" in data else data
    creators = [Creator.from_dict(d) for d in records]
    for c in creators:
        analyze_reviews(c)

    # Hard authenticity gate: direct, personal interaction only.
    authentic = [c for c in creators if authenticity_check(c, args.allow_suspected)]
    bots = [c for c in creators if c not in authentic]

    kept, filtered = apply_filters(authentic, args)
    for c in kept:
        compute_score(c, weights)
    kept.sort(key=SORT_KEYS[args.sort])
    ranked = kept[: args.top] if args.top else kept

    print(f"Loaded {len(creators)} creators from {args.dataset}")
    print(f"Excluded {len(bots)} for automation/no direct interaction, "
          f"{len(filtered)} by filters. Ranking {len(kept)}.\n")

    if ranked:
        print(render_table(ranked))
    else:
        print("No creators match the current filters.")

    if args.show_excluded and (bots or filtered):
        print("\nExcluded creators:")
        for c in bots + filtered:
            print(f"  @{c.username}: {'; '.join(c.exclusion_reasons)}")

    if args.explain:
        match = next((c for c in kept if c.username.lower() == args.explain.lower()), None)
        if match:
            print("\n" + render_explain(match, weights))
        else:
            dropped = next((c for c in bots + filtered
                            if c.username.lower() == args.explain.lower()), None)
            if dropped:
                print(f"\n@{dropped.username} was excluded: "
                      f"{'; '.join(dropped.exclusion_reasons)}")
            else:
                print(f"\nNo creator '{args.explain}' in dataset.")

    if args.export and ranked:
        export_results(ranked, args.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
