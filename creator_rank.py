#!/usr/bin/env python3
"""
creator_rank.py
================

A transparent ranking engine for adult-content creators (e.g. OnlyFans).

The whole point of this tool is to surface creators who *personally* engage with
their subscribers -- real human conversation -- and to push down or exclude
accounts that are run by AI chatbots, "chatter" agencies, or other automation.
You can also filter by spoken language, review volume and review score.

----------------------------------------------------------------------------
IMPORTANT - WHERE THE DATA COMES FROM
----------------------------------------------------------------------------
This tool does NOT scrape OnlyFans. OnlyFans has no public API, requires an
authenticated session, and its Terms of Service prohibit scraping. Doing so
would also raise privacy/consent problems for the creators involved.

Instead, this engine ranks a dataset that *you* maintain and supply (a CSV or
JSON file). You populate it from sources you are allowed to use: public
creator self-reported info, your own subscription experience, opt-in creator
submissions, or legitimate review aggregators. The "is this a real human or a
bot" signal therefore comes from the attributes in your dataset
(`human_verified`, `ai_bot_detected`, `personal_reply_rate`, ...), not from any
covert detection of a private platform.

The value this script adds is the *ranking methodology*: a defensible,
tunable, explainable score - not the data collection.

----------------------------------------------------------------------------
SCORING MODEL (high level)
----------------------------------------------------------------------------
Each creator gets a composite score in [0, 100] built from three pillars:

  1. Review quality   - a Bayesian (shrinkage) weighted rating so that a
                        creator with 5000 reviews at 4.8 outranks one with
                        3 reviews at 5.0.
  2. Human engagement - how much the creator personally interacts: verified
                        human status, share of DMs answered personally, reply
                        personalization, and responsiveness.
  3. Trust / safety   - penalties for scam/chargeback reports.

A creator flagged as using an AI bot (or not verified as human, when
--require-human is on) is excluded by default, because that is the explicit
requirement: only rank people who genuinely talk to their customers.

Run `python3 creator_rank.py --help` for usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Creator:
    """A single creator profile and the signals we rank on.

    Only `handle` is strictly required; everything else has sensible defaults
    so partially-filled datasets still load. Missing engagement signals simply
    pull a creator toward the neutral middle of the model rather than crashing.
    """

    handle: str
    display_name: str = ""
    languages: list[str] = field(default_factory=list)

    # Review signals
    num_reviews: int = 0
    avg_rating: float = 0.0  # on a 0-5 scale

    # Human-engagement signals (the core of what the user asked for)
    human_verified: bool = False        # confirmed a real person handles the chats
    ai_bot_detected: bool = False       # evidence of AI chatbot / automation
    personal_reply_rate: float = 0.0    # 0..1, share of DMs answered personally
    personalization: float = 0.0        # 0..1, how tailored replies are (vs canned)
    median_response_minutes: Optional[float] = None  # responsiveness; lower is better

    # Trust / safety
    scam_reports: int = 0               # chargeback/scam/cat-fishing complaints

    # Free-form
    notes: str = ""

    def normalized_languages(self) -> list[str]:
        return [str(l).strip().lower() for l in self.languages if str(l).strip()]


# ---------------------------------------------------------------------------
# Scoring weights (tunable)
# ---------------------------------------------------------------------------
@dataclass
class Weights:
    """Relative weights of each pillar. They are normalized internally, so only
    the *ratios* matter. Defaults emphasize genuine human engagement, which is
    the headline requirement."""

    review_quality: float = 0.40
    human_engagement: float = 0.45
    trust_safety: float = 0.15

    # Sub-weights inside the human-engagement pillar (also normalized).
    w_verified: float = 0.30
    w_reply_rate: float = 0.30
    w_personalization: float = 0.25
    w_responsiveness: float = 0.15

    # Bayesian prior strength: how many "average" reviews to blend in.
    # Larger -> low-review creators are pulled harder toward the global mean.
    prior_strength: float = 25.0

    # Response time (minutes) that maps to a "good" responsiveness score.
    # Used in an exponential decay: faster than this scores high.
    response_halflife_minutes: float = 60.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _bayesian_rating(num_reviews: int, avg_rating: float, global_mean: float,
                     prior_strength: float) -> float:
    """IMDb-style weighted rating with shrinkage toward the global mean.

    WR = (v/(v+m)) * R + (m/(v+m)) * C
    where v = num_reviews, R = avg_rating, C = global mean, m = prior strength.
    Returns a value on the same 0-5 scale.
    """
    v = max(0, num_reviews)
    m = max(0.0, prior_strength)
    if v + m == 0:
        return global_mean
    return (v / (v + m)) * avg_rating + (m / (v + m)) * global_mean


def _responsiveness_score(median_minutes: Optional[float], halflife: float) -> float:
    """Map response time to [0,1]. Unknown -> neutral 0.5.

    Exponential decay: at `halflife` minutes the score is 0.5, instant is ~1.0,
    very slow trends to 0.
    """
    if median_minutes is None:
        return 0.5
    if median_minutes <= 0:
        return 1.0
    return math.exp(-math.log(2) * median_minutes / max(1e-9, halflife))


def _engagement_score(c: Creator, w: Weights) -> float:
    """Human-engagement pillar in [0,1]."""
    verified = 1.0 if c.human_verified else 0.0
    reply_rate = _clamp01(c.personal_reply_rate)
    personalization = _clamp01(c.personalization)
    responsiveness = _responsiveness_score(c.median_response_minutes,
                                           w.response_halflife_minutes)

    sub_total = (w.w_verified + w.w_reply_rate +
                 w.w_personalization + w.w_responsiveness) or 1.0
    score = (
        w.w_verified * verified
        + w.w_reply_rate * reply_rate
        + w.w_personalization * personalization
        + w.w_responsiveness * responsiveness
    ) / sub_total
    return _clamp01(score)


def _trust_score(c: Creator) -> float:
    """Trust/safety pillar in [0,1]. Decays as scam reports accumulate."""
    # Each report meaningfully erodes trust; saturates rather than going negative.
    return 1.0 / (1.0 + 0.5 * max(0, c.scam_reports))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass
class ScoredCreator:
    creator: Creator
    score: float                 # 0..100 composite
    review_component: float      # 0..1
    engagement_component: float  # 0..1
    trust_component: float       # 0..1
    bayesian_rating: float       # 0..5


def score_creators(creators: list[Creator], weights: Weights) -> list[ScoredCreator]:
    """Compute composite scores for a list of creators."""
    rated = [c for c in creators if c.num_reviews > 0 and c.avg_rating > 0]
    if rated:
        global_mean = sum(c.avg_rating for c in rated) / len(rated)
    else:
        global_mean = 4.0  # neutral-ish prior on a 5-point scale

    total_w = (weights.review_quality + weights.human_engagement +
               weights.trust_safety) or 1.0

    results: list[ScoredCreator] = []
    for c in creators:
        bayes = _bayesian_rating(c.num_reviews, c.avg_rating, global_mean,
                                 weights.prior_strength)
        review_component = _clamp01(bayes / 5.0)
        engagement_component = _engagement_score(c, weights)
        trust_component = _trust_score(c)

        composite = (
            weights.review_quality * review_component
            + weights.human_engagement * engagement_component
            + weights.trust_safety * trust_component
        ) / total_w

        results.append(ScoredCreator(
            creator=c,
            score=round(100.0 * composite, 2),
            review_component=round(review_component, 4),
            engagement_component=round(engagement_component, 4),
            trust_component=round(trust_component, 4),
            bayesian_rating=round(bayes, 3),
        ))

    results.sort(key=lambda s: s.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
@dataclass
class Filters:
    language: Optional[str] = None
    min_reviews: int = 0
    min_rating: float = 0.0
    require_human: bool = True       # only creators verified as real humans
    exclude_ai_bots: bool = True     # drop accounts flagged as AI/automated
    min_reply_rate: float = 0.0      # 0..1


def apply_filters(creators: Iterable[Creator], f: Filters) -> list[Creator]:
    out: list[Creator] = []
    lang = f.language.strip().lower() if f.language else None
    for c in creators:
        if f.exclude_ai_bots and c.ai_bot_detected:
            continue
        if f.require_human and not c.human_verified:
            continue
        if lang and lang not in c.normalized_languages():
            continue
        if c.num_reviews < f.min_reviews:
            continue
        if c.avg_rating < f.min_rating:
            continue
        if c.personal_reply_rate < f.min_reply_rate:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------
def _coerce_creator(d: dict) -> Creator:
    """Build a Creator from a loose dict, tolerating string/None values."""
    def as_bool(v, default=False):
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}

    def as_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def as_int(v, default=0):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    langs = d.get("languages", [])
    if isinstance(langs, str):
        langs = [s for s in (p.strip() for p in langs.replace("|", ",").split(",")) if s]

    resp = d.get("median_response_minutes", None)
    resp = as_float(resp, None) if resp not in (None, "") else None

    return Creator(
        handle=str(d.get("handle") or d.get("username") or d.get("name") or "").strip(),
        display_name=str(d.get("display_name") or d.get("name") or "").strip(),
        languages=list(langs),
        num_reviews=as_int(d.get("num_reviews")),
        avg_rating=as_float(d.get("avg_rating")),
        human_verified=as_bool(d.get("human_verified")),
        ai_bot_detected=as_bool(d.get("ai_bot_detected")),
        personal_reply_rate=as_float(d.get("personal_reply_rate")),
        personalization=as_float(d.get("personalization")),
        median_response_minutes=resp,
        scam_reports=as_int(d.get("scam_reports")),
        notes=str(d.get("notes") or "").strip(),
    )


def load_creators(path: str) -> list[Creator]:
    """Load creators from a .json or .csv file."""
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = data.get("creators", [])
        return [_coerce_creator(d) for d in data]

    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [_coerce_creator(row) for row in reader]

    raise ValueError(f"Unsupported file type (use .json or .csv): {path}")


def export_results(scored: list[ScoredCreator], path: str) -> None:
    rows = []
    for rank, s in enumerate(scored, start=1):
        row = asdict(s.creator)
        row["languages"] = ",".join(s.creator.languages)
        row.update({
            "rank": rank,
            "score": s.score,
            "bayesian_rating": s.bayesian_rating,
            "review_component": s.review_component,
            "engagement_component": s.engagement_component,
            "trust_component": s.trust_component,
        })
        rows.append(row)

    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
    elif path.lower().endswith(".csv"):
        if not rows:
            open(path, "w").close()
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError(f"Unsupported export type (use .json or .csv): {path}")


# ---------------------------------------------------------------------------
# Sample data (for demo / --demo)
# ---------------------------------------------------------------------------
def sample_creators() -> list[Creator]:
    """Illustrative, fictional dataset so the tool is runnable out of the box.
    These are NOT real people; values are made up to demonstrate the ranking."""
    return [
        Creator("aurora_real", "Aurora", ["english", "spanish"],
                num_reviews=1840, avg_rating=4.8, human_verified=True,
                ai_bot_detected=False, personal_reply_rate=0.95,
                personalization=0.9, median_response_minutes=12, scam_reports=0,
                notes="Replies personally, voice notes."),
        Creator("bella_bot", "Bella", ["english"],
                num_reviews=5200, avg_rating=4.9, human_verified=False,
                ai_bot_detected=True, personal_reply_rate=0.1,
                personalization=0.2, median_response_minutes=1, scam_reports=3,
                notes="Suspected agency-run AI chatter."),
        Creator("chiara_it", "Chiara", ["italian", "english"],
                num_reviews=430, avg_rating=4.7, human_verified=True,
                ai_bot_detected=False, personal_reply_rate=0.88,
                personalization=0.85, median_response_minutes=40, scam_reports=0),
        Creator("dasha_ru", "Dasha", ["russian", "english"],
                num_reviews=95, avg_rating=5.0, human_verified=True,
                ai_bot_detected=False, personal_reply_rate=0.8,
                personalization=0.7, median_response_minutes=90, scam_reports=1),
        Creator("emma_new", "Emma", ["english", "french"],
                num_reviews=6, avg_rating=5.0, human_verified=True,
                ai_bot_detected=False, personal_reply_rate=0.7,
                personalization=0.6, median_response_minutes=120, scam_reports=0,
                notes="New creator, tiny sample size."),
        Creator("fleur_fr", "Fleur", ["french"],
                num_reviews=760, avg_rating=4.5, human_verified=True,
                ai_bot_detected=False, personal_reply_rate=0.6,
                personalization=0.55, median_response_minutes=180, scam_reports=0),
        Creator("gina_agency", "Gina", ["english", "spanish"],
                num_reviews=2100, avg_rating=4.6, human_verified=False,
                ai_bot_detected=True, personal_reply_rate=0.3,
                personalization=0.3, median_response_minutes=5, scam_reports=2,
                notes="Managed account."),
    ]


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _yn(b: bool) -> str:
    return "yes" if b else "no"


def print_ranking(scored: list[ScoredCreator], top: Optional[int] = None,
                  verbose: bool = False) -> None:
    if not scored:
        print("No creators matched the given filters.")
        return

    rows = scored[:top] if top else scored
    header = f"{'#':>3}  {'score':>6}  {'rating':>6}  {'revs':>6}  {'langs':<18}  handle"
    print(header)
    print("-" * max(len(header), 60))
    for i, s in enumerate(rows, start=1):
        c = s.creator
        langs = ",".join(c.normalized_languages())[:18]
        name = c.display_name or c.handle
        print(f"{i:>3}  {s.score:>6.1f}  {s.bayesian_rating:>6.2f}  "
              f"{c.num_reviews:>6}  {langs:<18}  @{c.handle} ({name})")
        if verbose:
            print(f"      reviews={s.review_component:.2f} "
                  f"engagement={s.engagement_component:.2f} "
                  f"trust={s.trust_component:.2f} | "
                  f"human={_yn(c.human_verified)} ai_bot={_yn(c.ai_bot_detected)} "
                  f"reply_rate={c.personal_reply_rate:.0%} "
                  f"resp={c.median_response_minutes if c.median_response_minutes is not None else '?'}min")
            if c.notes:
                print(f"      note: {c.notes}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="creator_rank",
        description="Rank creators who genuinely (human, non-AI) interact with "
                    "their subscribers. Filter by language, reviews and rating.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 creator_rank.py --demo --verbose\n"
            "  python3 creator_rank.py -i creators.csv --language english --min-reviews 100\n"
            "  python3 creator_rank.py -i creators.json --min-rating 4.5 --export ranked.csv\n"
            "  python3 creator_rank.py --write-template template.csv\n"
        ),
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("-i", "--input", help="Path to creators dataset (.csv or .json).")
    src.add_argument("--demo", action="store_true",
                     help="Use the built-in fictional sample dataset.")

    # Filters
    p.add_argument("--language", help="Only include creators who speak this language.")
    p.add_argument("--min-reviews", type=int, default=0,
                   help="Minimum number of reviews (default 0).")
    p.add_argument("--min-rating", type=float, default=0.0,
                   help="Minimum average rating on a 0-5 scale (default 0).")
    p.add_argument("--min-reply-rate", type=float, default=0.0,
                   help="Minimum personal reply rate, 0..1 (default 0).")
    p.add_argument("--allow-ai-bots", action="store_true",
                   help="Do NOT exclude accounts flagged as AI/automated "
                        "(off by default; the point is to exclude them).")
    p.add_argument("--no-require-human", action="store_true",
                   help="Include creators not verified as real humans "
                        "(by default only verified humans are ranked).")

    # Output
    p.add_argument("--top", type=int, default=None, help="Show only the top N.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show score breakdown per creator.")
    p.add_argument("--export", help="Write the ranked results to .csv or .json.")
    p.add_argument("--write-template", metavar="PATH",
                   help="Write an empty CSV template with the expected columns and exit.")

    # Weight overrides (advanced)
    p.add_argument("--w-reviews", type=float, help="Weight of review-quality pillar.")
    p.add_argument("--w-engagement", type=float, help="Weight of human-engagement pillar.")
    p.add_argument("--w-trust", type=float, help="Weight of trust/safety pillar.")
    p.add_argument("--prior-strength", type=float,
                   help="Bayesian prior strength for low-review shrinkage (default 25).")
    return p


TEMPLATE_COLUMNS = [
    "handle", "display_name", "languages", "num_reviews", "avg_rating",
    "human_verified", "ai_bot_detected", "personal_reply_rate",
    "personalization", "median_response_minutes", "scam_reports", "notes",
]


def write_template(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(TEMPLATE_COLUMNS)
        writer.writerow([
            "example_handle", "Example Name", "english|spanish", 250, 4.7,
            "true", "false", 0.9, 0.85, 30, 0, "personally replies",
        ])


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.write_template:
        write_template(args.write_template)
        print(f"Wrote CSV template to {args.write_template}")
        return 0

    if args.demo:
        creators = sample_creators()
    elif args.input:
        try:
            creators = load_creators(args.input)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"Error loading '{args.input}': {e}", file=sys.stderr)
            return 2
    else:
        print("No input given. Use --demo, -i/--input FILE, or --write-template.\n"
              "Run with --help for details.", file=sys.stderr)
        return 2

    if not creators:
        print("Dataset is empty.", file=sys.stderr)
        return 1

    weights = Weights()
    if args.w_reviews is not None:
        weights.review_quality = args.w_reviews
    if args.w_engagement is not None:
        weights.human_engagement = args.w_engagement
    if args.w_trust is not None:
        weights.trust_safety = args.w_trust
    if args.prior_strength is not None:
        weights.prior_strength = args.prior_strength

    filters = Filters(
        language=args.language,
        min_reviews=args.min_reviews,
        min_rating=args.min_rating,
        require_human=not args.no_require_human,
        exclude_ai_bots=not args.allow_ai_bots,
        min_reply_rate=args.min_reply_rate,
    )

    filtered = apply_filters(creators, filters)
    scored = score_creators(filtered, weights)

    total = len(creators)
    kept = len(filtered)
    print(f"Loaded {total} creator(s); {kept} passed filters "
          f"(require_human={_yn(filters.require_human)}, "
          f"exclude_ai_bots={_yn(filters.exclude_ai_bots)}"
          + (f", language={filters.language}" if filters.language else "")
          + (f", min_reviews={filters.min_reviews}" if filters.min_reviews else "")
          + (f", min_rating={filters.min_rating}" if filters.min_rating else "")
          + ").")
    print()
    print_ranking(scored, top=args.top, verbose=args.verbose)

    if args.export:
        try:
            export_results(scored, args.export)
            print(f"\nExported {len(scored)} ranked creator(s) to {args.export}")
        except (OSError, ValueError) as e:
            print(f"Export failed: {e}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
