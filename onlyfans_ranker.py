#!/usr/bin/env python3
"""
OnlyFans Creator Ranker

Ranks creators based on reviews, language, and direct customer interaction quality.
Automatically excludes profiles that use AI bots, chatting agencies, or show
automation signals (instant replies, templated messages, low personalization).

OnlyFans has no public API — supply creator profiles as JSON or CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA = Path(__file__).parent / "onlyfans_ranker_data" / "sample_creators.json"

# Interaction authenticity is weighted highest — direct human engagement matters most.
WEIGHTS = {
    "interaction_authenticity": 0.40,
    "reviews": 0.25,
    "engagement_quality": 0.20,
    "reliability": 0.15,
}

SUPPORTED_LANGUAGES = {
    "english",
    "spanish",
    "french",
    "german",
    "italian",
    "portuguese",
    "russian",
    "japanese",
    "korean",
    "chinese",
    "dutch",
    "polish",
    "arabic",
    "turkish",
}


@dataclass
class CreatorProfile:
    username: str
    display_name: str
    languages: list[str]
    review_avg: float
    review_count: int
    review_personal_touch_ratio: float
    avg_response_time_minutes: float
    response_time_variance: float
    personalization_score: float
    message_uniqueness_score: float
    uses_chatting_agency: bool
    uses_ai_bot: bool
    verified_personal_dm: bool
    live_video_available: bool
    custom_content_offered: bool
    posts_per_week: float
    subscriber_satisfaction: float
    price_monthly_usd: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreatorProfile:
        languages = [lang.strip().lower() for lang in data.get("languages", [])]
        return cls(
            username=str(data["username"]),
            display_name=str(data.get("display_name", data["username"])),
            languages=languages,
            review_avg=float(data.get("review_avg", 0)),
            review_count=int(data.get("review_count", 0)),
            review_personal_touch_ratio=float(data.get("review_personal_touch_ratio", 0)),
            avg_response_time_minutes=float(data.get("avg_response_time_minutes", 0)),
            response_time_variance=float(data.get("response_time_variance", 0)),
            personalization_score=float(data.get("personalization_score", 0)),
            message_uniqueness_score=float(data.get("message_uniqueness_score", 0)),
            uses_chatting_agency=bool(data.get("uses_chatting_agency", False)),
            uses_ai_bot=bool(data.get("uses_ai_bot", False)),
            verified_personal_dm=bool(data.get("verified_personal_dm", False)),
            live_video_available=bool(data.get("live_video_available", False)),
            custom_content_offered=bool(data.get("custom_content_offered", False)),
            posts_per_week=float(data.get("posts_per_week", 0)),
            subscriber_satisfaction=float(data.get("subscriber_satisfaction", 0)),
            price_monthly_usd=float(data.get("price_monthly_usd", 0)),
        )


@dataclass
class RankedCreator:
    profile: CreatorProfile
    rank: int
    total_score: float
    interaction_score: float
    review_score: float
    engagement_score: float
    reliability_score: float
    authenticity_score: float
    exclusion_reason: str | None = None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def compute_authenticity_score(profile: CreatorProfile) -> float:
    """
    Detect likely AI bots, chatting agencies, and automated messaging.

    Signals:
    - Hard flags: AI bot or agency usage
    - Response speed: sub-minute avg with near-zero variance suggests automation
    - Personalization and message uniqueness from manual review / subscriber feedback
    - Review mentions of personal touch vs generic copy-paste replies
    """
    if profile.uses_ai_bot or profile.uses_chatting_agency:
        return 0.0

    score = 50.0

    # Human response patterns: variable timing, not instant every time.
    if profile.avg_response_time_minutes < 1.0 and profile.response_time_variance < 1.0:
        score -= 35.0
    elif profile.avg_response_time_minutes < 3.0 and profile.response_time_variance < 2.0:
        score -= 20.0
    elif 5.0 <= profile.avg_response_time_minutes <= 120.0:
        score += 10.0

    if profile.response_time_variance >= 8.0:
        score += 10.0
    elif profile.response_time_variance < 1.0:
        score -= 15.0

    score += (profile.personalization_score / 10.0) * 15.0
    score += (profile.message_uniqueness_score / 10.0) * 15.0
    score += profile.review_personal_touch_ratio * 20.0

    if profile.verified_personal_dm:
        score += 8.0
    if profile.live_video_available:
        score += 7.0
    if profile.custom_content_offered:
        score += 5.0

    return _clamp(score)


def get_exclusion_reason(profile: CreatorProfile, min_authenticity: float) -> str | None:
    if profile.uses_ai_bot:
        return "uses AI bot"
    if profile.uses_chatting_agency:
        return "uses chatting agency (not direct creator)"
    authenticity = compute_authenticity_score(profile)
    if authenticity < min_authenticity:
        return f"low authenticity score ({authenticity:.1f} < {min_authenticity:.1f})"
    return None


def compute_review_score(profile: CreatorProfile) -> float:
    rating_component = (profile.review_avg / 5.0) * 60.0
    # More reviews = more reliable signal, capped at 100 reviews for full weight.
    volume_component = min(profile.review_count / 100.0, 1.0) * 25.0
    personal_touch_component = profile.review_personal_touch_ratio * 15.0
    return _clamp(rating_component + volume_component + personal_touch_component)


def compute_engagement_score(profile: CreatorProfile) -> float:
    score = (profile.subscriber_satisfaction / 5.0) * 50.0
    if profile.custom_content_offered:
        score += 15.0
    if profile.live_video_available:
        score += 20.0
    if profile.verified_personal_dm:
        score += 15.0
    return _clamp(score)


def compute_reliability_score(profile: CreatorProfile) -> float:
    # Consistent posting without spam-level frequency.
    if profile.posts_per_week <= 0:
        posting_score = 0.0
    elif profile.posts_per_week <= 7:
        posting_score = 40.0
    elif profile.posts_per_week <= 12:
        posting_score = 25.0
    else:
        posting_score = 10.0

    value_score = 30.0
    if profile.price_monthly_usd > 0:
        satisfaction_per_dollar = profile.subscriber_satisfaction / profile.price_monthly_usd
        value_score = _clamp(satisfaction_per_dollar * 40.0, 0.0, 40.0)

    return _clamp(posting_score + value_score + 30.0)


def rank_creators(
    profiles: list[CreatorProfile],
    *,
    language: str | None = None,
    min_reviews: int = 0,
    min_review_avg: float = 0.0,
    min_authenticity: float = 55.0,
    top_n: int | None = None,
) -> tuple[list[RankedCreator], list[RankedCreator]]:
    language_filter = language.lower().strip() if language else None
    ranked: list[RankedCreator] = []
    excluded: list[RankedCreator] = []

    for profile in profiles:
        if language_filter and language_filter not in profile.languages:
            continue
        if profile.review_count < min_reviews:
            continue
        if profile.review_avg < min_review_avg:
            continue

        authenticity = compute_authenticity_score(profile)
        exclusion = get_exclusion_reason(profile, min_authenticity)

        interaction_score = authenticity
        review_score = compute_review_score(profile)
        engagement_score = compute_engagement_score(profile)
        reliability_score = compute_reliability_score(profile)

        total_score = (
            interaction_score * WEIGHTS["interaction_authenticity"]
            + review_score * WEIGHTS["reviews"]
            + engagement_score * WEIGHTS["engagement_quality"]
            + reliability_score * WEIGHTS["reliability"]
        )

        entry = RankedCreator(
            profile=profile,
            rank=0,
            total_score=round(total_score, 2),
            interaction_score=round(interaction_score, 2),
            review_score=round(review_score, 2),
            engagement_score=round(engagement_score, 2),
            reliability_score=round(reliability_score, 2),
            authenticity_score=round(authenticity, 2),
            exclusion_reason=exclusion,
        )

        if exclusion:
            excluded.append(entry)
        else:
            ranked.append(entry)

    ranked.sort(key=lambda item: item.total_score, reverse=True)
    for index, entry in enumerate(ranked, start=1):
        entry.rank = index

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked, excluded


def load_profiles(path: Path) -> list[CreatorProfile]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError("JSON input must be a list of creator objects")
        return [CreatorProfile.from_dict(item) for item in raw]

    if suffix == ".csv":
        profiles: list[CreatorProfile] = []
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "languages" in row and isinstance(row["languages"], str):
                    row["languages"] = [
                        part.strip() for part in row["languages"].split("|") if part.strip()
                    ]
                for bool_field in (
                    "uses_chatting_agency",
                    "uses_ai_bot",
                    "verified_personal_dm",
                    "live_video_available",
                    "custom_content_offered",
                ):
                    if bool_field in row:
                        row[bool_field] = str(row[bool_field]).lower() in {"1", "true", "yes"}
                profiles.append(CreatorProfile.from_dict(row))
        return profiles

    raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")


def export_rankings(path: Path, ranked: list[RankedCreator]) -> None:
    rows = []
    for entry in ranked:
        row = {
            "rank": entry.rank,
            "username": entry.profile.username,
            "display_name": entry.profile.display_name,
            "languages": "|".join(entry.profile.languages),
            "total_score": entry.total_score,
            "authenticity_score": entry.authenticity_score,
            "review_avg": entry.profile.review_avg,
            "review_count": entry.profile.review_count,
            "interaction_score": entry.interaction_score,
            "review_score": entry.review_score,
            "engagement_score": entry.engagement_score,
            "reliability_score": entry.reliability_score,
        }
        rows.append(row)

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        return

    if suffix == ".csv":
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return

    raise ValueError(f"Unsupported export format: {suffix}")


def print_rankings(ranked: list[RankedCreator], excluded: list[RankedCreator], verbose: bool) -> None:
    if not ranked:
        print("No creators matched your filters (or all were excluded as bots/agencies).")
    else:
        print(f"\n{'=' * 72}")
        print("ONLYFANS CREATOR RANKINGS — Direct Interaction Only")
        print(f"{'=' * 72}")
        print(
            f"{'Rank':<5} {'Username':<18} {'Score':<7} {'Auth':<6} "
            f"{'Reviews':<12} {'Languages':<20} {'Name'}"
        )
        print("-" * 72)
        for entry in ranked:
            reviews = f"{entry.profile.review_avg:.1f}★ ({entry.profile.review_count})"
            langs = ", ".join(entry.profile.languages)
            print(
                f"{entry.rank:<5} {entry.profile.username:<18} {entry.total_score:<7.1f} "
                f"{entry.authenticity_score:<6.1f} {reviews:<12} {langs:<20} "
                f"{entry.profile.display_name}"
            )
            if verbose:
                print(
                    f"       breakdown: interaction={entry.interaction_score:.1f}, "
                    f"reviews={entry.review_score:.1f}, engagement={entry.engagement_score:.1f}, "
                    f"reliability={entry.reliability_score:.1f}"
                )

    if excluded:
        print(f"\n{'=' * 72}")
        print(f"EXCLUDED ({len(excluded)} profiles — bots, agencies, or low authenticity)")
        print(f"{'=' * 72}")
        for entry in excluded:
            print(
                f"  {entry.profile.username:<18} auth={entry.authenticity_score:.1f}  "
                f"reason: {entry.exclusion_reason}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank OnlyFans creators by reviews, language, and direct interaction quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python onlyfans_ranker.py
  python onlyfans_ranker.py --language english --min-reviews 30
  python onlyfans_ranker.py --language spanish --min-review-avg 4.5 --top 5
  python onlyfans_ranker.py --input my_creators.json --output rankings.csv -v

Profile fields (JSON/CSV):
  username, display_name, languages, review_avg, review_count,
  review_personal_touch_ratio, avg_response_time_minutes, response_time_variance,
  personalization_score, message_uniqueness_score,
  uses_chatting_agency, uses_ai_bot, verified_personal_dm,
  live_video_available, custom_content_offered, posts_per_week,
  subscriber_satisfaction, price_monthly_usd
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to creator profiles JSON or CSV (default: sample data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Export rankings to JSON or CSV",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help=f"Filter by language ({', '.join(sorted(SUPPORTED_LANGUAGES))})",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=0,
        help="Minimum number of reviews required (default: 0)",
    )
    parser.add_argument(
        "--min-review-avg",
        type=float,
        default=0.0,
        help="Minimum average review rating 0-5 (default: 0)",
    )
    parser.add_argument(
        "--min-authenticity",
        type=float,
        default=55.0,
        help="Minimum authenticity score to include (default: 55)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N results",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show score breakdown per creator",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported language filters and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_languages:
        print("Supported language filters:")
        for lang in sorted(SUPPORTED_LANGUAGES):
            print(f"  - {lang}")
        return 0

    if args.language and args.language.lower() not in SUPPORTED_LANGUAGES:
        print(
            f"Warning: '{args.language}' is not in the known language list. "
            "Filtering will still match if profiles use the same tag.",
            file=sys.stderr,
        )

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        profiles = load_profiles(args.input)
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"Failed to load profiles: {exc}", file=sys.stderr)
        return 1

    ranked, excluded = rank_creators(
        profiles,
        language=args.language,
        min_reviews=args.min_reviews,
        min_review_avg=args.min_review_avg,
        min_authenticity=args.min_authenticity,
        top_n=args.top,
    )

    print_rankings(ranked, excluded, args.verbose)

    if args.output:
        try:
            export_rankings(args.output, ranked)
            print(f"\nRankings exported to {args.output}")
        except ValueError as exc:
            print(f"Export failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
