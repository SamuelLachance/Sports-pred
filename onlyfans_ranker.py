#!/usr/bin/env python3
"""
Rank OnlyFans creators with strict human-interaction filtering.

The tool expects user-provided data in CSV or JSON format and ranks creators
using weighted factors:
- interaction quality (most important by default)
- reviews quality and volume
- optional priority score for other "important" factors
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "n", "f"}


@dataclass
class Creator:
    name: str
    language: str
    review_count: int
    average_rating: float
    direct_interaction_verified: bool
    uses_ai_bot: bool
    uses_automation: bool
    response_rate: float
    personalized_reply_ratio: float
    avg_response_minutes: float
    priority_score: float
    profile_url: str


@dataclass
class ScoredCreator:
    creator: Creator
    interaction_score: float
    review_score: float
    priority_score: float
    overall_score: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    value_str = str(value).strip().lower()
    if value_str in TRUE_VALUES:
        return True
    if value_str in FALSE_VALUES:
        return False
    return default


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def get_first(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def parse_ratio(value: Any, default: float = 0.0) -> float:
    ratio = to_float(value, default=default)
    if ratio > 1.0:
        ratio = ratio / 100.0
    return clamp(ratio, 0.0, 1.0)


def parse_creator(record: dict[str, Any], index: int) -> Creator:
    name = str(
        get_first(record, ["name", "creator_name", "display_name"], f"creator_{index + 1}")
    )
    language = str(get_first(record, ["language", "lang"], "unknown")).strip().lower()
    review_count = to_int(get_first(record, ["review_count", "reviews", "total_reviews"], 0), 0)
    average_rating = clamp(
        to_float(get_first(record, ["average_rating", "rating"], 0.0), 0.0), 0.0, 5.0
    )
    direct_interaction_verified = to_bool(
        get_first(
            record,
            ["direct_interaction_verified", "verified_human_chat", "human_chat_verified"],
            False,
        ),
        False,
    )
    uses_ai_bot = to_bool(get_first(record, ["uses_ai_bot", "ai_bot", "uses_ai"], False), False)
    uses_automation = to_bool(
        get_first(record, ["uses_automation", "automated_messages", "is_automated"], False),
        False,
    )
    response_rate = parse_ratio(get_first(record, ["response_rate", "reply_rate"], 0.0), 0.0)
    personalized_reply_ratio = parse_ratio(
        get_first(record, ["personalized_reply_ratio", "custom_reply_ratio"], 0.0),
        0.0,
    )
    avg_response_minutes = clamp(
        to_float(get_first(record, ["avg_response_minutes", "response_minutes"], 1440), 1440),
        1.0,
        1440.0,
    )
    priority_score = clamp(
        to_float(
            get_first(record, ["priority_score", "important_score", "quality_score"], average_rating * 20.0),
            average_rating * 20.0,
        ),
        0.0,
        100.0,
    )
    profile_url = str(get_first(record, ["profile_url", "url"], "")).strip()

    return Creator(
        name=name,
        language=language,
        review_count=review_count,
        average_rating=average_rating,
        direct_interaction_verified=direct_interaction_verified,
        uses_ai_bot=uses_ai_bot,
        uses_automation=uses_automation,
        response_rate=response_rate,
        personalized_reply_ratio=personalized_reply_ratio,
        avg_response_minutes=avg_response_minutes,
        priority_score=priority_score,
        profile_url=profile_url,
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            return list(csv.DictReader(file_handle))
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict) and isinstance(data.get("creators"), list):
            return [row for row in data["creators"] if isinstance(row, dict)]
        raise ValueError("JSON input must be a list or an object with a 'creators' list.")
    raise ValueError("Unsupported file type. Use CSV or JSON.")


def score_creator(
    creator: Creator,
    max_reviews: int,
    w_interaction: float,
    w_reviews: float,
    w_priority: float,
) -> ScoredCreator:
    response_speed_score = 1.0 - (creator.avg_response_minutes / 1440.0)
    interaction_score = (
        0.50 * creator.response_rate
        + 0.35 * creator.personalized_reply_ratio
        + 0.15 * response_speed_score
    ) * 100.0

    rating_score = clamp((creator.average_rating - 1.0) / 4.0, 0.0, 1.0)
    review_volume_score = math.log1p(max(creator.review_count, 0)) / math.log1p(max(max_reviews, 1))
    review_score = (0.70 * rating_score + 0.30 * review_volume_score) * 100.0

    overall = (
        w_interaction * interaction_score
        + w_reviews * review_score
        + w_priority * creator.priority_score
    )

    return ScoredCreator(
        creator=creator,
        interaction_score=interaction_score,
        review_score=review_score,
        priority_score=creator.priority_score,
        overall_score=overall,
    )


def filter_creators(
    creators: list[Creator],
    language_filters: set[str],
    min_reviews: int,
    require_direct_interaction: bool,
    exclude_ai_or_automation: bool,
) -> list[Creator]:
    filtered: list[Creator] = []
    for creator in creators:
        if language_filters and creator.language not in language_filters:
            continue
        if creator.review_count < min_reviews:
            continue
        if require_direct_interaction and not creator.direct_interaction_verified:
            continue
        if exclude_ai_or_automation and (creator.uses_ai_bot or creator.uses_automation):
            continue
        filtered.append(creator)
    return filtered


def parse_language_filters(value: str) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def build_table(rows: list[ScoredCreator], top: int) -> str:
    headers = [
        "Rank",
        "Name",
        "Lang",
        "Reviews",
        "Rating",
        "Interact",
        "Review",
        "Priority",
        "Overall",
    ]
    table_rows: list[list[str]] = []
    for rank, scored in enumerate(rows[:top], start=1):
        table_rows.append(
            [
                str(rank),
                scored.creator.name,
                scored.creator.language,
                str(scored.creator.review_count),
                f"{scored.creator.average_rating:.2f}",
                f"{scored.interaction_score:.1f}",
                f"{scored.review_score:.1f}",
                f"{scored.priority_score:.1f}",
                f"{scored.overall_score:.1f}",
            ]
        )

    if not table_rows:
        return "No creators matched the filters."

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in table_rows))
        for column in range(len(headers))
    ]
    line_parts = []
    for header, width in zip(headers, widths):
        line_parts.append(header.ljust(width))
    lines = [" | ".join(line_parts)]
    lines.append("-+-".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
    return "\n".join(lines)


def build_json(rows: list[ScoredCreator], top: int) -> str:
    data = []
    for rank, scored in enumerate(rows[:top], start=1):
        data.append(
            {
                "rank": rank,
                "name": scored.creator.name,
                "language": scored.creator.language,
                "review_count": scored.creator.review_count,
                "average_rating": round(scored.creator.average_rating, 3),
                "interaction_score": round(scored.interaction_score, 3),
                "review_score": round(scored.review_score, 3),
                "priority_score": round(scored.priority_score, 3),
                "overall_score": round(scored.overall_score, 3),
                "profile_url": scored.creator.profile_url,
            }
        )
    return json.dumps(data, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank creators by language, reviews, and interaction quality while "
            "excluding AI/bot-assisted accounts."
        )
    )
    parser.add_argument("--input", required=True, help="Path to CSV or JSON dataset.")
    parser.add_argument(
        "--language",
        default="",
        help="Comma-separated languages to include (e.g. english,spanish).",
    )
    parser.add_argument("--min-reviews", type=int, default=10, help="Minimum review count.")
    parser.add_argument("--top", type=int, default=10, help="Number of rows to show.")
    parser.add_argument(
        "--w-interaction",
        type=float,
        default=0.60,
        help="Weight for interaction quality (default: 0.60).",
    )
    parser.add_argument(
        "--w-reviews",
        type=float,
        default=0.25,
        help="Weight for review quality and volume (default: 0.25).",
    )
    parser.add_argument(
        "--w-priority",
        type=float,
        default=0.15,
        help="Weight for custom priority score (default: 0.15).",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Include creators without verified direct interaction.",
    )
    parser.add_argument(
        "--allow-ai",
        action="store_true",
        help="Include creators that use AI bots or automation.",
    )
    parser.add_argument(
        "--output-format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )
    return parser.parse_args()


def validate_weights(w_interaction: float, w_reviews: float, w_priority: float) -> None:
    total = w_interaction + w_reviews + w_priority
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    validate_weights(args.w_interaction, args.w_reviews, args.w_priority)
    language_filters = parse_language_filters(args.language)

    raw_records = load_records(input_path)
    creators = [parse_creator(record, index) for index, record in enumerate(raw_records)]

    filtered = filter_creators(
        creators=creators,
        language_filters=language_filters,
        min_reviews=max(args.min_reviews, 0),
        require_direct_interaction=not args.allow_unverified,
        exclude_ai_or_automation=not args.allow_ai,
    )

    if not filtered:
        print("No creators matched the filters.")
        return

    max_reviews = max(creator.review_count for creator in filtered)
    scored = [
        score_creator(
            creator,
            max_reviews=max_reviews,
            w_interaction=args.w_interaction,
            w_reviews=args.w_reviews,
            w_priority=args.w_priority,
        )
        for creator in filtered
    ]
    ranked = sorted(scored, key=lambda item: item.overall_score, reverse=True)

    if args.output_format == "json":
        print(build_json(ranked, args.top))
    else:
        print(build_table(ranked, args.top))


if __name__ == "__main__":
    main()
