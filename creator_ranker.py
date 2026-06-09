import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TRUE_VALUES = {"1", "true", "yes", "y", "verified", "confirmed"}
FALSE_VALUES = {"0", "false", "no", "n", "none", "not verified", "unverified", ""}
AUTOMATION_KEYWORDS = {
    "ai",
    "bot",
    "chatbot",
    "auto",
    "automated",
    "automation",
    "agency",
    "assistant",
    "outsourced",
}

COLUMN_ALIASES = {
    "name": ("name", "creator", "creator_name", "display_name", "username"),
    "profile_url": ("profile_url", "url", "link", "onlyfans_url"),
    "languages": ("languages", "language", "spoken_languages"),
    "average_rating": ("average_rating", "rating", "avg_rating", "reviews_average"),
    "review_count": ("review_count", "reviews", "num_reviews", "total_reviews"),
    "direct_interaction_confirmed": (
        "direct_interaction_confirmed",
        "direct_interaction",
        "directly_interacts",
        "creator_interacts_directly",
    ),
    "direct_interaction_score": (
        "direct_interaction_score",
        "interaction_score",
        "personal_interaction_score",
    ),
    "creator_reply_ratio": (
        "creator_reply_ratio",
        "reply_ratio",
        "personal_reply_ratio",
        "creator_reply_percent",
    ),
    "response_rate": ("response_rate", "dm_response_rate", "message_response_rate"),
    "median_response_time_hours": (
        "median_response_time_hours",
        "response_time_hours",
        "avg_response_time_hours",
    ),
    "verified_no_ai_bot": (
        "verified_no_ai_bot",
        "no_ai_bot_verified",
        "bot_free_verified",
        "human_verified",
    ),
    "uses_ai_bot": ("uses_ai_bot", "ai_bot", "bot_detected", "uses_bot"),
    "uses_automation": ("uses_automation", "automation_used", "uses_auto_tools"),
    "automation_disclosure": (
        "automation_disclosure",
        "bot_disclosure",
        "automation_notes",
    ),
    "subscriber_retention": (
        "subscriber_retention",
        "retention_rate",
        "renewal_rate",
    ),
    "content_update_frequency_per_week": (
        "content_update_frequency_per_week",
        "posts_per_week",
        "updates_per_week",
    ),
    "last_review_days_ago": (
        "last_review_days_ago",
        "days_since_last_review",
        "review_recency_days",
    ),
    "value_rating": ("value_rating", "value_score", "price_value_rating"),
    "subscription_price": ("subscription_price", "price", "monthly_price"),
    "direct_interaction_evidence": (
        "direct_interaction_evidence",
        "interaction_evidence",
        "evidence",
    ),
    "bot_free_evidence": (
        "bot_free_evidence",
        "no_bot_evidence",
        "human_verified_evidence",
    ),
}

REQUIRED_COLUMNS = {
    "name",
    "languages",
    "average_rating",
    "review_count",
    "direct_interaction_confirmed",
    "verified_no_ai_bot",
}

DEFAULT_WEIGHTS = {
    "review_quality": 0.25,
    "direct_interaction": 0.30,
    "response_reliability": 0.20,
    "customer_retention": 0.10,
    "freshness": 0.10,
    "value": 0.05,
}


@dataclass
class CreatorScore:
    rank: int
    name: str
    profile_url: str
    languages: str
    total_score: float
    review_quality: float
    direct_interaction: float
    response_reliability: float
    customer_retention: float
    freshness: float
    value: float
    average_rating: float
    review_count: int
    creator_reply_ratio: float
    response_rate: float
    median_response_time_hours: float
    direct_interaction_evidence: str
    bot_free_evidence: str

    def as_row(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "name": self.name,
            "profile_url": self.profile_url,
            "languages": self.languages,
            "total_score": round(self.total_score, 2),
            "review_quality": round(self.review_quality, 2),
            "direct_interaction": round(self.direct_interaction, 2),
            "response_reliability": round(self.response_reliability, 2),
            "customer_retention": round(self.customer_retention, 2),
            "freshness": round(self.freshness, 2),
            "value": round(self.value, 2),
            "average_rating": round(self.average_rating, 2),
            "review_count": self.review_count,
            "creator_reply_ratio": round(self.creator_reply_ratio, 2),
            "response_rate": round(self.response_rate, 2),
            "median_response_time_hours": round(self.median_response_time_hours, 2),
            "direct_interaction_evidence": self.direct_interaction_evidence,
            "bot_free_evidence": self.bot_free_evidence,
        }


def normalize_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def parse_bool(value: object) -> Optional[bool]:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def parse_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    cleaned = str(value).strip().replace("%", "").replace("$", "").replace(",", "")
    if cleaned == "":
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def normalize_percent(value: object, default: float = 0.0) -> float:
    parsed = parse_float(value, default)
    if 0 <= parsed <= 1:
        return parsed * 100
    return clamp(parsed)


def normalize_rating(value: object) -> float:
    parsed = parse_float(value)
    if parsed <= 5:
        return clamp((parsed / 5) * 100)
    return clamp(parsed)


def split_languages(value: str) -> List[str]:
    separators = [",", ";", "|", "/"]
    normalized = value
    for separator in separators:
        normalized = normalized.replace(separator, ",")
    return [language.strip().lower() for language in normalized.split(",") if language.strip()]


def header_map(fieldnames: Sequence[str]) -> Dict[str, str]:
    lookup = {normalize_key(field): field for field in fieldnames}
    mapped = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                mapped[canonical] = lookup[alias]
                break
    return mapped


def require_columns(mapped_headers: Dict[str, str]) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(mapped_headers))
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(missing)
            + ". Required fields are: "
            + ", ".join(sorted(REQUIRED_COLUMNS))
        )


def get_value(row: Dict[str, str], mapped_headers: Dict[str, str], key: str, default: str = "") -> str:
    column = mapped_headers.get(key)
    if not column:
        return default
    value = row.get(column, default)
    return default if value is None else str(value).strip()


def has_automation_disclosure(row: Dict[str, str], mapped_headers: Dict[str, str]) -> bool:
    disclosure = get_value(row, mapped_headers, "automation_disclosure").lower()
    if not disclosure or disclosure in FALSE_VALUES:
        return False
    return any(keyword in disclosure for keyword in AUTOMATION_KEYWORDS) or parse_bool(disclosure) is True


def is_bot_free(row: Dict[str, str], mapped_headers: Dict[str, str]) -> bool:
    verified_no_bot = parse_bool(get_value(row, mapped_headers, "verified_no_ai_bot"))
    uses_ai_bot = parse_bool(get_value(row, mapped_headers, "uses_ai_bot"))
    uses_automation = parse_bool(get_value(row, mapped_headers, "uses_automation"))
    return (
        verified_no_bot is True
        and uses_ai_bot is not True
        and uses_automation is not True
        and not has_automation_disclosure(row, mapped_headers)
    )


def matches_language(row_languages: str, requested_language: Optional[str]) -> bool:
    if not requested_language:
        return True
    requested = requested_language.strip().lower()
    return requested in split_languages(row_languages)


def review_quality_score(average_rating: float, review_count: int, min_reviews: int) -> float:
    rating_score = normalize_rating(average_rating)
    confidence_target = max(min_reviews * 5, 250)
    confidence = min(math.log10(review_count + 1) / math.log10(confidence_target + 1), 1.0)
    return clamp((rating_score * 0.70) + (confidence * 100 * 0.30))


def response_reliability_score(response_rate: float, median_response_time_hours: float) -> float:
    speed_score = 100 - (min(max(median_response_time_hours, 0), 72) / 72 * 100)
    return clamp((response_rate * 0.65) + (speed_score * 0.35))


def freshness_score(posts_per_week: float, last_review_days_ago: float) -> float:
    update_score = clamp((max(posts_per_week, 0) / 7) * 100)
    recency_score = 100 - (min(max(last_review_days_ago, 0), 120) / 120 * 100)
    return clamp((update_score * 0.45) + (recency_score * 0.55))


def value_score(value_rating: float, subscription_price: float) -> float:
    if value_rating > 0:
        return normalize_rating(value_rating)
    price_score = 100 - (min(max(subscription_price, 0), 50) / 50 * 100)
    return clamp(price_score)


def parse_weights(raw_weights: Optional[str]) -> Dict[str, float]:
    weights = DEFAULT_WEIGHTS.copy()
    if not raw_weights:
        return weights

    for item in raw_weights.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f'Invalid weight "{item}". Use component=value.')
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in weights:
            raise ValueError(
                f'Unknown weight "{key}". Allowed weights: {", ".join(sorted(weights))}'
            )
        weights[key] = parse_float(raw_value)

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one ranking weight must be greater than zero.")
    return {key: value / total for key, value in weights.items()}


def rank_creators(
    rows: Iterable[Dict[str, str]],
    mapped_headers: Dict[str, str],
    language: Optional[str],
    min_reviews: int,
    min_rating: float,
    min_interaction_score: float,
    weights: Dict[str, float],
) -> List[CreatorScore]:
    scores: List[CreatorScore] = []
    for row in rows:
        languages = get_value(row, mapped_headers, "languages")
        review_count = int(parse_float(get_value(row, mapped_headers, "review_count")))
        average_rating = parse_float(get_value(row, mapped_headers, "average_rating"))
        direct_confirmed = parse_bool(get_value(row, mapped_headers, "direct_interaction_confirmed"))

        if not matches_language(languages, language):
            continue
        if review_count < min_reviews or average_rating < min_rating:
            continue
        if direct_confirmed is not True or not is_bot_free(row, mapped_headers):
            continue

        direct_interaction = normalize_percent(
            get_value(row, mapped_headers, "direct_interaction_score"),
            default=75,
        )
        creator_reply_ratio = normalize_percent(
            get_value(row, mapped_headers, "creator_reply_ratio"),
            default=direct_interaction,
        )
        direct_score = clamp((direct_interaction * 0.55) + (creator_reply_ratio * 0.45))
        if direct_score < min_interaction_score:
            continue

        response_rate = normalize_percent(get_value(row, mapped_headers, "response_rate"), default=50)
        median_response_time_hours = parse_float(
            get_value(row, mapped_headers, "median_response_time_hours"),
            default=24,
        )
        retention = normalize_percent(get_value(row, mapped_headers, "subscriber_retention"), default=50)
        posts_per_week = parse_float(
            get_value(row, mapped_headers, "content_update_frequency_per_week"),
            default=3,
        )
        last_review_days_ago = parse_float(
            get_value(row, mapped_headers, "last_review_days_ago"),
            default=30,
        )
        creator_value = value_score(
            parse_float(get_value(row, mapped_headers, "value_rating")),
            parse_float(get_value(row, mapped_headers, "subscription_price")),
        )

        components = {
            "review_quality": review_quality_score(average_rating, review_count, min_reviews),
            "direct_interaction": direct_score,
            "response_reliability": response_reliability_score(
                response_rate,
                median_response_time_hours,
            ),
            "customer_retention": retention,
            "freshness": freshness_score(posts_per_week, last_review_days_ago),
            "value": creator_value,
        }
        total_score = sum(components[key] * weights[key] for key in components)

        scores.append(
            CreatorScore(
                rank=0,
                name=get_value(row, mapped_headers, "name"),
                profile_url=get_value(row, mapped_headers, "profile_url"),
                languages=languages,
                total_score=total_score,
                average_rating=average_rating,
                review_count=review_count,
                creator_reply_ratio=creator_reply_ratio,
                response_rate=response_rate,
                median_response_time_hours=median_response_time_hours,
                direct_interaction_evidence=get_value(
                    row,
                    mapped_headers,
                    "direct_interaction_evidence",
                ),
                bot_free_evidence=get_value(row, mapped_headers, "bot_free_evidence"),
                **components,
            )
        )

    scores.sort(key=lambda score: (-score.total_score, -score.review_count, score.name.lower()))
    for index, score in enumerate(scores, start=1):
        score.rank = index
    return scores


def read_rankings(
    input_path: Path,
    language: Optional[str],
    min_reviews: int,
    min_rating: float,
    min_interaction_score: float,
    weights: Dict[str, float],
) -> List[CreatorScore]:
    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")
        mapped_headers = header_map(reader.fieldnames)
        require_columns(mapped_headers)
        return rank_creators(
            reader,
            mapped_headers,
            language,
            min_reviews,
            min_rating,
            min_interaction_score,
            weights,
        )


def write_rankings(scores: Sequence[CreatorScore], output_path: Optional[Path]) -> None:
    fieldnames = list(CreatorScore(0, "", "", "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", "").as_row())
    output_file = output_path.open("w", newline="", encoding="utf-8") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            writer.writerow(score.as_row())
    finally:
        if output_path:
            output_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank subscription creators from a CSV, filtering to profiles with direct "
            "customer interaction and verified no AI/bot/automation use."
        )
    )
    parser.add_argument("input_csv", type=Path, help="CSV file containing creator metrics.")
    parser.add_argument("-o", "--output", type=Path, help="Write ranked CSV to this path.")
    parser.add_argument("--language", help="Only include creators who list this language.")
    parser.add_argument("--min-reviews", type=int, default=20, help="Minimum review count.")
    parser.add_argument("--min-rating", type=float, default=4.0, help="Minimum average rating.")
    parser.add_argument(
        "--min-interaction-score",
        type=float,
        default=70,
        help="Minimum direct interaction score after reply ratio weighting.",
    )
    parser.add_argument(
        "--weights",
        help=(
            "Optional comma-separated component weights, for example "
            "review_quality=.25,direct_interaction=.35,response_reliability=.20,"
            "customer_retention=.10,freshness=.05,value=.05"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        weights = parse_weights(args.weights)
        scores = read_rankings(
            args.input_csv,
            args.language,
            args.min_reviews,
            args.min_rating,
            args.min_interaction_score,
            weights,
        )
        write_rankings(scores, args.output)
    except (OSError, ValueError) as error:
        parser.exit(status=1, message=f"creator_ranker: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
