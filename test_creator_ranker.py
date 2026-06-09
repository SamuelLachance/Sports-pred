import csv
import tempfile
import unittest
from pathlib import Path

from creator_ranker import (
    DEFAULT_WEIGHTS,
    header_map,
    parse_weights,
    rank_creators,
    read_rankings,
    require_columns,
)


class CreatorRankerTests(unittest.TestCase):
    def test_ranks_only_direct_human_verified_language_matches(self):
        rows = [
            {
                "name": "Creator A",
                "languages": "English, Spanish",
                "average_rating": "4.8",
                "review_count": "150",
                "direct_interaction_confirmed": "yes",
                "direct_interaction_score": "94",
                "creator_reply_ratio": "91",
                "response_rate": "88",
                "median_response_time_hours": "6",
                "verified_no_ai_bot": "yes",
                "subscriber_retention": "82",
                "content_update_frequency_per_week": "5",
                "last_review_days_ago": "10",
                "value_rating": "4.6",
            },
            {
                "name": "Creator B",
                "languages": "English",
                "average_rating": "5",
                "review_count": "500",
                "direct_interaction_confirmed": "yes",
                "direct_interaction_score": "100",
                "creator_reply_ratio": "99",
                "response_rate": "99",
                "median_response_time_hours": "1",
                "verified_no_ai_bot": "no",
                "uses_ai_bot": "yes",
            },
            {
                "name": "Creator C",
                "languages": "French",
                "average_rating": "4.9",
                "review_count": "200",
                "direct_interaction_confirmed": "yes",
                "direct_interaction_score": "95",
                "creator_reply_ratio": "95",
                "response_rate": "95",
                "median_response_time_hours": "2",
                "verified_no_ai_bot": "yes",
            },
        ]

        mapped_headers = header_map(rows[0].keys())
        rankings = rank_creators(
            rows,
            mapped_headers,
            language="english",
            min_reviews=20,
            min_rating=4.0,
            min_interaction_score=70,
            weights=DEFAULT_WEIGHTS,
        )

        self.assertEqual([score.name for score in rankings], ["Creator A"])
        self.assertEqual(rankings[0].rank, 1)

    def test_automation_disclosure_excludes_creator(self):
        row = {
            "name": "Creator A",
            "languages": "English",
            "average_rating": "4.8",
            "review_count": "150",
            "direct_interaction_confirmed": "yes",
            "direct_interaction_score": "94",
            "creator_reply_ratio": "91",
            "response_rate": "88",
            "median_response_time_hours": "6",
            "verified_no_ai_bot": "yes",
            "automation_disclosure": "agency assistant handles messages",
        }

        rankings = rank_creators(
            [row],
            header_map(row.keys()),
            language="english",
            min_reviews=20,
            min_rating=4.0,
            min_interaction_score=70,
            weights=DEFAULT_WEIGHTS,
        )

        self.assertEqual(rankings, [])

    def test_required_columns_are_validated(self):
        mapped_headers = header_map(["name", "languages"])

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            require_columns(mapped_headers)

    def test_read_rankings_accepts_alias_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "creators.csv"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "creator_name",
                        "language",
                        "rating",
                        "reviews",
                        "directly_interacts",
                        "bot_free_verified",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "creator_name": "Creator A",
                        "language": "English",
                        "rating": "4.7",
                        "reviews": "80",
                        "directly_interacts": "true",
                        "bot_free_verified": "true",
                    }
                )

            rankings = read_rankings(
                input_path,
                language="english",
                min_reviews=20,
                min_rating=4.0,
                min_interaction_score=70,
                weights=DEFAULT_WEIGHTS,
            )

        self.assertEqual(len(rankings), 1)
        self.assertEqual(rankings[0].name, "Creator A")

    def test_custom_weights_are_normalized(self):
        weights = parse_weights("review_quality=2,direct_interaction=1")

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["review_quality"], weights["direct_interaction"])


if __name__ == "__main__":
    unittest.main()
