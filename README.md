ty to Neil Paine, the goat : https://github.com/Neil-Paine-1

## OnlyFans Creator Ranking Tool

`onlyfans_ranker.py` ranks creators from your CSV/JSON data using:
- language filter
- review count + rating
- interaction quality (highest default weight)
- custom priority score for any additional "important" factors

By default it **only includes creators with verified direct interaction** and
**excludes accounts using AI bots/automation**.

### Input fields

Supported fields (CSV headers or JSON keys):
- `name`
- `language`
- `review_count`
- `average_rating` (0-5)
- `direct_interaction_verified` (true/false)
- `uses_ai_bot` (true/false)
- `uses_automation` (true/false)
- `response_rate` (0-1 or 0-100)
- `personalized_reply_ratio` (0-1 or 0-100)
- `avg_response_minutes`
- `priority_score` (0-100)
- `profile_url` (optional)

### Quick start

```bash
python3 onlyfans_ranker.py --input sample_onlyfans_creators.csv --language english,spanish --min-reviews 50 --top 5
```

### JSON output

```bash
python3 onlyfans_ranker.py --input sample_onlyfans_creators.csv --output-format json
```

### Tune weights

Defaults: interaction `0.60`, reviews `0.25`, priority `0.15`

```bash
python3 onlyfans_ranker.py \
  --input sample_onlyfans_creators.csv \
  --w-interaction 0.70 \
  --w-reviews 0.20 \
  --w-priority 0.10
```

### Optional relaxed filters

Not recommended for your stated goal, but available:
- `--allow-unverified` (include unverified direct interaction)
- `--allow-ai` (include AI/automation accounts)
