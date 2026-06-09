ty to Neil Paine, the goat : https://github.com/Neil-Paine-1

## Creator ranking tool

`creator_ranker.py` ranks subscription creators from a CSV file. It does not scrape any
site or independently prove who sends messages. Instead, it only ranks rows where your
data explicitly confirms:

- the creator directly interacts with customers
- no AI bot, chatbot, agency assistant, or other automation is used for customer replies
- enough reviews and rating data exist to meet the configured thresholds
- the requested language is listed

Example:

```bash
python creator_ranker.py creators.csv --language english --min-reviews 25 --output ranked_creators.csv
```

Required CSV columns, with common aliases supported:

- `name`
- `languages`
- `average_rating` (`rating` also works)
- `review_count` (`reviews` also works)
- `direct_interaction_confirmed`
- `verified_no_ai_bot`

Recommended columns for better scoring:

- `direct_interaction_score`
- `creator_reply_ratio`
- `response_rate`
- `median_response_time_hours`
- `subscriber_retention`
- `content_update_frequency_per_week`
- `last_review_days_ago`
- `value_rating`
- `subscription_price`
- `direct_interaction_evidence`
- `bot_free_evidence`

Default ranking weights:

- review quality: 25%
- direct interaction: 30%
- response reliability: 20%
- customer retention: 10%
- freshness: 10%
- value: 5%

Override weights with:

```bash
python creator_ranker.py creators.csv --weights review_quality=.25,direct_interaction=.35,response_reliability=.20,customer_retention=.10,freshness=.05,value=.05
```
