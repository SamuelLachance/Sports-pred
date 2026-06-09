ty to Neil Paine, the goat : https://github.com/Neil-Paine-1

## creator_ranker.py

Ranks OnlyFans-style creators from a local JSON dataset, filtering by language,
reviews, price and responsiveness. Creators using AI chatbots, auto-DM tools or
agency "chatters" are hard-excluded - only creators who personally and directly
interact with their customers get ranked. Automation is caught via explicit
profile flags plus heuristic analysis of review text. Stdlib only, no installs.

```bash
python3 creator_ranker.py creators_sample.json                       # full ranking
python3 creator_ranker.py creators_sample.json -l en es --min-reviews 3
python3 creator_ranker.py creators_sample.json --show-excluded       # who got cut & why
python3 creator_ranker.py creators_sample.json --explain nina_fit    # score breakdown
python3 creator_ranker.py creators_sample.json --export ranked.csv
```

Note: there is no public OnlyFans API and scraping violates their ToS, so the
tool works on a dataset you maintain yourself (`creators_sample.json` shows the
schema with fictional sample data).
