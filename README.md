## Week 1 Results (MovieLens 1M)

Dataset: 1,000,209 ratings, 6,040 users, 3,706 rated movies (3,883 in full catalog), 95.53% sparse.

Split: temporal leave-one-out per user (last rating = test, second-to-last = validation, rest = train).
988,129 train / 6,040 val / 6,040 test — zero train/test overlap confirmed.

| Model | Hit Rate@10 | NDCG@10 | Catalog Coverage | Intra-list Diversity |
|---|---|---|---|---|
| Popularity baseline | 0.0368 | 0.0179 | 0.0507 | 0.7491 |
| User-based CF | 0.0747 | 0.0376 | 0.2846 | 0.7409 |

**Takeaways:**
- User-based CF roughly **doubles** both Hit Rate@10 and NDCG@10 over the non-personalized popularity baseline — confirms personalization is doing real work, not just noise.
- Catalog coverage jumps from ~5% to ~28%, showing CF draws recommendations from a much wider slice of the catalog rather than serving the same shortlist to everyone.
- Intra-list diversity stayed roughly flat (~0.75 for both) — both models already spread each individual user's list across a reasonable genre mix, so this metric wasn't the differentiator here. Included regardless, since it's tracked from Week 1 onward specifically to catch models that might quietly sacrifice per-list variety for accuracy — worth watching as later models (ALS, NCF) are added.
- Note: our Hit Rate/NDCG numbers run somewhat below commonly-cited reference ranges for this dataset, likely due to differences in evaluation protocol (e.g., k, candidate pool size). The comparison *between our own models*, on a fixed shared split, is the reliable signal here.

├── src/
│   ├── data/            # load_data.py, split.py
│   ├── models/          # popularity.py, user_cf.py
│   ├── eval/            # metrics.py (NDCG, Hit Rate), diversity.py
│   └── utils/