# Synthetic discovery corpora

Four independent synthetic datasets, each simulating the qualitative discovery
data a PM would feed into Discovery Lens. They exist to test that the pipeline
generalises across domains rather than overfitting one dataset.

| Corpus  | Domain           | Source types present                                  |
|---------|------------------|-------------------------------------------------------|
| airbnb  | Travel / lodging | interview, review, ticket, usability, social, internal |
| figma   | Design tool      | interview, review, usability, social, internal        |
| notion  | Workspace / wiki | interview, review, ticket, usability, social, internal |
| revolut | Fintech          | interview, review, ticket, usability, social, internal |

Notes:
- CSV files intentionally use several different column schemas across corpora.
  The extractor keys off column dtype, not column name, and is tested against
  all of them — a deliberate robustness property.
- `revolut/reviews_revolut.csv` is real Google Play data (500 rows) pulled via
  a scrape script; the rest are hand-authored synthetic material.
- All content is synthetic or public-review-derived; no private data.
