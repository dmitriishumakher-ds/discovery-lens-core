# Discovery Lens v2

Turn qualitative product-discovery data — user interviews, reviews, support
tickets, usability notes, social threads, internal call notes — into an
evidence-backed [Opportunity-Solution Tree](https://www.producttalk.org/2023/12/opportunity-solution-trees/).

This is an independent, production-focused rebuild of a bootcamp capstone.
The original ([discovery-lens](https://github.com/dmitriishumakher-ds/discovery-lens),
[live demo](https://huggingface.co/spaces/DiscoveryLens/discovery-lens)) was a
four-person team project with a Streamlit UI; this repository is a solo
follow-on that re-architects the pipeline for testability, measurement, and
deployment.

> **Status:** active rebuild. Stage 0 (clean package + tests + CI) and Stage 1
> (eval harness) are in place. Stage 2 (hybrid retrieval) is implemented and
> measured — see [Retrieval evaluation](#retrieval-evaluation); reranking is
> the next step. Stages 3–4 (FastAPI + Docker, showcase) are open.

---

## Why a v2

The v1 pipeline worked and shipped, but carried the usual bootcamp-deadline
debt. v2 fixes it deliberately, and each fix maps to a production concern:

| v1 | v2 | Why it matters |
|----|----|----------------|
| Pipeline logic lived inside a Streamlit button handler | `run_pipeline()` — one framework-agnostic orchestrator | Same core runs under the UI, a future API, and the eval harness |
| `import streamlit` inside `embedder`/`scorer` | Zero UI imports in the pipeline; models injected as callables | Domain logic is decoupled from the delivery layer |
| No tests, only `__main__` smoke blocks | Unit-tested pure logic, ML stubbed out | Correctness is verifiable; CI runs in seconds |
| Constants scattered across five modules | One frozen `PipelineConfig` | A run is fully described by its config |
| Config flags as globals | Chunking strategy, weights, thresholds are parameters | The eval harness can sweep them |
| TensorFlow **and** PyTorch in the image | Sentiment on torch only | ~1.5 GB smaller container |
| `print()` for diagnostics | `logging` | Structured, controllable output |
| Docs drifted from code | Contracts as typed `TypedDict`s, updated with the code | The contract can't silently lie |

## Architecture

```
files ─▶ extractor ─▶ chunker ─▶ [embed] ─▶ [cluster] ─▶ scorer ─▶ source_map ─▶ [build_ost] ─▶ OST
                                    │            │                                     │
                              injected callables (ml / llm extras) — stubbed in tests and eval
```

The steps in brackets are injected into `run_pipeline`, so importing the
package is cheap and the full orchestration is unit-testable without loading
a single model.

## Retrieval evaluation

Twelve hand-labelled queries over the synthetic Revolut corpus (351 chunks,
38 relevant passages). The gold set marks relevant passages by their exact
text, not by chunk id, so it survives a change of chunking strategy — the
labels stay valid when the thing being measured moves.

```bash
python eval/run_retrieval_eval.py --corpus revolut --retriever bm25
python eval/run_retrieval_eval.py --corpus revolut --retriever dense    # needs [ml]
python eval/run_retrieval_eval.py --corpus revolut --retriever hybrid   # needs [ml]
```

| Retriever | MRR | recall@5 | recall@10 |
|---|---|---|---|
| BM25 (lexical baseline) | 0.405 | 0.200 | 0.383 |
| Dense (`all-MiniLM-L6-v2`) | **0.473** | 0.277 | 0.449 |
| Hybrid (RRF over both) | 0.413 | **0.350** | **0.454** |

Raw per-query reports live in `eval/results/`.

### Reading the numbers

**Dense wins on ranking quality, hybrid wins on coverage.** Fusion lifts
recall@5 by 75% over the lexical baseline, but its MRR sits 13% *below* dense
alone. That is not noise, it is what reciprocal rank fusion does: it rewards
passages both retrievers agree on, and in doing so it can displace a passage
that one retriever alone put first. Coverage and top-1 precision pull in
opposite directions here, and no single number would have shown that.

**Three queries carry the whole story.** BM25 scores a flat zero on q01, q04
and q11 — all three phrased abstractly, sharing no vocabulary with the
passages that answer them. This is the lexical gap measured rather than
assumed, and each of the three fails differently:

| Query | BM25 | Dense | Hybrid | What it shows |
|---|---|---|---|---|
| q01 — account freezes and trust | 0.000 | **1.000** | 0.333 | Semantics recovers what wording hides |
| q04 — uncertainty about transfer arrival | 0.000 | **1.000** | 0.000 | Fusion *loses* a query dense solved |
| q11 — accounting integrations and churn | 0.000 | 0.000 | **0.100** | Fusion finds what neither found alone |

*(MRR per query.)*

q04 is the interesting failure. BM25 returns noise for it, but roughly a dozen
passages happen to appear in both rankings at middling positions. Each of
those collects `2 / (60 + rank)`, which outweighs the single correct passage
dense ranked first at `1 / 61` — so the right answer is pushed out of the top
ten by consensus on irrelevant ones. Plain RRF has no way to know that one of
its inputs is uninformative for a given query.

### What this changes

The measured trade-off sets the next step rather than a guess about one:

* Keep hybrid as the **candidate generator** — it has the best recall@10, and
  recall is what a reranker needs to work with.
* Add a **cross-encoder reranker** over those candidates to recover the top-1
  precision fusion gives away. This is the change the numbers actually argue
  for; before this run, "add a reranker" would have been cargo cult.
* Weight the fusion by per-query retriever confidence, so a retriever that is
  uninformative for a query contributes less. q04 is the test case.

### Caveats, stated up front

The corpus is synthetic and the gold set is small — 12 queries, one annotator,
no adjudication. These numbers are a reliable signal for *comparing retrievers
against each other on this corpus*, and nothing more. They are not an estimate
of production quality, and the absolute values should not be quoted as one.

## Install

```bash
pip install -e ".[dev]"            # pure-logic dev: tests, lint, types
pip install -e ".[ml,llm,parsing,app]"   # full runtime
```

## Develop

```bash
pytest          # unit tests (no model downloads)
ruff check src tests
mypy src
```

## Roadmap

- **Stage 0 — clean package + tests + CI** ✅
- **Stage 1 — eval harness** ✅: gold dataset, retrieval metrics (recall@k, MRR).
  Measure *before* improving. Four independent synthetic corpora (Airbnb, Figma,
  Notion, Revolut), each spanning the six source types, let us show metrics hold
  across domains rather than overfitting one dataset. Clustering quality and a
  hallucination check are still open.
- **Stage 2 — hybrid retrieval** 🔶: BM25, dense and RRF fusion implemented and
  scored against Stage 1 — see [Retrieval evaluation](#retrieval-evaluation).
  Cross-encoder reranking and a multilingual embedding-model comparison are next.
- **Stage 3 — FastAPI + Docker**: REST API, `docker compose up`.
- **Stage 4 — showcase**: architecture diagram, before/after metrics table, demo.

## Credit

Built on the original team capstone by Lucas, Mengda, Asma, and Dmitrii
(Neue Fische / Spiced Academy). v2 rebuild by Dmitrii Shumakher.

## License

MIT
