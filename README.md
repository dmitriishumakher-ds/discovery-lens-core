# Discovery Lens v2

Turn qualitative product-discovery data — user interviews, reviews, support
tickets, usability notes, social threads, internal call notes — into an
evidence-backed [Opportunity-Solution Tree](https://www.producttalk.org/2023/12/opportunity-solution-trees/).

This is an independent, production-focused rebuild of a bootcamp capstone.
The original ([discovery-lens](https://huggingface.co/spaces/DiscoveryLens/discovery-lens))
was a four-person team project; this repository is a solo follow-on that
re-architects the pipeline for testability, measurement, and deployment.

> **Status:** active rebuild. Stage 0 (clean package + test harness + CI) is
> in place. Stages 1–4 (eval harness, hybrid retrieval, FastAPI + Docker,
> showcase) are in progress — see the roadmap below.

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
- **Stage 1 — eval harness**: gold dataset, retrieval metrics (recall@k, MRR),
  clustering quality, hallucination check. Measure *before* improving. Four
  independent synthetic corpora (Airbnb, Figma, Notion, Revolut), each spanning
  the six source types, let us show metrics hold across domains rather than
  overfitting one dataset.
- **Stage 2 — hybrid retrieval**: BM25 + dense + reranker; embedding-model
  comparison incl. multilingual, each change scored against Stage 1.
- **Stage 3 — FastAPI + Docker**: REST API, `docker compose up`.
- **Stage 4 — showcase**: architecture diagram, before/after metrics table, demo.

## Credit

Built on the original team capstone by Lucas, Mengda, Asma, and Dmitrii
(Neue Fische / Spiced Academy). v2 rebuild by Dmitrii Shumakher.

## License

MIT
