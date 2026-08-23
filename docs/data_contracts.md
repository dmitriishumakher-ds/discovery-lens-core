# Discovery Lens — Data Contracts

Source of truth for all pipeline module I/O. Last updated: Apr 29, 2026.
If a contract needs to change, open a GitHub Issue and tag Lucas first.

---

## Pipeline flow

```
UploadedFile → extractor.py → raw_text (str)
→ chunker.py → chunks (list[dict])
→ embedder.py → embeddings (np.ndarray, n_chunks × 384)
→ clusterer.py → clusters (list[dict])
→ source_map.py → source_map (dict) → st.session_state["source_map"]
→ odi_scorer.py → scored_clusters (list[dict]) → st.session_state["scored_clusters"]
→ llm.py → ost (dict) → st.session_state["ost"]
```

---

## extractor.py

```python
# Input
file: UploadedFile   # Streamlit UploadedFile object
source_type: str     # one of: "interview" | "review" | "ticket" | "usability" | "social" | "internal"

# Output
raw_text: str        # full extracted text, plain string
```

---

## chunker.py

```python
# Input
raw_text: str
filename: str
source_type: str     # same enum as extractor.py

# Output — list of dicts, one per chunk
[
  {
    "chunk_id": str,        # format: "{safe_filename}_{zero_padded_index}" e.g. "interview_01_001"
    "text": str,            # 2–4 sentences
    "filename": str,
    "source_type": str
  },
  ...
]
```

---

## embedder.py

```python
# Input
chunks: list[dict]   # output of chunker.py

# Output
embeddings: np.ndarray   # shape: (n_chunks, 384)
```

Notes: `chunks[i]` and `embeddings[i]` share the same index — never reorder independently.

---

## clusterer.py

```python
# Input
chunks: list[dict]        # output of chunker.py
embeddings: np.ndarray    # output of embedder.py

# Output — list of dicts, one per cluster
[
  {
    "cluster_id": int,
    "representative_chunks": list[dict],   # top 3 chunks by HDBSCAN membership probability (cluster core)
    "boundary_chunks": list[dict],          # 1 chunk with lowest membership probability (outlier signal — consumed by T-10)
    "all_chunk_ids": list[str],             # all chunk_ids belonging to this cluster
    "membership_scores": dict[str, float],  # chunk_id → HDBSCAN membership probability, range 0.0–1.0
  },
  ...
]
```

Notes:
- **Algorithm:** BERTopic with UMAP (5 dims, cosine metric, `min_dist=0.0`) + HDBSCAN (`min_cluster_size=15`, `min_samples=5`, `cluster_selection_method="eom"`). Replaces KMeans+silhouette sweep (T-09, May 14 2026). Prototype notebook: `notebooks/bertopic_hdbscan_prototype.ipynb`. PM sign-off: Lucas (pending PR review).
- **Why HDBSCAN:** density-based, no predefined `k`, handles variable-size themes, produces membership probabilities natively (feed into T-10 hybrid chunk selection). Prototype on Notion synthetic corpus: 8 clusters at silhouette 0.48 (UMAP-5D) vs KMeans 0.47 with only 4 forced-equal partitions.
- **Noise fallback:** HDBSCAN refuses to assign ~7–18% of chunks (`topic_id = -1`). Each noise chunk is reassigned to its nearest cluster by cosine similarity to the cluster mean embedding, with membership capped at 0.5 so true core members stay ranked higher in `representative_chunks`. This preserves the "every chunk has a cluster_id" contract that `source_map.py` and `odi_scorer.py` rely on.
- **KMeans fallback:** Corpora with fewer than 50 chunks fall back to the legacy KMeans path (same output shape, rank-based pseudo-membership in `[0.1, 1.0]`). HDBSCAN cannot find density-rich regions reliably below this size.
- **`representative_chunks` semantics changed.** Previously "top 3 closest to centroid" (KMeans). Now "top 3 by HDBSCAN membership probability". Type and length are unchanged — `llm.py` continues to work without modification.
- **`boundary_chunks` and `membership_scores` are new** (T-09). Both will be consumed by T-10 for hybrid chunk selection. Older modules can ignore these fields safely (they are extra keys in the dict, no contract break).
- **Determinism:** `random_state=42` fixed. UMAP's parallel implementation introduces ±2–3% silhouette variance run-to-run; cluster membership of any given chunk is stable for the same input.

---

## source_map.py

```python
# Input
chunks: list[dict]    # output of chunker.py
clusters: list[dict]  # output of clusterer.py

# Output — flat dict for chunk-level traceability
{
  "<chunk_id>": {
    "text": str,
    "filename": str,
    "source_type": str,
    "cluster_id": int | None   # None if chunk was not assigned to any cluster
  },
  ...
}
```

Notes:
- Must be called after clusterer.py and before llm.py.
- Stored in st.session_state["source_map"].
- Used by the results page to show source quotes per opportunity.
- No LLM, no external API. Pure dict construction.

---

## odi_scorer.py

```python
# Input
clusters:         list[dict]        # output of clusterer.py
chunks:           list[dict]        # output of chunker.py
goal_embedding:   np.ndarray | None # 384-dim goal embedding — pass None to skip goal_relevance
chunk_embeddings: np.ndarray | None # shape (n_chunks, 384), same index order as chunks
                                    # pass None to skip goal_relevance
# total_chunks and total_source_types are derived internally — no extra args needed

# Output — list of dicts, one per cluster, sorted by priority_score descending
[
  {
    "cluster_id": int,
    "cluster_size": int,
    # --- Raw signals (available for UI display and debugging) ---
    "importance": float,             # cluster_size / total_chunks, range 0.0–1.0
    "avg_sentiment": float,          # lxyuan compound mean (positive→+score, negative→-score, neutral→0), range -1.0 to 1.0

    "satisfaction": float,           # (avg_sentiment + 1) / 2, range 0.0–1.0
    "source_type_diversity": float,  # unique source types in cluster / 6 (fixed denominator), range 0.0–1.0
    # --- Four scores shown independently in UI ---
    "odi_score": float,              # importance * (1 - satisfaction), range 0.0–1.0
    "evidence_robustness": float,    # (source_type_diversity * 0.65) + (importance * 0.35), range 0.0–1.0
    "goal_relevance": float,         # mean cosine_sim(goal_embedding, chunk_embeddings) per cluster, range 0.0–1.0
                                     # 1.0 if goal_embedding=None (no dampening)
    "priority_score": float,         # [(odi_score * 0.60) + (evidence_robustness * 0.40)]
                                     # × max(goal_relevance, 0.20), range 0.0–1.0
    # --- Recommendation label (D-02) ---
    "recommendation": str,           # "Act" | "Validate" | "Monitor" | "Deprioritise"
  },
  ...
]
```

Notes:

- Deterministic — no LLM, no external API.
- Sentiment model: lxyuan/distilbert-base-multilingual-cased-sentiments-student (replaced VADER May 13 2026, T-08).
- `source_type_diversity` uses a fixed denominator of 6 (all recognised source types). Stable across sessions.
- `goal_relevance` uses unweighted mean cosine similarity until T-09 (BERTopic + HDBSCAN) lands.
  After T-09, replace with membership-weighted mean. Requires revalidation of D-03 floor value. 
- `priority_score` incorporates goal_relevance as a multiplicative dampening factor floored at 0.20.
  A cluster is never zeroed out — it stays visible in the UI but ranked lower. D-03, May 14 2026.
- Sort key is `priority_score` descending.
- Weights confirmed stable by T-16 sensitivity analysis (May 14 2026): min tau=0.9048, mean tau=0.9947.

### Score definitions

| Score | Formula | What it answers |
|-------|---------|-----------------|
| `odi_score` | `importance × (1 - satisfaction)` | How underserved is this need? |
| `evidence_robustness` | `(source_type_diversity × 0.65) + (importance × 0.35)` | How robustly evidenced across source types? |
| `goal_relevance` | `mean cosine_sim(goal_embedding, chunk_embeddings)` clipped to [0, 1] | How directly does this cluster address the stated goal? |
| `priority_score` | `[(odi_score × 0.60) + (evidence_robustness × 0.40)] × max(goal_relevance, 0.20)` | What should a PM act on first? |

### Recommendation label thresholds (D-02, May 14 2026)

| Label | Condition | Meaning |
|-------|-----------|---------|
| `Act` | `odi_score ≥ 0.10` AND `evidence_robustness ≥ 0.40` | High unmet need, well-evidenced — prioritise for roadmap |
| `Validate` | `odi_score ≥ 0.10` AND `evidence_robustness < 0.40` | Strong signal, thin evidence — run more research first |
| `Monitor` | `odi_score < 0.10` AND `evidence_robustness ≥ 0.40` | Well-evidenced but not urgently underserved — keep on radar |
| `Deprioritise` | `odi_score < 0.10` AND `evidence_robustness < 0.40` | Weak signal, sparse evidence — not worth roadmap space now |

Thresholds calibrated to synthetic dataset score distributions (May 14 2026).
Re-evaluate when real PM data is loaded. 

---

## llm.py

```python
# Input
clusters: list[dict]          # output of clusterer.py
scored_clusters: list[dict]   # output of odi_scorer.py
goal: str                     # from st.session_state["goal"]
context_block: str            # from st.session_state["context_block"], default ""
                              # max 500 words — truncated at upload step before storage
                              # injected into user message after cluster evidence; omitted if empty

# LLM generates via Groq — JTBD and solutions only, no score fields
{
  "goal": str,
  "opportunities": [
    {
      "jtbd": str,                    # strictly: "When I [situation], I want to [motivation], so I can [outcome]."
      "job_type": str,                # "functional" | "emotional" | "social" — LLM-generated, not injected
      "jtbd_confidence": str,         # "high" | "medium" | "low" — LLM-generated; overridden to "low" for clusters with <= 3 chunks
      "jtbd_confidence_reason": str,  # one sentence — LLM-generated; deterministic for overridden clusters (states chunk count)
      "cluster_id": int,
      "solutions": [
        {
          "label": str,
          "assumptions": [
            {
              "text": str,
              "risk": str   # "low" | "medium" | "high"
            }
          ]
        }
      ]
    }
  ]
}

# After parsing, llm.py merges scored_clusters on cluster_id to produce the final OST:
{
  "goal": str,
  "opportunities": [
    {
      "jtbd": str,
      "job_type": str,                # passed through from LLM
      "jtbd_confidence": str,         # passed through from LLM, or "low" if overridden
      "jtbd_confidence_reason": str,  # passed through from LLM, or chunk-count sentence if overridden
      "cluster_id": int,
      # --- Injected from scored_clusters, never LLM-generated ---
      "importance": float | None,
      "satisfaction": float | None,
      "source_type_diversity": float | None,
      "odi_score": float | None,
      "evidence_robustness": float | None,
      "priority_score": float | None,
      "solutions": [...]
    }
  ]
}
```

Notes:
- Score fields are **never generated by the LLM** — injected post-parse by merging with `scored_clusters` on `cluster_id`.
- `job_type` **is** LLM-generated and passed through unchanged. Valid values: `functional` | `emotional` | `social`. Rubric in `prompts/system_prompt.txt` Rule 8. PM sign-off: Lucas (May 14 2026).
- If a `cluster_id` from the LLM has no match in `scored_clusters`, set all score fields to `null` — do not crash.
- Always instruct the model to return only valid JSON — no preamble, no markdown fences.
- On JSON parse or validation failure, retry once with `llama-3.1-8b-instant` (fallback).

---

## session_state keys

```python
st.session_state["goal"]            # str — product goal statement
st.session_state["product_name"]    # str — product name
st.session_state["context_block"]   # str — stakeholder/constraint context, max 500 words, default ""
st.session_state["chunks"]          # list[dict] — output of chunker.py
st.session_state["embeddings"]      # np.ndarray — output of embedder.py
st.session_state["clusters"]        # list[dict] — output of clusterer.py
st.session_state["scored_clusters"] # list[dict] — output of odi_scorer.py
st.session_state["ost"]             # dict — merged OST JSON (LLM output + injected scores)
st.session_state["source_map"]      # dict — chunk_id → {text, filename, source_type, cluster_id}
```

Notes: `scored_clusters` must be populated **before** `llm.py` is called so the merge step can do a simple dict lookup without a second pass through raw data.
