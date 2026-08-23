#!/usr/bin/env python3
"""Run the retrieval evaluation on a corpus.

Usage:
    python eval/run_retrieval_eval.py --corpus revolut --retriever bm25
    python eval/run_retrieval_eval.py --corpus revolut --retriever dense    # needs [ml]
    python eval/run_retrieval_eval.py --corpus revolut --retriever hybrid   # needs [ml]

Loads the corpus through the same extractor+chunker the pipeline uses, so
the eval measures the system as it actually runs, not a lab replica.
Results are printed and saved to eval/results/<corpus>_<retriever>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from discovery_lens.chunker import chunk_text
from discovery_lens.evaluation import evaluate_retriever, load_gold
from discovery_lens.extractor import extract_text
from discovery_lens.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parent.parent

# Retriever name → factory. Adding an implementation to the comparison means
# adding one line here; the harness itself never needs to change.
RETRIEVERS = {
    "bm25": BM25Retriever,
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
}

SOURCE_TYPE_HINTS = [
    ("interview", "interview"),
    ("g2", "review"),
    ("review", "review"),
    ("ticket", "ticket"),
    ("usability", "usability"),
    ("reddit", "social"),
    ("social", "social"),
    ("sales", "internal"),
    ("cs_notes", "internal"),
    ("internal", "internal"),
]


def infer_source_type(filename: str) -> str | None:
    name = filename.lower()
    for hint, stype in SOURCE_TYPE_HINTS:
        if hint in name:
            return stype
    return None


def load_corpus_chunks(corpus: str) -> list[dict]:
    corpus_dir = ROOT / "data" / "synthetic" / corpus
    if not corpus_dir.is_dir():
        sys.exit(f"No such corpus: {corpus_dir}")
    chunks: list[dict] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix not in {".txt", ".csv"}:
            continue
        stype = infer_source_type(path.name)
        if stype is None:
            print(f"  ! skipping {path.name}: cannot infer source type", file=sys.stderr)
            continue
        raw = extract_text(path, stype)
        chunks.extend(chunk_text(raw, path.name, stype))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="revolut")
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), default="bm25")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()

    gold_path = ROOT / "eval" / "gold" / f"{args.corpus}.json"
    if not gold_path.exists():
        sys.exit(f"No gold set for corpus '{args.corpus}' at {gold_path}")

    print(f"Corpus: {args.corpus} | Retriever: {args.retriever} | k={args.k}")
    chunks = load_corpus_chunks(args.corpus)
    print(f"Loaded {len(chunks)} chunks")

    gold = load_gold(gold_path)
    retriever = RETRIEVERS[args.retriever]()

    report = evaluate_retriever(retriever, chunks, gold, ks=tuple(args.k))

    if report["warnings"]:
        print("\n--- WARNINGS (gold data quality) ---")
        for w in report["warnings"]:
            print(" ", w)

    print(f"\n--- PER-QUERY ({report['n_queries']} queries) ---")
    header = f"{'query_id':10s} {'n_rel':>5s} {'mrr':>6s}" + "".join(
        f" {'r@' + str(k):>7s}" for k in args.k
    )
    print(header)
    for row in report["per_query"]:
        line = f"{row['query_id']:10s} {row['n_relevant']:5d} {row['mrr']:6.3f}" + "".join(
            f" {row[f'recall@{k}']:7.3f}" for k in args.k
        )
        print(line)

    print("\n--- AGGREGATE ---")
    for name, value in report["aggregate"].items():
        print(f"  {name:10s} {value:.3f}")

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.corpus}_{args.retriever}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
