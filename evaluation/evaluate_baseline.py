"""Reproducible functional evaluation for the local grounded baseline.

This suite checks that known, curated symptom prompts retrieve their expected
condition. It is a regression test for the RAG pipeline, not a clinical
validation study or an estimate of diagnosis accuracy on unseen patients.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from rag_llm_API.retriever import TfidfFaissRetriever


DEFAULT_CASES_PATH = Path(__file__).with_name("baseline_regression_cases.csv")


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).replace("\ufffd", "'").lower())


def evaluate(cases_path: Path, top_k: int) -> dict[str, Any]:
    retriever = TfidfFaissRetriever()
    with cases_path.open(encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle))
    if not cases:
        raise ValueError("The evaluation suite contains no cases.")

    results: list[dict[str, Any]] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        expected = case["expected_condition"].strip()
        candidates = retriever.search(case["prompt"], top_k=top_k)
        labels = [candidate["condition"] for candidate in candidates]
        expected_normalised = _normalise(expected)
        rank = next(
            (index + 1 for index, label in enumerate(labels) if _normalise(label) == expected_normalised),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        results.append(
            {
                "case_id": case["case_id"],
                "focus": case["focus"],
                "expected_condition": expected,
                "top_1": labels[0] if labels else None,
                "top_k_candidates": labels,
                "rank": rank,
                "top_1_correct": rank == 1,
                "top_k_correct": rank is not None,
            }
        )

    count = len(results)
    return {
        "suite": cases_path.name,
        "model": "TF-IDF + FAISS grounded symptom retrieval baseline",
        "case_count": count,
        "top_1_accuracy": sum(item["top_1_correct"] for item in results) / count,
        f"top_{top_k}_recall": sum(item["top_k_correct"] for item in results) / count,
        f"mrr_at_{top_k}": sum(reciprocal_ranks) / count,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    print(json.dumps(evaluate(args.cases, args.top_k), indent=2))


if __name__ == "__main__":
    main()
