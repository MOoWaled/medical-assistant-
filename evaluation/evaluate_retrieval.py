"""Backward-compatible command for the canonical baseline evaluation suite.

This measures grounded candidate retrieval only. It does not measure clinical
diagnostic accuracy and it does not call external LLMs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluate_baseline import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()

    summary = evaluate(
        Path(__file__).with_name("baseline_regression_cases.csv"), args.top_k
    )
    print(f"Cases: {summary['case_count']}")
    print(f"Top-1 accuracy: {summary['top_1_accuracy']:.1%}")
    print(f"Top-{args.top_k} recall: {summary[f'top_{args.top_k}_recall']:.1%}")
    print(f"MRR@{args.top_k}: {summary[f'mrr_at_{args.top_k}']:.2f}")
    print()
    for result in summary["results"]:
        rank = result["rank"] if result["rank"] is not None else "not retrieved"
        candidates = " | ".join(result["top_k_candidates"])
        print(
            f"{result['case_id']}: expected={result['expected_condition']}; "
            f"rank={rank}; candidates={candidates}"
        )


if __name__ == "__main__":
    main()
