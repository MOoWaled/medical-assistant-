"""Inference from trained RNN/LSTM checkpoints for the Flask API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch

from model_API.deep_learning_pipeline import CHECKPOINTS_DIR, PAD_TOKEN, UNK_TOKEN, create_model, tokenize


class DeepLearningModelUnavailable(RuntimeError):
    """Raised when Streamlit requests a model that has not been trained."""


@lru_cache(maxsize=2)
def _load_model(kind: str):
    checkpoint_path = CHECKPOINTS_DIR / kind / "best.pt"
    vocabulary_path = CHECKPOINTS_DIR / kind / "vocabulary.json"
    if not checkpoint_path.exists() or not vocabulary_path.exists():
        raise DeepLearningModelUnavailable(
            f"{kind.upper()} has not been trained. Run the Deep Learning notebook or "
            "`python model_API/deep_learning_pipeline.py --model all` first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    labels = checkpoint["labels"]
    architecture = checkpoint["architecture"]
    model = create_model(kind, len(vocabulary), len(labels), architecture["embedding_dim"], architecture["hidden_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary, labels, int(checkpoint["max_length"]), checkpoint.get("metrics", {})


def predict(
    kind: str,
    text: str,
    top_k: int = 3,
    *,
    allowed_conditions: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Rank labels from a trained sequence model.

    When ``allowed_conditions`` is supplied, the softmax is restricted to the
    current RAG result set.  This keeps the deep model from selecting a label
    that the local medical knowledge base did not retrieve for this request.
    """
    if kind not in {"rnn", "lstm"}:
        raise ValueError("Deep Learning model must be rnn or lstm.")
    model, vocabulary, labels, max_length, training_metrics = _load_model(kind)
    ids = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokenize(text)[:max_length]]
    length = max(1, len(ids))
    ids.extend([vocabulary[PAD_TOKEN]] * (max_length - len(ids)))
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor([ids], dtype=torch.long), torch.tensor([length], dtype=torch.long)), dim=1
        )[0]

    indexes_to_rank = list(range(len(labels)))
    if allowed_conditions is not None:
        allowed = {str(condition).casefold().strip() for condition in allowed_conditions}
        indexes_to_rank = [index for index, label in enumerate(labels) if label.casefold().strip() in allowed]
        if not indexes_to_rank:
            raise DeepLearningModelUnavailable(
                "None of the local RAG candidates matches labels in the trained deep-learning checkpoint."
            )

    ranked_indexes = sorted(indexes_to_rank, key=lambda index: float(probabilities[index]), reverse=True)
    ranked_indexes = ranked_indexes[: min(top_k, len(ranked_indexes))]
    candidates = [
        {"condition": labels[index], "score": round(float(probabilities[index]), 6)} for index in ranked_indexes
    ]
    return {
        "condition": candidates[0]["condition"],
        "confidence": candidates[0]["score"],
        "candidates": candidates,
        "training_metrics": training_metrics,
    }
