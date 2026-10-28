"""Runnable artifact for the Logistic Regression architecture in ``baseline_model.ipynb``.

The notebook is useful for exploration, but notebook variables disappear after a
kernel stops.  This module saves the fitted vectorizer and classifier so the
Streamlit application can use the same architecture reliably.

It is intentionally used as a *reranker* of local RAG candidates.  The source
dataset contains approximately one source document per condition, so fitting a
classifier on all labels is not clinical validation and must not be presented
as a diagnostic probability.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Support both ``python -m model_API.logistic_regression_notebook`` and the
# convenient direct command ``python model_API/logistic_regression_notebook.py``.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_API.train_baseline import DATASET_PATH, SUPPLEMENTAL_DATASET_PATH, _clean, _dataset_hash, _load_records


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "logistic_regression_notebook.joblib"
TEXT_COLUMNS = ("Symptoms", "Causes", "Warnings", "Recommendations")


class LogisticNotebookModelUnavailable(RuntimeError):
    """Raised when the notebook model cannot be built or used."""


def _normalise_name(value: str) -> str:
    return "".join(character for character in _clean(value).lower() if character.isalnum())


def _combined_features(frame: pd.DataFrame) -> pd.Series:
    """Reproduce the notebook's rich-text input, with safe raw-text fallbacks."""
    values: list[pd.Series] = []
    for column in TEXT_COLUMNS:
        cleaned_column = f"cleaned_{column}"
        raw = frame[column].map(_clean) if column in frame else pd.Series("", index=frame.index)
        if cleaned_column in frame:
            cleaned = frame[cleaned_column].map(_clean)
            values.append(cleaned.where(cleaned != "", raw))
        else:
            values.append(raw)
    return pd.concat(values, axis=1).agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()


def train(force: bool = False) -> Path:
    """Fit and save the Logistic Regression model represented by the notebook."""
    dataset_hash = _dataset_hash((DATASET_PATH, SUPPLEMENTAL_DATASET_PATH))
    if ARTIFACT_PATH.exists() and not force:
        artifact = joblib.load(ARTIFACT_PATH)
        if artifact.get("dataset_hash") == dataset_hash:
            print(f"Notebook Logistic Regression model is up to date: {ARTIFACT_PATH}")
            return ARTIFACT_PATH

    frame = _load_records().drop_duplicates().reset_index(drop=True)
    if "Condition_name" not in frame:
        raise LogisticNotebookModelUnavailable("Dataset must contain the Condition_name column.")
    frame["_condition"] = frame["Condition_name"].map(_clean)
    frame["_features"] = _combined_features(frame)
    raw_symptoms = frame["Symptoms"].map(_clean) if "Symptoms" in frame else pd.Series("", index=frame.index)
    if "cleaned_Symptoms" in frame:
        cleaned_symptoms = frame["cleaned_Symptoms"].map(_clean)
        frame["_symptom_features"] = cleaned_symptoms.where(cleaned_symptoms != "", raw_symptoms)
    else:
        frame["_symptom_features"] = raw_symptoms
    frame = frame[(frame["_condition"] != "") & (frame["_features"] != "")]
    frame = frame.drop_duplicates(subset=["_condition"], keep="first").reset_index(drop=True)
    if frame["_condition"].nunique() < 2:
        raise LogisticNotebookModelUnavailable("At least two condition labels are required for Logistic Regression.")

    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), lowercase=True, strip_accents="unicode")
    # The notebook's combined text is preserved, and each label also receives
    # a symptom-only training example.  The latter aligns the model with the
    # symptom-only query typed in the GUI instead of rewarding causes or advice
    # that users normally do not enter.
    training_text = pd.concat([frame["_features"], frame["_symptom_features"]], ignore_index=True)
    training_labels = pd.concat([frame["_condition"], frame["_condition"]], ignore_index=True)
    matrix = vectorizer.fit_transform(training_text)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(matrix, training_labels)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "kind": "notebook_logistic_regression_reranker",
            "dataset_hash": dataset_hash,
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": classifier.classes_.tolist(),
            "metrics": {
                "source_condition_rows": int(len(frame)),
                "training_documents": int(len(training_text)),
                "label_count": int(frame["_condition"].nunique()),
                "note": "Fitted on full source documents. It is used only to rank RAG candidates; no held-out clinical accuracy is claimed.",
            },
        },
        ARTIFACT_PATH,
    )
    print(f"Built Notebook Logistic Regression artifact from {len(frame)} conditions: {ARTIFACT_PATH}")
    return ARTIFACT_PATH


def predict(text: str, *, allowed_conditions: Iterable[str]) -> dict:
    """Select only among RAG-provided conditions and return their relative scores."""
    if not str(text).strip():
        raise LogisticNotebookModelUnavailable("Symptoms must not be empty.")
    if not ARTIFACT_PATH.exists():
        train()
    artifact = joblib.load(ARTIFACT_PATH)
    probabilities = artifact["classifier"].predict_proba(artifact["vectorizer"].transform([text]))[0]
    score_by_label = dict(zip(artifact["classifier"].classes_, probabilities, strict=True))
    allowed = { _normalise_name(name): str(name) for name in allowed_conditions }
    candidates = [
        {"condition": display_name, "score": float(score_by_label.get(label, 0.0))}
        for label, probability in score_by_label.items()
        if (display_name := allowed.get(_normalise_name(label))) is not None
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        raise LogisticNotebookModelUnavailable(
            "None of the local RAG candidates matches labels in the Logistic Regression artifact."
        )
    total = sum(item["score"] for item in candidates)
    for item in candidates:
        item["conditional_score"] = round((item["score"] / total) if total else 0.0, 6)
    return {
        "condition": candidates[0]["condition"],
        "confidence": candidates[0]["conditional_score"],
        "candidates": candidates,
        "training_metrics": artifact["metrics"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even when the dataset is unchanged.")
    args = parser.parse_args()
    train(force=args.force)
