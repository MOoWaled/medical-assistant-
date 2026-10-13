"""Grounded baseline API.

The former Logistic Regression memorised one long document per condition.  This
service now uses the same symptom-only retrieval knowledge base as the LLMs and
returns source fields verbatim from the selected record.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
from flask import Flask, jsonify, request

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from rag_llm_API.retriever import get_retriever, grounded_result, match_candidate
from model_API.deep_learning_inference import DeepLearningModelUnavailable, predict as predict_deep_model
from model_API.train_baseline import ARTIFACT_PATH, train


app = Flask(__name__)


def _fallback_tfidf_result(text: str) -> dict:
    """Use the locally fitted symptom baseline if FAISS cannot be loaded."""
    if not ARTIFACT_PATH.exists():
        train()
    artifact = joblib.load(ARTIFACT_PATH)
    query_vector = artifact["vectorizer"].transform([text])
    scores = (artifact["symptom_matrix"] @ query_vector.T).toarray().ravel()
    index = int(np.argmax(scores))
    record = artifact["records"][index]
    candidate = {
        "condition": str(record["Condition_name"]),
        "symptoms": str(record.get("Symptoms", "")),
        "causes": str(record.get("Causes", "")),
        "warnings": str(record.get("Warnings", "")),
        "recommendations": str(record.get("Recommendations", "")),
        "score": float(scores[index]),
    }
    return grounded_result(candidate)


def _predict(text: str) -> dict:
    try:
        candidates = get_retriever().search(text, top_k=3)
        if candidates:
            return grounded_result(candidates[0], candidates=candidates)
    except Exception as exc:
        app.logger.warning("FAISS retrieval unavailable; using local TF-IDF fallback: %s", exc)
    return _fallback_tfidf_result(text)


def _patient_context(data: dict) -> dict[str, int | str | None]:
    """Validate optional patient metadata without inventing medical rules for it."""
    context: dict[str, int | str | None] = {"age": None, "gender": None}

    raw_age = data.get("age")
    if raw_age not in (None, ""):
        try:
            age = int(raw_age)
        except (TypeError, ValueError) as error:
            raise ValueError("Age must be a whole number from 0 to 120.") from error
        if not 0 <= age <= 120:
            raise ValueError("Age must be between 0 and 120.")
        context["age"] = age

    raw_gender = data.get("gender")
    if raw_gender not in (None, ""):
        gender = str(raw_gender).strip()
        if len(gender) > 64:
            raise ValueError("Gender value is too long.")
        context["gender"] = gender or None
    return context


def _deep_prediction(text: str, model_name: str) -> dict:
    """Rank only conditions retrieved by RAG and expose only their source fields."""
    retriever = get_retriever()
    rag_candidates = retriever.search(text, top_k=8)
    if not rag_candidates:
        raise DeepLearningModelUnavailable(
            "No matching condition was retrieved from the local RAG knowledge base."
        )
    prediction = predict_deep_model(
        model_name,
        text,
        top_k=len(rag_candidates),
        allowed_conditions=[candidate["condition"] for candidate in rag_candidates],
    )
    deep_record = match_candidate(prediction["condition"], rag_candidates)
    if deep_record is None:  # Defensive guard: the inference allow-list must make this unreachable.
        raise DeepLearningModelUnavailable("The deep-learning selection was not present in local RAG candidates.")
    held_out_top_1 = float(prediction["training_metrics"].get("held_out_chunk_top_1", 0.0))
    # One document per condition cannot demonstrate a dependable supervised classifier.
    # Until an independently evaluated checkpoint clears this conservative guard,
    # retain the RAG top match while still returning the sequence-model ranking.
    use_deep_selection = held_out_top_1 >= 0.20
    record = deep_record if use_deep_selection else rag_candidates[0]
    result = grounded_result(record, candidates=rag_candidates)
    if use_deep_selection:
        # This is conditional within RAG candidates, never a clinical probability.
        result["confidence"] = round(prediction["confidence"] * 100, 2)
    result["model_candidates"] = prediction["candidates"]
    result["selection_mode"] = "deep_reranked_rag" if use_deep_selection else "rag_quality_guard"
    result["training_metrics"] = prediction["training_metrics"]
    return result


@app.get("/health")
def health_check():
    return jsonify(
        {
            "status": "Active",
            "model_name": "grounded_symptom_retrieval_baseline",
            "artifact_path": str(ARTIFACT_PATH),
            "rag_index_available": (ROOT_DIR / "rag_llm_API" / "faiss_index" / "faiss_store.bin").exists(),
        }
    )


@app.post("/predict")
def predict_disease():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please provide non-empty text in the JSON body."}), 400
    try:
        patient_context = _patient_context(data)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    result = _predict(text)
    details = {key: result[key] for key in ("condition", "symptoms", "causes", "warnings", "recommendations")}
    return jsonify(
        {
            "status": "success",
            "predicted_condition": result["condition"],
            "confidence": result["confidence"],
            "medical_details": details,
            "retrieval_candidates": result["retrieval_candidates"],
            "patient_context": patient_context,
            "metadata_note": (
                "Age and gender are optional request context. The baseline retrieves only from the local "
                "knowledge base, which does not currently contain demographic eligibility fields."
            ),
        }
    )


@app.post("/predict/deep-learning")
def predict_with_deep_learning():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Please provide non-empty text in the JSON body."}), 400
    model_name = str(data.get("deep_model", "lstm")).lower().strip()
    try:
        patient_context = _patient_context(data)
        result = _deep_prediction(text, model_name)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except DeepLearningModelUnavailable as error:
        return jsonify({"error": str(error)}), 409
    details = {key: result[key] for key in ("condition", "symptoms", "causes", "warnings", "recommendations")}
    return jsonify(
        {
            "status": "success",
            "model_name": f"deep_learning_{model_name}",
            "predicted_condition": result["condition"],
            "confidence": result["confidence"],
            "medical_details": details,
            "retrieval_candidates": result["retrieval_candidates"],
            "model_candidates": result["model_candidates"],
            "training_metrics": result["training_metrics"],
            "selection_mode": result["selection_mode"],
            "patient_context": patient_context,
            "metadata_note": (
                "RAG retrieves the only eligible condition records first; the deep model ranks that closed set. "
                "If checkpoint validation is too weak, a quality guard keeps RAG's top match. Symptoms, causes, "
                "warnings, and recommendations always come from the local knowledge base."
            ),
        }
    )


if __name__ == "__main__":
    train()  # Build/update the local, symptom-only baseline when the dataset changes.
    app.run(host="0.0.0.0", port=5000, debug=False)
