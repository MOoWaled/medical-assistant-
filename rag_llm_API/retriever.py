"""Grounded retrieval shared by the baseline API and both LLM clients.

The retriever is the source of truth for the chatbot.  Models may select from
its candidates, but no model is allowed to invent a condition or clinical text.
"""

from __future__ import annotations

import os
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index" / "faiss_store.bin"
CHUNKS_PATH = BASE_DIR / "faiss_index" / "chunks.pkl"
METADATA_PATH = BASE_DIR / "faiss_index" / "metadata.pkl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SECTION_NAMES = ("Symptoms", "Causes", "Warnings", "Recommendations")
BASELINE_DIR = BASE_DIR.parent / "model_API" / "artifacts"
TFIDF_ARTIFACT_PATH = BASELINE_DIR / "symptom_retrieval_baseline.joblib"
TFIDF_FAISS_INDEX_PATH = BASELINE_DIR / "symptom_tfidf_faiss.bin"
TFIDF_RECORDS_PATH = BASELINE_DIR / "symptom_tfidf_records.pkl"


def _clean_text(value: Any) -> str:
    """Normalise text exported with Windows/CSV character replacement marks."""
    return str(value or "").replace("\ufffd", "'").strip()


def _field(chunk: str, name: str) -> str:
    following = "|".join(SECTION_NAMES)
    pattern = rf"(?:^|\n){re.escape(name)}:\s*(.*?)(?=\n(?:{following}):|\Z)"
    match = re.search(pattern, chunk, flags=re.IGNORECASE | re.DOTALL)
    return _clean_text(match.group(1)) if match else ""


def _condition_from_chunk(chunk: str) -> str:
    match = re.search(r"(?:^|\n)Medical Condition:\s*(.*?)(?:\.\s*$|$)", chunk, re.IGNORECASE)
    return _clean_text(match.group(1)) if match else ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _lexical_score(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(text)) / len(query_tokens)


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_text(name).lower())


class ConditionRetriever:
    """Hybrid FAISS + keyword retriever over one chunk per known condition."""

    def __init__(self) -> None:
        # Keep the default offline TF-IDF mode light-weight. This optional
        # dependency is needed only when RAG_EMBEDDING_BACKEND=sentence_transformer.
        from sentence_transformers import SentenceTransformer

        if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "RAG index is missing. Run `python rag_llm_API/build_faiss_index.py` first."
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        with CHUNKS_PATH.open("rb") as handle:
            self.chunks: list[str] = pickle.load(handle)
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError("FAISS index and chunks file have different record counts. Rebuild the index.")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query = _clean_text(query)
        if not query:
            return []

        vector = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(vector)
        vector_scores, indexes = self.index.search(vector, self.index.ntotal)

        candidates: list[dict[str, Any]] = []
        for index, vector_score in zip(indexes[0], vector_scores[0], strict=True):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            lexical_score = _lexical_score(query, chunk)
            # Semantic similarity ranks paraphrases; lexical overlap preserves exact symptom matches.
            score = 0.70 * float(vector_score) + 0.30 * lexical_score
            candidates.append(
                {
                    "condition": _condition_from_chunk(chunk),
                    "symptoms": _field(chunk, "Symptoms"),
                    "causes": _field(chunk, "Causes"),
                    "warnings": _field(chunk, "Warnings"),
                    "recommendations": _field(chunk, "Recommendations"),
                    "score": round(score, 6),
                    "vector_score": round(float(vector_score), 6),
                    "lexical_score": round(lexical_score, 6),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]


class TfidfFaissRetriever:
    """Offline FAISS retriever fitted only on the dataset's symptom field."""

    def __init__(self) -> None:
        if not TFIDF_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                "Local TF-IDF baseline artefact is missing. Run `python model_API/train_baseline.py`."
            )
        artifact = joblib.load(TFIDF_ARTIFACT_PATH)
        self.vectorizer = artifact["vectorizer"]
        self.records: list[dict[str, Any]] = artifact["records"]
        self.symptom_matrix = artifact["symptom_matrix"]
        self.index = faiss.read_index(str(TFIDF_FAISS_INDEX_PATH)) if TFIDF_FAISS_INDEX_PATH.exists() else None
        if self.index is not None and self.index.ntotal != len(self.records):
            raise RuntimeError("Local TF-IDF FAISS index and records have different counts. Rebuild the baseline.")

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query = _clean_text(query)
        if not query:
            return []
        query_sparse = self.vectorizer.transform([query])
        if self.index is not None:
            vector = query_sparse.astype(np.float32).toarray()
            faiss.normalize_L2(vector)
            scores, indexes = self.index.search(vector, min(top_k, self.index.ntotal))
            ranked = zip(indexes[0], scores[0], strict=True)
        else:
            # The batch launcher normally creates the FAISS binary. Keep a read-only
            # sparse fallback so an existing local artifact remains usable if it cannot.
            sparse_scores = (self.symptom_matrix @ query_sparse.T).toarray().ravel()
            top_indexes = np.argsort(sparse_scores)[-top_k:][::-1]
            ranked = ((int(index), float(sparse_scores[index])) for index in top_indexes)
        candidates = []
        for index, score in ranked:
            record = self.records[int(index)]
            candidates.append(
                {
                    "condition": _clean_text(record.get("Condition_name")),
                    "symptoms": _clean_text(record.get("Symptoms")),
                    "causes": _clean_text(record.get("Causes")),
                    "warnings": _clean_text(record.get("Warnings")),
                    "recommendations": _clean_text(record.get("Recommendations")),
                    "score": round(float(score), 6),
                    "vector_score": round(float(score), 6),
                    "lexical_score": round(_lexical_score(query, _clean_text(record.get("Symptoms"))), 6),
                }
            )
        return candidates

    def get_condition(self, condition_name: str) -> dict[str, Any] | None:
        """Return source fields for an exact condition label from the local records."""
        expected = _normalise_name(condition_name)
        for record in self.records:
            if _normalise_name(_clean_text(record.get("Condition_name"))) == expected:
                return {
                    "condition": _clean_text(record.get("Condition_name")),
                    "symptoms": _clean_text(record.get("Symptoms")),
                    "causes": _clean_text(record.get("Causes")),
                    "warnings": _clean_text(record.get("Warnings")),
                    "recommendations": _clean_text(record.get("Recommendations")),
                }
        return None

def grounded_result(candidate: dict[str, Any], *, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return only fields stored in the knowledge base, never generated clinical text."""
    return {
        "condition": candidate["condition"],
        "symptoms": candidate.get("symptoms") or "Not available",
        "causes": candidate.get("causes") or "Not available",
        "warnings": candidate.get("warnings") or "Not available",
        "recommendations": candidate.get("recommendations") or "Not available",
        "confidence": round(float(candidate.get("score", 0.0)) * 100, 2),
        "retrieval_candidates": [
            {"condition": item["condition"], "score": item["score"]}
            for item in (candidates or [candidate])
        ],
    }


def match_candidate(name: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalised = _normalise_name(name)
    return next((item for item in candidates if _normalise_name(item["condition"]) == normalised), None)


@lru_cache(maxsize=1)
def get_retriever() -> ConditionRetriever:
    backend = os.getenv("RAG_EMBEDDING_BACKEND", "tfidf_faiss").lower()
    if backend == "sentence_transformer":
        return ConditionRetriever()
    return TfidfFaissRetriever()
