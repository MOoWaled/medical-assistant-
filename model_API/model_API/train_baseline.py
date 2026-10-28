"""Build a non-overfitting symptom-retrieval baseline from the project dataset.

There is one source record per condition, so a supervised multi-class classifier
would memorise the labels and cannot be validly evaluated.  This baseline fits a
TF-IDF symptom space and retrieves the nearest known condition instead.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
from pathlib import Path

# Configure numerical libraries before NumPy/FAISS are imported. This applies
# when the trainer is launched directly, not only through the batch launcher.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "usable_dataset.csv"
SUPPLEMENTAL_DATASET_PATH = ROOT_DIR / "dataset" / "supplemental_conditions.csv"
RETRIEVAL_CUES_PATH = ROOT_DIR / "dataset" / "retrieval_cues.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "symptom_retrieval_baseline.joblib"
FAISS_INDEX_PATH = ARTIFACT_PATH.parent / "symptom_tfidf_faiss.bin"
FAISS_RECORDS_PATH = ARTIFACT_PATH.parent / "symptom_tfidf_records.pkl"


def _clean(value: object) -> str:
    return str(value or "").replace("\ufffd", "'").strip()


def _dataset_hash(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_records() -> pd.DataFrame:
    """Load the scraped dataset plus any reviewed NHS supplemental records."""
    sources = [pd.read_csv(DATASET_PATH)]
    if SUPPLEMENTAL_DATASET_PATH.exists():
        sources.append(pd.read_csv(SUPPLEMENTAL_DATASET_PATH))
    return pd.concat(sources, ignore_index=True, sort=False).fillna("")


def _load_retrieval_cues() -> dict[str, str]:
    """Return reviewed, source-linked search phrases keyed by condition name.

    Cues improve matching of common symptom phrasing.  They are used only to
    build the search representation: displayed medical fields still come from
    the original condition records.
    """
    if not RETRIEVAL_CUES_PATH.exists():
        return {}
    cues = pd.read_csv(RETRIEVAL_CUES_PATH).fillna("")
    required = {"Condition_name", "Retrieval_cues", "source_url"}
    if not required.issubset(cues.columns):
        raise ValueError(f"Retrieval cues must contain {sorted(required)}")
    return {
        _clean(row.Condition_name).casefold(): _clean(row.Retrieval_cues)
        for row in cues.itertuples(index=False)
        if _clean(row.Condition_name) and _clean(row.Retrieval_cues) and _clean(row.source_url)
    }


def train(force: bool = False) -> Path:
    dataset_hash = _dataset_hash((DATASET_PATH, SUPPLEMENTAL_DATASET_PATH, RETRIEVAL_CUES_PATH))
    if ARTIFACT_PATH.exists() and FAISS_INDEX_PATH.exists() and FAISS_RECORDS_PATH.exists() and not force:
        existing = joblib.load(ARTIFACT_PATH)
        if existing.get("dataset_hash") == dataset_hash:
            print(f"Baseline is up to date: {ARTIFACT_PATH}")
            return ARTIFACT_PATH

    frame = _load_records()
    required = {"Condition_name", "Symptoms"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dataset must contain {sorted(required)}")

    if "cleaned_Symptoms" in frame.columns:
        cleaned_symptoms = frame["cleaned_Symptoms"].map(_clean)
        raw_symptoms = frame["Symptoms"].map(_clean)
        # Reviewed supplemental records may not have preprocessing columns yet.
        # Preserve them by falling back to their source symptom text.
        frame["_symptoms"] = cleaned_symptoms.where(cleaned_symptoms != "", raw_symptoms)
    else:
        frame["_symptoms"] = frame["Symptoms"].map(_clean)
    frame["_condition"] = frame["Condition_name"].map(_clean)
    frame = frame[(frame["_condition"] != "") & (frame["_symptoms"] != "")]
    frame = frame.drop_duplicates(subset=["_condition"], keep="first").reset_index(drop=True)
    cue_map = _load_retrieval_cues()
    frame["_retrieval_text"] = frame.apply(
        lambda row: " ".join(
            part
            for part in (
                row["_symptoms"],
                cue_map.get(row["_condition"].casefold(), ""),
            )
            if part
        ),
        axis=1,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    symptom_matrix = vectorizer.fit_transform(frame["_retrieval_text"])
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )
    char_symptom_matrix = char_vectorizer.fit_transform(frame["_retrieval_text"])
    records = frame[["_condition", "Symptoms", "Causes", "Warnings", "Recommendations"]].rename(
        columns={"_condition": "Condition_name"}
    ).to_dict(orient="records")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dense_matrix = symptom_matrix.astype(np.float32).toarray()
    faiss.normalize_L2(dense_matrix)
    faiss_index = faiss.IndexFlatIP(dense_matrix.shape[1])
    faiss_index.add(dense_matrix)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    with FAISS_RECORDS_PATH.open("wb") as handle:
        pickle.dump(records, handle)
    joblib.dump(
        {
            "kind": "tfidf_symptom_retrieval",
            "dataset_hash": dataset_hash,
            "vectorizer": vectorizer,
            "symptom_matrix": symptom_matrix,
            "char_vectorizer": char_vectorizer,
            "char_symptom_matrix": char_symptom_matrix,
            "records": records,
            "faiss_index_path": str(FAISS_INDEX_PATH),
            "metrics": {
                "method": "hybrid word and character TF-IDF cosine retrieval over symptom-only records",
                "training_rows": len(records),
                "reviewed_retrieval_cues": len(cue_map),
                "note": "No train/test score is reported because each condition has one source record; a label split would be misleading.",
            },
        },
        ARTIFACT_PATH,
    )
    print(f"Built symptom-only TF-IDF FAISS baseline from {len(records)} conditions: {ARTIFACT_PATH}")
    return ARTIFACT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even if the source dataset is unchanged.")
    args = parser.parse_args()
    train(force=args.force)
