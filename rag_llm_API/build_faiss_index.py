"""
build_faiss_index.py
====================
Builds a FAISS vector index from NHS medical conditions stored in MongoDB.

Strategy:
  • Each MongoDB record (one medical condition) → ONE chunk.
  • This guarantees that a similarity search for patient symptoms returns
    the *specific* condition(s) most relevant, not a mix of 50 unrelated ones.
  • Vectors are L2-normalised so FAISS Inner-Product search ≡ cosine similarity.
  • Both the index and a metadata file are saved for the RAG pipeline.
"""

import os

# Thread limits — must be set before numpy/faiss are imported
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import time
import sys
from pathlib import Path
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# =============================================================
# Logging Helpers
# =============================================================
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"
LINE   = "═" * 64

def header(text):
    print(f"\n{CYAN}{LINE}{RESET}")
    print(f"{CYAN}{BOLD}  {text}{RESET}")
    print(f"{CYAN}{LINE}{RESET}")

def step(num, text):
    print(f"\n{YELLOW}▸ Step {num}:{RESET} {BOLD}{text}{RESET}")

def info(text):
    print(f"  {DIM}ℹ {text}{RESET}")

def success(text):
    print(f"  {GREEN}✅ {text}{RESET}")

def warn(text):
    print(f"  {YELLOW}⚠️  {text}{RESET}")

def error(text):
    print(f"  {RED}❌ {text}{RESET}")


# =============================================================
# Directory & File Paths
# =============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR  = os.path.join(BASE_DIR, "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH    = os.path.join(INDEX_DIR, "faiss_store.bin")
CHUNKS_PATH   = os.path.join(INDEX_DIR, "chunks.pkl")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

# MongoDB Settings
MONGO_URI       = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME         = "nhs_conditions_db"
COLLECTION_NAME = "conditions"
DATASET_PATH = Path(BASE_DIR).parent / "dataset" / "usable_dataset.csv"


# =============================================================
# 1. Fetch Records from MongoDB
# =============================================================
def fetch_records():
    """Return a list of dicts — one per medical condition."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        records = list(client[DB_NAME][COLLECTION_NAME].find())
        if records:
            success(f"Using {len(records)} records from MongoDB")
            return [{
                "Condition_name": str(doc.get("Condition_name", "")).replace("\ufffd", "'").strip(),
                "Symptoms": str(doc.get("Symptoms", "")).replace("\ufffd", "'").strip(),
                "Causes": str(doc.get("Causes", "")).replace("\ufffd", "'").strip(),
                "Warnings": str(doc.get("Warnings", "")).replace("\ufffd", "'").strip(),
                "Recommendations": str(doc.get("Recommendations", "")).replace("\ufffd", "'").strip(),
            } for doc in records]
    except PyMongoError as exc:
        warn(f"MongoDB unavailable ({exc.__class__.__name__}); using the tracked CSV dataset instead.")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No MongoDB records and no dataset found at {DATASET_PATH}")
    frame = pd.read_csv(DATASET_PATH).fillna("")
    required = ["Condition_name", "Symptoms", "Causes", "Warnings", "Recommendations"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.drop_duplicates(subset=["Condition_name"], keep="first")
    success(f"Using {len(frame)} records from {DATASET_PATH.name}")
    return [{field: str(row[field]).replace("\ufffd", "'").strip() for field in required} for _, row in frame.iterrows()]


# =============================================================
# 2. Build Chunks — One Condition Per Chunk
# =============================================================
# 2. Build Chunks — One Condition Per Chunk
# =============================================================
def build_chunks(records):
    """
    Convert each medical record into:
      1. full_chunks: Rich, detailed text stored for LLM context generation.
      2. search_chunks: Concise text focused on Condition Name + Symptoms
         to avoid embedding dilution & truncation (all-MiniLM-L6-v2 limit).
    """
    full_chunks   = []   # Complete text (saved to chunks.pkl for LLM prompt)
    search_chunks = []   # Concise text (used for FAISS vector embeddings)
    metadata      = []   # Structured metadata for each chunk

    for rec in records:
        name = rec["Condition_name"]
        if not name:
            continue

        symptoms = rec["Symptoms"]
        causes   = rec["Causes"]
        warnings = rec["Warnings"]
        recs     = rec["Recommendations"]

        # Search chunk: Focus ONLY on Condition Name & Symptoms so vector search pinpoints exact symptom matches
        s_parts = [f"Medical Condition: {name}."]
        if symptoms:
            s_parts.append(f"Symptoms: {symptoms}")
        search_chunks.append("\n".join(s_parts))

        # Full chunk: Rich, complete medical information passed to the LLM
        f_parts = [f"Medical Condition: {name}."]
        if symptoms:
            f_parts.append(f"Symptoms: {symptoms}")
        if causes:
            f_parts.append(f"Causes: {causes}")
        if warnings:
            f_parts.append(f"Warnings: {warnings}")
        if recs:
            f_parts.append(f"Recommendations: {recs}")

        chunk_text = "\n".join(f_parts)
        full_chunks.append(chunk_text)
        metadata.append({
            "condition": name,
            "chunk_index": len(full_chunks) - 1,
            "char_length": len(chunk_text),
            "symptoms": symptoms,
            "causes": causes,
            "warnings": warnings,
            "recommendations": recs,
        })

    return full_chunks, search_chunks, metadata


# =============================================================
# 3. Embed Chunks
# =============================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def embed_chunks(chunks):
    """Encode all search chunks and return (model, embeddings_matrix)."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    )
    return model, embeddings


# =============================================================
# 4. Normalise & Create FAISS Index (Cosine Similarity)
# =============================================================
def create_faiss_index(embeddings):
    """
    Normalise vectors → use IndexFlatIP (inner product).
    After L2-normalisation, inner product == cosine similarity.
    """
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# =============================================================
# 5. Sanity Check — Run a Test Query
# =============================================================
def sanity_check(index, chunks, model):
    """Run a few test queries to verify the index works."""
    test_queries = [
        "Difficulty falling asleep at night, waking up frequently",
        "Chest pain, shortness of breath, dizziness",
        "Persistent cough that lasts more than three weeks",
        "Severe headache with sensitivity to light",
        "fatigue shortness of breath headaches chest pains bleeding bruising reoccurring severe infections"
    ]

    print(f"\n    {'Query':<55} {'Top Match':<35} {'Score'}")
    print(f"    {'─'*55} {'─'*35} {'─'*8}")

    for q in test_queries:
        vec = model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(vec)
        scores, idxs = index.search(vec, 1)
        best_idx = idxs[0][0]
        score = scores[0][0]

        first_line = chunks[best_idx].split("\n")[0]
        condition = first_line.replace("Medical Condition: ", "").rstrip(".")

        q_short = (q[:52] + "…") if len(q) > 52 else q
        c_short = (condition[:32] + "…") if len(condition) > 32 else condition
        print(f"    {q_short:<55} {c_short:<35} {score:.4f}")


# =============================================================
# 6. Save Everything
# =============================================================
def save_index(index, chunks, metadata):
    """Persist the FAISS index, text chunks, and metadata to disk."""
    faiss.write_index(index, INDEX_PATH)
    info(f"FAISS index   → {INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    info(f"Text chunks   → {CHUNKS_PATH}")

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    info(f"Metadata      → {METADATA_PATH}")


# =============================================================
# Main Pipeline
# =============================================================
if __name__ == "__main__":
    header("Build FAISS Index — NHS Medical Conditions")
    pipeline_start = time.time()

    # ── Step 1: Fetch ──
    step(1, "Fetching records from MongoDB")
    info(f"URI: {MONGO_URI}  |  DB: {DB_NAME}  |  Collection: {COLLECTION_NAME}")
    records = fetch_records()
    success(f"Fetched {len(records)} medical condition records")

    if not records:
        error("No records found in MongoDB! Aborting.")
        exit(1)

    # ── Step 2: Chunk ──
    step(2, "Building chunks (1 condition = 1 chunk)")
    full_chunks, search_chunks, metadata = build_chunks(records)
    success(f"Created {len(full_chunks)} chunks")

    char_lengths = [m["char_length"] for m in metadata]
    info(f"Chunk sizes — min: {min(char_lengths)} chars  |  max: {max(char_lengths)} chars  |  avg: {sum(char_lengths)//len(char_lengths)} chars")

    # ── Step 3: Embed ──
    step(3, f"Generating embeddings ({EMBEDDING_MODEL})")
    t0 = time.time()
    model, embeddings = embed_chunks(search_chunks)
    success(f"Embedded {embeddings.shape[0]} search chunks → {embeddings.shape[1]}-dim vectors in {time.time()-t0:.2f}s")

    # ── Step 4: Index ──
    step(4, "Creating FAISS index (normalised cosine similarity)")
    index = create_faiss_index(embeddings)
    success(f"FAISS IndexFlatIP created  |  {index.ntotal} vectors  |  dim={index.d}")

    # ── Step 5: Sanity Check ──
    step(5, "Running sanity-check queries")
    sanity_check(index, full_chunks, model)

    # ── Step 6: Save ──
    step(6, "Saving index, chunks & metadata to disk")
    save_index(index, full_chunks, metadata)
    success("All files saved!")

    # ── Summary ──
    total = time.time() - pipeline_start
    header(f"Done!  {len(full_chunks)} conditions indexed in {total:.1f}s")

    # Clean up temp file if exists
    temp_file = os.path.join(BASE_DIR, "_inspect_db.py")
    if os.path.exists(temp_file):
        os.remove(temp_file)
