"""MongoDB GridFS model registry for the Medical Condition Assistant.

Every trained model (TF-IDF/FAISS baseline, RNN, LSTM, a future BERT, ...) is
stored as ONE zip "bundle" in GridFS, plus ONE metadata document in the
``models`` collection that follows the project schema:

    name, type, gridfs_id, labels, metrics, created

Extra bookkeeping fields (version, files, size_bytes, sha256, ...) are added
on top of that schema; they never replace it.

Typical use
-----------
    handler = ModelGridFSHandler()                      # MONGO_URI env var is honoured

    handler.save_bundle(
        name="lstm",
        model_type="LSTM",
        files={"best.pt": torch_to_bytes(checkpoint),
               "vocabulary.json": json.dumps(vocabulary).encode("utf-8")},
        labels=labels,
        metrics=best_metrics,
        extra={"dataset_hash": dataset_hash},
        keep_versions=3,
    )

    files, document = handler.load_bundle("lstm")       # latest version
    checkpoint = torch_from_bytes(files["best.pt"])

The old ``save_model`` / ``load_latest_model`` API (model + vectorizer pair) is
kept so ``baseline_model.ipynb`` keeps working, and models that were saved by
the old code can still be loaded.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from datetime import datetime, timezone
from typing import Any, Mapping

import gridfs
import joblib
from pymongo import DESCENDING, MongoClient
from pymongo.errors import PyMongoError

DEFAULT_MONGO_URI = "mongodb://localhost:27017/"
DEFAULT_DB_NAME = "nhs_conditions_db"
MODELS_COLLECTION = "models"

# Newest first. `version` is the primary key because BSON datetimes only have
# millisecond precision: two saves in the same millisecond would tie on `created`.
# Documents written by the old handler have no `version`; they sort last.
_NEWEST_FIRST = [("version", DESCENDING), ("created", DESCENDING)]

# Names used inside a bundle by the backwards-compatible save_model().
_LEGACY_PAIR_FILE = "model_bundle.joblib"
# Name given to a raw (non-zip) GridFS file written by the old handler.
_LEGACY_RAW_FILE = "legacy.joblib"


class ModelRegistryError(RuntimeError):
    """A model could not be saved to, or read back from, the registry."""


# --------------------------------------------------------------------------- #
# Serialisation helpers (heavy libraries are imported lazily on purpose, so   #
# importing this module never requires torch or faiss).                       #
# --------------------------------------------------------------------------- #
def joblib_to_bytes(obj: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


def joblib_from_bytes(data: bytes) -> Any:
    return joblib.load(io.BytesIO(data))


def faiss_to_bytes(index: Any) -> bytes:
    """A FAISS index cannot be joblib-dumped; it has its own serialiser."""
    import faiss

    return faiss.serialize_index(index).tobytes()


def faiss_from_bytes(data: bytes) -> Any:
    import faiss
    import numpy as np

    return faiss.deserialize_index(np.frombuffer(data, dtype="uint8"))


def torch_to_bytes(obj: Any) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def torch_from_bytes(data: bytes, *, weights_only: bool = True) -> Any:
    """Load a checkpoint made of tensors and plain Python containers.

    ``weights_only=True`` refuses arbitrary pickled objects, which is the safe
    default for data that came out of a database.
    """
    import torch

    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=weights_only)


def _to_builtin(value: Any) -> Any:
    """Turn numpy scalars/arrays into plain Python so BSON can store them."""
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "tolist"):  # numpy scalar or ndarray
        return _to_builtin(value.tolist())
    return value


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
class ModelGridFSHandler:
    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str | None = None,
        *,
        client: Any = None,
        timeout_ms: int = 3000,
    ) -> None:
        """Connect to MongoDB.

        ``mongo_uri`` / ``db_name`` default to the MONGO_URI / MONGO_DB_NAME
        environment variables, then to localhost. A short server-selection
        timeout means an unavailable MongoDB fails fast instead of hanging the
        API for 30 seconds. ``client`` lets tests inject a fake client.
        """
        uri = mongo_uri or os.getenv("MONGO_URI", DEFAULT_MONGO_URI)
        name = db_name or os.getenv("MONGO_DB_NAME", DEFAULT_DB_NAME)
        self.client = client or MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        self.db = self.client[name]
        self.fs = gridfs.GridFS(self.db)
        self.models_collection = self.db[MODELS_COLLECTION]

    # ----------------------------------------------------------------- health
    def is_available(self) -> bool:
        """True when MongoDB answers; lets callers fall back to local files."""
        try:
            self.client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    # ------------------------------------------------------------------- save
    def save_bundle(
        self,
        name: str,
        model_type: str,
        files: Mapping[str, bytes],
        labels: list[str],
        metrics: Mapping[str, Any],
        *,
        extra: Mapping[str, Any] | None = None,
        keep_versions: int | None = None,
    ) -> Any:
        """Zip ``files`` into GridFS and register them in the models collection.

        Each call creates a new version of ``name``. ``keep_versions`` (when
        given) deletes the oldest versions beyond that count. Returns the
        inserted metadata document's ``_id``.
        """
        if not name:
            raise ValueError("A model name is required.")
        if not files:
            raise ValueError("A model bundle needs at least one file.")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, data in files.items():
                archive.writestr(filename, data)
        payload = buffer.getvalue()

        version = self._next_version(name)
        file_id = self.fs.put(
            payload,
            filename=f"{name}_v{version}.zip",
            model_name=name,
            version=version,
        )

        # Schema fields come last so `extra` can never overwrite them.
        document = {
            **_to_builtin(extra or {}),
            "name": name,
            "type": model_type,
            "gridfs_id": file_id,
            "labels": [str(label) for label in labels],
            "metrics": _to_builtin(metrics),
            "created": datetime.now(timezone.utc),
            "version": version,
            "files": sorted(files),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        try:
            inserted = self.models_collection.insert_one(document)
        except Exception:
            self.fs.delete(file_id)  # never leave an orphan file without metadata
            raise

        if keep_versions is not None:
            self.prune(name, keep=keep_versions)
        print(f"Model '{name}' v{version} saved to GridFS (file id {file_id}, {len(payload):,} bytes)")
        return inserted.inserted_id

    # ------------------------------------------------------------------- load
    def load_bundle(self, name: str, version: int | None = None) -> tuple[dict[str, bytes], dict[str, Any]]:
        """Return ``(files, metadata_document)`` for the latest (or given) version."""
        query: dict[str, Any] = {"name": name}
        if version is not None:
            query["version"] = int(version)
        document = self.models_collection.find_one(query, sort=_NEWEST_FIRST)
        if document is None:
            label = f"{name} v{version}" if version is not None else name
            raise FileNotFoundError(f"No model found with name: {label}")

        try:
            payload = self.fs.get(document["gridfs_id"]).read()
        except gridfs.errors.NoFile as error:
            raise ModelRegistryError(
                f"Metadata for '{name}' exists but its GridFS file is missing."
            ) from error
        except gridfs.errors.CorruptGridFile as error:
            # GridFS itself noticed a damaged/truncated chunk.
            raise ModelRegistryError(f"The stored model '{name}' is corrupted in GridFS: {error}") from error

        # Catches damage GridFS cannot see (e.g. bytes changed but chunk sizes intact).
        expected = document.get("sha256")
        if expected and hashlib.sha256(payload).hexdigest() != expected:
            raise ModelRegistryError(f"Checksum mismatch for '{name}': the stored model is corrupted.")

        if not zipfile.is_zipfile(io.BytesIO(payload)):
            # Written by the old handler: one raw joblib file, not a zip bundle.
            return {_LEGACY_RAW_FILE: payload}, document
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            files = {member: archive.read(member) for member in archive.namelist()}
        return files, document

    # ------------------------------------------------------------------- list
    def list_models(self, name: str | None = None) -> list[dict[str, Any]]:
        """JSON-friendly metadata of stored models, newest first (for GET /models)."""
        query = {"name": name} if name else {}
        rows = []
        # Listing mixes different models, so order by date first; version breaks same-millisecond ties.
        for document in self.models_collection.find(query, sort=[("created", DESCENDING), ("version", DESCENDING)]):
            row = dict(document)
            row["_id"] = str(row["_id"])
            row["gridfs_id"] = str(row.get("gridfs_id"))
            created = row.get("created")
            row["created"] = created.isoformat() if hasattr(created, "isoformat") else created
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ prune
    def prune(self, name: str, keep: int) -> int:
        """Delete all but the newest ``keep`` versions of ``name``. Returns how many were removed."""
        if keep < 1:
            raise ValueError("keep must be at least 1.")
        documents = list(self.models_collection.find({"name": name}, sort=_NEWEST_FIRST))
        stale = documents[keep:]
        for document in stale:
            self.fs.delete(document["gridfs_id"])
            self.models_collection.delete_one({"_id": document["_id"]})
        return len(stale)

    def _next_version(self, name: str) -> int:
        latest = self.models_collection.find_one({"name": name}, sort=_NEWEST_FIRST)
        return int(latest.get("version") or 0) + 1 if latest else 1

    # ------------------------------------------------------------------------
    # Backwards-compatible API (used by baseline_model.ipynb)
    # ------------------------------------------------------------------------
    def save_model(self, model_object, vectorizer_object, model_name, model_type, labels, metrics):
        """Save a ``model + vectorizer`` pair as a one-file bundle."""
        return self.save_bundle(
            name=model_name,
            model_type=model_type,
            files={_LEGACY_PAIR_FILE: joblib_to_bytes({"model": model_object, "vectorizer": vectorizer_object})},
            labels=list(labels),
            metrics=metrics,
        )

    def load_latest_model(self, model_name):
        """Return ``(model, vectorizer, labels)``; also reads models saved by the old handler."""
        files, document = self.load_bundle(model_name)
        raw = files.get(_LEGACY_PAIR_FILE) or files.get(_LEGACY_RAW_FILE)
        if raw is None:
            raise ModelRegistryError(
                f"'{model_name}' is not a model+vectorizer pair. Use load_bundle() instead."
            )
        saved = joblib_from_bytes(raw)
        return saved["model"], saved["vectorizer"], document["labels"]
