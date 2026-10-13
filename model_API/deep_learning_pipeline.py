"""RNN/LSTM training pipeline for the project's symptom-condition dataset.

The source dataset has one NHS document for each condition. To expose the
sequence models to more than one input window per condition, the symptom field
is split into overlapping token chunks. Validation therefore measures held-out
chunks from known condition documents; it is an engineering regression metric,
not clinical diagnostic accuracy or independent patient validation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT_DIR / "dataset" / "usable_dataset.csv"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints" / "deep_learning"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class SymptomChunk:
    tokens: tuple[str, ...]
    label: str


def _clean(value: object) -> str:
    return str(value or "").replace("\ufffd", "'").strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_condition_documents(path: Path = DATASET_PATH) -> list[tuple[str, list[str]]]:
    frame = pd.read_csv(path).fillna("")
    required = {"Condition_name", "Symptoms"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dataset must contain {sorted(required)}")
    documents = []
    seen = set()
    for _, row in frame.iterrows():
        label = _clean(row["Condition_name"])
        tokens = tokenize(_clean(row["Symptoms"]))
        if label and tokens and label not in seen:
            documents.append((label, tokens))
            seen.add(label)
    if len(documents) < 2:
        raise ValueError("At least two labelled condition documents are required.")
    return documents


def build_chunks(
    documents: list[tuple[str, list[str]]], *, chunk_size: int, stride: int, max_chunks_per_condition: int
) -> list[SymptomChunk]:
    if chunk_size < 8 or stride < 1 or max_chunks_per_condition < 1:
        raise ValueError("Chunk size, stride, and max chunks must be positive (chunk size >= 8).")
    chunks: list[SymptomChunk] = []
    for label, tokens in documents:
        starts = list(range(0, max(1, len(tokens) - chunk_size + 1), stride))
        final_start = max(0, len(tokens) - chunk_size)
        if final_start not in starts:
            starts.append(final_start)
        for start in sorted(set(starts))[:max_chunks_per_condition]:
            chunks.append(SymptomChunk(tuple(tokens[start : start + chunk_size]), label))
    return chunks


def split_chunks_by_condition(
    chunks: list[SymptomChunk], validation_fraction: float, seed: int
) -> tuple[list[SymptomChunk], list[SymptomChunk]]:
    """Hold out one or more chunks where a document supplies multiple chunks."""
    grouped: dict[str, list[SymptomChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.label].append(chunk)
    rng = random.Random(seed)
    train, validation = [], []
    for label_chunks in grouped.values():
        shuffled = list(label_chunks)
        rng.shuffle(shuffled)
        validation_count = 0
        if len(shuffled) >= 2:
            validation_count = min(max(1, round(len(shuffled) * validation_fraction)), len(shuffled) - 1)
        validation.extend(shuffled[:validation_count])
        train.extend(shuffled[validation_count:])
    if not validation:
        raise ValueError("No held-out chunks were created. Increase max chunks per condition or reduce chunk size.")
    return train, validation


def build_vocabulary(chunks: list[SymptomChunk], minimum_frequency: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        for token in chunk.tokens:
            counts[token] += 1
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(token for token, count in counts.items() if count >= minimum_frequency):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def create_model(kind: str, vocabulary_size: int, label_count: int, embedding_dim: int, hidden_dim: int):
    """Create Embedding -> RNN/LSTM -> Dropout -> Dense classifier."""
    import torch.nn as nn

    if kind not in {"rnn", "lstm"}:
        raise ValueError("Model kind must be 'rnn' or 'lstm'.")

    class SequenceConditionClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
            self.dropout = nn.Dropout(0.30)
            recurrent_layer = nn.RNN if kind == "rnn" else nn.LSTM
            self.recurrent = recurrent_layer(embedding_dim, hidden_dim, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, label_count)

        def forward(self, input_ids, lengths):
            embedded = self.dropout(self.embedding(input_ids))
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.recurrent(packed)
            hidden_state = hidden[0] if isinstance(hidden, tuple) else hidden
            return self.classifier(self.dropout(hidden_state[-1]))

    return SequenceConditionClassifier()


def create_dataset(chunks: list[SymptomChunk], vocabulary: dict[str, int], label_to_id: dict[str, int], max_length: int):
    import torch
    from torch.utils.data import Dataset

    class ChunkDataset(Dataset):
        def __len__(self) -> int:
            return len(chunks)

        def __getitem__(self, index: int) -> dict[str, Any]:
            chunk = chunks[index]
            ids = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in chunk.tokens[:max_length]]
            length = max(1, len(ids))
            ids.extend([vocabulary[PAD_TOKEN]] * (max_length - len(ids)))
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "lengths": torch.tensor(length, dtype=torch.long),
                "labels": torch.tensor(label_to_id[chunk.label], dtype=torch.long),
            }

    return ChunkDataset()


def evaluate(model: Any, data_loader: Any, device: Any) -> dict[str, float]:
    import torch

    model.eval()
    total = top_1 = top_3 = 0
    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device)
            logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
            top_indexes = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
            top_1 += int((top_indexes[:, 0] == labels).sum().item())
            top_3 += int((top_indexes == labels.unsqueeze(1)).any(dim=1).sum().item())
            total += labels.shape[0]
    return {"held_out_chunk_top_1": top_1 / total, "held_out_chunk_top_3": top_3 / total}


def train(kind: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    documents = load_condition_documents(Path(args.data))
    chunks = build_chunks(
        documents,
        chunk_size=args.chunk_size,
        stride=args.chunk_stride,
        max_chunks_per_condition=args.max_chunks_per_condition,
    )
    train_chunks, validation_chunks = split_chunks_by_condition(chunks, args.validation_fraction, args.seed)
    labels = sorted(label for label, _ in documents)
    label_to_id = {label: index for index, label in enumerate(labels)}
    vocabulary = build_vocabulary(train_chunks, args.minimum_token_frequency)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device auto or --device cpu.")
    model = create_model(kind, len(vocabulary), len(labels), args.embedding_dim, args.hidden_dim).to(device)
    train_loader = DataLoader(
        create_dataset(train_chunks, vocabulary, label_to_id, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        create_dataset(validation_chunks, vocabulary, label_to_id, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    output_dir = CHECKPOINTS_DIR / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    best_score = float("-inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, validation_loader, device)
        metrics.update({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 6)})
        history.append(metrics)
        checkpoint = {
            "model_kind": kind,
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "architecture": {"embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim},
            "max_length": args.max_length,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if metrics["held_out_chunk_top_1"] >= best_score:
            best_score = metrics["held_out_chunk_top_1"]
            torch.save(checkpoint, output_dir / "best.pt")
        print(json.dumps(metrics))
    metadata = {
        "model_kind": kind,
        "dataset": str(Path(args.data)),
        "document_count": len(documents),
        "train_chunk_count": len(train_chunks),
        "validation_chunk_count": len(validation_chunks),
        "label_count": len(labels),
        "checkpoint": "best.pt",
        "history": history,
        "warning": (
            "Validation is performed on held-out chunks from source condition documents. "
            "It is not independent patient validation or clinical diagnostic accuracy."
        ),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"checkpoint_dir": str(output_dir), "best_metrics": max(history, key=lambda item: item["held_out_chunk_top_1"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rnn", "lstm", "all"), default="all")
    parser.add_argument("--data", default=str(DATASET_PATH))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=96)
    parser.add_argument("--chunk-stride", type=int, default=72)
    parser.add_argument("--max-chunks-per-condition", type=int, default=4)
    parser.add_argument("--minimum-token-frequency", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5.")
    requested = ("rnn", "lstm") if args.model == "all" else (args.model,)
    try:
        print(json.dumps([train(kind, args) for kind in requested], indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
