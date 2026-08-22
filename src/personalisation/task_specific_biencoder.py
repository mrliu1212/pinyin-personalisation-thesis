"""Shared training/runtime helpers for the task-specific context bi-encoder."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


CONTEXT_CHARS = 64
MAX_LENGTH = 128
TEMPERATURE = 0.05
SEED = 1729


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> tuple[str, list[dict[str, Any]]]:
    files = []
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(root).parts
    ):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest()
    return digest, files


def context64(text: str) -> str:
    return str(text)[-CONTEXT_CHARS:]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(canonical_json(dict(row)) + "\n")
    temporary.replace(path)


def refuse_closed_path(path: Path) -> None:
    lowered = str(path).lower()
    if "dev3000" in lowered or "test" in lowered:
        raise ValueError(f"closed evaluation path is forbidden: {path}")


def split_position_cutoffs(
    queries: Sequence[tuple[str, str, int]], fit_fraction: float = 0.9
) -> dict[str, int]:
    """Return the first gate position per author without splitting a position block."""

    by_author: dict[str, dict[str, int]] = defaultdict(dict)
    for query_id, author, position in queries:
        previous = by_author[author].setdefault(query_id, position)
        if previous != position:
            raise ValueError(f"query position changed: {query_id}")
    cutoffs: dict[str, int] = {}
    for author, values in by_author.items():
        by_position: dict[int, int] = defaultdict(int)
        for position in values.values():
            by_position[position] += 1
        target = math.floor(len(values) * fit_fraction)
        fitted = 0
        cutoff = None
        for position, count in sorted(by_position.items()):
            if fitted + count > target:
                cutoff = position
                break
            fitted += count
        if cutoff is None or fitted == 0:
            raise ValueError(f"cannot form chronological inner split for {author}")
        cutoffs[author] = cutoff
    return cutoffs


def assign_inner_split(author: str, position: int, cutoffs: Mapping[str, int]) -> str:
    return "inner_fit" if position < int(cutoffs[author]) else "inner_gate"


def selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = record["metrics"]
    return (
        -float(metrics["macro_author_recall_at_1"]),
        -float(metrics["micro_recall_at_1"]),
        -float(metrics["mrr"]),
        int(record["epoch"]),
    )


def select_epoch(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if {int(record["epoch"]) for record in records} != {1, 2}:
        raise ValueError("the frozen checkpoint surface must contain epochs 1 and 2")
    return min(records, key=selection_key)


def group_retrieval_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no retrieval rows")
    per_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        per_author[str(row["author"])].append(row)

    def summary(values: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "n": len(values),
            "recall_at_1": statistics.fmean(float(row["rank"] <= 1) for row in values),
            "recall_at_5": statistics.fmean(float(row["rank"] <= 5) for row in values),
            "recall_at_10": statistics.fmean(float(row["rank"] <= 10) for row in values),
            "mrr": statistics.fmean(1.0 / int(row["rank"]) for row in values),
        }

    author_metrics = {author: summary(values) for author, values in sorted(per_author.items())}
    overall = summary(rows)
    overall["macro_author_recall_at_1"] = statistics.fmean(
        float(value["recall_at_1"]) for value in author_metrics.values()
    )
    overall["macro_author_mrr"] = statistics.fmean(
        float(value["mrr"]) for value in author_metrics.values()
    )
    overall["micro_recall_at_1"] = overall["recall_at_1"]
    return {"overall": overall, "per_author": author_metrics}


def ranking_metrics(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    per_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        rank = row.get(rank_key)
        rank = None if rank is None else int(rank)
        ranks.append(rank)
        per_author[str(row["author"])].append(rank)

    def summary(values: Sequence[int | None]) -> dict[str, float | int]:
        n = len(values)
        return {
            "n": n,
            "micro_top1": sum(rank == 1 for rank in values) / n,
            "top3": sum(rank is not None and rank <= 3 for rank in values) / n,
            "top5": sum(rank is not None and rank <= 5 for rank in values) / n,
            "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in values) / n,
            "missing10": sum(rank is None for rank in values) / n,
        }

    author_metrics = {author: summary(values) for author, values in sorted(per_author.items())}
    overall = summary(ranks)
    overall["macro_author_top1"] = statistics.fmean(
        float(value["micro_top1"]) for value in author_metrics.values()
    )
    overall["per_author_top1"] = {
        author: float(value["micro_top1"]) for author, value in author_metrics.items()
    }
    return overall


def transition_counts(
    rows: Sequence[Mapping[str, Any]], baseline_key: str, new_key: str
) -> dict[str, int]:
    rescue = sum(row.get(baseline_key) != 1 and row.get(new_key) == 1 for row in rows)
    harm = sum(row.get(baseline_key) == 1 and row.get(new_key) != 1 for row in rows)
    return {"n": len(rows), "rescue": rescue, "harm": harm, "net": rescue - harm}


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


class SharedContextEncoder:
    def __init__(self, model_path: Path, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)

    def tokenize(self, texts: Sequence[str]) -> Mapping[str, Any]:
        return self.tokenizer(
            [context64(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    def encode_tensor(self, texts: Sequence[str]) -> Any:
        tokens = {key: value.to(self.device) for key, value in self.tokenize(texts).items()}
        output = self.model(**tokens)
        return mean_pool(output.last_hidden_state, tokens["attention_mask"])

    def embed(self, texts: Sequence[str], *, batch_size: int = 128) -> Any:
        import numpy as np
        import torch

        self.model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                values.append(self.encode_tensor(texts[start : start + batch_size]).float().cpu().numpy())
        return np.vstack(values) if values else np.empty((0, 0), dtype=np.float32)

    def save(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output, safe_serialization=True)
        self.tokenizer.save_pretrained(output)


def load_groups(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return rows


def batched_group_indices(count: int, batch_size: int, *, seed: int) -> Iterable[list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    for start in range(0, count, batch_size):
        yield indices[start : start + batch_size]


def group_batch_loss(encoder: SharedContextEncoder, groups: Sequence[Mapping[str, Any]]) -> Any:
    import torch

    query_texts = [str(group["query_context"]) for group in groups]
    history_texts = [str(text) for group in groups for text in group["history_contexts"]]
    vectors = encoder.encode_tensor([*query_texts, *history_texts])
    query_vectors = vectors[: len(groups)]
    history_vectors = vectors[len(groups) :]
    offset = 0
    losses = []
    for index, group in enumerate(groups):
        size = len(group["history_contexts"])
        logits = (history_vectors[offset : offset + size] @ query_vectors[index]) / TEMPERATURE
        losses.append(torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=logits.device)))
        offset += size
    return torch.stack(losses).mean()


def evaluate_groups(
    encoder: SharedContextEncoder,
    groups: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    import torch

    encoder.model.eval()
    ranked = []
    with torch.inference_mode():
        for start in range(0, len(groups), batch_size):
            batch = groups[start : start + batch_size]
            queries = [str(group["query_context"]) for group in batch]
            histories = [str(text) for group in batch for text in group["history_contexts"]]
            vectors = encoder.encode_tensor([*queries, *histories])
            query_vectors = vectors[: len(batch)]
            history_vectors = vectors[len(batch) :]
            offset = 0
            for index, group in enumerate(batch):
                size = len(group["history_contexts"])
                scores = history_vectors[offset : offset + size] @ query_vectors[index]
                order = sorted(range(size), key=lambda item: (-float(scores[item]), item))
                ranked.append({"author": group["author"], "rank": order.index(0) + 1})
                offset += size
    return group_retrieval_metrics(ranked)
