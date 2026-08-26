from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(r"C:\Users\chiar\Desktop\LBH")
MODEL_REPO = ROOT / "thesis-model-level"
EM_REPO = ROOT / "thesis-external-memory-next"

VAL_SHA256 = "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
AUTHOR = "Agent Phage"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source_split", "")).lower() == "test":
                raise RuntimeError(f"Test row encountered: {path}:{number}")
            if bool(row.get("used_test", False)):
                raise RuntimeError(f"used_test=true encountered: {path}:{number}")
            yield row


def find_val() -> Path:
    matches: list[Path] = []

    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", ".venv", "node_modules", "__pycache__"}
        ]
        if "clean3_train_val_v1.jsonl" in files:
            matches.append(Path(base) / "clean3_train_val_v1.jsonl")

    for path in matches:
        try:
            if sha256_file(path) == VAL_SHA256:
                return path
        except OSError:
            pass

    raise FileNotFoundError(
        "Could not find clean3_train_val_v1.jsonl with the frozen SHA256."
    )


def find_lambdamart_predictions() -> Path:
    root = EM_REPO / "results" / "personalisation" / "external_memory_next"
    matches: list[Path] = []

    for result_path in root.rglob("result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if result.get("experiment") == "lambdamart_external_memory_fusion_v1":
            predictions = result_path.parent / "selected_predictions.jsonl"
            if predictions.is_file():
                matches.append(predictions)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one frozen LambdaMART prediction artifact, found {len(matches)}:\n"
            + "\n".join(map(str, matches))
        )

    return matches[0]


def text_of(item: Any) -> str:
    if isinstance(item, str):
        return item

    if isinstance(item, Mapping):
        for key in ("candidate", "text", "target"):
            if item.get(key) is not None:
                return str(item[key])

    for key in ("candidate", "text", "target"):
        value = getattr(item, key, None)
        if value is not None:
            return str(value)

    raise RuntimeError(f"Cannot extract candidate text from: {item!r}")


def generated_top10(result: Any) -> list[str]:
    payload = result.to_dict() if hasattr(result, "to_dict") else result

    if isinstance(payload, Mapping):
        for key in ("candidates", "top10", "top_k", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                texts = [text_of(item) for item in value]
                if texts:
                    return texts[:10]

    for key in ("candidates", "top10", "results"):
        value = getattr(result, key, None)
        if value is not None:
            texts = [text_of(item) for item in value]
            if texts:
                return texts[:10]

    raise RuntimeError(
        "Cannot locate generated candidates. "
        f"Payload keys={list(payload) if isinstance(payload, Mapping) else type(payload)}"
    )


def pinyin_of(row: Mapping[str, Any]) -> str | Sequence[str]:
    for key in ("pinyin_segments", "pinyin", "typed_pinyin"):
        value = row.get(key)
        if value:
            return value
    raise RuntimeError(f"No pinyin field for row {row.get('row_id')}")


def context_of(row: Mapping[str, Any]) -> str:
    for key in ("context", "preceding_context", "left_context"):
        if key in row:
            return str(row.get(key) or "")
    return ""


def rank_of(gold: str, top10: Sequence[str]) -> int | None:
    try:
        return list(top10).index(gold) + 1
    except ValueError:
        return None


def borda_fuse(adapter_top10: Sequence[str], em_top10: Sequence[str]) -> list[str]:
    adapter_rank = {candidate: rank for rank, candidate in enumerate(adapter_top10, start=1)}
    em_rank = {candidate: rank for rank, candidate in enumerate(em_top10, start=1)}

    surface = set(adapter_rank) | set(em_rank)

    def key(candidate: str) -> tuple[Any, ...]:
        ar = adapter_rank.get(candidate)
        er = em_rank.get(candidate)

        # Top10 Borda points: rank1=10 ... rank10=1, absent=0.
        adapter_points = 0 if ar is None else 11 - ar
        em_points = 0 if er is None else 11 - er
        total = adapter_points + em_points

        # Frozen EM first as deterministic tie-break, then Adapter.
        return (
            -total,
            er if er is not None else 11,
            ar if ar is not None else 11,
            candidate,
        )

    return sorted(surface, key=key)[:10]


def metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    n = len(ranks)

    def top(k: int) -> float:
        return sum(rank is not None and rank <= k for rank in ranks) / n

    return {
        "n": n,
        "top1": top(1),
        "top3": top(3),
        "top5": top(5),
        "mrr_at_10": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / n,
        "missing10": sum(rank is None for rank in ranks) / n,
    }


def transition(before: Sequence[int | None], after: Sequence[int | None]) -> dict[str, int]:
    out = {
        "n": len(before),
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }

    for old, new in zip(before, after):
        old_correct = old == 1
        new_correct = new == 1

        if not old_correct and new_correct:
            out["rescue"] += 1
        elif old_correct and not new_correct:
            out["harm"] += 1
        elif old_correct:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1

    out["net"] = out["rescue"] - out["harm"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/personalisation/hybrid_zero_shot_v1"),
    )
    args = parser.parse_args()

    val_path = find_val()
    em_predictions_path = find_lambdamart_predictions()

    base_checkpoint = (
        MODEL_REPO
        / "local_artifacts"
        / "base_model"
        / "pinyingpt2-concat"
    )
    adapter_path = (
        MODEL_REPO
        / "local_artifacts"
        / "adapters"
        / "static"
        / "agent_full55926.safetensors"
    )

    for path in (base_checkpoint, adapter_path, val_path, em_predictions_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("VAL:", val_path)
    print("VAL SHA256:", sha256_file(val_path))
    print("EM:", em_predictions_path)
    print("BASE:", base_checkpoint)
    print("ADAPTER:", adapter_path)

    em_rows = [
        row for row in iter_jsonl(em_predictions_path)
        if str(row["author"]) == AUTHOR
    ]

    if args.limit is not None:
        em_rows = em_rows[: args.limit]

    wanted_ids = {str(row["row_id"]) for row in em_rows}

    val_by_id = {
        str(row["row_id"]): row
        for row in iter_jsonl(val_path)
        if str(row.get("row_id")) in wanted_ids
    }

    if len(val_by_id) != len(em_rows):
        missing = sorted(wanted_ids - set(val_by_id))
        raise RuntimeError(
            f"Could not resolve all EM row IDs in frozen Train-Val: "
            f"{len(val_by_id)}/{len(em_rows)}; first missing={missing[:5]}"
        )

    sys.path.insert(0, str(MODEL_REPO / "src"))

    import torch

    from model_level.pinyingpt_adapter import (
        SerialAdapterConfig,
        install_adapters,
        load_adapter_bundle,
        set_adapter_training_mode,
    )
    from reference_backend_pinyingpt.backend import PinyinGPTConcatBackend

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    backend = PinyinGPTConcatBackend(base_checkpoint, device=device)

    install_info = install_adapters(
        backend.model,
        config=SerialAdapterConfig(reduction_factor=16),
        expected_layers=12,
    )
    metadata = load_adapter_bundle(backend.model, adapter_path)
    set_adapter_training_mode(backend.model, enabled=False)
    backend.model.eval()

    print("INSTALL:", install_info)
    print(
        "ADAPTER METADATA:",
        {
            key: metadata.get(key)
            for key in (
                "author",
                "global_steps",
                "processed_rows",
                "used_dev3000",
                "used_test",
                "formal_training_checkpoint",
            )
        },
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    suffix = f"smoke_{args.limit}" if args.limit is not None else "agent_full"
    prediction_path = args.output_root / f"predictions_{suffix}.jsonl"
    result_path = args.output_root / f"result_{suffix}.json"

    adapter_ranks: list[int | None] = []
    em_ranks: list[int | None] = []
    hybrid_ranks: list[int | None] = []

    started = time.perf_counter()

    with prediction_path.open("w", encoding="utf-8", newline="\n") as sink:
        with torch.inference_mode():
            for index, em_row in enumerate(em_rows, start=1):
                row_id = str(em_row["row_id"])
                source = val_by_id[row_id]
                gold = str(em_row["gold"])

                em_top10 = list(map(str, em_row["LambdaMART_top10"]))
                if len(em_top10) > 10:
                    em_top10 = em_top10[:10]

                generated = backend.generate(
                    context=context_of(source),
                    typed_pinyin=pinyin_of(source),
                    top_k=10,
                    beam_size=16,
                )
                adapter_top10 = generated_top10(generated)

                hybrid_top10 = borda_fuse(adapter_top10, em_top10)

                ar = rank_of(gold, adapter_top10)
                er = rank_of(gold, em_top10)
                hr = rank_of(gold, hybrid_top10)

                adapter_ranks.append(ar)
                em_ranks.append(er)
                hybrid_ranks.append(hr)

                sink.write(
                    json.dumps(
                        {
                            "row_id": row_id,
                            "author": AUTHOR,
                            "gold": gold,
                            "Adapter_rank": ar,
                            "LambdaMART_rank": er,
                            "HybridH0_rank": hr,
                            "Adapter_top10": adapter_top10,
                            "LambdaMART_top10": em_top10,
                            "HybridH0_top10": hybrid_top10,
                            "fusion": "equal_weight_top10_borda_rank_fusion",
                            "used_dev3000": False,
                            "used_test": False,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

                if index == 1 or index % 32 == 0 or index == len(em_rows):
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"{index}/{len(em_rows)} "
                        f"rate={index / elapsed:.2f} rows/s",
                        flush=True,
                    )

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "hybrid_zero_shot_agent_adapter_lambdamart_borda_v1",
        "author": AUTHOR,
        "population": "clean3_train_val_v1",
        "rows": len(em_rows),
        "protocol": {
            "adapter": "agent_full55926",
            "external_memory": "frozen_selected_LambdaMART",
            "fusion": "equal_weight_top10_borda_rank_fusion",
            "borda_points": "rank1=10,...,rank10=1,absent=0",
            "training": False,
            "tuning": False,
            "candidate_union": True,
        },
        "metrics": {
            "Adapter": metrics(adapter_ranks),
            "LambdaMART": metrics(em_ranks),
            "HybridH0": metrics(hybrid_ranks),
        },
        "transitions": {
            "LambdaMART_to_HybridH0": transition(em_ranks, hybrid_ranks),
            "Adapter_to_HybridH0": transition(adapter_ranks, hybrid_ranks),
        },
        "provenance": {
            "train_val": str(val_path),
            "train_val_sha256": sha256_file(val_path),
            "external_memory_predictions": str(em_predictions_path),
            "external_memory_predictions_sha256": sha256_file(em_predictions_path),
            "base_checkpoint": str(base_checkpoint),
            "adapter_checkpoint": str(adapter_path),
            "adapter_checkpoint_sha256": sha256_file(adapter_path),
        },
        "used_dev3000": False,
        "used_test": False,
    }

    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print(json.dumps(result["metrics"], indent=2))
    print()
    print(json.dumps(result["transitions"], indent=2))
    print()
    print("RESULT:", result_path.resolve())
    print("PREDICTIONS:", prediction_path.resolve())


if __name__ == "__main__":
    main()
