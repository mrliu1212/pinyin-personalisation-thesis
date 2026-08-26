from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(r"C:\Users\chiar\Desktop\LBH")
MODEL_REPO = ROOT / "thesis-model-level"
LF_REPO = ROOT / "thesis-learned-fusion-lab"

AUTHOR = "Agent Phage"

VAL = (
    ROOT
    / "thesis-context-compare"
    / "results"
    / "personalisation"
    / "context_comparison_v2"
    / "clean3_train_val_v1.jsonl"
)

META = (
    LF_REPO
    / "results"
    / "personalisation"
    / "initial_learned_fusion_transfer"
    / "ilt001_matrix_v1"
    / "val_query_meta.jsonl"
)

ILT004 = (
    LF_REPO
    / "results"
    / "personalisation"
    / "initial_learned_fusion_transfer"
    / "ilt004_task_fusion_v1"
    / "predictions.jsonl"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source_split", "")).lower() == "test":
                raise RuntimeError(f"Test row encountered: {path}:{number}")
            if bool(row.get("used_test", False)):
                raise RuntimeError(f"used_test=true: {path}:{number}")
            yield row


def context_of(row: Mapping[str, Any]) -> str:
    for key in ("context", "preceding_context", "left_context"):
        if key in row:
            return str(row.get(key) or "")
    return ""


def pinyin_segments(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("pinyin_segments", row.get("pinyin"))
    if not value:
        raise RuntimeError(f"No Pinyin: {row.get('row_id')}")
    if isinstance(value, str):
        return tuple(value.split())
    return tuple(map(str, value))


def decode_score(item: Any) -> tuple[str, float]:
    payload = item.to_dict() if hasattr(item, "to_dict") else item

    if isinstance(payload, Mapping):
        text = next(
            (
                str(payload[k])
                for k in ("candidate", "text", "target")
                if payload.get(k) is not None
            ),
            None,
        )
        score = next(
            (
                float(payload[k])
                for k in ("score", "log_probability", "logprob", "log_prob")
                if payload.get(k) is not None
            ),
            None,
        )
        if text is not None and score is not None:
            return text, score

    text = next(
        (
            str(getattr(item, k))
            for k in ("candidate", "text", "target")
            if getattr(item, k, None) is not None
        ),
        None,
    )
    score = next(
        (
            float(getattr(item, k))
            for k in ("score", "log_probability", "logprob", "log_prob")
            if getattr(item, k, None) is not None
        ),
        None,
    )

    if text is None or score is None:
        raise RuntimeError(f"Cannot decode score: {item!r}")

    return text, score


def rank_of(gold: str, top10: Sequence[str]) -> int | None:
    try:
        return list(top10).index(gold) + 1
    except ValueError:
        return None


def metrics(ranks: Sequence[int | None]) -> dict[str, float | int]:
    n = len(ranks)

    def top(k: int) -> float:
        return sum(r is not None and r <= k for r in ranks) / n

    return {
        "n": n,
        "top1": top(1),
        "top3": top(3),
        "top5": top(5),
        "mrr_at_10": sum(0.0 if r is None else 1.0 / r for r in ranks) / n,
        "missing10": sum(r is None for r in ranks) / n,
    }


def transition(before, after):
    out = {
        "n": len(before),
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }

    for old, new in zip(before, after):
        oc = old == 1
        nc = new == 1

        if nc and not oc:
            out["rescue"] += 1
        elif oc and not nc:
            out["harm"] += 1
        elif oc:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1

    out["net"] = out["rescue"] - out["harm"]
    return out


def borda(left: Sequence[str], right: Sequence[str]) -> list[str]:
    lr = {x: i for i, x in enumerate(left, start=1)}
    rr = {x: i for i, x in enumerate(right, start=1)}

    surface = set(lr) | set(rr)

    return sorted(
        surface,
        key=lambda x: (
            -(
                (0 if x not in lr else 11 - lr[x])
                + (0 if x not in rr else 11 - rr[x])
            ),
            rr.get(x, 11),
            lr.get(x, 11),
            x,
        ),
    )[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()

    meta_rows = [
        row for row in iter_jsonl(META)
        if str(row["author"]) == AUTHOR
    ]

    if args.limit:
        meta_rows = meta_rows[:args.limit]

    wanted_ids = {str(row["row_id"]) for row in meta_rows}

    ilt = {
        str(row["row_id"]): row
        for row in iter_jsonl(ILT004)
        if str(row["row_id"]) in wanted_ids
    }

    # Matrix IDs differ from original Train-Val IDs.
    # Match by frozen author + chronological position.
    wanted_keys = {
        (str(row["author"]), int(row["chronological_position"]))
        for row in meta_rows
    }

    source_by_key = {}
    for row in iter_jsonl(VAL):
        key = (
            str(row.get("author")),
            int(row.get("chronological_position")),
        )
        if key in wanted_keys:
            if key in source_by_key:
                raise RuntimeError(f"Duplicate Train-Val key: {key}")
            source_by_key[key] = row

    if len(source_by_key) != len(meta_rows):
        raise RuntimeError(
            f"Train-Val alignment failed: "
            f"{len(source_by_key)}/{len(meta_rows)}"
        )

    if len(ilt) != len(meta_rows):
        raise RuntimeError(
            f"ILT004 alignment failed: {len(ilt)}/{len(meta_rows)}"
        )

    sys.path.insert(0, str(MODEL_REPO / "src"))

    import torch

    from model_level.pinyin_modes import full_to_initial
    from model_level.pinyingpt_adapter import (
        SerialAdapterConfig,
        install_adapters,
        load_adapter_bundle,
        set_adapter_training_mode,
    )
    from reference_backend_pinyingpt.backend import PinyinGPTConcatBackend

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    backend = PinyinGPTConcatBackend(base_checkpoint, device=device)

    install_adapters(
        backend.model,
        config=SerialAdapterConfig(reduction_factor=16),
        expected_layers=12,
    )
    metadata = load_adapter_bundle(backend.model, adapter_path)
    set_adapter_training_mode(backend.model, enabled=False)
    backend.model.eval()

    print("DEVICE:", device)
    print(
        "ADAPTER:",
        {
            k: metadata.get(k)
            for k in (
                "author",
                "processed_rows",
                "used_dev3000",
                "used_test",
            )
        },
    )
    print("ROWS:", len(meta_rows))

    adapter_ranks = []
    ilt_ranks = []
    hybrid_ranks = []

    both_correct = 0
    adapter_only_correct = 0
    ilt_only_correct = 0
    neither_correct = 0

    output_rows = []
    started = time.perf_counter()

    with torch.inference_mode():
        for number, meta in enumerate(meta_rows, start=1):
            row_id = str(meta["row_id"])
            gold = str(meta["gold"])

            key = (
                str(meta["author"]),
                int(meta["chronological_position"]),
            )
            source = source_by_key[key]

            candidates = list(map(str, meta["candidates"]))
            frozen = ilt[row_id]
            ilt_top10 = list(map(str, frozen["generic_task_shape_top10"]))

            if set(candidates) != set(ilt_top10):
                raise RuntimeError(
                    f"ILT004 candidate surface changed: {row_id}"
                )

            initials = full_to_initial(pinyin_segments(source))

            scores = backend.score_candidates(
                context=context_of(source),
                typed_pinyin=initials,
                candidates=candidates,
            )
            score_map = dict(decode_score(item) for item in scores)

            if set(score_map) != set(candidates):
                raise RuntimeError(
                    f"Adapter candidate surface mismatch: {row_id}"
                )

            baseline_order = {
                x: i
                for i, x in enumerate(
                    map(str, meta["baseline_top10"]),
                    start=1,
                )
            }

            adapter_top10 = sorted(
                candidates,
                key=lambda x: (
                    -score_map[x],
                    baseline_order.get(x, 11),
                    x,
                ),
            )

            hybrid_top10 = borda(adapter_top10, ilt_top10)

            ar = rank_of(gold, adapter_top10)
            ir = rank_of(gold, ilt_top10)
            hr = rank_of(gold, hybrid_top10)

            adapter_ranks.append(ar)
            ilt_ranks.append(ir)
            hybrid_ranks.append(hr)

            ac = ar == 1
            ic = ir == 1

            if ac and ic:
                both_correct += 1
            elif ac:
                adapter_only_correct += 1
            elif ic:
                ilt_only_correct += 1
            else:
                neither_correct += 1

            output_rows.append(
                {
                    "row_id": row_id,
                    "author": AUTHOR,
                    "gold": gold,
                    "initial_pinyin": list(initials),
                    "Adapter_rank": ar,
                    "ILT004D_rank": ir,
                    "HybridBorda_rank": hr,
                    "Adapter_top10": adapter_top10,
                    "ILT004D_top10": ilt_top10,
                    "HybridBorda_top10": hybrid_top10,
                    "used_dev3000": False,
                    "used_test": False,
                }
            )

            if (
                number == 1
                or number % 32 == 0
                or number == len(meta_rows)
            ):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"{number}/{len(meta_rows)} "
                    f"rate={number / elapsed:.2f} rows/s",
                    flush=True,
                )

    oracle_top1 = (
        both_correct
        + adapter_only_correct
        + ilt_only_correct
    ) / len(meta_rows)

    result = {
        "status": "complete",
        "experiment": "initial_zero_shot_adapter_ilt004d_borda_v1",
        "author": AUTHOR,
        "rows": len(meta_rows),
        "metrics": {
            "AdapterOnFixedInitialSurface": metrics(adapter_ranks),
            "FrozenILT004D": metrics(ilt_ranks),
            "InitialBordaH0": metrics(hybrid_ranks),
        },
        "complementarity": {
            "both_top1_correct": both_correct,
            "adapter_only_top1_correct": adapter_only_correct,
            "ilt004_only_top1_correct": ilt_only_correct,
            "neither_top1_correct": neither_correct,
            "oracle_either_system_top1": oracle_top1,
        },
        "transitions": {
            "ILT004D_to_Hybrid": transition(ilt_ranks, hybrid_ranks),
            "Adapter_to_Hybrid": transition(adapter_ranks, hybrid_ranks),
        },
        "training": False,
        "tuning": False,
        "used_dev3000": False,
        "used_test": False,
    }

    root = Path(
        "results/personalisation/"
        "initial_hybrid_zero_shot_v1"
    )
    root.mkdir(parents=True, exist_ok=True)

    suffix = f"smoke_{args.limit}" if args.limit else "agent_full"

    (root / f"result_{suffix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (root / f"predictions_{suffix}.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("===== METRICS =====")
    print(json.dumps(result["metrics"], indent=2))

    print()
    print("===== COMPLEMENTARITY =====")
    print(json.dumps(result["complementarity"], indent=2))

    print()
    print("===== TRANSITIONS =====")
    print(json.dumps(result["transitions"], indent=2))


if __name__ == "__main__":
    main()
