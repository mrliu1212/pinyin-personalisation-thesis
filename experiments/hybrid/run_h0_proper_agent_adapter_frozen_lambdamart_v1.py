from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(r"C:\Users\chiar\Desktop\LBH")
MODEL_REPO = ROOT / "thesis-model-level"
EM_REPO = ROOT / "thesis-external-memory-next"
LF_REPO = ROOT / "thesis-learned-fusion-lab"

AUTHOR = "Agent Phage"

VAL_PATH = (
    ROOT
    / "thesis-context-compare"
    / "results"
    / "personalisation"
    / "context_comparison_v2"
    / "clean3_train_val_v1.jsonl"
)

VAL_SHA256 = "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
STAGE1_SHA256 = "e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13"

FEATURE_NAMES = (
    "base_score", "base_rank", "source_personal", "has_generic",
    "generic_score", "normalized_generic_score", "generic_rank",
    "frequency_count", "personal_score", "has_personal",
    "personal_candidate_rank", "p_ng", "choice_share",
    "entropy_concentration", "log1p_same_pinyin_history",
    "log1p_raw_history", "ngram_support", "bge_support",
    "log1p_bge_history_count", "ngram_effective_n",
    "log1p_ngram_matched_history", "base_gap_to_top",
    "ngram_gap_to_top", "bge_gap_to_top", "frozen_linear_score",
)

FREQUENCY_LAMBDA = 4.0
W_P = 2.0
W_CS = 6.0
W_E = 4.0
LAMBDA_N = 6.0
LAMBDA_B = 6.0


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
                raise RuntimeError(f"used_test=true: {path}:{number}")
            yield row


def candidate_text(row: Mapping[str, Any]) -> str:
    for key in ("candidate", "text", "target"):
        if row.get(key) is not None:
            return str(row[key])
    raise RuntimeError(f"Candidate text missing: {row}")


def pinyin_of(row: Mapping[str, Any]) -> str | Sequence[str]:
    for key in ("pinyin_segments", "pinyin", "typed_pinyin"):
        value = row.get(key)
        if value:
            return value
    raise RuntimeError(f"Pinyin missing: {row.get('row_id')}")


def context_of(row: Mapping[str, Any]) -> str:
    for key in ("context", "preceding_context", "left_context"):
        if key in row:
            return str(row.get(key) or "")
    return ""


def find_stage1() -> Path:
    roots = [
        EM_REPO / "results",
        LF_REPO / "results",
    ]

    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*.jsonl"):
            name = path.name.lower()
            if "stage1" not in name:
                continue
            if not any(token in name for token in ("val", "feature", "train")):
                continue
            candidates.append(path)

    for path in sorted(candidates):
        try:
            if sha256_file(path) == STAGE1_SHA256:
                return path
        except OSError:
            pass

    raise FileNotFoundError(
        "Could not locate frozen Train-Val Stage-1 artifact "
        f"with SHA256 {STAGE1_SHA256}"
    )


def find_matrix_manifest() -> Path:
    matches: list[Path] = []

    for root in (EM_REPO / "results", LF_REPO / "results"):
        if not root.exists():
            continue

        for path in root.rglob("matrix_manifest.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
                continue

            val = value.get("val", {})
            if int(val.get("groups", -1)) != 34416:
                continue

            artifact = value.get("artifacts", {}).get("val_X.npy", {})
            artifact_path = Path(str(artifact.get("path", "")))
            if artifact_path.is_file():
                matches.append(path)

    if not matches:
        raise FileNotFoundError("Frozen 25-feature LambdaMART matrix manifest not found.")

    return sorted(matches)[0]


def find_lambdamart_result() -> Path:
    matches: list[Path] = []

    root = EM_REPO / "results" / "personalisation" / "external_memory_next"
    for path in root.rglob("result.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if value.get("experiment") == "lambdamart_external_memory_fusion_v1":
            if (path.parent / "selected_predictions.jsonl").is_file():
                matches.append(path)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one frozen LambdaMART result; found {len(matches)}:\n"
            + "\n".join(map(str, matches))
        )

    return matches[0]


def selected_model_path(result_path: Path) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selected = result["selected"]

    config_id = selected.get("config_id")
    if config_id is None:
        config_id = selected.get("config", {}).get("config_id")

    if not config_id:
        raise RuntimeError("Selected LambdaMART config_id missing.")

    model = result_path.parent / "models" / f"{config_id}.txt"
    if not model.is_file():
        raise FileNotFoundError(model)

    return model


def extract_score(item: Any) -> tuple[str, float]:
    payload = item.to_dict() if hasattr(item, "to_dict") else item

    if isinstance(payload, Mapping):
        text = None
        for key in ("candidate", "text", "target"):
            if payload.get(key) is not None:
                text = str(payload[key])
                break

        score = None
        for key in (
            "score",
            "log_probability",
            "logprob",
            "log_prob",
            "candidate_score",
        ):
            if payload.get(key) is not None:
                score = float(payload[key])
                break

        if text is not None and score is not None:
            return text, score

    text = None
    for key in ("candidate", "text", "target"):
        value = getattr(item, key, None)
        if value is not None:
            text = str(value)
            break

    score = None
    for key in ("score", "log_probability", "logprob", "log_prob"):
        value = getattr(item, key, None)
        if value is not None:
            score = float(value)
            break

    if text is None or score is None:
        raise RuntimeError(f"Cannot decode CandidateScore: {item!r}")

    return text, score


def make_candidate(candidate_cls: Any, text: str, score: float, rank: int) -> Any:
    sig = inspect.signature(candidate_cls)
    kwargs: dict[str, Any] = {}

    for name, parameter in sig.parameters.items():
        lname = name.lower()

        if lname in ("text", "candidate", "target"):
            kwargs[name] = text
        elif lname in (
            "score",
            "generic_score",
            "logprob",
            "log_probability",
            "log_prob",
        ):
            kwargs[name] = float(score)
        elif lname in ("rank", "generic_rank", "original_rank"):
            kwargs[name] = int(rank)
        elif parameter.default is not inspect._empty:
            continue
        else:
            raise RuntimeError(
                f"Unsupported Candidate constructor parameter: "
                f"{name}; signature={sig}"
            )

    return candidate_cls(**kwargs)


def normalized_adapter_scores(
    context_memory: Any,
    raw: Mapping[str, float],
    original_generic_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    candidate_cls = getattr(context_memory, "Candidate", None)
    if candidate_cls is None:
        raise RuntimeError("context_memory.Candidate is unavailable.")

    original_rank = {
        candidate_text(row): int(row.get("generic_rank") or row.get("rank") or 0)
        for row in original_generic_rows
    }

    order = sorted(
        raw,
        key=lambda text: (
            -float(raw[text]),
            original_rank.get(text, 999),
            text,
        ),
    )
    adapter_rank = {text: rank for rank, text in enumerate(order, start=1)}

    candidates = [
        make_candidate(candidate_cls, text, raw[text], adapter_rank[text])
        for text in order
    ]

    values = context_memory.normalize_generic_scores(candidates)

    if isinstance(values, Mapping):
        result: dict[str, float] = {}
        for key, value in values.items():
            if isinstance(key, str):
                result[key] = float(value)
            else:
                result[str(getattr(key, "text"))] = float(value)
        return result

    values = list(values)
    if len(values) != len(candidates):
        raise RuntimeError(
            "normalize_generic_scores returned unexpected length: "
            f"{len(values)} != {len(candidates)}"
        )

    if all(isinstance(value, (int, float, np.floating)) for value in values):
        return {
            text: float(value)
            for text, value in zip(order, values)
        }

    result = {}
    for candidate, value in zip(candidates, values):
        if isinstance(value, Mapping):
            found = None
            for key in ("normalized_generic_score", "score", "value"):
                if value.get(key) is not None:
                    found = float(value[key])
                    break
            if found is None:
                raise RuntimeError(
                    f"Cannot decode normalized score mapping: {value}"
                )
            result[str(getattr(candidate, "text"))] = found
        else:
            raise RuntimeError(
                "Unsupported normalize_generic_scores output type: "
                f"{type(value)}"
            )

    return result


def rank_top10(
    scores: Sequence[float],
    candidates: Sequence[str],
    baseline_top10: Sequence[str],
) -> list[str]:
    baseline_order = {
        candidate: rank
        for rank, candidate in enumerate(map(str, baseline_top10), start=1)
    }

    if set(baseline_order) != set(map(str, candidates)):
        raise RuntimeError("Fixed candidate surface changed.")

    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -float(scores[i]),
            baseline_order[str(candidates[i])],
            str(candidates[i]),
        ),
    )

    return [str(candidates[i]) for i in order]


def gold_rank(gold: str, top10: Sequence[str]) -> int | None:
    try:
        return list(top10).index(gold) + 1
    except ValueError:
        return None


def metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    n = len(ranks)

    def top(k: int) -> float:
        return sum(rank is not None and rank <= k for rank in ranks) / n

    return {
        "n": n,
        "top1": top(1),
        "top3": top(3),
        "top5": top(5),
        "mrr_at_10": sum(
            0.0 if rank is None else 1.0 / rank
            for rank in ranks
        ) / n,
        "missing10": sum(rank is None for rank in ranks) / n,
    }


def transition(
    before: Sequence[int | None],
    after: Sequence[int | None],
) -> dict[str, int]:
    result = {
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
            result["rescue"] += 1
        elif oc and not nc:
            result["harm"] += 1
        elif oc:
            result["unchanged_correct"] += 1
        else:
            result["unchanged_wrong"] += 1

    result["net"] = result["rescue"] - result["harm"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/personalisation/"
            "hybrid_zero_shot_proper_v1"
        ),
    )
    args = parser.parse_args()

    if sha256_file(VAL_PATH) != VAL_SHA256:
        raise RuntimeError("Frozen Train-Val SHA256 changed.")

    stage1_path = Path(r"C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune\train_val_stage1_features.jsonl")
    if sha256_file(stage1_path) != STAGE1_SHA256:
        raise RuntimeError("Frozen Stage-1 SHA256 changed.")
    matrix_manifest_path = find_matrix_manifest()
    lambdamart_result_path = find_lambdamart_result()
    model_path = selected_model_path(lambdamart_result_path)

    matrix_manifest = json.loads(
        matrix_manifest_path.read_text(encoding="utf-8")
    )

    if tuple(matrix_manifest["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("Feature schema changed.")

    val_x_path = Path(
        matrix_manifest["artifacts"]["val_X.npy"]["path"]
    )
    meta_path = Path(
        matrix_manifest["artifacts"]["val_query_meta.jsonl"]["path"]
    )

    selected_predictions_path = (
        lambdamart_result_path.parent / "selected_predictions.jsonl"
    )

    print("VAL:", VAL_PATH)
    print("STAGE1:", stage1_path)
    print("MATRIX:", val_x_path)
    print("META:", meta_path)
    print("LAMBDAMART:", model_path)
    print("FROZEN PREDICTIONS:", selected_predictions_path)

    meta_all = list(iter_jsonl(meta_path))
    agent_meta = [
        row for row in meta_all
        if str(row["author"]) == AUTHOR
    ]

    if args.limit is not None:
        agent_meta = agent_meta[:args.limit]

    row_ids = {str(row["row_id"]) for row in agent_meta}

    val_by_id = {
        str(row["row_id"]): row
        for row in iter_jsonl(VAL_PATH)
        if str(row.get("row_id")) in row_ids
    }

    stage1_by_id = {
        str(row["row_id"]): row
        for row in iter_jsonl(stage1_path)
        if str(row.get("row_id")) in row_ids
    }

    frozen_prediction_by_id = {
        str(row["row_id"]): row
        for row in iter_jsonl(selected_predictions_path)
        if str(row.get("row_id")) in row_ids
    }

    if not (
        len(agent_meta)
        == len(val_by_id)
        == len(stage1_by_id)
        == len(frozen_prediction_by_id)
    ):
        raise RuntimeError(
            "Agent row alignment failed: "
            f"meta={len(agent_meta)} "
            f"val={len(val_by_id)} "
            f"stage1={len(stage1_by_id)} "
            f"frozen={len(frozen_prediction_by_id)}"
        )

    sys.path.insert(0, str(EM_REPO))
    import src.personalisation.context_memory as context_memory

    sys.path.insert(0, str(MODEL_REPO / "src"))
    import torch

    from model_level.pinyingpt_adapter import (
        SerialAdapterConfig,
        install_adapters,
        load_adapter_bundle,
        set_adapter_training_mode,
    )
    from reference_backend_pinyingpt.backend import (
        PinyinGPTConcatBackend,
    )

    import lightgbm as lgb

    print("LightGBM:", lgb.__version__)
    print("Torch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())

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

    backend = PinyinGPTConcatBackend(
        base_checkpoint,
        device=device,
    )

    install_info = install_adapters(
        backend.model,
        config=SerialAdapterConfig(reduction_factor=16),
        expected_layers=12,
    )
    adapter_metadata = load_adapter_bundle(
        backend.model,
        adapter_path,
    )
    set_adapter_training_mode(
        backend.model,
        enabled=False,
    )
    backend.model.eval()

    print("DEVICE:", device)
    print("INSTALL:", install_info)
    print(
        "ADAPTER:",
        {
            key: adapter_metadata.get(key)
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

    booster = lgb.Booster(model_file=str(model_path))
    val_x = np.load(val_x_path, mmap_mode="r")

    feature_index = {
        name: index
        for index, name in enumerate(FEATURE_NAMES)
    }

    hybrid_groups: list[np.ndarray] = []
    original_groups: list[np.ndarray] = []
    group_records: list[dict[str, Any]] = []

    adapter_surface_ranks: list[int | None] = []
    frozen_ranks: list[int | None] = []

    started = time.perf_counter()

    with torch.inference_mode():
        for number, meta in enumerate(agent_meta, start=1):
            row_id = str(meta["row_id"])
            source = val_by_id[row_id]
            feature = stage1_by_id[row_id]

            offset = int(meta["offset"])
            count = int(meta["candidate_count"])
            fixed_candidates = list(map(str, meta["candidates"]))

            if count != len(fixed_candidates):
                raise RuntimeError(
                    f"Candidate count mismatch: {row_id}"
                )

            original_x = np.asarray(
                val_x[offset:offset + count],
                dtype=np.float32,
            ).copy()

            generic_rows = [
                dict(row)
                for row in feature["generic_frequency_candidates"]
            ]
            generic_names = [
                candidate_text(row)
                for row in generic_rows
            ]

            score_surface = list(
                dict.fromkeys(
                    generic_names + fixed_candidates
                )
            )

            scored = backend.score_candidates(
                context=context_of(source),
                typed_pinyin=pinyin_of(source),
                candidates=score_surface,
            )

            raw_scores = dict(
                extract_score(item)
                for item in scored
            )

            if set(raw_scores) != set(score_surface):
                raise RuntimeError(
                    f"Adapter score surface mismatch: {row_id}"
                )

            normalized = normalized_adapter_scores(
                context_memory,
                {name: raw_scores[name] for name in generic_names},
                generic_rows,
            )

            if set(normalized) != set(generic_names):
                raise RuntimeError(
                    f"Adapter normalization mismatch: {row_id}"
                )

            old_generic_rank = {
                candidate_text(row):
                    int(row.get("generic_rank") or row.get("rank") or 0)
                for row in generic_rows
            }

            adapter_generic_order = sorted(
                generic_names,
                key=lambda name: (
                    -raw_scores[name],
                    old_generic_rank[name],
                    name,
                ),
            )
            adapter_generic_rank = {
                name: rank
                for rank, name in enumerate(
                    adapter_generic_order,
                    start=1,
                )
            }

            new_generic_base: dict[str, float] = {}

            for row in generic_rows:
                name = candidate_text(row)
                old_norm = float(
                    row["normalized_generic_score"]
                )
                old_final = float(row["final_score"])

                frozen_frequency_support = (
                    old_final - old_norm
                ) / FREQUENCY_LAMBDA

                new_generic_base[name] = (
                    float(normalized[name])
                    + FREQUENCY_LAMBDA
                    * frozen_frequency_support
                )

            boundary = min(
                float(normalized[name])
                for name in generic_names
            )

            hybrid_x = original_x.copy()
            base_scores: list[float] = []

            for local_index, name in enumerate(fixed_candidates):
                x = hybrid_x[local_index]

                has_generic = (
                    float(
                        x[
                            feature_index["has_generic"]
                        ]
                    )
                    > 0.5
                )

                if has_generic:
                    if name not in new_generic_base:
                        raise RuntimeError(
                            f"Fixed generic candidate absent "
                            f"from original Generic surface: "
                            f"{row_id} {name}"
                        )

                    base_score = new_generic_base[name]

                    x[
                        feature_index["generic_score"]
                    ] = raw_scores[name]

                    x[
                        feature_index[
                            "normalized_generic_score"
                        ]
                    ] = normalized[name]

                    x[
                        feature_index["generic_rank"]
                    ] = adapter_generic_rank[name]

                else:
                    p_ng = float(
                        x[feature_index["p_ng"]]
                    )
                    choice = float(
                        x[feature_index["choice_share"]]
                    )
                    entropy = float(
                        x[
                            feature_index[
                                "entropy_concentration"
                            ]
                        ]
                    )

                    base_score = (
                        boundary
                        + W_P * p_ng
                        + W_CS * choice
                        + W_E * entropy
                    )

                base_scores.append(float(base_score))

            generic_flag = [
                float(
                    hybrid_x[i][
                        feature_index["has_generic"]
                    ]
                ) > 0.5
                for i in range(count)
            ]

            stage1_order = sorted(
                range(count),
                key=lambda i: (
                    -base_scores[i],
                    0 if generic_flag[i] else 1,
                    (
                        int(
                            hybrid_x[i][
                                feature_index["generic_rank"]
                            ]
                        )
                        if generic_flag[i]
                        else int(
                            hybrid_x[i][
                                feature_index[
                                    "personal_candidate_rank"
                                ]
                            ]
                        )
                    ),
                    fixed_candidates[i],
                ),
            )

            new_base_rank = {
                i: rank
                for rank, i in enumerate(
                    stage1_order,
                    start=1,
                )
            }

            base_top = max(base_scores)

            for i in range(count):
                x = hybrid_x[i]
                base_score = base_scores[i]

                x[
                    feature_index["base_score"]
                ] = base_score

                x[
                    feature_index["base_rank"]
                ] = new_base_rank[i]

                x[
                    feature_index["base_gap_to_top"]
                ] = base_score - base_top

                ngram = float(
                    x[feature_index["ngram_support"]]
                )
                bge = float(
                    x[feature_index["bge_support"]]
                )

                x[
                    feature_index["frozen_linear_score"]
                ] = (
                    base_score
                    + LAMBDA_N * ngram
                    + LAMBDA_B * bge
                )

            adapter_surface_order = sorted(
                fixed_candidates,
                key=lambda name: (
                    -raw_scores[name],
                    name,
                ),
            )

            gold = str(meta["gold"])

            adapter_surface_ranks.append(
                gold_rank(
                    gold,
                    adapter_surface_order,
                )
            )

            frozen_row = frozen_prediction_by_id[row_id]
            frozen_rank = frozen_row.get("LambdaMART_rank")
            frozen_ranks.append(
                None
                if frozen_rank is None
                else int(frozen_rank)
            )

            original_groups.append(original_x)
            hybrid_groups.append(hybrid_x)
            group_records.append(
                {
                    "row_id": row_id,
                    "gold": gold,
                    "candidates": fixed_candidates,
                    "baseline_top10":
                        list(map(str, meta["baseline_top10"])),
                }
            )

            if (
                number == 1
                or number % 32 == 0
                or number == len(agent_meta)
            ):
                elapsed = max(
                    time.perf_counter() - started,
                    1e-9,
                )
                print(
                    f"{number}/{len(agent_meta)} "
                    f"rate={number / elapsed:.2f} rows/s",
                    flush=True,
                )

    original_matrix = np.concatenate(
        original_groups,
        axis=0,
    )
    hybrid_matrix = np.concatenate(
        hybrid_groups,
        axis=0,
    )

    frozen_scores = booster.predict(
        original_matrix,
        num_iteration=booster.current_iteration(),
    )
    hybrid_scores = booster.predict(
        hybrid_matrix,
        num_iteration=booster.current_iteration(),
    )

    frozen_rebuilt_ranks: list[int | None] = []
    hybrid_ranks: list[int | None] = []
    predictions: list[dict[str, Any]] = []

    cursor = 0

    for record in group_records:
        count = len(record["candidates"])
        sl = slice(cursor, cursor + count)

        rebuilt_top10 = rank_top10(
            frozen_scores[sl],
            record["candidates"],
            record["baseline_top10"],
        )
        hybrid_top10 = rank_top10(
            hybrid_scores[sl],
            record["candidates"],
            record["baseline_top10"],
        )

        fr = gold_rank(
            record["gold"],
            rebuilt_top10,
        )
        hr = gold_rank(
            record["gold"],
            hybrid_top10,
        )

        frozen_rebuilt_ranks.append(fr)
        hybrid_ranks.append(hr)

        predictions.append(
            {
                "row_id": record["row_id"],
                "author": AUTHOR,
                "gold": record["gold"],
                "FrozenLambdaMART_rank": fr,
                "HybridH0_rank": hr,
                "FrozenLambdaMART_top10":
                    rebuilt_top10,
                "HybridH0_top10":
                    hybrid_top10,
                "candidate_surface_fixed": True,
                "training": False,
                "tuning": False,
                "used_dev3000": False,
                "used_test": False,
            }
        )

        cursor += count

    if frozen_rebuilt_ranks != frozen_ranks:
        mismatches = [
            (
                group_records[i]["row_id"],
                frozen_ranks[i],
                frozen_rebuilt_ranks[i],
            )
            for i in range(len(frozen_ranks))
            if frozen_ranks[i] != frozen_rebuilt_ranks[i]
        ]

        raise RuntimeError(
            "Frozen LambdaMART reproduction failed. "
            f"mismatches={len(mismatches)} "
            f"first={mismatches[:5]}"
        )

    output_root = args.output_root
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"smoke_{args.limit}"
        if args.limit is not None
        else "agent_full"
    )

    prediction_path = (
        output_root
        / f"predictions_{suffix}.jsonl"
    )
    result_path = (
        output_root
        / f"result_{suffix}.json"
    )

    with prediction_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for row in predictions:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment":
            "proper_zero_shot_agent_adapter_frozen_lambdamart_v1",
        "author": AUTHOR,
        "rows": len(agent_meta),
        "protocol": {
            "adapter":
                "agent_full55926",
            "candidate_surface":
                "frozen original EM Stage-1 Top10",
            "generic_neural_signal":
                "replaced by Agent Adapter",
            "frequency_support":
                "frozen; recovered exactly from old "
                "(final_score-normalized_generic_score)/4",
            "personal_recovery_weights": {
                "w_p": W_P,
                "w_cs": W_CS,
                "w_e": W_E,
            },
            "stage2_support":
                "frozen original NGram/BGE",
            "lambdamart":
                "frozen selected model",
            "training": False,
            "tuning": False,
        },
        "metrics": {
            "AdapterOnFixedEMSurface":
                metrics(adapter_surface_ranks),
            "FrozenLambdaMART":
                metrics(frozen_ranks),
            "ProperHybridH0":
                metrics(hybrid_ranks),
        },
        "transitions": {
            "FrozenLambdaMART_to_ProperHybridH0":
                transition(
                    frozen_ranks,
                    hybrid_ranks,
                ),
            "AdapterSurface_to_ProperHybridH0":
                transition(
                    adapter_surface_ranks,
                    hybrid_ranks,
                ),
        },
        "provenance": {
            "train_val": str(VAL_PATH),
            "train_val_sha256":
                sha256_file(VAL_PATH),
            "stage1": str(stage1_path),
            "stage1_sha256":
                sha256_file(stage1_path),
            "matrix_manifest":
                str(matrix_manifest_path),
            "lambdamart_model":
                str(model_path),
            "lambdamart_model_sha256":
                sha256_file(model_path),
            "adapter_checkpoint":
                str(adapter_path),
            "adapter_checkpoint_sha256":
                sha256_file(adapter_path),
        },
        "used_dev3000": False,
        "used_test": False,
    }

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== METRICS =====")
    print(
        json.dumps(
            result["metrics"],
            indent=2,
        )
    )

    print()
    print("===== TRANSITIONS =====")
    print(
        json.dumps(
            result["transitions"],
            indent=2,
        )
    )

    print()
    print("RESULT:", result_path.resolve())
    print(
        "PREDICTIONS:",
        prediction_path.resolve(),
    )


if __name__ == "__main__":
    main()
