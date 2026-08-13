"""Prepare and run Frozen LiveChat Development Evaluation Set V1."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping
from collections import Counter

from src.datasets.livechat.baseline import (
    canonical_json,
    prepare_livechat_baseline,
    sha256_file,
    write_json,
)
from src.evaluation.ranking import compute_metrics, context_gain, evaluate_breakdown
from src.reference_backend_pinyingpt import PinyinGPTConcatBackend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/livechat_pinyingpt_generic_baseline_v1.json"
PREDICTION_FILES = {
    "pinyin_only": "pinyin_only_predictions.jsonl",
    "contextual": "contextual_predictions.jsonl",
}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _prediction(
    interaction: Mapping[str, Any],
    condition: str,
    backend: PinyinGPTConcatBackend,
    *,
    top_k: int,
    beam_size: int,
) -> dict[str, Any]:
    context = "" if condition == "pinyin_only" else str(interaction["effective_context"])
    typed_pinyin = list(interaction["segmented_pinyin"])
    started = time.perf_counter()
    generated = backend.generate(
        context,
        typed_pinyin,
        top_k=top_k,
        beam_size=beam_size,
    )
    gold_score = backend.score_candidates(
        context,
        typed_pinyin,
        [str(interaction["gold"])],
    )[0].log_probability
    elapsed = time.perf_counter() - started
    candidates = [candidate.text for candidate in generated.candidates]
    scores = [candidate.log_probability for candidate in generated.candidates]
    gold = str(interaction["gold"])
    rank = candidates.index(gold) + 1 if gold in candidates else None
    return {
        "schema_version": 1,
        "interaction_id": interaction["interaction_id"],
        "user_id": interaction["user_id"],
        "gold": gold,
        "segmented_pinyin": typed_pinyin,
        "typed_pinyin": " ".join(typed_pinyin),
        "context_condition": condition,
        "context": context,
        "top10_candidates": candidates,
        "top10_candidate_scores": scores,
        "top1_score": scores[0] if scores else None,
        "top2_score": scores[1] if len(scores) > 1 else None,
        "top1_minus_top2_score_margin": scores[0] - scores[1] if len(scores) > 1 else None,
        "exact_gold_teacher_forced_score": gold_score,
        "gold_top10_rank": rank,
        "top1_correct": rank == 1,
        "top3_correct": rank is not None and rank <= 3,
        "top5_correct": rank is not None and rank <= 5,
        "top10_present": rank is not None,
        "reciprocal_rank_at_10": 0.0 if rank is None else 1.0 / rank,
        "top1_minus_gold_score_gap": scores[0] - gold_score if scores else None,
        "inference_seconds": elapsed,
        "beam_size": beam_size,
        "top_k": top_k,
        "runtime_device": generated.runtime_device,
    }


def run_inference_condition(
    condition: str,
    interactions: list[dict[str, Any]],
    backend: PinyinGPTConcatBackend,
    output_path: Path,
    *,
    top_k: int,
    beam_size: int,
    force: bool,
    flush_every: int = 10,
) -> dict[str, Any]:
    if condition not in PREDICTION_FILES:
        raise ValueError(f"unknown condition {condition!r}")
    existing = {} if force else {
        row["interaction_id"]: row for row in load_jsonl(output_path)
    }
    valid_ids = {row["interaction_id"] for row in interactions}
    unexpected = set(existing) - valid_ids
    if unexpected:
        raise ValueError(f"prediction file has IDs outside frozen set: {sorted(unexpected)[:5]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "a"
    completed_now = 0
    started = time.perf_counter()
    with output_path.open(mode, encoding="utf-8", newline="\n", buffering=1) as destination:
        for index, interaction in enumerate(interactions, start=1):
            if interaction["interaction_id"] in existing:
                continue
            row = _prediction(
                interaction,
                condition,
                backend,
                top_k=top_k,
                beam_size=beam_size,
            )
            destination.write(canonical_json(row) + "\n")
            completed_now += 1
            if completed_now % flush_every == 0:
                destination.flush()
            if completed_now % 100 == 0:
                print(
                    f"{condition}: {len(existing) + completed_now}/{len(interactions)} "
                    f"({time.perf_counter() - started:.1f}s this run)",
                    flush=True,
                )
    rows = load_jsonl(output_path)
    rows.sort(key=lambda row: row["interaction_id"])
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(canonical_json(row) + "\n")
    temporary.replace(output_path)
    return {
        "condition": condition,
        "expected_interactions": len(interactions),
        "completed_interactions": len(rows),
        "completed_this_run": completed_now,
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "complete": len(rows) == len(interactions),
    }


def _context_bin(length: int) -> str:
    if length == 0:
        return "0"
    if length <= 5:
        return "1-5"
    if length <= 10:
        return "6-10"
    if length <= 20:
        return "11-20"
    return "21+"


def _joined_rows(
    interactions: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {row["interaction_id"]: row for row in interactions}
    result = []
    for prediction in predictions:
        interaction = by_id[prediction["interaction_id"]]
        result.append({**interaction, **prediction})
    return result


def target_length_groups(
    interactions: Iterable[Mapping[str, Any]],
    *,
    minimum_long_tail_count: int,
) -> dict[int, str]:
    """Define length bins from frozen inputs alone, before consulting outcomes."""

    counts = Counter(int(row["target_length"]) for row in interactions)
    if not counts:
        return {}
    maximum = max(counts)
    tail_total = 0
    tail_start = maximum
    for length in range(maximum, 0, -1):
        tail_total += counts.get(length, 0)
        tail_start = length
        if tail_total >= minimum_long_tail_count:
            break
    mapping = {}
    for length in counts:
        mapping[length] = f"{tail_start}+" if length >= tail_start and tail_start < maximum else str(length)
    return mapping


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values = sorted(values)
    percentile = lambda probability: values[round((len(values) - 1) * probability)]
    return {
        "count": len(values),
        "min": values[0],
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def score_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    groups = {
        "top1_correct": [row for row in rows if row["top1_correct"]],
        "top1_incorrect": [row for row in rows if not row["top1_correct"]],
        "gold_missing_from_top10": [row for row in rows if row["gold_top10_rank"] is None],
    }
    fields = (
        "top1_score",
        "top2_score",
        "top1_minus_top2_score_margin",
        "exact_gold_teacher_forced_score",
        "top1_minus_gold_score_gap",
    )
    return {
        group: {
            field: _distribution([float(row[field]) for row in group_rows if row[field] is not None])
            for field in fields
        }
        for group, group_rows in groups.items()
    }


def create_plots(
    output: Path,
    only_metrics: Mapping[str, Any],
    contextual_metrics: Mapping[str, Any],
    gain: Mapping[str, Any],
    target_breakdown: Mapping[str, Any],
    context_breakdown: Mapping[str, Any],
    ambiguity_breakdown: Mapping[str, Any],
    contextual_rows: list[Mapping[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_bar(name: str, labels: list[str], values: list[float], title: str, ylabel: str = "Accuracy") -> None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(labels, values, color="#3568A8")
        axis.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / name, dpi=160)
        plt.close(figure)

    labels = ["Top-1", "Top-3", "Top-5", "Top-10", "MRR@10"]
    only_values = [only_metrics["micro"][key] for key in ("top1", "top3", "top5", "top10", "mrr_at_10")]
    contextual_values = [contextual_metrics["micro"][key] for key in ("top1", "top3", "top5", "top10", "mrr_at_10")]
    x = range(len(labels))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([i - 0.2 for i in x], only_values, width=0.4, label="Pinyin-only")
    axis.bar([i + 0.2 for i in x], contextual_values, width=0.4, label="Contextual")
    axis.set_xticks(list(x), labels)
    axis.set_ylim(0, 1)
    axis.set_title("Frozen generic PinyinGPT baseline")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(output / "topk_baseline.png", dpi=160)
    plt.close(figure)

    differences = gain["paired_micro_differences_contextual_minus_pinyin_only"]
    save_bar("context_gain.png", list(differences), list(differences.values()), "Contextual minus Pinyin-only")
    target_labels = sorted(
        target_breakdown,
        key=lambda label: int(label.rstrip("+")),
    )
    save_bar("target_length_top1.png", target_labels, [target_breakdown[label]["top1"] for label in target_labels], "Contextual Top-1 by target length")
    context_labels = [label for label in ("0", "1-5", "6-10", "11-20", "21+") if label in context_breakdown]
    save_bar("context_length_top1.png", context_labels, [context_breakdown[label]["top1"] for label in context_labels], "Contextual Top-1 by context length")
    save_bar("pinyin_ambiguity_top1.png", list(ambiguity_breakdown), [value["top1"] for value in ambiguity_breakdown.values()], "Contextual Top-1 by Pinyin ambiguity")

    per_user = contextual_metrics["per_user"]
    ordered_users = sorted(per_user, key=lambda user: per_user[user]["top1"])
    figure, axis = plt.subplots(figsize=(10, 4.5))
    user_positions = list(range(1, len(ordered_users) + 1))
    axis.bar(user_positions, [per_user[user]["top1"] for user in ordered_users], color="#3568A8")
    axis.set_ylim(0, 1)
    axis.set_title("Contextual per-user Top-1")
    axis.set_xlabel("Users ordered by Top-1 accuracy")
    axis.set_ylabel("Accuracy")
    axis.set_xticks([1, 20, 40, 60, 80, 100])
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "per_user_top1.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    correct = [row["top1_minus_top2_score_margin"] for row in contextual_rows if row["top1_correct"] and row["top1_minus_top2_score_margin"] is not None]
    incorrect = [row["top1_minus_top2_score_margin"] for row in contextual_rows if not row["top1_correct"] and row["top1_minus_top2_score_margin"] is not None]
    axis.hist([correct, incorrect], bins=30, label=["Top-1 correct", "Top-1 incorrect"], alpha=0.7)
    axis.set_title("Contextual Top-1 score margin")
    axis.set_xlabel("Top-1 minus Top-2 cumulative log probability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "top1_score_margin.png", dpi=160)
    plt.close(figure)


def recompute_metrics(config: Mapping[str, Any]) -> dict[str, Any]:
    output = ROOT / config["output_dir"]
    interactions = load_jsonl(output / "frozen_interactions.jsonl")
    only = load_jsonl(output / PREDICTION_FILES["pinyin_only"])
    contextual = load_jsonl(output / PREDICTION_FILES["contextual"])
    expected = {row["interaction_id"] for row in interactions}
    if {row["interaction_id"] for row in only} != expected or {row["interaction_id"] for row in contextual} != expected:
        raise RuntimeError("both prediction files must exactly align with frozen interactions")
    only_metrics = compute_metrics(only)
    contextual_metrics = compute_metrics(contextual)
    gain = context_gain(
        only,
        contextual,
        seed=int(config["seed"]),
        bootstrap_resamples=int(config["analysis"]["bootstrap_resamples"]),
    )
    joined = _joined_rows(interactions, contextual)
    length_groups = target_length_groups(
        interactions,
        minimum_long_tail_count=int(config["analysis"]["target_length_long_tail_min_interactions"]),
    )
    target_breakdown = evaluate_breakdown(joined, lambda row: length_groups[int(row["target_length"])])
    context_breakdown = evaluate_breakdown(joined, lambda row: _context_bin(int(row["context_length_chinese_characters"])))
    ambiguity_breakdown = evaluate_breakdown(joined, lambda row: str(row["ambiguity_quartile"]))
    user_breakdown = contextual_metrics["per_user"]
    diagnostics = score_diagnostics(joined)
    write_json(output / "pinyin_only_metrics.json", only_metrics)
    write_json(output / "contextual_metrics.json", contextual_metrics)
    write_json(output / "context_gain.json", gain)
    write_json(output / "breakdown_target_length.json", target_breakdown)
    write_json(output / "breakdown_context_length.json", context_breakdown)
    write_json(output / "breakdown_pinyin_ambiguity.json", ambiguity_breakdown)
    write_json(output / "breakdown_user.json", user_breakdown)
    write_json(output / "score_diagnostics.json", diagnostics)
    refresh_persisted_runtime(output)
    create_plots(output, only_metrics, contextual_metrics, gain, target_breakdown, context_breakdown, ambiguity_breakdown, joined)
    write_analysis(output, config, only_metrics, contextual_metrics, gain, target_breakdown, context_breakdown, ambiguity_breakdown, diagnostics)
    write_checksums(output, ROOT / "configs/livechat_pinyingpt_generic_baseline_v1.json")
    return {"pinyin_only": only_metrics, "contextual": contextual_metrics, "context_gain": gain}


def write_checksums(output: Path, config_path: Path) -> None:
    files = {
        "baseline_config": config_path,
        "selected_users": output / "selected_users.json",
        "split_manifest": output / "split_manifest.json",
        "frozen_interactions": output / "frozen_interactions.jsonl",
        "pinyin_only_predictions": output / "pinyin_only_predictions.jsonl",
        "contextual_predictions": output / "contextual_predictions.jsonl",
    }
    write_json(output / "checksums.json", {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in files.items()})


def _fmt(value: Any) -> str:
    return "n/a" if value is None else (f"{value:.6f}" if isinstance(value, float) else str(value))


def write_analysis(
    output: Path,
    config: Mapping[str, Any],
    only: Mapping[str, Any],
    contextual: Mapping[str, Any],
    gain: Mapping[str, Any],
    target: Mapping[str, Any],
    context: Mapping[str, Any],
    ambiguity: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> None:
    dataset = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    users = json.loads((output / "selected_users.json").read_text(encoding="utf-8"))
    chronology = json.loads((output / "chronology_audit.json").read_text(encoding="utf-8"))
    construction = json.loads((output / "interaction_construction_summary.json").read_text(encoding="utf-8"))
    quality = json.loads((output / "pinyin_quality_audit.json").read_text(encoding="utf-8"))
    selected = users["selected_users"]
    hardest = sorted(contextual["per_user"].items(), key=lambda item: item[1]["top1"])[:5]
    easiest = sorted(contextual["per_user"].items(), key=lambda item: (-item[1]["top1"], item[0]))[:5]
    train = dataset["files"][config["dataset"]["source_split"]]
    text = f"""# LiveChat PinyinGPT Generic Baseline V1

This is a development generic-baseline analysis. No personalisation was implemented or evaluated.

1. **Download:** successful from the official repository-linked Google Drive.
2. **Actual schema:** each pickle is a `list`; every row is a three-element `list[str]`: `streamer_id`, `audience_comment`, `streamer_response`. No timestamp/order/session field is present.
3. **Train scale:** {train['row_count']} rows and {train['unique_streamers']} streamers.
4. **Deep users:** {users['qualifying_user_count']} users have at least {users['threshold']} usable train responses.
5. **Selected users:** {users['selected_user_count']} users, ranked by usable response count. Depth ranges from {min(item['usable_train_responses'] for item in selected)} to {max(item['usable_train_responses'] for item in selected)}.
6. **Chronology:** Grade {chronology['grade']}. Released data have no timestamp, and official code/docs do not establish that serialization preserves chronological order.
7. **Split:** `{chronology['label']}` using deterministic response/session-level stable hashing with seed {config['seed']}. E7 temporal/prequential evaluation is unavailable.
8. **Frozen interactions:** {construction['frozen_interaction_count']} across {users['selected_user_count']} users, at most {config['selection']['max_evaluation_interactions_per_user']} per user.
9. **Pinyin quality:** candidate-target exclusion rate {_fmt(quality['candidate_target_exclusion_rate'])}; exclusions {quality['candidate_target_exclusion_reasons']}; frozen polyphonic rate {_fmt(quality['polyphonic_target_rate'])}.
10. **Pinyin-only micro:** Top-1 {_fmt(only['micro']['top1'])}, Top-3 {_fmt(only['micro']['top3'])}, Top-5 {_fmt(only['micro']['top5'])}, Coverage@10 {_fmt(only['micro']['top10'])}, MRR@10 {_fmt(only['micro']['mrr_at_10'])}, MeanRank|Top10 {_fmt(only['micro']['mean_rank_given_top10'])}, Missing@10 {only['micro']['missing_at_10_count']} ({_fmt(only['micro']['missing_at_10_rate'])}). Macro-user Top-1 {_fmt(only['macro_user']['top1'])}.
11. **Contextual micro:** Top-1 {_fmt(contextual['micro']['top1'])}, Top-3 {_fmt(contextual['micro']['top3'])}, Top-5 {_fmt(contextual['micro']['top5'])}, Coverage@10 {_fmt(contextual['micro']['top10'])}, MRR@10 {_fmt(contextual['micro']['mrr_at_10'])}, MeanRank|Top10 {_fmt(contextual['micro']['mean_rank_given_top10'])}, Missing@10 {contextual['micro']['missing_at_10_count']} ({_fmt(contextual['micro']['missing_at_10_rate'])}). Primary macro-user Top-1 {_fmt(contextual['macro_user']['top1'])}.
12. **Context gain:** paired micro differences {gain['paired_micro_differences_contextual_minus_pinyin_only']}; outcomes {gain['top1_outcome_counts']}.
13. **Bootstrap:** contextual minus Pinyin-only macro-user Top-1 {_fmt(gain['macro_user_top1_difference'])}, 95% development CI [{_fmt(gain['paired_user_bootstrap']['lower'])}, {_fmt(gain['paired_user_bootstrap']['upper'])}], 10,000 resamples, seed 40408.
14. **Target length:** {target}.
15. **Context length:** {context}.
16. **Pinyin ambiguity:** {ambiguity}.
17. **Users:** easiest {easiest}; hardest {hardest}. These are descriptive development results, not user selection criteria.
18. **Missing targets:** {contextual['micro']['missing_at_10_count']} contextual targets are absent from Generic Top-10. Their exact frozen teacher-forced gold scores remain stored; absence is not assigned an unrestricted rank.
19. **Score diagnostics:** {diagnostics}.
20. **Room for future personalisation:** interactions that are compatible but missing from contextual Top-10, low-margin errors, longer targets, and users/ambiguity strata with lower generic Top-1 provide frozen diagnostic strata. No personalisation claim follows from this baseline.
21. **Future layers:** E1-E6 and E8 can reuse the frozen IDs, generic candidates, scores, and non-temporal history proxy under separately frozen methods. E7 is unavailable because chronology is Grade C.
22. **Limitations:** reconstructed rather than real keystroke Pinyin; Jieba-reconstructed composition boundaries; ambiguous polyphonic conversion; no proven chronology; audience comments excluded from the main IME context; deep-user sample is development-only; dataset usage terms are not explicitly established by the repository license alone.

Metric note: MRR@10 is zero for missing gold. MeanRank|Top10 is computed only where gold is returned in Top-10. No unrestricted MRR or mean rank is reported.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")


def runtime_diagnostics(
    output: Path,
    backend: PinyinGPTConcatBackend,
    inference_summaries: list[Mapping[str, Any]],
) -> None:
    torch = backend.torch
    predictions = sum(summary["completed_interactions"] for summary in inference_summaries)
    total_seconds = sum(summary["elapsed_seconds_this_run"] for summary in inference_summaries)
    value = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runtime": backend.runtime_info(),
        "transformers_version": importlib.metadata.version("transformers"),
        "conditions": inference_summaries,
        "prediction_rows_across_conditions": predictions,
        "total_inference_seconds_this_run": total_seconds,
        "average_seconds_per_prediction_this_run": total_seconds / predictions if predictions else None,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if backend.device.type == "cuda" else None,
    }
    write_json(output / "runtime_diagnostics.json", value)


def refresh_persisted_runtime(output: Path) -> None:
    path = output / "runtime_diagnostics.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    persisted = {}
    total_rows = 0
    total_seconds = 0.0
    for condition, filename in PREDICTION_FILES.items():
        rows = load_jsonl(output / filename)
        seconds = sum(float(row["inference_seconds"]) for row in rows)
        persisted[condition] = {
            "prediction_rows": len(rows),
            "summed_per_interaction_inference_seconds": seconds,
            "average_seconds_per_interaction": seconds / len(rows) if rows else None,
        }
        total_rows += len(rows)
        total_seconds += seconds
    value["persisted_prediction_runtime"] = persisted
    value["persisted_prediction_rows_across_conditions"] = total_rows
    value["persisted_summed_inference_seconds_across_conditions"] = total_seconds
    value["persisted_average_seconds_per_prediction"] = total_seconds / total_rows if total_rows else None
    write_json(path, value)


def probe_peak_gpu_memory(config: Mapping[str, Any]) -> dict[str, Any]:
    """Measure an inference-path peak without modifying frozen predictions."""

    output = ROOT / config["output_dir"]
    interactions = load_jsonl(output / "frozen_interactions.jsonl")
    if not interactions:
        raise RuntimeError("prepare frozen interactions before the runtime probe")
    backend = PinyinGPTConcatBackend(ROOT / config["model"]["checkpoint_path"], device="auto")
    if backend.device.type != "cuda":
        raise RuntimeError("CUDA peak-memory probe requested without a CUDA device")
    probe_interaction = max(
        interactions,
        key=lambda row: (
            len(backend.tokenizer.encode(row["effective_context"], add_special_tokens=False))
            * len(row["segmented_pinyin"]),
            row["interaction_id"],
        ),
    )
    context_token_count = len(
        backend.tokenizer.encode(probe_interaction["effective_context"], add_special_tokens=False)
    )
    backend.torch.cuda.reset_peak_memory_stats(backend.device)
    for condition in PREDICTION_FILES:
        _prediction(
            probe_interaction,
            condition,
            backend,
            top_k=int(config["model"]["top_k"]),
            beam_size=int(config["model"]["beam_size"]),
        )
    peak = backend.torch.cuda.max_memory_allocated(backend.device)
    path = output / "runtime_diagnostics.json"
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    value["peak_gpu_memory_bytes"] = max(int(value.get("peak_gpu_memory_bytes") or 0), peak)
    value["peak_gpu_memory_probe"] = {
        "conditions": list(PREDICTION_FILES),
        "interaction_count": 1,
        "interaction_id": probe_interaction["interaction_id"],
        "effective_context_token_count": context_token_count,
        "target_length": len(probe_interaction["segmented_pinyin"]),
        "prediction_artifacts_modified": False,
        "observed_peak_gpu_memory_bytes": peak,
    }
    write_json(path, value)
    return value["peak_gpu_memory_probe"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--download", action="store_true", help="Download official Drive folder before auditing")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--recompute-metrics", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete prediction files (default behavior)")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--runtime-probe", action="store_true")
    parser.add_argument("--condition", choices=("all", "pinyin_only", "contextual"), default="all")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    dataset_root = ROOT / config["dataset"]["root"]
    output = ROOT / config["output_dir"]

    if args.runtime_probe:
        print(json.dumps(probe_peak_gpu_memory(config), ensure_ascii=False, indent=2))
        return

    if args.download:
        import gdown
        gdown.download_folder(
            url=config["dataset"]["download_folder_url"],
            output=str(dataset_root),
            resume=True,
            quiet=False,
        )
    if args.recompute_metrics:
        result = recompute_metrics(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    prepared = prepare_livechat_baseline(config, root=ROOT)
    print(json.dumps(prepared["manifest"], ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    interactions = load_jsonl(output / "frozen_interactions.jsonl")
    backend = PinyinGPTConcatBackend(ROOT / config["model"]["checkpoint_path"], device="auto")
    if backend.device.type == "cuda":
        backend.torch.cuda.reset_peak_memory_stats(backend.device)
    conditions = list(PREDICTION_FILES) if args.condition == "all" else [args.condition]
    summaries = []
    for condition in conditions:
        summaries.append(
            run_inference_condition(
                condition,
                interactions,
                backend,
                output / PREDICTION_FILES[condition],
                top_k=int(config["model"]["top_k"]),
                beam_size=int(config["model"]["beam_size"]),
                force=args.force_recompute,
            )
        )
    runtime_diagnostics(output, backend, summaries)
    if all((output / filename).is_file() for filename in PREDICTION_FILES.values()):
        result = recompute_metrics(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"inference": summaries, "next": "resume remaining condition"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
