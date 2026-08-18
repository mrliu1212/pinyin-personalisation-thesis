"""Audit gate and context policy for the focused Multi3/H5000 v2 experiment.

This module performs no model inference.  It freezes the interaction-specific
128-position policy and prepares deterministic human-review artifacts before a
later, explicitly approved formal run.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from src.evaluation.deep_author_v2 import AUTHORS, sha256_file
from src.personalisation.h5000 import T1_MANIFEST_SHA256, T1_PREDICTIONS_SHA256


EXPERIMENT_NAME = "multi3_128_h5000_v2"
EXPERIMENT_STATUS = "HUMAN_AUDIT_REQUIRED"
EXPERIMENT_POSITION_CAP = 128
AUDIT_SEED = 40408
CONDITIONS = ("full_multi3", "initial_multi3")
HISTORY_BUDGET = 5000
PLANNED_METHODS = ("Generic", "F", "M1", "M2")
PERSONALISATION_METHODS = ("F", "M1", "M2")
SCHEMA_VERSION = 1

ROW_DIAGNOSTIC_FIELDS = (
    "anchor_id", "author", "condition", "pinyin_input", "pinyin_segments",
    "gold", "stored_context", "effective_context_128",
    "original_context_tokens", "effective_context_tokens", "context_truncated",
    "history_available", "ambiguous", "conflict", "visible_history_count",
    "same_pinyin_visible_history_count", "history_target_frequency_counts",
    "frequency_winner", "frequency_winner_count", "gold_historical_count",
    "generic_top1", "generic_gold_rank", "generic_top10", "f_top1",
    "f_gold_rank", "m1_top1", "m1_gold_rank", "m2_top1", "m2_gold_rank",
    "f_correct", "m1_correct", "m2_correct", "f_wrong_m2_correct",
    "f_correct_m2_wrong", "f_wrong_m1_correct", "f_correct_m1_wrong",
)

EVIDENCE_FIELDS = (
    "method", "retrieved_historical_interaction_id",
    "historical_effective_context_128", "historical_target",
    "retrieval_rank", "similarity_score", "m2_cross_encoder_score",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(rows[0]) if rows else ["audit_case_index"]
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            rendered = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(rendered)
    temporary.replace(path)


def _normalized_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, encoding="utf-8"
    ).strip()


@dataclass(frozen=True)
class EffectiveContext:
    stored_context: str
    effective_context_128: str
    stored_context_characters: int
    original_context_tokens: int
    effective_context_tokens: int
    pinyin_target_tokens: int
    prompt_tokens: int
    complete_generation_positions: int
    effective_maximum_positions: int
    context_truncated: bool


class InteractionContextPolicy:
    """PinyinGPT-Concat budget applied independently to each interaction."""

    def __init__(
        self,
        tokenizer: Any,
        model_maximum_positions: int,
        *,
        experimental_cap: int = EXPERIMENT_POSITION_CAP,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_maximum_positions = int(model_maximum_positions)
        self.experimental_cap = int(experimental_cap)
        self.effective_maximum_positions = min(
            self.model_maximum_positions, self.experimental_cap
        )
        if self.effective_maximum_positions < 1:
            raise ValueError("effective maximum positions must be positive")

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> "InteractionContextPolicy":
        try:
            from transformers import BertTokenizer, GPT2Config
        except ImportError as error:  # pragma: no cover - environment setup
            raise RuntimeError("transformers is required for the audit tokenizer") from error
        tokenizer = BertTokenizer.from_pretrained(checkpoint)
        config = GPT2Config.from_pretrained(checkpoint)
        return cls(tokenizer, int(config.n_positions))

    def apply(self, context: str, pinyin: Sequence[str]) -> EffectiveContext:
        segments = tuple(str(value) for value in pinyin)
        if not segments:
            raise ValueError("an interaction must contain at least one Pinyin segment")
        original_ids = self.tokenizer.encode(context, add_special_tokens=False)
        # Complete Concat generation consumes context + 2 + 2*k positions:
        # [CLS], context, [SEP], k Pinyin, [SEP], and k-1 later target forwards.
        available = self.effective_maximum_positions - (2 + 2 * len(segments))
        if available < 0:
            raise ValueError("interaction Pinyin target exceeds the 128-position policy")
        if len(original_ids) <= available:
            effective = context
            used_tokens = len(original_ids)
            truncated = False
        else:
            low, high = 0, len(context)
            while low < high:
                middle = (low + high) // 2
                length = len(
                    self.tokenizer.encode(context[middle:], add_special_tokens=False)
                )
                if length <= available:
                    high = middle
                else:
                    low = middle + 1
            effective = context[low:]
            used_tokens = len(
                self.tokenizer.encode(effective, add_special_tokens=False)
            )
            truncated = True
        prompt_tokens = used_tokens + len(segments) + 3
        complete_positions = used_tokens + 2 + 2 * len(segments)
        if complete_positions > self.effective_maximum_positions:
            raise AssertionError("effective context exceeds the experimental cap")
        if truncated and not context.endswith(effective):
            raise AssertionError("context policy did not preserve the most recent suffix")
        return EffectiveContext(
            stored_context=context,
            effective_context_128=effective,
            stored_context_characters=len(context),
            original_context_tokens=len(original_ids),
            effective_context_tokens=used_tokens,
            pinyin_target_tokens=len(segments),
            prompt_tokens=prompt_tokens,
            complete_generation_positions=complete_positions,
            effective_maximum_positions=self.effective_maximum_positions,
            context_truncated=truncated,
        )

    def contextualize_interaction(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Return a derived v2 row while retaining the original stored context."""

        segments = tuple(row.get("pinyin_segments") or str(row["pinyin_input"]).split())
        result = self.apply(str(row["context"]), segments)
        derived = dict(row)
        derived["stored_context"] = str(row["context"])
        derived["context"] = result.effective_context_128
        derived["effective_context_128"] = result.effective_context_128
        derived["context_policy"] = "pinyingpt-concat-total-positions-128-recent-suffix-v1"
        derived["context_diagnostics"] = asdict(result)
        return derived


def _percentile(values: Sequence[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


@dataclass
class Multi3AuditRunner:
    root: Path
    dataset_root: Path
    pinyingpt_model: Path
    t1_predictions: Path
    output_root: Path

    @property
    def condition_manifest_path(self) -> Path:
        return self.root / "results/evaluation/deep_author_v2/design/t1_condition_manifest.jsonl"

    @property
    def version_manifest_path(self) -> Path:
        return self.output_root / "design/version_manifest.json"

    def _conditions(self) -> list[dict[str, Any]]:
        digest = _normalized_sha256(self.condition_manifest_path)
        if digest != T1_MANIFEST_SHA256:
            raise RuntimeError(
                f"frozen Test hash mismatch: expected {T1_MANIFEST_SHA256}, got {digest}"
            )
        rows = [
            json.loads(line)
            for line in self.condition_manifest_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != 24_000:
            raise RuntimeError("frozen T1 condition manifest is not 24,000 rows")
        counts = Counter(str(row["condition"]) for row in rows)
        if any(counts[name] != 6000 for name in CONDITIONS):
            raise RuntimeError("frozen Multi3 Test population differs")
        return [row for row in rows if row["condition"] in CONDITIONS]

    def _verify_frozen_inputs(self) -> dict[str, Any]:
        if not self.t1_predictions.is_file():
            raise RuntimeError(f"frozen predecessor T1 predictions are absent: {self.t1_predictions}")
        predictions_hash = sha256_file(self.t1_predictions)
        if predictions_hash != T1_PREDICTIONS_SHA256:
            raise RuntimeError("frozen predecessor T1 predictions hash mismatch")
        return {
            "test_condition_manifest_normalized_sha256": T1_MANIFEST_SHA256,
            "predecessor_t1_predictions_sha256": predictions_hash,
        }

    def _source_parts(self, row: Mapping[str, Any]) -> dict[str, str]:
        work_path = (
            self.dataset_root / "data/processed/deep_author/works" / f"{row['work_id']}.json"
        )
        work = json.loads(work_path.read_text(encoding="utf-8"))
        text = str(work["cleaned_text"])
        start, end = int(row["source_position_start"]), int(row["source_position_end"])
        gold = str(row["gold"])
        if text[start:end] != gold:
            raise RuntimeError(f"source Gold mismatch for {row['condition_id']}")
        return {
            "source_context_before": text[max(0, start - 48):start],
            "source_gold": text[start:end],
            "source_context_after": text[end:min(len(text), end + 48)],
            "source_local_snippet": text[max(0, start - 48):min(len(text), end + 48)],
        }

    @staticmethod
    def _sample(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        selected: list[Mapping[str, Any]] = []
        for condition in CONDITIONS:
            for author_index, author in enumerate(AUTHORS):
                population = sorted(
                    (row for row in rows if row["condition"] == condition and row["author"] == author),
                    key=lambda row: str(row["condition_id"]),
                )
                rng = random.Random(AUDIT_SEED + 100 * CONDITIONS.index(condition) + author_index)
                selected.extend(sorted(rng.sample(population, 5), key=lambda row: str(row["condition_id"])))
        return selected

    @staticmethod
    def _context_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        stored = [int(row["stored_context_characters"]) for row in rows]
        original = [int(row["original_context_tokens"]) for row in rows]
        effective = [int(row["effective_context_tokens"]) for row in rows]
        complete = [int(row["complete_generation_positions"]) for row in rows]
        truncated = sum(bool(row["context_truncated"]) for row in rows)
        return {
            "n": len(rows),
            "stored_context_characters": _distribution(stored),
            "original_tokenizer_context_tokens": _distribution(original),
            "effective_context_tokens": _distribution(effective),
            "complete_generation_positions": _distribution(complete),
            "truncation_count": truncated,
            "truncation_rate": truncated / len(rows) if rows else None,
            "rows_exceeding_128_positions": sum(value > EXPERIMENT_POSITION_CAP for value in complete),
        }

    def _write_schema(self) -> None:
        _atomic_json(
            self.output_root / "design/row_diagnostic_schema.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "schema_only_no_predictions",
                "row_fields": list(ROW_DIAGNOSTIC_FIELDS),
                "evidence_fields": list(EVIDENCE_FIELDS),
                "post_run_exports": [
                    "audit/conflict_cases.jsonl", "audit/conflict_cases.md",
                    "audit/f_to_m2_rescues.jsonl", "audit/f_to_m2_rescues.md",
                    "audit/m2_harms_f.jsonl", "audit/m2_harms_f.md",
                    "diagnostics/all_test_diagnostics.jsonl", "audit/audit_summary.md",
                ],
                "conflict_definition": {
                    "ambiguous": True,
                    "unique_frequency_winner_required": True,
                    "gold_differs_from_winner": True,
                    "frequency_ties_excluded": True,
                    "history_budget": "H5000 strictly-prior same-user interactions before exact segmented-Pinyin filtering",
                },
            },
        )

    def _write_markdown(self, cases: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> None:
        lines = [
            "# Multi3 128-Position Human Audit", "", f"Status: **{EXPERIMENT_STATUS}**", "",
            "This report is inspection-only. Text after the Gold span is never model input.", "",
            "## Context summary", "", "```json", json.dumps(stats, ensure_ascii=False, indent=2), "```", "",
            "## Cases", "",
        ]
        for row in cases:
            lines.extend([
                f"### {row['audit_case_index']:02d}. {row['condition']} / {row['author']}", "",
                f"- Anchor: `{row['anchor_id']}`", f"- Pinyin: `{row['pinyin_input']}`",
                f"- Gold Multi3: `{row['gold']}` ({row['gold_char_length']} characters)",
                f"- Stored context: {row['stored_context_characters']} chars / {row['original_context_tokens']} tokens",
                f"- Effective context: {row['effective_context_tokens']} tokens; truncated = `{str(row['context_truncated']).lower()}`",
                f"- Complete position use: {row['complete_generation_positions']} / {row['effective_maximum_positions']}", "",
                "Stored/raw context:", "", f"> {str(row['stored_context']).replace(chr(10), '<br>')}", "",
                "Effective context supplied to the future model:", "", f"> {str(row['effective_context_128']).replace(chr(10), '<br>')}", "",
                "Human-only source check (before **Gold** after):", "",
                f"> {str(row['source_context_before']).replace(chr(10), '<br>')} **{row['source_gold']}** {str(row['source_context_after']).replace(chr(10), '<br>')}", "",
                "Review: [ ] context correct  [ ] Pinyin/segmentation correct  [ ] Gold follows context", "",
            ])
        path = self.output_root / "audit/random_test_cases.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    def prepare_audit(self) -> dict[str, Any]:
        frozen = self._verify_frozen_inputs()
        rows = self._conditions()
        policy = InteractionContextPolicy.from_checkpoint(self.pinyingpt_model)
        context_cache: dict[tuple[str, tuple[str, ...]], EffectiveContext] = {}
        prepared: list[dict[str, Any]] = []
        for row in rows:
            segments = tuple(str(row["pinyin_input"]).split())
            key = (str(row["context"]), segments)
            result = context_cache.get(key)
            if result is None:
                result = policy.apply(key[0], segments)
                context_cache[key] = result
            prepared.append({**row, **asdict(result), "pinyin_segments": list(segments)})
        if len(prepared) != 12_000:
            raise AssertionError("focused Test audit must contain 12,000 rows")
        if any(row["complete_generation_positions"] > EXPERIMENT_POSITION_CAP for row in prepared):
            raise AssertionError("a prepared interaction exceeds 128 positions")

        selected_rows = self._sample(prepared)
        cases = []
        for index, row in enumerate(selected_rows, start=1):
            cases.append({
                "audit_case_index": index,
                "anchor_id": row["anchor_id"], "condition_id": row["condition_id"],
                "author": row["author"], "condition": row["condition"], "split": "Test",
                "stored_context": row["stored_context"],
                "stored_context_characters": row["stored_context_characters"],
                "effective_context_128": row["effective_context_128"],
                "effective_context_tokens": row["effective_context_tokens"],
                "original_context_tokens": row["original_context_tokens"],
                "context_truncated": row["context_truncated"],
                "effective_maximum_positions": row["effective_maximum_positions"],
                "prompt_tokens": row["prompt_tokens"],
                "complete_generation_positions": row["complete_generation_positions"],
                "pinyin_input": row["pinyin_input"], "pinyin_segments": row["pinyin_segments"],
                "gold": row["gold"], "gold_char_length": row["gold_char_length"],
                "target_type": row["target_type"], "work_id": row["work_id"],
                "source_position_start": row["source_position_start"],
                "source_position_end": row["source_position_end"],
                **self._source_parts(row),
            })
        expected_strata = Counter((row["condition"], row["author"]) for row in cases)
        if set(expected_strata.values()) != {5} or len(expected_strata) != 12:
            raise AssertionError("audit sample is not 5 cases per condition/author stratum")

        stats = {
            condition: self._context_stats([row for row in prepared if row["condition"] == condition])
            for condition in CONDITIONS
        }
        stats["combined"] = self._context_stats(prepared)
        stats["policy"] = {
            "model_n_positions": policy.model_maximum_positions,
            "experimental_cap": EXPERIMENT_POSITION_CAP,
            "effective_maximum_positions": policy.effective_maximum_positions,
            "truncation_side": "left/oldest; most recent suffix retained",
        }

        jsonl_path = self.output_root / "audit/random_test_cases.jsonl"
        csv_path = self.output_root / "audit/random_test_cases.csv"
        stats_path = self.output_root / "audit/context_length_statistics.json"
        _write_jsonl(jsonl_path, cases)
        _write_csv(csv_path, cases)
        _atomic_json(stats_path, stats)
        self._write_markdown(cases, stats)
        self._write_schema()

        prior_approval = False
        if self.version_manifest_path.is_file():
            prior = json.loads(self.version_manifest_path.read_text(encoding="utf-8"))
            prior_approval = bool(prior.get("human_audit_approved", False))
        artifacts = {
            "audit_jsonl": str(jsonl_path.relative_to(self.root)).replace("\\", "/"),
            "audit_csv": str(csv_path.relative_to(self.root)).replace("\\", "/"),
            "audit_markdown": str((self.output_root / "audit/random_test_cases.md").relative_to(self.root)).replace("\\", "/"),
            "context_statistics": str(stats_path.relative_to(self.root)).replace("\\", "/"),
            "row_diagnostic_schema": str((self.output_root / "design/row_diagnostic_schema.json").relative_to(self.root)).replace("\\", "/"),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION, "experiment_name": EXPERIMENT_NAME,
            "date": datetime.now().date().isoformat(), "git_branch": _git_value(self.root, "branch", "--show-current"),
            "implementation_commit": None, "worktree_head_at_preparation": _git_value(self.root, "rev-parse", "HEAD"),
            "status": EXPERIMENT_STATUS, "human_audit_required": True,
            "human_audit_approved": prior_approval, "audit_seed": AUDIT_SEED,
            "test_hash": frozen["test_condition_manifest_normalized_sha256"],
            "frozen_inputs": frozen, "conditions": list(CONDITIONS), "history_budget": "H5000",
            "history_budget_interactions": HISTORY_BUDGET, "planned_methods": list(PLANNED_METHODS),
            "context_policy": {
                "name": "pinyingpt-concat-total-positions-128-recent-suffix-v1",
                "total_position_cap": EXPERIMENT_POSITION_CAP,
                "effective_maximum": "min(model.config.n_positions, 128)",
                "truncation": "left-truncate oldest context; retain most recent suffix",
                "interaction_specific": True,
                "stored_512_character_context_preserved": True,
                "generic_m1_m2_use_same_per_interaction_effective_context": True,
                "m2_pair_serialization_safety_unchanged": True,
                "f_semantics_unchanged": True,
            },
            "predecessor": {
                "name": "v1_long_context_matrix", "classification": "long-context exploratory baseline",
                "tag": "personalisation-v1-long-context-matrix", "commit": "617a20fc7e0c08d3d04eafd9f29302d0a9d1193e",
                "result_root": "results/personalisation/reranking_matrix", "status": "intentionally_stopped",
            },
            "audit_case_condition_ids": [row["condition_id"] for row in cases],
            "artifacts": artifacts,
            "artifact_sha256": {name: sha256_file(self.root / path) for name, path in artifacts.items()},
            "formal_cells": [
                {"condition": condition, "history_budget": "H5000", "method": method, "state": "blocked_human_audit"}
                for condition in CONDITIONS for method in PERSONALISATION_METHODS
            ],
            "generic_test_inference_rows": 0, "prepared_at": _utc_now(),
        }
        _atomic_json(self.version_manifest_path, manifest)
        return {
            "status": EXPERIMENT_STATUS, "test_rows_audited": len(prepared),
            "human_audit_cases": len(cases), "test_hash": manifest["test_hash"],
            "manifest": str(self.version_manifest_path), "artifacts": artifacts,
        }

    def preflight(self) -> dict[str, Any]:
        if not self.version_manifest_path.is_file():
            raise RuntimeError("run --phase prepare-audit first")
        manifest = json.loads(self.version_manifest_path.read_text(encoding="utf-8"))
        frozen = self._verify_frozen_inputs()
        self._conditions()
        artifact_checks = {}
        for name, relative in manifest["artifacts"].items():
            path = self.root / relative
            actual = sha256_file(path) if path.is_file() else None
            expected = manifest["artifact_sha256"][name]
            artifact_checks[name] = {"exists": path.is_file(), "sha256_matches": actual == expected}
        approval = bool(manifest.get("human_audit_approved", False))
        passed = all(value["sha256_matches"] for value in artifact_checks.values())
        return {
            "status": "READY_FOR_FORMAL_RUN" if passed and approval else EXPERIMENT_STATUS,
            "eligible_for_formal_run": passed and approval,
            "human_audit_approved": approval,
            "test_hash_matches": frozen["test_condition_manifest_normalized_sha256"] == manifest["test_hash"],
            "artifacts": artifact_checks,
            "generic_test_inference_rows": 0,
        }

    def formal_run(self) -> dict[str, Any]:
        preflight = self.preflight()
        if not preflight["human_audit_approved"]:
            raise RuntimeError(
                "formal run refused: design/version_manifest.json has human_audit_approved != true"
            )
        if not preflight["eligible_for_formal_run"]:
            raise RuntimeError("formal run refused: preflight failed")
        raise RuntimeError(
            "formal run is intentionally not activated by the audit-preparation stage; "
            "implement and review the six-cell execution adapter after human approval"
        )
