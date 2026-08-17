"""Frozen Deep Author Evaluation V2 design and T1 scoring utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import random
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

from src.datasets.deep_author.pipeline import full_pinyin, initial_pinyin, is_han, stable_hash


DATASET_V1_SHA256 = "8d1a98e18a5f7ed997930b65bbd1149c3d52daaa22ac2c59771256a966648da2"
DATASET_V1_BYTES = 2_048_557_493
SEED = 40408
AUTHORS = ("Re_spectators", "MScarlet", "Etinjat", "Agent Phage", "QBLevi", "breaddddd")
CONDITIONS = ("full_short", "initial_short", "full_multi3", "initial_multi3")
CHECKPOINT_REVISION = "76dd20dc92d8236a350fb732e99dde6fa15e2263"
OFFICIAL_CODE_REVISION = "8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1"
BACKEND_SOURCE_REVISION = "07a79f301a094d3db88780f00fcf85a4abf80d7f"
BACKEND_INTEGRATION_REVISION = "8c608f106ee7bb49ca5573e72de3da5eeb2290af"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(canonical_json(row) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def anchor_id(work_id: str, start: int) -> str:
    return "da-v2-anchor-" + stable_hash(work_id, start, "multi3")[:24]


def condition_id(anchor: str, condition: str) -> str:
    return "da-v2-condition-" + stable_hash(anchor, condition)[:24]


def load_tokens(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def valid_anchors_for_work(work: Mapping[str, Any], tokens: Sequence[Mapping[str, Any]], compatibility: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    text = str(work["cleaned_text"])
    rows = []
    for index in range(len(tokens) - 2):
        trio = tokens[index : index + 3]
        if int(trio[0]["start"]) <= 0 or not all(row["is_han"] for row in trio):
            continue
        if any(int(trio[offset]["end"]) != int(trio[offset + 1]["start"]) for offset in (0, 1)):
            continue
        short = str(trio[0]["text"])
        multi = "".join(str(row["text"]) for row in trio)
        if not all(is_han(character) for character in multi):
            continue
        try:
            short_full = full_pinyin(short)
            multi_full = full_pinyin(multi)
        except ValueError:
            continue
        short_initial = initial_pinyin(short_full)
        multi_initial = initial_pinyin(multi_full)
        full_ok = all(character in compatibility.get(pinyin, ()) for character, pinyin in zip(multi, multi_full))
        initial_ok = all(character in compatibility.get(pinyin, ()) for character, pinyin in zip(multi, multi_initial))
        if not full_ok or not initial_ok:
            continue
        start = int(trio[0]["start"])
        end = int(trio[-1]["end"])
        context = text[max(0, start - 512) : start]
        row_id = anchor_id(str(work["work_id"]), start)
        rows.append(
            {
                "anchor_id": row_id,
                "author": work["author_name"],
                "author_id": work["author_id"],
                "work_id": work["work_id"],
                "work_title": work["page_title"],
                "creation_date": work["creation_date"],
                "source_hash": work["SHA256"],
                "cleaned_text_hash": work["cleaned_sha256"],
                "source_position_start": start,
                "source_position_end": end,
                "context_source_position_start": max(0, start - 512),
                "context": context,
                "short_gold": short,
                "multi3_gold": multi,
                "short_full_pinyin": " ".join(short_full),
                "short_initial_pinyin": " ".join(short_initial),
                "multi3_full_pinyin": " ".join(multi_full),
                "multi3_initial_pinyin": " ".join(multi_initial),
                "multi3_token_count": 3,
                "short_gold_char_length": len(short),
                "multi3_gold_char_length": len(multi),
            }
        )
    return rows


def choose_split(works: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Choose chronological History/Dev/Test cuts closest to 70/10/20 volume."""

    total = sum(int(row["eligible_anchor_count"]) for row in works)
    best: tuple[float, int, int] | None = None
    for history_end in range(5, len(works) - 5 + 1):
        for dev_end in range(history_end + 2, len(works) - 3 + 1):
            volumes = (
                sum(int(row["eligible_anchor_count"]) for row in works[:history_end]),
                sum(int(row["eligible_anchor_count"]) for row in works[history_end:dev_end]),
                sum(int(row["eligible_anchor_count"]) for row in works[dev_end:]),
            )
            if volumes[2] < 1000:
                continue
            fractions = tuple(value / total for value in volumes)
            error = sum((actual - target) ** 2 for actual, target in zip(fractions, (0.7, 0.1, 0.2)))
            candidate = (error, history_end, dev_end)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("no chronological split satisfies the frozen minima")
    return best[1], best[2]


def balanced_sample(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    by_work: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_work[str(row["work_id"])].append(dict(row))
    rng = random.Random(seed)
    for work_rows in by_work.values():
        work_rows.sort(key=lambda row: (int(row["source_position_start"]), row["anchor_id"]))
        rng.shuffle(work_rows)
    work_ids = sorted(by_work)
    rng.shuffle(work_ids)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < count:
        progress = False
        for work in work_ids:
            if offset < len(by_work[work]):
                selected.append(by_work[work][offset])
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            raise ValueError(f"only {len(selected)} anchors available; {count} required")
        offset += 1
    return sorted(selected, key=lambda row: row["anchor_id"])


def conditions_for_anchor(anchor: Mapping[str, Any]) -> list[dict[str, Any]]:
    definitions = (
        ("full_short", "short", "full", "short_gold", "short_full_pinyin"),
        ("initial_short", "short", "initial", "short_gold", "short_initial_pinyin"),
        ("full_multi3", "multi3", "full", "multi3_gold", "multi3_full_pinyin"),
        ("initial_multi3", "multi3", "initial", "multi3_gold", "multi3_initial_pinyin"),
    )
    rows = []
    for condition, target_type, pinyin_type, gold_key, pinyin_key in definitions:
        rows.append(
            {
                "condition_id": condition_id(str(anchor["anchor_id"]), condition),
                "anchor_id": anchor["anchor_id"],
                "author": anchor["author"],
                "work_id": anchor["work_id"],
                "condition": condition,
                "target_type": target_type,
                "pinyin_type": pinyin_type,
                "context": anchor["context"],
                "pinyin_input": anchor[pinyin_key],
                "gold": anchor[gold_key],
                "gold_char_length": len(str(anchor[gold_key])),
                "source_position_start": anchor["source_position_start"],
                "source_position_end": anchor["source_position_start"] + len(str(anchor[gold_key])),
                "source_hash": anchor["source_hash"],
                "cleaned_text_hash": anchor["cleaned_text_hash"],
            }
        )
    return rows


@dataclass
class DesignBuilder:
    root: Path

    def run(self) -> dict[str, Any]:
        source_root = self.root / ".build/dataset-v1-reconstruction"
        dataset = source_root / "data/processed/deep_author/interactions_t1_ready.jsonl"
        if dataset.stat().st_size != DATASET_V1_BYTES or sha256_file(dataset) != DATASET_V1_SHA256:
            raise RuntimeError("Dataset V1 source does not match the frozen byte size and SHA-256")
        works_root = source_root / "data/processed/deep_author/works"
        compatibility = json.loads((self.root / ".build/pinyingpt2-concat/pinyin2char.json").read_text(encoding="utf-8"))
        work_rows = []
        anchors_by_work: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(works_root.glob("da-work-*.json")):
            work = json.loads(path.read_text(encoding="utf-8"))
            tokens = load_tokens(path.with_name(path.stem + ".tokens.jsonl"))
            anchors = valid_anchors_for_work(work, tokens, compatibility)
            anchors_by_work[work["work_id"]] = anchors
            work_rows.append(
                {
                    "author": work["author_name"], "author_id": work["author_id"],
                    "work_id": work["work_id"], "work_title": work["page_title"],
                    "creation_date": work["creation_date"], "source_hash": work["SHA256"],
                    "cleaned_text_hash": work["cleaned_sha256"], "eligible_anchor_count": len(anchors),
                }
            )
        if tuple(sorted({row["author"] for row in work_rows})) != tuple(sorted(AUTHORS)):
            raise RuntimeError("author set differs from the frozen six authors")

        split_rows = []
        summary_authors = {}
        sampled = []
        for author_index, author in enumerate(AUTHORS):
            author_works = sorted((row for row in work_rows if row["author"] == author), key=lambda row: (row["creation_date"], row["work_id"]))
            history_end, dev_end = choose_split(author_works)
            assignments = ["history"] * history_end + ["dev"] * (dev_end - history_end) + ["test"] * (len(author_works) - dev_end)
            for row, split in zip(author_works, assignments):
                split_rows.append({**row, "chronological_index": len([x for x in split_rows if x["author"] == author]), "split": split})
            test_anchors = [anchor for row, split in zip(author_works, assignments) if split == "test" for anchor in anchors_by_work[row["work_id"]]]
            author_sample = balanced_sample(test_anchors, 1000, SEED + author_index)
            sampled.extend(author_sample)
            volumes = {split: sum(int(row["eligible_anchor_count"]) for row, assigned in zip(author_works, assignments) if assigned == split) for split in ("history", "dev", "test")}
            works_count = {split: assignments.count(split) for split in ("history", "dev", "test")}
            summary_authors[author] = {"works": works_count, "eligible_anchors": volumes, "sampled_test_anchors": len(author_sample)}

        conditions = [condition for anchor in sampled for condition in conditions_for_anchor(anchor)]
        self.validate(split_rows, sampled, conditions)
        output = self.root / "results/evaluation/deep_author_v2/design"
        write_csv(output / "work_split_manifest.csv", split_rows, list(split_rows[0]))
        split_summary = {"dataset_sha256": DATASET_V1_SHA256, "dataset_bytes": DATASET_V1_BYTES, "authors": summary_authors}
        write_json(output / "work_split_summary.json", split_summary)
        write_csv(output / "t1_anchor_manifest.csv", sampled, list(sampled[0]))
        write_jsonl(output / "t1_condition_manifest.jsonl", conditions)
        counts = Counter(row["condition"] for row in conditions)
        write_json(output / "t1_sampling_summary.json", {"seed": SEED, "anchors": len(sampled), "conditions": len(conditions), "by_condition": counts, "authors": Counter(row["author"] for row in sampled), "sampling": "deterministic round-robin work-balanced within Test"})
        write_json(output / "t2_history_pool_summary.json", {"performance_run": False, "by_author": {author: summary_authors[author]["eligible_anchors"]["history"] for author in AUTHORS}})
        write_json(output / "t3_feasibility_summary.json", {"performance_run": False, "scope": "not executed in Evaluation V2 T1"})
        review = [{**row, "split_ok": "", "context_ok": "", "pinyin_ok": "", "multi3_ok": "", "notes": ""} for author in AUTHORS for row in [next(item for item in sampled if item["author"] == author)]]
        write_csv(output / "t1_manual_review.csv", review, list(review[0]))
        checksums = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(output.iterdir()) if path.is_file()}
        write_json(output / "manifest.json", {"dataset": "Deep Author Dataset V1", "dataset_sha256": DATASET_V1_SHA256, "seed": SEED, "authors": list(AUTHORS), "anchors": 6000, "conditions": 24000, "model_inference": False, "outputs": checksums})
        return {"works": len(work_rows), "anchors": len(sampled), "conditions": len(conditions), "authors": summary_authors}

    @staticmethod
    def validate(split_rows: Sequence[Mapping[str, Any]], anchors: Sequence[Mapping[str, Any]], conditions: Sequence[Mapping[str, Any]]) -> None:
        if len(anchors) != 6000 or Counter(row["author"] for row in anchors) != Counter({author: 1000 for author in AUTHORS}):
            raise AssertionError("expected exactly 1,000 anchors per author")
        if len({row["anchor_id"] for row in anchors}) != 6000:
            raise AssertionError("anchor IDs are not unique")
        if len(conditions) != 24000 or len({row["condition_id"] for row in conditions}) != 24000:
            raise AssertionError("expected exactly 24,000 unique conditions")
        if Counter(row["condition"] for row in conditions) != Counter({condition: 6000 for condition in CONDITIONS}):
            raise AssertionError("condition balance differs from frozen design")
        split_by_work = {row["work_id"]: row["split"] for row in split_rows}
        if any(split_by_work[row["work_id"]] != "test" for row in anchors):
            raise AssertionError("T1 anchors must come only from Test works")
        for author in AUTHORS:
            rows = [row for row in split_rows if row["author"] == author]
            counts = Counter(row["split"] for row in rows)
            if counts["history"] < 5 or counts["dev"] < 2 or counts["test"] < 3:
                raise AssertionError(f"work minimum failed for {author}")
            order = {"history": 0, "dev": 1, "test": 2}
            if [order[row["split"]] for row in rows] != sorted(order[row["split"]] for row in rows):
                raise AssertionError(f"chronology failed for {author}")
        grouped = defaultdict(set)
        for row in conditions:
            grouped[row["anchor_id"]].add(row["condition"])
            if str(row["context"]).endswith(str(row["gold"])):
                pass  # valid natural repetition; leakage is checked by absolute source offsets instead
        if any(values != set(CONDITIONS) for values in grouped.values()):
            raise AssertionError("an anchor is missing a paired condition")


def metric_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    count = len(rows)
    found = [int(row["gold_rank"]) for row in rows if row.get("gold_rank") not in (None, "")]
    return {
        "n": count,
        "top1": sum(bool(row["top1_correct"]) for row in rows) / count,
        "top3": sum(bool(row["top3_correct"]) for row in rows) / count,
        "mrr_at_10": sum(float(row["reciprocal_rank"]) for row in rows) / count,
        "missing_at_10": sum(bool(row["missing_at_10"]) for row in rows) / count,
        "mean_rank_given_top10": statistics.fmean(found) if found else None,
    }


def aggregate_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_author_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_author[str(row["author"])].append(row)
        by_author_condition[(str(row["author"]), str(row["condition"]))].append(row)
        by_condition[str(row["condition"])].append(row)
    per_author = {author: metric_values(by_author[author]) for author in AUTHORS}
    per_author_condition = {
        f"{author}|{condition}": metric_values(by_author_condition[(author, condition)])
        for author in AUTHORS
        for condition in CONDITIONS
    }
    metric_names = ("top1", "top3", "mrr_at_10", "missing_at_10", "mean_rank_given_top10")
    overall_macro = {
        key: statistics.fmean(float(per_author[author][key]) for author in AUTHORS if per_author[author][key] is not None)
        for key in metric_names
    }
    per_condition_macro = {}
    per_condition_micro = {}
    for condition in CONDITIONS:
        author_values = [per_author_condition[f"{author}|{condition}"] for author in AUTHORS]
        per_condition_macro[condition] = {
            key: statistics.fmean(float(value[key]) for value in author_values if value[key] is not None)
            for key in metric_names
        }
        per_condition_micro[condition] = metric_values(by_condition[condition])
    return {
        "primary_macro_author": overall_macro,
        "per_condition_macro_author": per_condition_macro,
        "secondary_micro": {
            "overall": metric_values(predictions),
            "per_condition": per_condition_micro,
        },
        "per_author": per_author,
        "per_author_condition": per_author_condition,
    }


def paired_rows(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_anchor_condition = {(str(row["anchor_id"]), str(row["condition"])): row for row in predictions}
    output = []
    for target, full_name, initial_name in (("short", "full_short", "initial_short"), ("multi3", "full_multi3", "initial_multi3")):
        for author in AUTHORS:
            anchors = sorted({anchor for anchor, condition in by_anchor_condition if condition == full_name and by_anchor_condition[(anchor, condition)]["author"] == author})
            counts = Counter()
            for anchor in anchors:
                full = bool(by_anchor_condition[(anchor, full_name)]["top1_correct"])
                initial = bool(by_anchor_condition[(anchor, initial_name)]["top1_correct"])
                counts[(full, initial)] += 1
            output.append({
                "target": target, "author": author, "n": len(anchors),
                "both_correct": counts[(True, True)],
                "full_correct_initial_wrong": counts[(True, False)],
                "full_wrong_initial_correct": counts[(False, True)],
                "both_wrong": counts[(False, False)],
                "initial_minus_full_top1": (counts[(False, True)] - counts[(True, False)]) / len(anchors),
            })
    return output


@dataclass
class T1Runner:
    root: Path

    @property
    def design_root(self) -> Path:
        return self.root / "results/evaluation/deep_author_v2/design"

    @property
    def output_root(self) -> Path:
        return self.root / "results/evaluation/deep_author_v2/t1"

    def load_conditions(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.design_root / "t1_condition_manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def validate_cached_prediction(row: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
        for key, frozen_value in frozen.items():
            if row.get(key) != frozen_value:
                raise RuntimeError(f"cached prediction differs from frozen manifest: {frozen['condition_id']} {key}")
        if row.get("checkpoint_revision") != CHECKPOINT_REVISION or row.get("official_code_revision") != OFFICIAL_CODE_REVISION:
            raise RuntimeError("cached prediction uses a non-frozen model or code revision")
        if row.get("beam_size") != 16 or row.get("top_k") != 10:
            raise RuntimeError("cached prediction uses non-frozen decoding parameters")
        candidates = row.get("top10_candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 10:
            raise RuntimeError(f"cached prediction has an invalid candidate surface: {frozen['condition_id']}")
        if [candidate.get("rank") for candidate in candidates] != list(range(1, len(candidates) + 1)):
            raise RuntimeError(f"cached candidate ranks are invalid: {frozen['condition_id']}")
        texts = [candidate.get("text") for candidate in candidates]
        if any(not isinstance(text, str) or not text for text in texts) or len(set(texts)) != len(texts):
            raise RuntimeError(f"cached candidate text is invalid: {frozen['condition_id']}")
        if any(not isinstance(candidate.get("log_probability"), (int, float)) for candidate in candidates):
            raise RuntimeError(f"cached candidate scores are invalid: {frozen['condition_id']}")
        expected_rank = next((index for index, text in enumerate(texts, start=1) if text == frozen["gold"]), None)
        expected_values = {
            "gold_rank": expected_rank,
            "top1_correct": expected_rank == 1,
            "top3_correct": expected_rank is not None and expected_rank <= 3,
            "top10_present": expected_rank is not None,
            "missing_at_10": expected_rank is None,
            "reciprocal_rank": 0.0 if expected_rank is None else 1.0 / expected_rank,
        }
        for key, expected_value in expected_values.items():
            if row.get(key) != expected_value:
                raise RuntimeError(f"cached derived result is invalid: {frozen['condition_id']} {key}")
        used_context = row.get("model_used_context")
        if not isinstance(used_context, str) or not str(frozen["context"]).endswith(used_context):
            raise RuntimeError(f"cached model context is not a suffix of frozen context: {frozen['condition_id']}")

    def load_completed(self, conditions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        prediction_path = self.output_root / "predictions.jsonl"
        if not prediction_path.exists():
            return {}
        expected = {str(row["condition_id"]): row for row in conditions}
        completed: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(prediction_path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            condition_id_value = str(row.get("condition_id", ""))
            if condition_id_value in completed:
                raise RuntimeError(f"duplicate cached condition ID at line {line_number}: {condition_id_value}")
            if condition_id_value not in expected:
                raise RuntimeError(f"unknown cached condition ID at line {line_number}: {condition_id_value}")
            self.validate_cached_prediction(row, expected[condition_id_value])
            completed[condition_id_value] = row
        return completed

    def run(self) -> dict[str, Any]:
        from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

        conditions = self.load_conditions()
        if len(conditions) != 24000:
            raise RuntimeError("frozen design does not contain exactly 24,000 conditions")
        self.output_root.mkdir(parents=True, exist_ok=True)
        prediction_path = self.output_root / "predictions.jsonl"
        completed = self.load_completed(conditions)
        existing_rows = len(completed)
        model_load_started = time.perf_counter()
        backend = PinyinGPTConcatBackend(self.root / ".build/pinyingpt2-concat")
        model_load_seconds = time.perf_counter() - model_load_started
        if backend.device.type == "cuda":
            backend.torch.cuda.reset_peak_memory_stats(backend.device)
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        added_rows = 0
        next_progress = min(24000, ((existing_rows // 100) + 1) * 100)
        mode = "a" if completed else "w"
        with prediction_path.open(mode, encoding="utf-8", newline="\n") as destination:
            pending = [condition for condition in conditions if condition["condition_id"] not in completed]
            for batch_start in range(0, len(pending), 16):
                raw_batch = pending[batch_start : batch_start + 16]
                prepared_batch = []
                for condition in raw_batch:
                    oracle_segments = str(condition["pinyin_input"]).split()
                    used_context, original_tokens, used_tokens, truncated = backend.truncate_context_for_generation(condition["context"], oracle_segments)
                    prepared_batch.append((condition, oracle_segments, used_context, original_tokens, used_tokens, truncated))
                # Equal prompt and target lengths allow exact padding-free shared forwards.
                groups: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
                for item in prepared_batch:
                    prompt_ids, _ = backend._prompt(item[2], item[1])
                    groups[(len(item[1]), len(prompt_ids))].append(item)
                for group in groups.values():
                    results = backend.generate_batch([(item[2], item[1]) for item in group], top_k=10, beam_size=16)
                    for item, result in zip(group, results):
                        condition, _, used_context, original_tokens, used_tokens, truncated = item
                        candidates = [candidate.to_dict() for candidate in result.candidates]
                        gold_rank = next((candidate["rank"] for candidate in candidates if candidate["text"] == condition["gold"]), None)
                        row = {
                            **condition, "top10_candidates": candidates, "gold_rank": gold_rank,
                            "top1_correct": gold_rank == 1, "top3_correct": gold_rank is not None and gold_rank <= 3,
                            "top10_present": gold_rank is not None, "missing_at_10": gold_rank is None,
                            "reciprocal_rank": 0.0 if gold_rank is None else 1.0 / gold_rank,
                            "original_stored_context_length": len(condition["context"]), "original_stored_context_tokens": original_tokens,
                            "model_used_context_length": len(used_context), "model_used_context_tokens": used_tokens,
                            "context_truncated": truncated, "model_used_context": used_context,
                            "beam_size": 16, "top_k": 10, "runtime_device": result.runtime_device,
                            "checkpoint_revision": CHECKPOINT_REVISION, "official_code_revision": OFFICIAL_CODE_REVISION,
                            "backend_source_revision": BACKEND_SOURCE_REVISION,
                            "backend_integration_revision": BACKEND_INTEGRATION_REVISION,
                            "inference_implementation": "semantic-equivalent KV-cache; independent beam-16 searches",
                        }
                        destination.write(canonical_json(row) + "\n")
                        completed[row["condition_id"]] = row
                        added_rows += 1
                destination.flush()
                if len(completed) >= next_progress:
                    elapsed = time.perf_counter() - started
                    throughput = added_rows / elapsed
                    eta = (24000 - len(completed)) / throughput
                    print(
                        f"predictions {len(completed)}/24000; elapsed={elapsed:.1f}s; "
                        f"throughput={throughput:.3f}/s; eta={eta:.1f}s",
                        flush=True,
                    )
                    next_progress = min(24000, ((len(completed) // 100) + 1) * 100)
        elapsed = time.perf_counter() - started
        runtime = backend.runtime_info()
        runtime.update({
            "schema_version": 2,
            "status": "complete",
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_load_seconds": model_load_seconds,
            "inference_seconds_latest_invocation": elapsed,
            "resume_existing_rows": existing_rows,
            "rows_added_latest_invocation": added_rows,
            "final_rows": len(completed),
            "conditions_per_second_latest_invocation": added_rows / elapsed if added_rows else 0.0,
            "dtype": str(next(backend.model.parameters()).dtype),
            "transformers_version": package_version("transformers"),
            "python_version": platform.python_version(),
            "backend_source_revision": BACKEND_SOURCE_REVISION,
            "backend_integration_revision": BACKEND_INTEGRATION_REVISION,
            "inference_implementation": "semantic-equivalent KV-cache; independent beam-16 searches; exact-length padding-free batch groups",
            "batch_window": 16,
            "beam_size": 16,
            "top_k": 10,
            "prediction_cache_sha256": sha256_file(prediction_path),
        })
        if backend.device.type == "cuda":
            properties = backend.torch.cuda.get_device_properties(backend.device)
            runtime["gpu_total_memory_bytes"] = properties.total_memory
            runtime["gpu_peak_allocated_bytes_latest_invocation"] = backend.torch.cuda.max_memory_allocated(backend.device)
            runtime["gpu_peak_reserved_bytes_latest_invocation"] = backend.torch.cuda.max_memory_reserved(backend.device)
        write_json(self.output_root / "runtime_summary.json", runtime)
        return self.metrics(runtime_seconds=elapsed, resumed_rows=existing_rows)

    def metrics(self, runtime_seconds: float | None = None, resumed_rows: int | None = None) -> dict[str, Any]:
        conditions = self.load_conditions()
        completed = self.load_completed(conditions)
        predictions = list(completed.values())
        expected = {row["condition_id"]: row for row in conditions}
        actual = {row["condition_id"]: row for row in predictions}
        if len(predictions) != 24000 or len(actual) != 24000 or set(actual) != set(expected):
            raise RuntimeError("predictions do not map one-to-one to the 24,000 frozen conditions")
        metrics = aggregate_metrics(predictions)
        runtime_path = self.output_root / "runtime_summary.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
        metrics.update({
            "schema_version": 2, "predictions": len(predictions),
            "checkpoint": "aihijo/transformers4ime-pinyingpt-concat",
            "checkpoint_revision": CHECKPOINT_REVISION, "official_code_revision": OFFICIAL_CODE_REVISION,
            "beam_size": 16, "top_k": 10, "oracle_pinyin_segmentation": True,
            "runtime_device": sorted({row["runtime_device"] for row in predictions}),
            "context_truncation_count": sum(bool(row["context_truncated"]) for row in predictions),
            "runtime_seconds_latest_invocation": runtime_seconds if runtime_seconds is not None else runtime.get("inference_seconds_latest_invocation"),
            "resume_existing_rows": resumed_rows if resumed_rows is not None else runtime.get("resume_existing_rows"),
            "backend_source_revision": BACKEND_SOURCE_REVISION,
            "backend_integration_revision": BACKEND_INTEGRATION_REVISION,
            "author_identity_used": False, "history_used": False, "personalisation_used": False, "dev_scored": False,
        })
        write_json(self.output_root / "metrics_summary.json", metrics)
        author_rows = [{"author": author, **metrics["per_author"][author]} for author in AUTHORS]
        write_csv(self.output_root / "metrics_by_author.csv", author_rows, list(author_rows[0]))
        write_json(self.output_root / "metrics_by_author.json", author_rows)
        author_condition_rows = []
        for author in AUTHORS:
            for condition in CONDITIONS:
                author_condition_rows.append({"author": author, "condition": condition, **metrics["per_author_condition"][f"{author}|{condition}"]})
        write_csv(self.output_root / "metrics_by_author_condition.csv", author_condition_rows, list(author_condition_rows[0]))
        condition_rows = [
            {"condition": condition, **metrics["per_condition_macro_author"][condition]}
            for condition in CONDITIONS
        ]
        write_csv(self.output_root / "metrics_by_condition.csv", condition_rows, list(condition_rows[0]))
        write_json(self.output_root / "metrics_by_condition.json", condition_rows)

        grouped_work: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped_work[(row["author"], row["work_id"], row["condition"])].append(row)
        work_rows = [{"author": key[0], "work_id": key[1], "condition": key[2], **metric_values(rows)} for key, rows in sorted(grouped_work.items())]
        write_csv(self.output_root / "metrics_by_work.csv", work_rows, list(work_rows[0]))
        paired = paired_rows(predictions)
        write_csv(self.output_root / "paired_full_initial.csv", paired, list(paired[0]))

        length_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        context_groups: dict[tuple[str, str, bool], list[Mapping[str, Any]]] = defaultdict(list)
        for row in predictions:
            length_groups[(row["condition"], int(row["gold_char_length"]))].append(row)
            tokens = int(row["original_stored_context_tokens"])
            bucket = "0-127" if tokens < 128 else "128-255" if tokens < 256 else "256-511" if tokens < 512 else "512+"
            context_groups[(row["condition"], bucket, bool(row["context_truncated"]))].append(row)
        length_rows = [{"condition": key[0], "gold_char_length": key[1], **metric_values(rows)} for key, rows in sorted(length_groups.items())]
        context_rows = [{"condition": key[0], "context_token_bucket": key[1], "truncated": key[2], **metric_values(rows)} for key, rows in sorted(context_groups.items())]
        write_csv(self.output_root / "diagnostics_gold_length.csv", length_rows, list(length_rows[0]))
        write_csv(self.output_root / "diagnostics_context_length.csv", context_rows, list(context_rows[0]))

        failures = [row for row in predictions if row["missing_at_10"]]
        failure_rows = [{key: row[key] for key in ("condition_id", "anchor_id", "author", "work_id", "condition", "context", "pinyin_input", "gold", "gold_rank")} | {"top10": "|".join(item["text"] for item in row["top10_candidates"])} for row in failures[:1000]]
        write_csv(self.output_root / "failure_examples.csv", failure_rows, list(failure_rows[0]) if failure_rows else ["condition_id"])

        review = []
        categories = (
            ("correct_top1", lambda row: row["gold_rank"] == 1),
            ("gold_rank_2_3", lambda row: row["gold_rank"] in (2, 3)),
            ("gold_rank_4_10", lambda row: row["gold_rank"] is not None and 4 <= row["gold_rank"] <= 10),
            ("missing_at_10", lambda row: row["missing_at_10"]),
        )
        for author in AUTHORS:
            author_rows_all = [row for row in predictions if row["author"] == author]
            for category, predicate in categories:
                choice = next((row for row in author_rows_all if predicate(row)), None)
                if choice:
                    review.append({"category": category, **{key: choice[key] for key in ("condition_id", "anchor_id", "author", "work_id", "condition", "context", "pinyin_input", "gold", "gold_rank")}, "top10": "|".join(item["text"] for item in choice["top10_candidates"]), "review_ok": "", "notes": ""})
        write_csv(self.output_root / "t1_prediction_review.csv", review, list(review[0]))
        cache_validation = {
            "schema_version": 2,
            "status": "valid",
            "manifest_rows": len(conditions),
            "prediction_rows": len(predictions),
            "unique_condition_ids": len(actual),
            "manifest_sha256": sha256_file(self.design_root / "t1_condition_manifest.jsonl"),
            "predictions_sha256": sha256_file(self.output_root / "predictions.jsonl"),
            "candidate_surfaces_below_top_k": sum(len(row["top10_candidates"]) < 10 for row in predictions),
            "inference_failures": 0,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "beam_size": 16,
            "top_k": 10,
        }
        write_json(self.output_root / "cache_validation.json", cache_validation)
        artifact_paths = [
            path for path in sorted(self.output_root.iterdir())
            if path.is_file() and path.name != "artifact_checksums.json" and path.suffix in {".json", ".jsonl", ".csv"}
        ]
        write_json(
            self.output_root / "artifact_checksums.json",
            {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifact_paths},
        )
        return metrics
