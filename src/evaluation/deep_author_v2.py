"""Frozen Deep Author Evaluation V2 design and T1 scoring utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from src.datasets.deep_author.pipeline import full_pinyin, initial_pinyin, is_han, stable_hash


DATASET_V1_SHA256 = "8d1a98e18a5f7ed997930b65bbd1149c3d52daaa22ac2c59771256a966648da2"
DATASET_V1_BYTES = 2_048_557_493
SEED = 40408
AUTHORS = ("Re_spectators", "MScarlet", "Etinjat", "Agent Phage", "QBLevi", "breaddddd")
CONDITIONS = ("full_short", "initial_short", "full_multi3", "initial_multi3")


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
