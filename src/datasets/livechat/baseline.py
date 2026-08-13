"""Prepare the frozen LiveChat development IME interaction set.

The official processed release contains no timestamp field and the official
repository does not include the construction/serialization code. Accordingly,
this adapter assigns chronology Grade C and uses the protocol's deterministic,
session-disjoint non-temporal proxy split. It never treats file order as time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import re
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import hanzidentifier
import jieba
from pypinyin import Style, lazy_pinyin, pinyin as heteronym_pinyin
from transformers import BertTokenizer


CHRONOLOGY_GRADE = "C"
CHRONOLOGY_LABEL = "non-temporal proxy split"
SOURCE_SCHEMA = ("streamer_id", "audience_comment", "streamer_response")
OFFICIAL_REVISION = "d06c90aae0cedc1d75792c84e6bc140828c90ded"
DRIVE_FOLDER_ID = "1q2GXfeNRN5bOr2Hc5aDneiBXXVfGN45V"
RAW_DRIVE_FILE_IDS = {
    "RawDialogueData/dev_data.pk": "1_0j1--iPJBu3spWrA_6boWLkiCRFNLeP",
    "RawDialogueData/test_data.pk": "1_5LHN1WItUm4_XpznzIgEtajrtBcwpEC",
    "RawDialogueData/train_data.pk": "1OmmMqjg_Ajmf1PqiD0XTglc8amrPgS6n",
}
_STREAMER_PATTERN = re.compile(r"^streamer\d+$")


@dataclass(frozen=True)
class LiveChatRow:
    streamer_id: str
    audience_comment: str
    streamer_response: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(*parts: Any, seed: int = 40408) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    for part in parts:
        digest.update(b"\x1f")
        digest.update(canonical_json(part).encode("utf-8"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(canonical_json(row) + "\n")
    temporary.replace(path)


def load_livechat_pickle(path: Path) -> list[LiveChatRow]:
    """Load an official pickle and reject undocumented/malformed rows."""

    with path.open("rb") as source:
        value = pickle.load(source)  # noqa: S301 - official pinned dataset artifact
    if not isinstance(value, list):
        raise ValueError(f"{path} container is {type(value).__name__}, expected list")
    rows: list[LiveChatRow] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"{path} row {index} does not have the documented 3 fields")
        if not all(isinstance(field, str) for field in item):
            raise ValueError(f"{path} row {index} contains non-string fields")
        rows.append(LiveChatRow(*item))
    return rows


def build_source_response_id(
    source_file: str,
    row: LiveChatRow,
    duplicate_occurrence: int,
) -> str:
    """Stable source ID based on immutable fields plus duplicate occurrence."""

    return "lcresp-" + stable_hash(
        source_file,
        row.streamer_id,
        row.audience_comment,
        row.streamer_response,
        duplicate_occurrence,
        seed=0,
    )[:24]


def session_partition(response_id: str, *, seed: int = 40408, history_ratio: float = 0.7) -> str:
    fraction = int(stable_hash("session-partition", response_id, seed=seed)[:16], 16) / 2**64
    return "history" if fraction < history_ratio else "evaluation"


def determine_chronology_grade(
    *,
    released_order_metadata: bool,
    official_order_preservation_evidence: bool,
    official_reordering_evidence: bool,
) -> str:
    if released_order_metadata:
        return "A"
    if official_reordering_evidence:
        return "D"
    if official_order_preservation_evidence:
        return "B"
    return "C"


def _is_han(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def _script_label(text: str) -> str:
    labels = {
        hanzidentifier.SIMPLIFIED: "simplified",
        hanzidentifier.TRADITIONAL: "traditional",
        hanzidentifier.BOTH: "shared_or_ambiguous",
        hanzidentifier.MIXED: "mixed",
        hanzidentifier.UNKNOWN: "no_identifiable_chinese",
    }
    return labels[hanzidentifier.identify(text)]


def _percentile(values: Sequence[int | float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _distribution(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p25": _percentile(values, 0.25),
        "median": median(values),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": mean(values),
    }


def segment_response(response: str) -> list[dict[str, Any]]:
    return [
        {"text": word, "start": start, "end": end}
        for word, start, end in jieba.tokenize(response, mode="default")
    ]


_READING_CACHE: dict[str, tuple[str, ...]] = {}


def _readings(character: str) -> tuple[str, ...]:
    if character not in _READING_CACHE:
        alternatives = heteronym_pinyin(
            character,
            style=Style.NORMAL,
            heteronym=True,
            strict=True,
            errors=lambda value: list(value),
        )[0]
        _READING_CACHE[character] = tuple(sorted(set(item.lower() for item in alternatives)))
    return _READING_CACHE[character]


def construct_eligible_targets(
    response: str,
    pinyin2char: Mapping[str, Sequence[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return segmentation, compatible targets, and recorded exclusions."""

    segmentation = segment_response(response)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for span in segmentation:
        target = span["text"]
        if not target or not all(_is_han(character) for character in target):
            continue
        reason = None
        try:
            syllables = [
                value.lower()
                for value in lazy_pinyin(
                    target,
                    style=Style.NORMAL,
                    strict=True,
                    errors=lambda value: list(value),
                )
            ]
        except Exception as error:  # pragma: no cover - defensive audit path
            syllables = []
            reason = f"pinyin_conversion_error:{type(error).__name__}"
        if reason is None and len(syllables) != len(target):
            reason = "pinyin_character_alignment_failure"
        incompatible_positions = []
        if reason is None:
            for position, (character, syllable) in enumerate(zip(target, syllables)):
                if syllable not in pinyin2char or character not in pinyin2char[syllable]:
                    incompatible_positions.append(
                        {"position": position, "character": character, "pinyin": syllable}
                    )
            if incompatible_positions:
                reason = "pinyingpt_compatibility_failure"
        polyphonic = any(len(_readings(character)) > 1 for character in target)
        record = {
            **span,
            "target": target,
            "pinyin": syllables,
            "polyphonic": polyphonic,
            "incompatible_positions": incompatible_positions,
        }
        if reason is None:
            eligible.append(record)
        else:
            record["exclusion_reason"] = reason
            excluded.append(record)
    return segmentation, eligible, excluded


def tokenizer_compatible_character_map(
    tokenizer: Any,
    pinyin2char: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Match the backend's effective map: published mapping plus real tokens."""

    compatible = {}
    for syllable, characters in pinyin2char.items():
        retained = []
        for character in characters:
            token_id = tokenizer.convert_tokens_to_ids(character)
            if (
                token_id != tokenizer.unk_token_id
                and tokenizer.convert_ids_to_tokens(token_id) == character
            ):
                retained.append(character)
        compatible[syllable] = tuple(sorted(set(retained)))
    return compatible


def choose_target(
    targets: Sequence[Mapping[str, Any]],
    *,
    streamer_id: str,
    source_response_id: str,
    seed: int,
) -> dict[str, Any]:
    if not targets:
        raise ValueError("cannot choose from an empty target sequence")
    return dict(
        min(
            targets,
            key=lambda target: stable_hash(
                "target-selection",
                streamer_id,
                source_response_id,
                target["start"],
                target["end"],
                target["target"],
                seed=seed,
            ),
        )
    )


def select_max_interactions(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
    seed: int,
    chronology_grade: str,
) -> list[dict[str, Any]]:
    if len(rows) <= maximum:
        return [dict(row) for row in rows]
    if chronology_grade in {"A", "B"}:
        indices = sorted(
            {round(index * (len(rows) - 1) / (maximum - 1)) for index in range(maximum)}
        )
        return [dict(rows[index]) for index in indices]
    return [
        dict(row)
        for row in sorted(
            rows,
            key=lambda row: stable_hash(
                "evaluation-sample", row["interaction_id"], seed=seed
            ),
        )[:maximum]
    ]


def _audit_file(path: Path, relative: str) -> tuple[dict[str, Any], list[LiveChatRow]]:
    rows = load_livechat_pickle(path)
    row_counter = Counter((row.streamer_id, row.audience_comment, row.streamer_response) for row in rows)
    response_counter = Counter(row.streamer_response for row in rows)
    user_counts = Counter(row.streamer_id for row in rows)
    script_counts = Counter(_script_label(row.streamer_response) for row in rows)
    examples = []
    for index, row in sorted(
        enumerate(rows), key=lambda item: stable_hash(relative, item[0], asdict(item[1]))
    )[:5]:
        examples.append({"row_index": index, **asdict(row)})
    return (
        {
            "path": relative,
            "download_file_id": RAW_DRIVE_FILE_IDS.get(relative),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "container_type": "list",
            "row_count": len(rows),
            "sample_structure": "list[str, str, str]",
            "sample_length": 3,
            "field_names": list(SOURCE_SCHEMA),
            "field_types": ["str", "str", "str"],
            "hidden_timestamp_order_session_fields": False,
            "malformed_rows": 0,
            "empty_streamer_responses": sum(not row.streamer_response.strip() for row in rows),
            "unique_streamers": len(user_counts),
            "streamer_id_all_strings": all(isinstance(row.streamer_id, str) for row in rows),
            "streamer_id_pattern_match_count": sum(bool(_STREAMER_PATTERN.fullmatch(row.streamer_id)) for row in rows),
            "duplicate_rows_beyond_first": sum(count - 1 for count in row_counter.values()),
            "duplicate_streamer_responses_beyond_first": sum(count - 1 for count in response_counter.values()),
            "responses_per_streamer": _distribution(list(user_counts.values())),
            "response_script_distribution": dict(sorted(script_counts.items())),
            "deterministic_examples": examples,
        },
        rows,
    )


def _ambiguity_quartiles(values: Sequence[float]) -> dict[str, float]:
    return {
        "q25": float(_percentile(values, 0.25)),
        "q50": float(_percentile(values, 0.50)),
        "q75": float(_percentile(values, 0.75)),
    }


def _ambiguity_bin(value: float, boundaries: Mapping[str, float]) -> str:
    if value <= boundaries["q25"]:
        return "Q1"
    if value <= boundaries["q50"]:
        return "Q2"
    if value <= boundaries["q75"]:
        return "Q3"
    return "Q4"


def prepare_livechat_baseline(
    config: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Audit official files and create Frozen LiveChat Development Set V1."""

    dataset_root = root / config["dataset"]["root"]
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs/livechat_pinyingpt_generic_baseline_v1.json"
    checkpoint = root / config["model"]["checkpoint_path"]
    pinyin2char = json.loads((checkpoint / "pinyin2char.json").read_text(encoding="utf-8"))
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": json.loads(
                (checkpoint / "additional_special_tokens.json").read_text(encoding="utf-8")
            )
        }
    )
    context_token_limit = int(config["model"]["context_token_limit"])
    compatible_pinyin2char = tokenizer_compatible_character_map(tokenizer, pinyin2char)
    allowed_counts = {key: len(value) for key, value in compatible_pinyin2char.items()}
    seed = int(config["seed"])

    file_audits: dict[str, Any] = {}
    loaded: dict[str, list[LiveChatRow]] = {}
    for relative in config["dataset"]["audit_splits"]:
        audit, rows = _audit_file(dataset_root / relative, relative)
        file_audits[relative] = audit
        loaded[relative] = rows
    train_relative = config["dataset"]["source_split"]
    train_rows = loaded[train_relative]

    usable_counts = Counter(
        row.streamer_id for row in train_rows if row.streamer_response.strip()
    )
    threshold = int(config["selection"]["deep_user_min_usable_responses"])
    qualifying = sorted(
        ((user, count) for user, count in usable_counts.items() if count >= threshold),
        key=lambda item: (-item[1], item[0]),
    )
    selected = qualifying[: int(config["selection"]["max_users"])]
    selected_ids = {user for user, _ in selected}

    depth_csv = output / "user_depth_characterisation.csv"
    with depth_csv.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["streamer_id", "usable_train_responses", "qualifies_2400", "selected"])
        for user, count in sorted(usable_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([user, count, count >= threshold, user in selected_ids])

    chronology = {
        "grade": determine_chronology_grade(
            released_order_metadata=False,
            official_order_preservation_evidence=False,
            official_reordering_evidence=False,
        ),
        "label": CHRONOLOGY_LABEL,
        "released_fields": list(SOURCE_SCHEMA),
        "timestamps_or_order_fields_present": False,
        "official_construction_or_serialization_code_present": False,
        "official_order_preservation_statement_present": False,
        "official_reordering_evidence_present": False,
        "empirical_file_order_note": (
            "Rows are interleaved across streamers and file order may be plausible, but "
            "empirical appearance is not proof of chronological preservation."
        ),
        "consequence": (
            "Use deterministic response/session-level stable-hash 70/30 history/evaluation "
            "partition; do not call it chronological; E7 is unavailable."
        ),
        "official_repository_search_terms": [
            "timestamp", "time", "sort", "order", "sequence", "shuffle",
            "random.shuffle", "train_test_split", "pickle", "RawDialogueData",
        ],
    }

    duplicate_occurrences: Counter[tuple[str, str, str]] = Counter()
    split_by_user: dict[str, dict[str, list[str]]] = {
        user: {"history": [], "evaluation": []} for user in selected_ids
    }
    candidate_interactions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    construction_counts = Counter()
    exclusion_counts = Counter()
    review_ordinary: list[dict[str, Any]] = []
    review_polyphonic: list[dict[str, Any]] = []
    review_failures: list[dict[str, Any]] = []

    jieba.initialize()
    for row_index, row in enumerate(train_rows):
        if row.streamer_id not in selected_ids or not row.streamer_response.strip():
            continue
        key = (row.streamer_id, row.audience_comment, row.streamer_response)
        occurrence = duplicate_occurrences[key]
        duplicate_occurrences[key] += 1
        response_id = build_source_response_id(train_relative, row, occurrence)
        partition = session_partition(
            response_id,
            seed=seed,
            history_ratio=float(config["selection"]["history_ratio"]),
        )
        split_by_user[row.streamer_id][partition].append(response_id)
        if partition != "evaluation":
            continue
        construction_counts["evaluation_responses_considered"] += 1
        segmentation, targets, exclusions = construct_eligible_targets(
            row.streamer_response, compatible_pinyin2char
        )
        construction_counts["segmented_han_targets"] += len(targets) + len(exclusions)
        construction_counts["compatible_targets"] += len(targets)
        construction_counts["excluded_targets"] += len(exclusions)
        for excluded in exclusions:
            exclusion_counts[excluded["exclusion_reason"]] += 1
            review_failures.append(
                {
                    "streamer_id": row.streamer_id,
                    "source_response_id": response_id,
                    "response": row.streamer_response,
                    **excluded,
                }
            )
        if not targets:
            construction_counts["evaluation_responses_without_eligible_target"] += 1
            continue
        construction_counts["evaluation_responses_with_eligible_target"] += 1
        target = choose_target(
            targets,
            streamer_id=row.streamer_id,
            source_response_id=response_id,
            seed=seed,
        )
        pinyin_values = list(target["pinyin"])
        ambiguity = sum(math.log2(allowed_counts[value]) for value in pinyin_values)
        interaction_id = "lcime-" + stable_hash(
            "interaction-v1",
            train_relative,
            response_id,
            target["start"],
            target["end"],
            target["target"],
            seed=0,
        )[:24]
        context = row.streamer_response[: int(target["start"])]
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        effective_context = (
            context
            if len(context_ids) <= context_token_limit
            else tokenizer.decode(
                context_ids[:context_token_limit],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        interaction = {
            "schema_version": 1,
            "benchmark": "FROZEN LIVECHAT DEVELOPMENT EVALUATION SET V1",
            "interaction_id": interaction_id,
            "user_id": row.streamer_id,
            "streamer_id": row.streamer_id,
            "dataset_source_file": train_relative,
            "source_row_index": row_index,
            "source_response_id": response_id,
            "duplicate_occurrence": occurrence,
            "chronology_grade": CHRONOLOGY_GRADE,
            "split_label": CHRONOLOGY_LABEL,
            "split_partition": "evaluation",
            "audience_comment_metadata_only": row.audience_comment,
            "streamer_response": row.streamer_response,
            "segmentation": segmentation,
            "target_span": {"start": target["start"], "end": target["end"]},
            "target_text": target["target"],
            "gold": target["target"],
            "preceding_context": context,
            "effective_context": effective_context,
            "segmented_pinyin": pinyin_values,
            "typed_pinyin": " ".join(pinyin_values),
            "target_length": len(target["target"]),
            "context_length_chinese_characters": sum(_is_han(char) for char in context),
            "ambiguity_score": ambiguity,
            "polyphonic": bool(target["polyphonic"]),
            "pinyin_compatible": True,
        }
        candidate_interactions[row.streamer_id].append(interaction)
        review_target = {
            "streamer_id": row.streamer_id,
            "source_response_id": response_id,
            "interaction_id": interaction_id,
            "response": row.streamer_response,
            "context": context,
            "target": target["target"],
            "pinyin": " ".join(pinyin_values),
            "polyphonic": bool(target["polyphonic"]),
        }
        (review_polyphonic if target["polyphonic"] else review_ordinary).append(review_target)

    selected_interactions: list[dict[str, Any]] = []
    max_per_user = int(config["selection"]["max_evaluation_interactions_per_user"])
    for user in sorted(selected_ids):
        chosen = select_max_interactions(
            candidate_interactions[user],
            maximum=max_per_user,
            seed=seed,
            chronology_grade=CHRONOLOGY_GRADE,
        )
        selected_interactions.extend(chosen)
    selected_interactions.sort(key=lambda row: (row["user_id"], row["source_row_index"], row["interaction_id"]))
    boundaries = _ambiguity_quartiles([row["ambiguity_score"] for row in selected_interactions])
    for row in selected_interactions:
        row["ambiguity_quartile"] = _ambiguity_bin(row["ambiguity_score"], boundaries)

    split_manifest = {
        "schema_version": 1,
        "seed": seed,
        "chronology_grade": CHRONOLOGY_GRADE,
        "split_mode": CHRONOLOGY_LABEL,
        "history_ratio": config["selection"]["history_ratio"],
        "session_disjoint": True,
        "users": {
            user: {
                "history_response_ids": partitions["history"],
                "evaluation_response_ids": partitions["evaluation"],
                "history_count": len(partitions["history"]),
                "evaluation_count": len(partitions["evaluation"]),
            }
            for user, partitions in sorted(split_by_user.items())
        },
    }
    selected_users = {
        "threshold": threshold,
        "qualifying_user_count": len(qualifying),
        "selection_rule": "highest usable train response count; streamer_id lexical tie break",
        "selected_user_count": len(selected),
        "selected_users": [
            {"streamer_id": user, "usable_train_responses": count}
            for user, count in selected
        ],
    }
    dataset_audit = {
        "official_documented_schema": list(SOURCE_SCHEMA),
        "actual_schema_matches_documentation": True,
        "files": file_audits,
        "cross_split_streamer_overlap": {
            "train_dev": len({r.streamer_id for r in loaded[train_relative]} & {r.streamer_id for r in loaded["RawDialogueData/dev_data.pk"]}),
            "train_test": len({r.streamer_id for r in loaded[train_relative]} & {r.streamer_id for r in loaded["RawDialogueData/test_data.pk"]}),
            "dev_test": len({r.streamer_id for r in loaded["RawDialogueData/dev_data.pk"]} & {r.streamer_id for r in loaded["RawDialogueData/test_data.pk"]}),
        },
        "license_note": (
            "The repository root contains an MIT license for software and associated "
            "documentation. Dataset usage terms are not explicitly established by the "
            "repository license alone."
        ),
    }
    construction_summary = {
        **dict(construction_counts),
        "exclusion_reasons": dict(exclusion_counts),
        "frozen_interaction_count": len(selected_interactions),
        "interactions_per_user": _distribution(
            [sum(row["user_id"] == user for row in selected_interactions) for user in sorted(selected_ids)]
        ),
        "ambiguity_quartile_boundaries": boundaries,
        "one_target_per_evaluation_response": True,
        "audience_comment_used_as_model_context": False,
    }
    quality = {
        "conversion_library": "pypinyin",
        "conversion_version": importlib.metadata.version("pypinyin"),
        "segmentation_library": "jieba",
        "segmentation_version": importlib.metadata.version("jieba"),
        "frozen_target_count": len(selected_interactions),
        "polyphonic_target_count": sum(row["polyphonic"] for row in selected_interactions),
        "polyphonic_target_rate": (
            sum(row["polyphonic"] for row in selected_interactions) / len(selected_interactions)
            if selected_interactions else None
        ),
        "target_length_distribution": _distribution([row["target_length"] for row in selected_interactions]),
        "candidate_target_exclusion_reasons": dict(exclusion_counts),
        "candidate_target_exclusion_rate": (
            construction_counts["excluded_targets"] / construction_counts["segmented_han_targets"]
            if construction_counts["segmented_han_targets"] else None
        ),
        "traditional_to_simplified_conversion": False,
        "published_map_character_assignments": sum(len(set(value)) for value in pinyin2char.values()),
        "tokenizer_compatible_character_assignments": sum(len(value) for value in compatible_pinyin2char.values()),
        "compatibility_definition": "character occurs in pinyin2char.json and is an exact non-UNK checkpoint tokenizer token",
    }
    provenance = {
        "official_repository_url": config["dataset"]["official_repository"],
        "official_repository_revision": OFFICIAL_REVISION,
        "download_source": config["dataset"]["download_folder_url"],
        "google_drive_folder_id": DRIVE_FOLDER_ID,
        "downloaded_files": file_audits,
        "repository_license_file": "LICENSE (MIT text)",
        "dataset_license_conclusion": (
            "dataset usage terms are not explicitly established by the repository license alone"
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "development_only": True,
        "personalisation_implemented": False,
        "official_dev_test_scored": False,
        "chronology_grade": CHRONOLOGY_GRADE,
        "split_mode": CHRONOLOGY_LABEL,
        "frozen_interaction_count": len(selected_interactions),
        "selected_user_count": len(selected),
        "model_checkpoint": config["model"]["checkpoint"],
        "model_revision": config["model"]["checkpoint_revision"],
        "config_sha256": sha256_file(config_path),
    }

    write_json(output / "manifest.json", manifest)
    write_json(output / "provenance.json", provenance)
    write_json(output / "dataset_audit.json", dataset_audit)
    write_json(output / "chronology_audit.json", chronology)
    write_json(output / "selected_users.json", selected_users)
    write_json(output / "split_manifest.json", split_manifest)
    write_json(output / "interaction_construction_summary.json", construction_summary)
    write_jsonl(output / "frozen_interactions.jsonl", selected_interactions)
    write_json(output / "pinyin_quality_audit.json", quality)

    review_groups = (
        ("ordinary", review_ordinary),
        ("polyphonic", review_polyphonic),
        ("compatibility_failure", review_failures),
    )
    review_path = output / "pinyin_manual_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as destination:
        fields = ["sample_type", "streamer_id", "source_response_id", "interaction_id", "response", "context", "target", "pinyin", "polyphonic", "exclusion_reason"]
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for label, candidates in review_groups:
            for row in sorted(candidates, key=lambda item: stable_hash("manual-review", item, seed=seed))[:25]:
                writer.writerow({"sample_type": label, **row})

    return {
        "manifest": manifest,
        "dataset_audit": dataset_audit,
        "chronology_audit": chronology,
        "selected_users": selected_users,
        "interaction_construction_summary": construction_summary,
        "pinyin_quality_audit": quality,
        "output_dir": str(output),
    }
