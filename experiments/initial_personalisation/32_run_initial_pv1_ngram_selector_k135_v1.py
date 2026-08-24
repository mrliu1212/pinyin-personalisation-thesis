"""PV1 reproduction with frozen Interpolated-NGram Personal-K5 selector K ablations.

Research question
-----------------
Keep the historical Initial-Pinyin PV1 recovery mechanism fixed, but replace
PV1's frequency Top1 personal selector with the frozen Interpolated-NGram
ordering and admit the first K selected personal candidates for K in {1,3,5}.

This is a selector/admission-width ablation, not a new recovery-strength model.

Frozen semantics
----------------
- Frozen Personal-only K5 candidate surface.
- Frozen Generic Frequency-F ranking (lambda_F=4).
- Frozen PV1 personal lambda = 4.
- Personal support for every admitted candidate is the original PV1 support:

      raw(c) = log(1 + count(c))
      support(c) = raw(c) / max_j raw(j)      over the frozen Personal K5

- Every admitted personal candidate receives:

      final_score(c) = generic_boundary + 4 * support(c)

- Generic candidates keep their frozen Frequency-F final scores.
- Generic wins exact score ties.
- Personal-personal ties are broken by NGram selector rank, then lexical text.
- Final list is truncated to Top10.

Ablations
---------
PV1 baseline:
    frequency-ordered Personal K5 -> take frequency Top1 -> PV1 merge

PV1+NGramSelector@K1:
    NGram-rank Personal K5 -> admit NGram Top1 -> same PV1 merge

PV1+NGramSelector@K3:
    NGram-rank Personal K5 -> admit NGram Top3 -> same PV1 merge

PV1+NGramSelector@K5:
    NGram-rank Personal K5 -> admit all available K5 -> same PV1 merge

Important K5 interpretation
---------------------------
Because the frozen candidate pool itself contains at most five personal
candidates, K=5 admits the whole Personal K5. NGram therefore no longer filters
candidate identity at K=5; it only supplies deterministic personal tie order
when PV1 final scores are exactly equal.

Protocol
--------
- Standardized Train-Val only (34,416 rows).
- Gold is used only for evaluation/diagnostics, never selection or scoring.
- Dev3000 is not read. Test is not read.
- No PinyinGPT/BGE/NGram inference is performed; frozen adaptive scores are reused.
- The script first requires an exact PV1 Top10/rank/score reproduction audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"
EXPECTED_VAL_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_K5_RECOVERABLE_MISSING = 4_910

FROZEN_LAMBDA_FREQUENCY = 4.0
FROZEN_K_PV = 1
FROZEN_LAMBDA_PV = 4.0
DEFAULT_SELECTOR_KS = (1, 3, 5)


# ---------------------------------------------------------------------------
# IO / utility
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as sink:
        for row in rows:
            sink.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def index_rows(rows: Iterable[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        row_id = str(row["row_id"])
        if row_id in output:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        output[row_id] = row
    return output


def safe_rate(n: float | int, d: int) -> float:
    return float(n) / d if d else 0.0


def candidate_text(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate", row.get("text", "")))


def rank_of(rows: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for rank, row in enumerate(rows, start=1):
        if candidate_text(row) == gold:
            return rank
    return None


def parse_selector_ks(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--selector-ks must contain at least one K")
    try:
        values = tuple(sorted(set(int(part) for part in parts)))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--selector-ks must be comma-separated integers") from exc
    if any(k < 1 or k > 5 for k in values):
        raise argparse.ArgumentTypeError("Every selector K must be in 1..5")
    return values


# ---------------------------------------------------------------------------
# Frozen Personal K5 / supports
# ---------------------------------------------------------------------------


def extract_personal_records(surface: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = surface.get("personal_candidates_top5", [])
    if not isinstance(raw, list):
        raise RuntimeError(f"Invalid personal_candidates_top5 at {surface.get('row_id')}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, value in enumerate(raw, start=1):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Invalid Personal-K5 record at {surface.get('row_id')}: {value!r}")
        text = str(value.get("text", ""))
        if not text:
            raise RuntimeError(f"Empty Personal-K5 target at {surface.get('row_id')}")
        if text in seen:
            raise RuntimeError(f"Duplicate Personal-K5 target at {surface.get('row_id')}: {text}")
        seen.add(text)

        count = int(value.get("frequency_count", 0))
        if count <= 0:
            raise RuntimeError(f"Non-positive frequency_count at {surface.get('row_id')}: {text}")

        recorded_rank = int(value.get("personal_rank", position))
        if recorded_rank != position:
            raise RuntimeError(
                f"Frozen Personal-K5 rank mismatch at {surface.get('row_id')}: "
                f"target={text} expected={position} got={recorded_rank}"
            )
        records.append(
            {
                "text": text,
                "frequency_count": count,
                "frequency_rank": position,
            }
        )

    texts = [record["text"] for record in records]
    frozen_texts = [str(value) for value in surface.get("personal_candidate_texts_top5", [])]
    if texts != frozen_texts:
        raise RuntimeError(
            f"Personal-K5 record/text fields differ at {surface.get('row_id')}: "
            f"records={texts} texts={frozen_texts}"
        )
    return tuple(records)


def pv1_frequency_support(personal_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Exact context_memory.frequency_support semantics on frozen Personal K5 counts."""
    raw = {
        str(record["text"]): math.log1p(int(record["frequency_count"]))
        for record in personal_records
    }
    maximum = max(raw.values(), default=0.0)
    return {text: value / maximum for text, value in raw.items()} if maximum else {}


def ngram_rank_order(candidates: Sequence[str], support: Sequence[float]) -> list[int]:
    """Exact adaptive-runner tie policy: descending support, original K5 index, text."""
    if len(candidates) != len(support):
        raise RuntimeError("NGram candidate/support length mismatch")
    return sorted(
        range(len(candidates)),
        key=lambda i: (-float(support[i]), i, str(candidates[i])),
    )


# ---------------------------------------------------------------------------
# Exact PV1-style boundary merge with variable admitted Personal K
# ---------------------------------------------------------------------------


def pv1_style_merge(
    *,
    frequency_candidates: Sequence[Mapping[str, Any]],
    selected_personal: Sequence[Mapping[str, Any]],
    selector_label: str,
) -> list[dict[str, Any]]:
    """PV1 boundary merge; supports/lambda are unchanged, admission width may vary."""

    rows: list[dict[str, Any]] = []
    generic_texts: set[str] = set()
    normalized_generic: list[float] = []

    for position, frozen in enumerate(frequency_candidates, start=1):
        target = candidate_text(frozen)
        if not target:
            raise RuntimeError("Cannot identify frozen Frequency candidate")
        if target in generic_texts:
            raise RuntimeError(f"Duplicate frozen Generic candidate: {target!r}")
        generic_texts.add(target)

        normalized = float(frozen["normalized_generic_score"])
        normalized_generic.append(normalized)

        row = dict(frozen)
        row["candidate"] = target
        row["source"] = "generic"
        row["frequency_support"] = float(frozen.get("personal_score", 0.0))
        row["context_support"] = 0.0
        row["final_score"] = float(frozen["final_score"])
        row["generic_rank"] = int(frozen.get("generic_rank") or position)
        rows.append(row)

    if not rows:
        raise RuntimeError("Frozen Frequency candidate list is empty")

    boundary = min(normalized_generic)
    seen_personal: set[str] = set()

    for admitted_rank, item in enumerate(selected_personal, start=1):
        target = str(item["target"])
        if target in generic_texts:
            raise RuntimeError(f"Selected Personal candidate overlaps Generic: {target!r}")
        if target in seen_personal:
            raise RuntimeError(f"Duplicate selected Personal candidate: {target!r}")
        seen_personal.add(target)

        support = float(item["frequency_support"])
        count = int(item["frequency_count"])
        original_frequency_rank = int(item["original_frequency_rank"])
        ngram_rank = int(item.get("ngram_rank", admitted_rank))

        rows.append(
            {
                "candidate": target,
                "generic_rank": None,
                "generic_score": None,
                "normalized_generic_score": boundary,
                "personal_score": support,
                "frequency_support": support,
                "context_support": 0.0,
                "frequency_count": count,
                "final_score": boundary + FROZEN_LAMBDA_PV * support,
                "source": "personal_vocabulary",
                # This is the admitted selector order. For K1 it exactly equals 1.
                "personal_candidate_rank": admitted_rank,
                "selector": selector_label,
                "ngram_rank": ngram_rank,
                "original_personal_frequency_rank": original_frequency_rank,
            }
        )

    # Same source priority as PV1. For Personal-personal ties, selector/admission rank
    # is deterministic. Generic still wins any exact Generic-Personal score tie.
    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row["source"] == "generic" else 1,
            int(row.get("generic_rank") or row.get("personal_candidate_rank", 0)),
            str(row["candidate"]),
        )
    )

    values = rows[:10]
    for rank, row in enumerate(values, start=1):
        row["rank"] = rank
    return values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_summary(rows: Sequence[Mapping[str, Any]], rank_key: str, method: str) -> dict[str, Any]:
    author_total: Counter[str] = Counter()
    author_top1: Counter[str] = Counter()
    top1 = top3 = top5 = missing = 0
    reciprocal = 0.0
    observed_ranks: list[int] = []

    for row in rows:
        author = str(row["author"])
        author_total[author] += 1
        rank = row.get(rank_key)
        if rank is None:
            missing += 1
            continue
        rank_i = int(rank)
        observed_ranks.append(rank_i)
        reciprocal += 1.0 / rank_i
        if rank_i == 1:
            top1 += 1
            author_top1[author] += 1
        if rank_i <= 3:
            top3 += 1
        if rank_i <= 5:
            top5 += 1

    n = len(rows)
    per_author = {
        author: safe_rate(author_top1[author], count)
        for author, count in sorted(author_total.items())
    }
    return {
        "method": method,
        "n": n,
        "authors": len(per_author),
        "macro_author_top1": statistics.fmean(per_author.values()) if per_author else 0.0,
        "micro_top1": safe_rate(top1, n),
        "top3": safe_rate(top3, n),
        "top5": safe_rate(top5, n),
        "mrr_at_10": safe_rate(reciprocal, n),
        "missing10": safe_rate(missing, n),
        "mean_rank_given_top10": statistics.fmean(observed_ranks) if observed_ranks else None,
        "per_author_top1": per_author,
    }


def transition_counts(
    rows: Sequence[Mapping[str, Any]],
    base_key: str,
    new_key: str,
) -> dict[str, int]:
    counts = {
        "n": 0,
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for row in rows:
        counts["n"] += 1
        before = row.get(base_key) == 1
        after = row.get(new_key) == 1
        if not before and after:
            counts["rescue"] += 1
        elif before and not after:
            counts["harm"] += 1
        elif before and after:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    counts["net"] = counts["rescue"] - counts["harm"]
    return counts


def recovery_summary(rows: Sequence[Mapping[str, Any]], rank_key: str) -> dict[str, Any]:
    generic_missing = [row for row in rows if bool(row["generic_missing"])]
    recoverable = [row for row in generic_missing if bool(row["gold_in_personal_k5"])]
    ranks = [None if row.get(rank_key) is None else int(row[rank_key]) for row in recoverable]

    def count_at(limit: int) -> int:
        return sum(rank is not None and rank <= limit for rank in ranks)

    recovered_ranks = [rank for rank in ranks if rank is not None and rank <= 10]
    reciprocal = sum(0.0 if rank is None else 1.0 / rank for rank in ranks)
    denom = len(recoverable)

    return {
        "generic_missing_n": len(generic_missing),
        "recoverable_n": denom,
        "recovered_at_1_n": count_at(1),
        "recovered_at_3_n": count_at(3),
        "recovered_at_5_n": count_at(5),
        "recovered_at_10_n": count_at(10),
        "recovered_at_1": safe_rate(count_at(1), denom),
        "recovered_at_3": safe_rate(count_at(3), denom),
        "recovered_at_5": safe_rate(count_at(5), denom),
        "recovered_at_10": safe_rate(count_at(10), denom),
        "recovery_mrr_at_10": safe_rate(reciprocal, denom),
        "mean_recovered_rank": statistics.fmean(recovered_ranks) if recovered_ranks else None,
    }


def delta_metrics(method: Mapping[str, Any], pv1: Mapping[str, Any]) -> dict[str, float]:
    keys = ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing10")
    return {key: float(method[key]) - float(pv1[key]) for key in keys}


# ---------------------------------------------------------------------------
# Frozen input verification
# ---------------------------------------------------------------------------


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    frozen = {
        "val": (args.val, EXPECTED_VAL_SHA256),
        "candidate_surface": (args.candidate_surface, EXPECTED_SURFACE_SHA256),
        "frequency_pv1_predictions": (
            args.frequency_pv1_predictions,
            EXPECTED_FREQUENCY_PV1_SHA256,
        ),
    }
    hashes: dict[str, str] = {}
    for label, (path, expected) in frozen.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        hashes[label] = actual
        if actual != expected:
            raise RuntimeError(
                f"SHA mismatch for {label}: expected={expected} actual={actual} path={path}"
            )

    scores = args.adaptive_dir / "scores.jsonl"
    comparison_path = args.adaptive_dir / "comparison.json"
    if not scores.is_file():
        raise FileNotFoundError(scores)
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)

    comparison = read_json(comparison_path)
    adaptive_provenance = comparison.get("provenance", {})
    if adaptive_provenance.get("val_sha256") != EXPECTED_VAL_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Train-Val artifact")
    if adaptive_provenance.get("frozen_candidate_surface_sha256") != EXPECTED_SURFACE_SHA256:
        raise RuntimeError("Adaptive NGram comparison uses another Personal-K5 surface")
    if adaptive_provenance.get("candidate_selection_used_gold") is not False:
        raise RuntimeError("Adaptive NGram candidate selection is not Gold-free")
    if adaptive_provenance.get("gold_used_for_scoring") is not False:
        raise RuntimeError("Adaptive NGram scoring is not Gold-free")
    if (
        adaptive_provenance.get("dev3000_used") is not False
        or adaptive_provenance.get("test_used") is not False
    ):
        raise RuntimeError("Adaptive NGram artifact crossed Dev3000/Test boundary")

    selected_by_k = comparison.get("selected_by_k", {})
    selected_key = str(selected_by_k.get("K5", {}).get("best_interpolated", ""))
    if not selected_key.startswith("K5|Interpolated|"):
        raise RuntimeError(f"Unexpected selected K5 Interpolated key: {selected_key!r}")

    actual_scores_sha = sha256_file(scores)
    recorded_scores_sha = adaptive_provenance.get("scores_sha256")
    if recorded_scores_sha and recorded_scores_sha != actual_scores_sha:
        raise RuntimeError("Adaptive scores.jsonl SHA differs from comparison provenance")

    return {
        "paths": {
            "val": str(args.val),
            "candidate_surface": str(args.candidate_surface),
            "frequency_pv1_predictions": str(args.frequency_pv1_predictions),
            "adaptive_scores": str(scores),
            "adaptive_comparison": str(comparison_path),
        },
        "sha256": {
            **hashes,
            "adaptive_scores": actual_scores_sha,
            "adaptive_comparison": sha256_file(comparison_path),
        },
        "selected_k5_interpolated_key": selected_key,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exact Initial PV1 reproduction with frozen K5 Interpolated-NGram "
            "selector admission-width ablations K=1,3,5"
        )
    )
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument(
        "--adaptive-dir",
        type=Path,
        required=True,
        help="adaptive_ngram_top10_v1 directory containing scores.jsonl and comparison.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--selector-ks",
        type=parse_selector_ks,
        default=DEFAULT_SELECTOR_KS,
        help="Comma-separated admitted NGram selector widths; default: 1,3,5",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    selector_ks: tuple[int, ...] = args.selector_ks
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = verify_inputs(args)
    selected_ngram_key = str(provenance["selected_k5_interpolated_key"])

    print("=== PV1 NGRAM-SELECTOR K1/K3/K5 ABLATION ===")
    print("Frozen PV1 baseline: Kpv=1, lambda_F=4, lambda_PV=4")
    print(f"Frozen K5 Interpolated method: {selected_ngram_key}")
    print(f"Reported selector widths: {selector_ks}")
    print("Each admitted candidate keeps exact PV1 frequency support; NGram score is not recovery strength")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    val_rows = read_jsonl(args.val)
    surface_by_id = index_rows(read_jsonl(args.candidate_surface), "candidate surface")
    pred_by_id = index_rows(
        read_jsonl(args.frequency_pv1_predictions),
        "Frequency/PV1 predictions",
    )
    ngram_by_id = index_rows(
        read_jsonl(args.adaptive_dir / "scores.jsonl"),
        "adaptive NGram scores",
    )
    val_by_id = index_rows(val_rows, "Train-Val")

    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val rows: {len(val_rows)}")
    if not (set(val_by_id) == set(surface_by_id) == set(pred_by_id) == set(ngram_by_id)):
        raise RuntimeError("Train-Val / surface / PV1 / NGram row IDs differ")

    outputs: list[dict[str, Any]] = []
    reproduction_mismatch_examples: list[dict[str, Any]] = []
    reproduced_rank_mismatch_n = 0
    reproduced_top10_mismatch_n = 0

    diagnostics: dict[int, dict[str, Any]] = {
        k: {
            "rows_with_personal_candidates": 0,
            "rows_admitting_fewer_than_requested_k": 0,
            "admitted_candidate_total": 0,
            "gold_in_admitted_n": 0,
            "gold_in_personal_k5_n": 0,
        }
        for k in selector_ks
    }

    # Preserve the existing K1 selector diagnostic for direct cross-check with the
    # previous PV1+NGramSelector script.
    k1_candidate_only = {
        "eligible_gold_in_k5_and_k_ge_2": 0,
        "frequency_top1_gold": 0,
        "ngram_top1_gold": 0,
        "ngram_rescue_vs_frequency": 0,
        "ngram_harm_vs_frequency": 0,
    }
    k1_selector_changed_n = 0
    k1_selector_rows_with_candidates = 0

    started = time.perf_counter()
    for number, row_id in enumerate(sorted(val_by_id), start=1):
        row = val_by_id[row_id]
        surface = surface_by_id[row_id]
        frozen = pred_by_id[row_id]
        ngram_row = ngram_by_id[row_id]

        gold = str(row.get("gold", row.get("target")))
        author = str(row["author"])

        if int(frozen.get("selected_k_pv", -1)) != FROZEN_K_PV:
            raise RuntimeError(
                f"Unexpected frozen selected_k_pv at {row_id}: {frozen.get('selected_k_pv')}"
            )
        if not math.isclose(
            float(frozen.get("selected_lambda_frequency", -1)),
            FROZEN_LAMBDA_FREQUENCY,
        ):
            raise RuntimeError(
                f"Unexpected frozen selected_lambda_frequency at {row_id}: "
                f"{frozen.get('selected_lambda_frequency')}"
            )
        if not math.isclose(
            float(frozen.get("selected_lambda_pv", -1)),
            FROZEN_LAMBDA_PV,
        ):
            raise RuntimeError(
                f"Unexpected frozen selected_lambda_pv at {row_id}: "
                f"{frozen.get('selected_lambda_pv')}"
            )

        frequency_candidates = frozen.get("frequency_candidates")
        frozen_pv1_candidates = frozen.get("pv1_candidates")
        if not isinstance(frequency_candidates, list) or not frequency_candidates:
            raise RuntimeError(f"Missing frozen frequency_candidates at {row_id}")
        if not isinstance(frozen_pv1_candidates, list) or not frozen_pv1_candidates:
            raise RuntimeError(f"Missing frozen pv1_candidates at {row_id}")

        personal_records = extract_personal_records(surface)
        personal_k5 = tuple(str(record["text"]) for record in personal_records)
        personal_k5_set = set(personal_k5)
        frequency_support = pv1_frequency_support(personal_records)

        methods = ngram_row.get("methods")
        if not isinstance(methods, Mapping) or selected_ngram_key not in methods:
            raise RuntimeError(f"Missing selected NGram method at {row_id}: {selected_ngram_key}")
        method = methods[selected_ngram_key]
        ngram_candidates = tuple(str(value) for value in method.get("candidates", []))
        ngram_support = tuple(float(value) for value in method.get("support", []))
        if ngram_candidates != personal_k5:
            raise RuntimeError(
                f"Frozen Personal K5 != NGram K5 at {row_id}: "
                f"surface={personal_k5} ngram={ngram_candidates}"
            )
        if len(ngram_support) != len(personal_k5):
            raise RuntimeError(f"NGram support length mismatch at {row_id}")
        if ngram_support and not math.isclose(
            sum(ngram_support),
            1.0,
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise RuntimeError(
                f"NGram support does not sum to 1 at {row_id}: {sum(ngram_support)}"
            )
        if ngram_row.get("gold_used_for_scoring") is not False:
            raise RuntimeError(f"NGram row has invalid Gold provenance at {row_id}")

        # Exact frozen PV1 reproduction: frequency Top1 only.
        original_target = personal_k5[0] if personal_k5 else None
        original_selected: list[dict[str, Any]] = []
        if original_target is not None:
            original_selected.append(
                {
                    "target": original_target,
                    "frequency_support": float(frequency_support[original_target]),
                    "frequency_count": int(personal_records[0]["frequency_count"]),
                    "original_frequency_rank": 1,
                    "ngram_rank": 1,
                }
            )

        reproduced = pv1_style_merge(
            frequency_candidates=frequency_candidates,
            selected_personal=original_selected,
            selector_label="frequency_top1",
        )

        frozen_texts = [candidate_text(value) for value in frozen_pv1_candidates]
        reproduced_texts = [candidate_text(value) for value in reproduced]
        frozen_rank = frozen.get("pv1_rank")
        reproduced_rank = rank_of(reproduced, gold)

        if reproduced_texts != frozen_texts:
            reproduced_top10_mismatch_n += 1
        if reproduced_rank != frozen_rank:
            reproduced_rank_mismatch_n += 1

        score_mismatches: list[dict[str, Any]] = []
        frozen_by_target = {candidate_text(value): value for value in frozen_pv1_candidates}
        for value in reproduced:
            target = candidate_text(value)
            frozen_value = frozen_by_target.get(target)
            if frozen_value is None:
                score_mismatches.append({"candidate": target, "reason": "missing_from_frozen"})
                continue
            if not math.isclose(
                float(value["final_score"]),
                float(frozen_value["final_score"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                score_mismatches.append(
                    {
                        "candidate": target,
                        "reproduced": float(value["final_score"]),
                        "frozen": float(frozen_value["final_score"]),
                    }
                )

        if (
            reproduced_texts != frozen_texts
            or reproduced_rank != frozen_rank
            or score_mismatches
        ) and len(reproduction_mismatch_examples) < 20:
            reproduction_mismatch_examples.append(
                {
                    "row_id": row_id,
                    "gold": gold,
                    "personal_k5": list(personal_k5),
                    "reproduced_texts": reproduced_texts,
                    "frozen_texts": frozen_texts,
                    "reproduced_rank": reproduced_rank,
                    "frozen_rank": frozen_rank,
                    "score_mismatches": score_mismatches[:5],
                }
            )

        ngram_order = ngram_rank_order(personal_k5, ngram_support) if personal_k5 else []
        ngram_ordered_targets = [personal_k5[i] for i in ngram_order]

        row_output: dict[str, Any] = {
            "schema_version": 1,
            "row_id": row_id,
            "author": author,
            "gold": gold,
            "generic_missing": bool(
                frozen.get("generic_missing", surface.get("generic_missing", False))
            ),
            "gold_in_personal_k5": gold in personal_k5_set,
            "generic_rank": frozen.get("generic_rank"),
            "frequency_rank": frozen.get("frequency_rank"),
            "pv1_rank": frozen.get("pv1_rank"),
            "pv1_reproduced_rank": reproduced_rank,
            "personal_k5": list(personal_k5),
            "personal_frequency_support": frequency_support,
            "frequency_selector_target": original_target,
            "ngram_support": list(ngram_support),
            "ngram_ordered_personal_k5": ngram_ordered_targets,
            "pv1_reproduced_candidates": reproduced,
            "gold_used_for_scoring": False,
            "dev3000_used": False,
            "test_used": False,
        }

        for k in selector_ks:
            chosen_indices = ngram_order[: min(k, len(ngram_order))]
            selected_personal: list[dict[str, Any]] = []
            for ngram_rank, index in enumerate(chosen_indices, start=1):
                target = personal_k5[index]
                selected_personal.append(
                    {
                        "target": target,
                        "frequency_support": float(frequency_support[target]),
                        "frequency_count": int(personal_records[index]["frequency_count"]),
                        "original_frequency_rank": index + 1,
                        "ngram_rank": ngram_rank,
                    }
                )

            ranking = pv1_style_merge(
                frequency_candidates=frequency_candidates,
                selected_personal=selected_personal,
                selector_label=f"interpolated_ngram_top{k}",
            )
            rank = rank_of(ranking, gold)

            rank_key = f"ngram_selector_k{k}_rank"
            candidates_key = f"ngram_selector_k{k}_candidates"
            selected_key = f"ngram_selector_k{k}_selected_personal"
            row_output[rank_key] = rank
            row_output[candidates_key] = ranking
            row_output[selected_key] = selected_personal

            diag = diagnostics[k]
            if personal_k5:
                diag["rows_with_personal_candidates"] += 1
            if len(personal_k5) < k:
                diag["rows_admitting_fewer_than_requested_k"] += 1
            diag["admitted_candidate_total"] += len(selected_personal)
            if gold in personal_k5_set:
                diag["gold_in_personal_k5_n"] += 1
                admitted_targets = {str(item["target"]) for item in selected_personal}
                diag["gold_in_admitted_n"] += int(gold in admitted_targets)

        # K1 cross-check diagnostic against the earlier selector experiment.
        if 1 in selector_ks and personal_k5:
            k1_selector_rows_with_candidates += 1
            ngram_top1 = ngram_ordered_targets[0]
            k1_selector_changed_n += int(ngram_top1 != original_target)
            if len(personal_k5) >= 2 and gold in personal_k5_set:
                k1_candidate_only["eligible_gold_in_k5_and_k_ge_2"] += 1
                freq_correct = original_target == gold
                ngram_correct = ngram_top1 == gold
                k1_candidate_only["frequency_top1_gold"] += int(freq_correct)
                k1_candidate_only["ngram_top1_gold"] += int(ngram_correct)
                k1_candidate_only["ngram_rescue_vs_frequency"] += int(
                    (not freq_correct) and ngram_correct
                )
                k1_candidate_only["ngram_harm_vs_frequency"] += int(
                    freq_correct and (not ngram_correct)
                )

        outputs.append(row_output)

        if args.progress_every > 0 and (
            number % args.progress_every == 0 or number == len(val_by_id)
        ):
            elapsed = time.perf_counter() - started
            print(f"ROWS {number}/{len(val_by_id)} elapsed={elapsed:.1f}s", flush=True)

    # Exact reproduction is a precondition for interpreting any K ablation.
    reproduction_ok = (
        reproduced_rank_mismatch_n == 0
        and reproduced_top10_mismatch_n == 0
        and not reproduction_mismatch_examples
    )
    reproduction_audit = {
        "status": "passed" if reproduction_ok else "failed",
        "rows": len(outputs),
        "pv1_rank_mismatch_n": reproduced_rank_mismatch_n,
        "pv1_top10_candidate_order_mismatch_n": reproduced_top10_mismatch_n,
        "examples": reproduction_mismatch_examples,
        "frozen_k_pv": FROZEN_K_PV,
        "frozen_lambda_frequency": FROZEN_LAMBDA_FREQUENCY,
        "frozen_lambda_pv": FROZEN_LAMBDA_PV,
    }
    audit_path = args.output_root / "pv1_reproduction_audit.json"
    write_json(audit_path, reproduction_audit)
    if not reproduction_ok:
        raise RuntimeError(
            "PV1 reproduction failed; K ablations are not scientifically interpretable. "
            f"See {audit_path}"
        )

    generic_missing_n = sum(bool(row["generic_missing"]) for row in outputs)
    recoverable_n = sum(
        bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"])
        for row in outputs
    )
    if generic_missing_n != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Unexpected Generic-Missing count: {generic_missing_n}")
    if recoverable_n != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError(f"Unexpected K5-recoverable Generic-Missing count: {recoverable_n}")

    pv1_metrics = metric_summary(outputs, "pv1_rank", "PV1")
    reproduced_metrics = metric_summary(outputs, "pv1_reproduced_rank", "PV1-Reproduced")

    # Metric-level PV1 reproduction cross-check.
    for key in (
        "n",
        "authors",
        "macro_author_top1",
        "micro_top1",
        "top3",
        "top5",
        "mrr_at_10",
        "missing10",
        "mean_rank_given_top10",
        "per_author_top1",
    ):
        if pv1_metrics[key] != reproduced_metrics[key]:
            raise RuntimeError(f"PV1 reproduced metric mismatch for {key}")

    pv1_recovery = recovery_summary(outputs, "pv1_rank")
    if pv1_recovery["generic_missing_n"] != EXPECTED_GENERIC_MISSING:
        raise RuntimeError("PV1 recovery denominator drift")
    if pv1_recovery["recoverable_n"] != EXPECTED_K5_RECOVERABLE_MISSING:
        raise RuntimeError("PV1 recoverable denominator drift")

    method_reports: dict[str, Any] = {}
    for k in selector_ks:
        label = f"PV1+NGramSelector@K{k}"
        rank_key = f"ngram_selector_k{k}_rank"
        metrics = metric_summary(outputs, rank_key, label)
        transition = transition_counts(outputs, "pv1_rank", rank_key)
        recovery = recovery_summary(outputs, rank_key)

        diag = diagnostics[k]
        diag["mean_admitted_candidates_per_row_with_personal"] = safe_rate(
            diag["admitted_candidate_total"],
            diag["rows_with_personal_candidates"],
        )
        diag["gold_admission_rate_given_gold_in_personal_k5"] = safe_rate(
            diag["gold_in_admitted_n"],
            diag["gold_in_personal_k5_n"],
        )

        method_reports[label] = {
            "selector_k": k,
            "rank_key": rank_key,
            "metrics": metrics,
            "delta_vs_pv1": delta_metrics(metrics, pv1_metrics),
            "pv1_to_method": transition,
            "recovery": recovery,
            "admission_diagnostics": diag,
        }

    if 1 in selector_ks:
        k1_candidate_only["net"] = (
            k1_candidate_only["ngram_rescue_vs_frequency"]
            - k1_candidate_only["ngram_harm_vs_frequency"]
        )
        denom = k1_candidate_only["eligible_gold_in_k5_and_k_ge_2"]
        k1_candidate_only["frequency_top1_accuracy"] = safe_rate(
            k1_candidate_only["frequency_top1_gold"],
            denom,
        )
        k1_candidate_only["ngram_top1_accuracy"] = safe_rate(
            k1_candidate_only["ngram_top1_gold"],
            denom,
        )

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_pv1_ngram_selector_k135_v1",
        "research_question": (
            "With exact PV1 recovery strength preserved, how does admitting the NGram-ranked "
            "Personal-K5 Top1 vs Top3 vs Top5 affect end-to-end ranking?"
        ),
        "primary_comparator": "PV1",
        "selection_policy": (
            "No automatic winner selection; K values are pre-specified ablations and all are reported"
        ),
        "selector_ks": list(selector_ks),
        "rows": len(outputs),
        "frozen_pv1": {
            "baseline_k_pv": FROZEN_K_PV,
            "lambda_frequency": FROZEN_LAMBDA_FREQUENCY,
            "lambda_pv": FROZEN_LAMBDA_PV,
            "frequency_support": "log1p(count) / max_log1p_count_over_frozen_PersonalK5",
            "merge": "PV1 boundary injection; Generic F unchanged; Generic wins exact ties",
        },
        "ablation_semantics": {
            "K1": "admit NGram Top1 personal candidate",
            "K3": "admit up to NGram Top3 personal candidates",
            "K5": (
                "admit all available Personal K5; NGram no longer filters identity, only tie order"
            ),
            "ngram_support_used_as_recovery_strength": False,
            "pv1_frequency_support_used_as_recovery_strength": True,
        },
        "pv1_reproduction_audit": reproduction_audit,
        "baseline_metrics": {
            "PV1": pv1_metrics,
            "PV1-Reproduced": reproduced_metrics,
        },
        "baseline_recovery": {"PV1": pv1_recovery},
        "methods": method_reports,
        "k1_selector_diagnostic": {
            "rows_with_personal_candidates": k1_selector_rows_with_candidates,
            "selector_changed_n": k1_selector_changed_n,
            "selector_changed_rate": safe_rate(
                k1_selector_changed_n,
                k1_selector_rows_with_candidates,
            ),
            "candidate_only_gold_in_k5_k_ge_2": k1_candidate_only,
        }
        if 1 in selector_ks
        else None,
        "generic_missing_n": generic_missing_n,
        "recoverable_generic_missing_k5_n": recoverable_n,
        "selected_k5_interpolated_key": selected_ngram_key,
        "gold_used_for_scoring_or_selection": False,
        "gold_used_for_train_val_evaluation_only": True,
        "dev3000_used": False,
        "test_used": False,
        "provenance": provenance,
    }

    predictions_path = args.output_root / "predictions.jsonl"
    write_jsonl(predictions_path, outputs)
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    artifact_paths = [audit_path, predictions_path, comparison_path]
    checksums = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in artifact_paths
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print("\n=== RESULTS (PRIMARY COMPARATOR = PV1) ===")
    print(
        "PV1                      "
        f"Macro={pv1_metrics['macro_author_top1']:.6f} "
        f"Top3={pv1_metrics['top3']:.6f} "
        f"MRR10={pv1_metrics['mrr_at_10']:.6f} "
        f"Missing10={pv1_metrics['missing10']:.6f} "
        f"Rec@1={pv1_recovery['recovered_at_1']:.4f} "
        f"Rec@10={pv1_recovery['recovered_at_10']:.4f}"
    )

    for k in selector_ks:
        label = f"PV1+NGramSelector@K{k}"
        report = method_reports[label]
        metrics = report["metrics"]
        delta = report["delta_vs_pv1"]
        transition = report["pv1_to_method"]
        recovery = report["recovery"]
        print(
            f"{label:<25} "
            f"Macro={metrics['macro_author_top1']:.6f} "
            f"dTop1={delta['macro_author_top1']:+.6f} "
            f"Top3={metrics['top3']:.6f} "
            f"dTop3={delta['top3']:+.6f} "
            f"MRR10={metrics['mrr_at_10']:.6f} "
            f"Missing10={metrics['missing10']:.6f} "
            f"rescue={transition['rescue']} harm={transition['harm']} net={transition['net']:+d} "
            f"Rec@1={recovery['recovered_at_1']:.4f} "
            f"Rec@3={recovery['recovered_at_3']:.4f} "
            f"Rec@5={recovery['recovered_at_5']:.4f} "
            f"Rec@10={recovery['recovered_at_10']:.4f}"
        )

    if 1 in selector_ks:
        diag = comparison["k1_selector_diagnostic"]
        cand = diag["candidate_only_gold_in_k5_k_ge_2"]
        print(
            "K1 selector diagnostic: "
            f"changed={diag['selector_changed_n']}/{diag['rows_with_personal_candidates']} "
            f"({diag['selector_changed_rate']:.4f}); "
            f"candidate-only F-top1={cand['frequency_top1_accuracy']:.6f} "
            f"NGram-top1={cand['ngram_top1_accuracy']:.6f} "
            f"net={cand['net']:+d}"
        )

    print("PV1 reproduction: PASSED exact Top10/rank/score cross-check")
    print(f"Saved: {comparison_path}")
    print("Gold used for scoring/selection: false")
    print("Dev3000 used: false")
    print("Test used: false")


if __name__ == "__main__":
    main()
