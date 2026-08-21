"""K5 Choice-Share + Concentration recovery experiment for Initial Pinyin.

Research question
-----------------
Without any context scorer (no NGram, BGE, or PinyinGPT candidate scoring), can a
transparent history-distribution signal recover Personal-K5 candidates into the
Frequency-reranked Generic Top10 while reducing harmful overrides?

Frozen semantics
----------------
- Clean3 Train-Fit + Train-Val only.
- Strictly-prior same-author H5000, selected BEFORE exact-Pinyin filtering.
- Earlier Train-Val rows may become history for later Train-Val rows.
- Frozen Personal-only K5 candidate surface; current Gold is never used to build it.
- Frozen Frequency-reranked Generic candidates (lambda_F=4) are the base ranking.
- No K10. No NGram. No BGE. No PinyinGPT inference.
- Gold is used only for Train-Val model selection/evaluation and diagnostics.
- Dev3000 and Test are not read.

Choice Share
------------
For the complete legal same-Pinyin history H(q), let n_c be the count of target c
and N = |H(q)|. For a Personal-K5 candidate c:

    CS(c) = n_c / N

The denominator is ALL same-Pinyin historical targets, not just Personal K5.
Therefore K=1 does not artificially force CS=1.

Two concentration definitions
-----------------------------
1) Entropy concentration (global distribution concentration):

    C_entropy = 1 - H_norm
    H_norm = -sum_i p_i log p_i / log(M)

where p_i is the full same-Pinyin target distribution and M is the number of
distinct historical targets. If M=1, C_entropy=1; if no history, 0.

2) Winner-margin concentration (head separation):

    C_margin = p_(1) - p_(2)

where p_(1) >= p_(2) are the largest two historical target shares. If M=1,
C_margin=1; if no history, 0.

A combined version considers both without adding another learned weight:

    C_dual = sqrt(C_entropy * C_margin)

This geometric mean is deliberately conservative: it is high only when the whole
distribution is concentrated AND the leading historical choice is clearly
separated from the runner-up.

Recovery scoring families
-------------------------
Generic candidates keep their frozen Frequency final_score. Let b(q) be the same
Generic boundary used by PV1: the minimum normalized Generic score before the
Frequency boost. A Personal-only K5 candidate receives:

    score(c) = b(q) + lambda_D * CS(c) * C(q)^gamma

We compare:

    CS                : C(q) is disabled (gate = 1)
    CS+EntropyConc    : C(q) = C_entropy
    CS+MarginConc     : C(q) = C_margin
    CS+DualConc       : C(q) = C_dual

For every concentration family, gamma=0 is explicitly included and is exactly the
no-concentration CS baseline. Thus concentration is allowed to lose.

Default grids
-------------
    lambda_D in {0, .25, .5, 1, 2, 4, 8, 16}
    gamma    in {0, .5, 1, 2}

Selection is done separately per family on ALL Train-Val rows using Macro-author
Top1, then MRR@10, then lower gamma, then lower lambda. A development-best family
is also reported, with exact ties preferring the simpler family.

Outputs
-------
<output-root>/
    features.jsonl
    feature_summary.json
    grid_results.json
    selected_predictions.jsonl
    comparison.json
    artifact_checksums.json

The comparison reports overall metrics, F/PV1 baselines, F->method rescue/harm,
Generic-Missing recovery@1/@3/@5/@10, and concentration-bin diagnostics.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_FREQUENCY_PV1_SHA256 = "7fd8aa158d8cd50bced36b55610f8d932bc65e3aae1dbbd5bd65907ff1707ea7"
EXPECTED_FIT_ROWS = 144526
EXPECTED_VAL_ROWS = 34416
HISTORY_BUDGET = 5000

DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
DEFAULT_GAMMAS = (0.0, 0.5, 1.0, 2.0)

FAMILY_ORDER = (
    "CS",
    "CS+EntropyConc",
    "CS+MarginConc",
    "CS+DualConc",
)
FAMILY_COMPLEXITY = {name: index for index, name in enumerate(FAMILY_ORDER)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
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
    out: dict[str, dict[str, Any]] = {}
    for value in rows:
        row = dict(value)
        row_id = str(row["row_id"])
        if row_id in out:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        out[row_id] = row
    return out


def parse_float_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("Empty numeric grid")
    if any(value < 0 for value in values):
        raise ValueError("Grid values must be non-negative")
    return tuple(dict.fromkeys(values))


@dataclass(frozen=True)
class HistoryRecord:
    row_id: str
    author: str
    position: int
    pinyin: tuple[str, ...]
    target: str


class CausalHistoryIndex:
    """Strictly-prior same-author H5000 before exact-Pinyin filtering."""

    def __init__(self, records: Sequence[HistoryRecord]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.author].append(record)

        self.positions: dict[str, tuple[int, ...]] = {}
        self.pinyin_records: dict[tuple[str, tuple[str, ...]], tuple[HistoryRecord, ...]] = {}
        self.pinyin_ordinals: dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}

        for author, values in grouped.items():
            # row_id is only a deterministic tie ordering for storage. Strict prior is
            # enforced below with bisect_left(position), so equal-position rows are
            # never visible to one another.
            ordered = tuple(sorted(values, key=lambda row: (row.position, row.row_id)))
            self.positions[author] = tuple(row.position for row in ordered)
            by_pinyin: dict[tuple[str, ...], list[tuple[int, HistoryRecord]]] = defaultdict(list)
            for ordinal, record in enumerate(ordered):
                by_pinyin[record.pinyin].append((ordinal, record))
            for pinyin, pairs in by_pinyin.items():
                key = (author, pinyin)
                self.pinyin_ordinals[key] = tuple(ordinal for ordinal, _ in pairs)
                self.pinyin_records[key] = tuple(record for _, record in pairs)

    def visible(
        self,
        *,
        author: str,
        position: int,
        pinyin: tuple[str, ...],
    ) -> tuple[HistoryRecord, ...]:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, position)
        start = max(0, stop - HISTORY_BUDGET)

        key = (author, pinyin)
        ordinals = self.pinyin_ordinals.get(key, ())
        records = self.pinyin_records.get(key, ())
        left = bisect.bisect_left(ordinals, start)
        right = bisect.bisect_left(ordinals, stop)
        return records[left:right]


def to_history_record(row: Mapping[str, Any]) -> HistoryRecord:
    target = row.get("target", row.get("gold"))
    if target is None:
        raise RuntimeError(f"History row has no target/gold: {row.get('row_id')}")
    return HistoryRecord(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=tuple(str(value) for value in row["pinyin_segments"]),
        target=str(target),
    )


def extract_personal_k5(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Read frozen K5 robustly across the known candidate-surface field variants."""

    for key in ("personal_candidate_texts_top5", "personal_k5"):
        value = row.get(key)
        if isinstance(value, list):
            result = tuple(str(item) for item in value)
            if len(result) > 5:
                raise RuntimeError(f"{key} contains more than 5 candidates at {row.get('row_id')}")
            return result

    value = row.get("personal_candidates_top5")
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
                continue
            if not isinstance(item, Mapping):
                raise RuntimeError(f"Unexpected personal_candidates_top5 item: {item!r}")
            target = item.get("target", item.get("candidate", item.get("text")))
            if target is None:
                raise RuntimeError(f"Cannot identify K5 target at {row.get('row_id')}")
            result.append(str(target))
        if len(result) > 5:
            raise RuntimeError(f"personal_candidates_top5 contains >5 at {row.get('row_id')}")
        return tuple(result)

    raise RuntimeError(
        "Frozen candidate surface does not expose a recognized K5 field at "
        f"{row.get('row_id')}; keys={sorted(row)}"
    )


def distribution_features(counts: Mapping[str, int]) -> dict[str, float | int]:
    positive = sorted((int(value) for value in counts.values() if int(value) > 0), reverse=True)
    total = sum(positive)
    distinct = len(positive)

    if total <= 0 or distinct == 0:
        return {
            "same_pinyin_history_count": 0,
            "distinct_targets": 0,
            "entropy_norm": 1.0,
            "conc_entropy": 0.0,
            "winner_share": 0.0,
            "runner_up_share": 0.0,
            "conc_margin": 0.0,
            "conc_dual": 0.0,
        }

    shares = [value / total for value in positive]
    winner_share = shares[0]
    runner_up_share = shares[1] if distinct >= 2 else 0.0

    if distinct == 1:
        entropy_norm = 0.0
        conc_entropy = 1.0
        conc_margin = 1.0
    else:
        entropy = -sum(p * math.log(p) for p in shares if p > 0)
        entropy_norm = entropy / math.log(distinct)
        entropy_norm = max(0.0, min(1.0, entropy_norm))
        conc_entropy = 1.0 - entropy_norm
        conc_margin = max(0.0, min(1.0, winner_share - runner_up_share))

    conc_dual = math.sqrt(max(0.0, conc_entropy) * max(0.0, conc_margin))

    return {
        "same_pinyin_history_count": total,
        "distinct_targets": distinct,
        "entropy_norm": entropy_norm,
        "conc_entropy": conc_entropy,
        "winner_share": winner_share,
        "runner_up_share": runner_up_share,
        "conc_margin": conc_margin,
        "conc_dual": conc_dual,
    }


def gate_for_family(feature: Mapping[str, Any], family: str, gamma: float) -> float:
    if family == "CS":
        return 1.0
    if gamma == 0.0:
        return 1.0
    if family == "CS+EntropyConc":
        concentration = float(feature["conc_entropy"])
    elif family == "CS+MarginConc":
        concentration = float(feature["conc_margin"])
    elif family == "CS+DualConc":
        concentration = float(feature["conc_dual"])
    else:
        raise ValueError(f"Unknown family: {family}")
    return concentration ** float(gamma)


def rank_merged(
    *,
    frequency_candidates: Sequence[Mapping[str, Any]],
    personal_candidates: Sequence[str],
    choice_shares: Sequence[float],
    gate: float,
    lambda_personal: float,
) -> list[dict[str, Any]]:
    if len(personal_candidates) != len(choice_shares):
        raise RuntimeError("Personal candidate/share length mismatch")

    generic_rows: list[dict[str, Any]] = []
    generic_texts: set[str] = set()
    normalized_generic: list[float] = []

    for position, value in enumerate(frequency_candidates, start=1):
        candidate = str(value.get("candidate", value.get("text")))
        if not candidate:
            raise RuntimeError("Cannot identify Frequency Generic candidate")
        if candidate in generic_texts:
            raise RuntimeError(f"Duplicate Generic candidate: {candidate!r}")
        generic_texts.add(candidate)
        normalized = float(value["normalized_generic_score"])
        normalized_generic.append(normalized)
        generic_rows.append(
            {
                "candidate": candidate,
                "source": "generic_frequency",
                "final_score": float(value["final_score"]),
                "generic_rank": int(value.get("generic_rank") or value.get("rank") or position),
                "personal_rank": None,
            }
        )

    if not generic_rows:
        raise RuntimeError("Frequency Generic candidate surface is empty")

    boundary = min(normalized_generic)
    rows = list(generic_rows)

    for personal_rank, (candidate, share) in enumerate(
        zip(personal_candidates, choice_shares), start=1
    ):
        if candidate in generic_texts:
            raise RuntimeError(f"Frozen Personal-only K5 overlaps Generic: {candidate!r}")
        rows.append(
            {
                "candidate": candidate,
                "source": "personal_k5",
                "final_score": boundary + float(lambda_personal) * float(gate) * float(share),
                "generic_rank": None,
                "personal_rank": personal_rank,
                "choice_share": float(share),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["final_score"]),
            0 if row["source"] == "generic_frequency" else 1,
            int(row["generic_rank"] or row["personal_rank"] or 0),
            str(row["candidate"]),
        )
    )
    return rows[:10]


def rank_of(ranking: Sequence[Mapping[str, Any]], gold: str) -> int | None:
    for index, row in enumerate(ranking, start=1):
        if str(row["candidate"]) == gold:
            return index
    return None


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
        author: author_top1[author] / count
        for author, count in sorted(author_total.items())
    }
    return {
        "method": method,
        "n": n,
        "authors": len(per_author),
        "macro_author_top1": statistics.fmean(per_author.values()) if per_author else 0.0,
        "micro_top1": top1 / n if n else 0.0,
        "top3": top3 / n if n else 0.0,
        "top5": top5 / n if n else 0.0,
        "mrr_at_10": reciprocal / n if n else 0.0,
        "missing10": missing / n if n else 0.0,
        "mean_rank_given_top10": statistics.fmean(observed_ranks) if observed_ranks else None,
        "per_author_top1": per_author,
    }


def transition_counts(
    rows: Sequence[Mapping[str, Any]],
    base_key: str,
    new_key: str,
    predicate=lambda row: True,
) -> dict[str, int]:
    counts = {
        "n": 0,
        "rescue": 0,
        "harm": 0,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    for row in rows:
        if not predicate(row):
            continue
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
    missing = [row for row in rows if bool(row["generic_missing"])]
    available = [row for row in missing if bool(row["gold_in_personal_k5"])]

    def count_at(limit: int) -> int:
        return sum(
            row.get(rank_key) is not None and int(row[rank_key]) <= limit
            for row in available
        )

    top1 = count_at(1)
    top3 = count_at(3)
    top5 = count_at(5)
    top10 = count_at(10)
    denom_avail = len(available)
    denom_missing = len(missing)
    return {
        "generic_missing_n": denom_missing,
        "gold_available_in_personal_k5_n": denom_avail,
        "gold_available_given_missing_rate": denom_avail / denom_missing if denom_missing else 0.0,
        "recovered_to_top1_n": top1,
        "recovered_to_top3_n": top3,
        "recovered_to_top5_n": top5,
        "recovered_to_top10_n": top10,
        "top1_given_available": top1 / denom_avail if denom_avail else 0.0,
        "top3_given_available": top3 / denom_avail if denom_avail else 0.0,
        "top5_given_available": top5 / denom_avail if denom_avail else 0.0,
        "top10_given_available": top10 / denom_avail if denom_avail else 0.0,
        "top1_given_missing": top1 / denom_missing if denom_missing else 0.0,
        "top3_given_missing": top3 / denom_missing if denom_missing else 0.0,
        "top5_given_missing": top5 / denom_missing if denom_missing else 0.0,
        "top10_given_missing": top10 / denom_missing if denom_missing else 0.0,
    }


def conc_bin(value: float) -> str:
    if value < 0.10:
        return "[0,.10)"
    if value < 0.25:
        return "[.10,.25)"
    if value < 0.50:
        return "[.25,.50)"
    if value < 0.75:
        return "[.50,.75)"
    return "[.75,1]"


def bin_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    rank_key: str,
    concentration_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[conc_bin(float(row[concentration_key]))].append(row)

    ordered_bins = ("[0,.10)", "[.10,.25)", "[.25,.50)", "[.50,.75)", "[.75,1]")
    output: list[dict[str, Any]] = []
    for label in ordered_bins:
        values = grouped.get(label, [])
        trans = transition_counts(values, "frequency_rank", rank_key)
        missing = sum(bool(row["generic_missing"]) for row in values)
        available = sum(
            bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"])
            for row in values
        )
        recovered_top1 = sum(
            bool(row["generic_missing"])
            and bool(row["gold_in_personal_k5"])
            and row.get(rank_key) == 1
            for row in values
        )
        output.append(
            {
                "bin": label,
                "n": len(values),
                "generic_missing_n": missing,
                "gold_available_k5_n": available,
                "recovered_top1_n": recovered_top1,
                **{key: trans[key] for key in ("rescue", "harm", "net")},
            }
        )
    return output


def verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "fit": args.fit,
        "val": args.val,
        "candidate_surface": args.candidate_surface,
        "frequency_pv1_predictions": args.frequency_pv1_predictions,
    }
    expected = {
        "fit": EXPECTED_FIT_SHA256,
        "val": EXPECTED_VAL_SHA256,
        "candidate_surface": EXPECTED_SURFACE_SHA256,
        "frequency_pv1_predictions": EXPECTED_FREQUENCY_PV1_SHA256,
    }
    hashes: dict[str, str] = {}
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        hashes[label] = digest
        if digest != expected[label]:
            raise RuntimeError(
                f"SHA mismatch for {label}:\nexpected={expected[label]}\nactual={digest}\npath={path}"
            )
    return {
        "paths": {key: str(path) for key, path in required.items()},
        "sha256": hashes,
    }


def choose_config(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Cannot select from empty grid")
    return dict(
        max(
            results,
            key=lambda row: (
                float(row["metrics"]["macro_author_top1"]),
                float(row["metrics"]["mrr_at_10"]),
                -float(row["gamma"]),
                -float(row["lambda_personal"]),
            ),
        )
    )


def choose_development_best(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(
        max(
            selected,
            key=lambda row: (
                float(row["metrics"]["macro_author_top1"]),
                float(row["metrics"]["mrr_at_10"]),
                -FAMILY_COMPLEXITY[str(row["family"])],
                -float(row["gamma"]),
                -float(row["lambda_personal"]),
            ),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K5 Choice-Share + entropy/margin/dual concentration recovery experiment"
    )
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--frequency-pv1-predictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--lambdas",
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated lambda_D grid",
    )
    parser.add_argument(
        "--gammas",
        default=",".join(str(value) for value in DEFAULT_GAMMAS),
        help="Comma-separated concentration exponent grid",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    lambdas = parse_float_grid(args.lambdas)
    gammas = parse_float_grid(args.gammas)
    if 0.0 not in gammas:
        raise RuntimeError("Gamma grid must include 0 so concentration can explicitly lose to CS")

    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = verify_inputs(args)

    print("=== K5 CHOICE-SHARE + CONCENTRATION RECOVERY ===")
    print("K: 5 only")
    print(f"lambda_D grid: {lambdas}")
    print(f"gamma grid: {gammas}")
    print("families:")
    for family in FAMILY_ORDER:
        print(f"  - {family}")
    print("Context scorer used: false")
    print("K10 used: false")
    print("Gold used for scoring/features: false")
    print("Gold used for Train-Val selection/evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false")
    print()

    fit_rows = read_jsonl(args.fit)
    val_rows = read_jsonl(args.val)
    surface_rows = index_rows(read_jsonl(args.candidate_surface), "candidate surface")
    pred_rows = index_rows(read_jsonl(args.frequency_pv1_predictions), "Frequency/PV1 predictions")
    val = index_rows(val_rows, "Train-Val")

    if len(fit_rows) != EXPECTED_FIT_ROWS:
        raise RuntimeError(f"Unexpected Train-Fit row count: {len(fit_rows)}")
    if len(val_rows) != EXPECTED_VAL_ROWS:
        raise RuntimeError(f"Unexpected Train-Val row count: {len(val_rows)}")
    if not (set(val) == set(surface_rows) == set(pred_rows)):
        raise RuntimeError("Train-Val / candidate surface / Frequency-PV1 row IDs differ")

    records = [to_history_record(row) for row in fit_rows]
    records.extend(to_history_record(row) for row in val_rows)
    history = CausalHistoryIndex(records)

    feature_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    candidate_count_distribution: Counter[int] = Counter()

    for number, row_id in enumerate(sorted(val), start=1):
        row = val[row_id]
        pred = pred_rows[row_id]
        surface = surface_rows[row_id]

        author = str(row["author"])
        gold = str(row.get("target", row.get("gold")))
        pinyin = tuple(str(value) for value in row["pinyin_segments"])
        visible = history.visible(
            author=author,
            position=int(row["chronological_position"]),
            pinyin=pinyin,
        )
        counts = Counter(record.target for record in visible)
        dist = distribution_features(counts)

        personal_k5 = extract_personal_k5(surface)
        candidate_count_distribution[len(personal_k5)] += 1

        frequency_candidates = list(pred.get("frequency_candidates", []))
        if not frequency_candidates:
            raise RuntimeError(f"Missing frequency_candidates at {row_id}")
        generic_texts = {
            str(value.get("candidate", value.get("text"))) for value in frequency_candidates
        }
        overlap = generic_texts.intersection(personal_k5)
        if overlap:
            raise RuntimeError(f"Personal K5 overlaps Generic at {row_id}: {sorted(overlap)}")

        history_total = int(dist["same_pinyin_history_count"])
        personal_counts = [int(counts.get(candidate, 0)) for candidate in personal_k5]
        choice_shares = [
            count / history_total if history_total > 0 else 0.0
            for count in personal_counts
        ]

        # Every frozen Personal-K5 candidate should be supported by visible history.
        if any(count <= 0 for count in personal_counts):
            raise RuntimeError(
                f"Frozen K5 candidate lacks legal visible same-Pinyin history at {row_id}: "
                f"{list(zip(personal_k5, personal_counts))}"
            )

        generic_missing = bool(row.get("generic_missing", pred.get("generic_missing", False)))
        frequency_rank = pred.get("frequency_rank")
        pv1_rank = pred.get("pv1_rank")
        generic_rank = pred.get("generic_rank", row.get("generic_rank"))

        feature_rows.append(
            {
                "row_id": row_id,
                "author": author,
                "gold": gold,
                "pinyin_segments": list(pinyin),
                "ambiguous": bool(row.get("ambiguous", False)),
                # IMPORTANT: use the Train-Val formal Conflict field, not the older
                # candidate-surface winner-mismatch field.
                "formal_conflict": bool(row.get("conflict", False)),
                "generic_missing": generic_missing,
                "generic_rank": generic_rank,
                "frequency_rank": frequency_rank,
                "pv1_rank": pv1_rank,
                "personal_k5": list(personal_k5),
                "personal_k5_count": len(personal_k5),
                "personal_counts": personal_counts,
                "choice_shares": choice_shares,
                "gold_in_personal_k5": gold in set(personal_k5),
                "frequency_candidates": frequency_candidates,
                **dist,
            }
        )

        if args.progress_every > 0 and (
            number % args.progress_every == 0 or number == len(val)
        ):
            elapsed = time.perf_counter() - start
            print(
                f"features {number}/{len(val)}  rate={number / elapsed:.1f} rows/s",
                flush=True,
            )

    feature_path = args.output_root / "features.jsonl"
    write_jsonl(feature_path, feature_rows)

    feature_summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_k5_choice_share_concentration_recovery_v1",
        "rows": len(feature_rows),
        "history_budget": HISTORY_BUDGET,
        "history_semantics": "strictly-prior same-author H5000 before exact-Pinyin filtering",
        "candidate_pool": "frozen Personal-only K5",
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_count_distribution.items())
        },
        "generic_missing_n": sum(row["generic_missing"] for row in feature_rows),
        "generic_missing_gold_in_k5_n": sum(
            row["generic_missing"] and row["gold_in_personal_k5"] for row in feature_rows
        ),
        "history_available_n": sum(row["same_pinyin_history_count"] > 0 for row in feature_rows),
        "mean_same_pinyin_history_count": statistics.fmean(
            int(row["same_pinyin_history_count"]) for row in feature_rows
        ),
        "mean_distinct_targets": statistics.fmean(
            int(row["distinct_targets"]) for row in feature_rows
        ),
        "mean_conc_entropy": statistics.fmean(float(row["conc_entropy"]) for row in feature_rows),
        "mean_conc_margin": statistics.fmean(float(row["conc_margin"]) for row in feature_rows),
        "mean_conc_dual": statistics.fmean(float(row["conc_dual"]) for row in feature_rows),
        "choice_share_denominator": "all legal visible same-Pinyin H5000 history",
        "concentration_entropy": "1-normalized_entropy(full same-Pinyin target distribution)",
        "concentration_margin": "top1_share-top2_share(full same-Pinyin target distribution)",
        "concentration_dual": "sqrt(conc_entropy*conc_margin)",
        "gold_used_for_feature_construction": False,
        "dev3000_used": False,
        "test_used": False,
        "provenance": provenance,
        "features_sha256": sha256_file(feature_path),
    }
    write_json(args.output_root / "feature_summary.json", feature_summary)

    # Baselines from the existing frozen prediction artifact.
    baseline_metrics = {
        "G": metric_summary(feature_rows, "generic_rank", "G"),
        "F": metric_summary(feature_rows, "frequency_rank", "F"),
        "PV1": metric_summary(feature_rows, "pv1_rank", "PV1"),
    }

    grid_results: list[dict[str, Any]] = []
    total_configs = len(lambdas) + 3 * len(lambdas) * len(gammas)
    config_number = 0
    grid_start = time.perf_counter()

    # Store ranks/top10 for all configs only transiently. Selected configs are
    # recomputed below so the durable artifact remains compact and auditable.
    for family in FAMILY_ORDER:
        family_gammas = (0.0,) if family == "CS" else gammas
        for gamma in family_gammas:
            for lambda_personal in lambdas:
                config_number += 1
                eval_rows: list[dict[str, Any]] = []
                for feature in feature_rows:
                    gate = gate_for_family(feature, family, gamma)
                    ranking = rank_merged(
                        frequency_candidates=feature["frequency_candidates"],
                        personal_candidates=feature["personal_k5"],
                        choice_shares=feature["choice_shares"],
                        gate=gate,
                        lambda_personal=lambda_personal,
                    )
                    eval_rows.append(
                        {
                            "row_id": feature["row_id"],
                            "author": feature["author"],
                            "rank": rank_of(ranking, str(feature["gold"])),
                            "generic_missing": feature["generic_missing"],
                            "gold_in_personal_k5": feature["gold_in_personal_k5"],
                            "frequency_rank": feature["frequency_rank"],
                        }
                    )

                metrics = metric_summary(eval_rows, "rank", family)
                # transition_counts expects the new rank under a stable key.
                transition_rows = [
                    {
                        "frequency_rank": row["frequency_rank"],
                        "new_rank": row["rank"],
                    }
                    for row in eval_rows
                ]
                grid_results.append(
                    {
                        "family": family,
                        "gamma": float(gamma),
                        "lambda_personal": float(lambda_personal),
                        "metrics": metrics,
                        "f_to_method": transition_counts(
                            transition_rows, "frequency_rank", "new_rank"
                        ),
                    }
                )
                print(
                    f"grid {config_number}/{total_configs} {family} "
                    f"gamma={gamma:g} lambda={lambda_personal:g} "
                    f"MacroTop1={metrics['macro_author_top1']:.6f} "
                    f"MicroTop1={metrics['micro_top1']:.6f} "
                    f"Missing10={metrics['missing10']:.6f}",
                    flush=True,
                )

    selected_by_family: dict[str, dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        selected_by_family[family] = choose_config(
            [row for row in grid_results if row["family"] == family]
        )

    development_best = choose_development_best(list(selected_by_family.values()))

    # Recompute selected family predictions with durable rankings/ranks.
    selected_rows: list[dict[str, Any]] = []
    for feature in feature_rows:
        row_out: dict[str, Any] = {
            "row_id": feature["row_id"],
            "author": feature["author"],
            "gold": feature["gold"],
            "ambiguous": feature["ambiguous"],
            "formal_conflict": feature["formal_conflict"],
            "generic_missing": feature["generic_missing"],
            "gold_in_personal_k5": feature["gold_in_personal_k5"],
            "same_pinyin_history_count": feature["same_pinyin_history_count"],
            "distinct_targets": feature["distinct_targets"],
            "conc_entropy": feature["conc_entropy"],
            "conc_margin": feature["conc_margin"],
            "conc_dual": feature["conc_dual"],
            "personal_k5": feature["personal_k5"],
            "choice_shares": feature["choice_shares"],
            "ranks": {
                "G": feature["generic_rank"],
                "F": feature["frequency_rank"],
                "PV1": feature["pv1_rank"],
            },
            "selected": {},
        }
        for family in FAMILY_ORDER:
            selected = selected_by_family[family]
            gamma = float(selected["gamma"])
            lambda_personal = float(selected["lambda_personal"])
            gate = gate_for_family(feature, family, gamma)
            ranking = rank_merged(
                frequency_candidates=feature["frequency_candidates"],
                personal_candidates=feature["personal_k5"],
                choice_shares=feature["choice_shares"],
                gate=gate,
                lambda_personal=lambda_personal,
            )
            rank = rank_of(ranking, str(feature["gold"]))
            row_out["ranks"][family] = rank
            row_out["selected"][family] = {
                "gamma": gamma,
                "lambda_personal": lambda_personal,
                "gate": gate,
                "top10": [str(item["candidate"]) for item in ranking],
            }
        selected_rows.append(row_out)

    selected_path = args.output_root / "selected_predictions.jsonl"
    write_jsonl(selected_path, selected_rows)

    selected_reports: dict[str, Any] = {}
    for family in FAMILY_ORDER:
        rank_key = family
        selected = selected_by_family[family]
        metrics = metric_summary(
            [
                {
                    "author": row["author"],
                    "rank": row["ranks"][family],
                }
                for row in selected_rows
            ],
            "rank",
            family,
        )
        # Convert selected_rows to flat rank keys for shared helpers.
        flat_rows = [
            {
                **row,
                "frequency_rank": row["ranks"]["F"],
                "pv1_rank": row["ranks"]["PV1"],
                "selected_rank": row["ranks"][family],
            }
            for row in selected_rows
        ]
        selected_reports[family] = {
            "selected_config": {
                "gamma": float(selected["gamma"]),
                "lambda_personal": float(selected["lambda_personal"]),
            },
            "metrics": metrics,
            "f_to_method": transition_counts(flat_rows, "frequency_rank", "selected_rank"),
            "pv1_to_method": transition_counts(flat_rows, "pv1_rank", "selected_rank"),
            "generic_missing_recovery": recovery_summary(flat_rows, "selected_rank"),
            "formal_conflict_transition_vs_f": transition_counts(
                flat_rows,
                "frequency_rank",
                "selected_rank",
                predicate=lambda row: bool(row["formal_conflict"]),
            ),
            "ambiguous_non_conflict_transition_vs_f": transition_counts(
                flat_rows,
                "frequency_rank",
                "selected_rank",
                predicate=lambda row: bool(row["ambiguous"]) and not bool(row["formal_conflict"]),
            ),
            "diagnostics": {
                "entropy_concentration_bins": bin_diagnostics(
                    flat_rows, "selected_rank", "conc_entropy"
                ),
                "margin_concentration_bins": bin_diagnostics(
                    flat_rows, "selected_rank", "conc_margin"
                ),
                "dual_concentration_bins": bin_diagnostics(
                    flat_rows, "selected_rank", "conc_dual"
                ),
            },
        }

    grid_payload = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_k5_choice_share_concentration_recovery_v1",
        "selection_population": "all 34416 standardized Clean3 Train-Val rows",
        "selection_metric": "Macro-author Top1; tie MRR@10, lower gamma, lower lambda",
        "lambda_grid": list(lambdas),
        "gamma_grid": list(gammas),
        "families": list(FAMILY_ORDER),
        "results": grid_results,
        "selected_by_family": selected_by_family,
        "development_best": development_best,
        "gold_used_for_scoring": False,
        "gold_used_for_train_val_selection": True,
        "dev3000_used": False,
        "test_used": False,
    }
    grid_path = args.output_root / "grid_results.json"
    write_json(grid_path, grid_payload)

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_k5_choice_share_concentration_recovery_v1",
        "research_scope": "K5 recovery/distribution only; no context scoring",
        "rows": len(feature_rows),
        "history_budget": HISTORY_BUDGET,
        "candidate_pool": "frozen Personal-only K5",
        "formulas": {
            "choice_share": "count(candidate) / all legal visible same-Pinyin history count",
            "conc_entropy": "1 - normalized entropy over full same-Pinyin target distribution",
            "conc_margin": "top1 historical target share - top2 historical target share",
            "conc_dual": "sqrt(conc_entropy * conc_margin)",
            "personal_score": "generic_boundary + lambda_D * CS(candidate) * concentration^gamma",
        },
        "baselines": baseline_metrics,
        "baseline_transitions": {
            "F_to_PV1": transition_counts(feature_rows, "frequency_rank", "pv1_rank"),
        },
        "baseline_recovery": {
            "PV1": recovery_summary(feature_rows, "pv1_rank"),
        },
        "selected_by_family": selected_reports,
        "development_best": {
            "family": development_best["family"],
            "gamma": development_best["gamma"],
            "lambda_personal": development_best["lambda_personal"],
            "metrics": development_best["metrics"],
        },
        "interpretation_contract": {
            "gamma_0": "No concentration; exact CS-only gate for concentration families",
            "entropy": "Global distribution concentration",
            "margin": "Top-1 vs Top-2 historical preference separation",
            "dual": "Both concentration views must be high; geometric-mean combination",
            "primary_goal": "Improve recovery while controlling harmful override relative to F/PV1",
        },
        "provenance": {
            **provenance,
            "features_sha256": sha256_file(feature_path),
            "grid_results_sha256": sha256_file(grid_path),
            "selected_predictions_sha256": sha256_file(selected_path),
            "gold_used_for_candidate_construction": False,
            "gold_used_for_feature_construction": False,
            "gold_used_for_scoring": False,
            "gold_used_for_train_val_selection_and_evaluation_only": True,
            "context_scorer_used": False,
            "k10_used": False,
            "dev3000_used": False,
            "test_used": False,
        },
        "runtime_seconds": time.perf_counter() - start,
        "grid_runtime_seconds": time.perf_counter() - grid_start,
    }
    comparison_path = args.output_root / "comparison.json"
    write_json(comparison_path, comparison)

    checksums = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (
            feature_path,
            args.output_root / "feature_summary.json",
            grid_path,
            selected_path,
            comparison_path,
        )
    }
    write_json(args.output_root / "artifact_checksums.json", checksums)

    print()
    print("=== COMPLETE ===")
    for name in ("G", "F", "PV1"):
        m = baseline_metrics[name]
        print(
            f"{name:>18}  MacroTop1={m['macro_author_top1']:.6f} "
            f"MicroTop1={m['micro_top1']:.6f} Top3={m['top3']:.6f} "
            f"MRR10={m['mrr_at_10']:.6f} Missing10={m['missing10']:.6f}"
        )
    print()
    for family in FAMILY_ORDER:
        report = selected_reports[family]
        cfg = report["selected_config"]
        m = report["metrics"]
        tr = report["f_to_method"]
        rec = report["generic_missing_recovery"]
        print(
            f"{family:>18}  gamma={cfg['gamma']:g} lambda={cfg['lambda_personal']:g}  "
            f"MacroTop1={m['macro_author_top1']:.6f} MicroTop1={m['micro_top1']:.6f} "
            f"Missing10={m['missing10']:.6f}  "
            f"F->method rescue={tr['rescue']} harm={tr['harm']} net={tr['net']}  "
            f"K5 recovery@1/@3/@5/@10="
            f"{rec['recovered_to_top1_n']}/{rec['recovered_to_top3_n']}/"
            f"{rec['recovered_to_top5_n']}/{rec['recovered_to_top10_n']}"
        )
    print()
    print(
        "DEVELOPMENT BEST: "
        f"{development_best['family']} gamma={development_best['gamma']:g} "
        f"lambda={development_best['lambda_personal']:g} "
        f"MacroTop1={development_best['metrics']['macro_author_top1']:.6f}"
    )
    print(f"saved: {comparison_path}")


if __name__ == "__main__":
    main()
