"""Adaptive lexical-context candidate scoring for Initial Pinyin personalisation.

This CPU-oriented experiment extends the earlier NGramRecency study in two ways:

1. Compare three transparent lexical context mechanisms:
   - HardBackoffNGramRecency
   - SoftSuffixRecency
   - InterpolatedNGramRecency
2. Evaluate both the frozen Personal K5 surface and an exploratory Personal K10
   surface reconstructed with the SAME causal-history, personal-frequency ordering,
   Generic-exclusion, and Frozen-PinyinGPT compatibility rules.

Scientific safeguards
---------------------
- H5000 is applied to strictly-prior same-author history BEFORE Pinyin filtering.
- Candidate construction never consults current Gold.
- The reconstructed K5 prefix MUST exactly match the frozen K5 candidate surface;
  otherwise the run aborts before producing research results.
- Dev3000 is not read.
- Test is not read.
- Gold is used only for Train-Val diagnostic selection/evaluation after all scores
  have been computed.

Method definitions
------------------
Let H be legal same-Pinyin history after the author-level H5000 cut, restricted to
rows whose historical target is in the current Personal candidate pool. Let
R_tau(h)=exp(-age(h)/tau), where age=0 is the immediately previous same-author
interaction inside the H5000 author stream.

HardBackoffNGramRecency(max_n, tau)
    Choose the largest k <= max_n with at least one candidate-target history row
    whose context has the same last-k Python Unicode code points as the query.
    If none exists, k=0. Score candidate c by the sum of R_tau(h) over matched
    history rows with target c.

SoftSuffixRecency(max_n, beta, tau)
    Keep ALL candidate-target history rows. For each history row, let m be the
    longest exact suffix match length between query and history context, capped
    at max_n. Weight the row by exp(beta*m) * R_tau(h). Thus m=0 still retains a
    recency-weighted long-term preference signal, while longer lexical matches
    receive smoothly increasing weight.

InterpolatedNGramRecency(max_n, kappa, tau)
    Build a recency-weighted candidate distribution P_hat_k at every exact suffix
    order k=0..max_n. Start with P_0. For k>=1, let mass_k be the total recency
    weight at that suffix order and set lambda_k=mass_k/(mass_k+kappa). Recursively
    interpolate P_k=lambda_k*P_hat_k+(1-lambda_k)*P_{k-1}. Sparse long contexts
    therefore back off smoothly rather than replacing shorter-context evidence.

Default grids
-------------
max_n in {2, 4, 8}
tau   in {512, 2048, 8192}
beta  in {0.25, 0.5, 1.0}
kappa in {1, 4, 16, 64}

Outputs
-------
<output-root>/adaptive_ngram_top10_v1/
    candidate_surface_k5_k10.jsonl
    surface_audit.json
    scores.jsonl
    score_summary.json
    comparison.json

The row-score file is resumable. The reconstructed candidate-surface file is
versioned separately and never overwrites the frozen K5 artifact.
"""

from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Frozen provenance
# ---------------------------------------------------------------------------

EXPECTED_SURFACE_SHA256 = "205c0ba01cd0678d7a4341c503fa2e74cf126a70182ff582687025e4946764b2"
EXPECTED_GENERIC_SHA256 = "bd0fb4dc304e0b266b90fae6fe3ac65424d2f52b23fedfa881212706ba2c2873"
EXPECTED_FIT_SHA256 = "162f5c98daa86cc69947571e6d8f20fc401f0a82cdd3fd6e517eb7be2addbdb4"
EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"

EXPECTED_SURFACE_ROWS = 34416
EXPECTED_FIT_ROWS = 144526
EXPECTED_VAL_ROWS = 34416
EXPECTED_FROZEN_K5_ELIGIBLE = 30509
EXPECTED_FROZEN_K5_PAIRS = 123738

HISTORY_BUDGET = 5000
DEFAULT_CANDIDATE_KS = (5, 10)
DEFAULT_MAX_NS = (2, 4, 8)
DEFAULT_TAUS = (512.0, 2048.0, 8192.0)
DEFAULT_BETAS = (0.25, 0.5, 1.0)
DEFAULT_KAPPAS = (1.0, 4.0, 16.0, 64.0)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def index_rows(rows: Iterable[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in out:
            raise RuntimeError(f"Duplicate {label} row_id: {row_id}")
        out[row_id] = row
    return out


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    return ordered[int((len(ordered) - 1) * fraction)]


def timing_summary(values_ms: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values_ms]
    mean = statistics.fmean(values) if values else None
    return {
        "n": len(values),
        "mean_ms": mean,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "queries_per_second_from_mean": 1000.0 / mean if mean and mean > 0 else None,
    }


def normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(0.0, float(v)) for v in values]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [v / total for v in clipped]


def rank_candidates(candidates: Sequence[str], distribution: Sequence[float]) -> list[str]:
    return [
        candidates[i]
        for i in sorted(
            range(len(candidates)),
            key=lambda i: (-float(distribution[i]), i, candidates[i]),
        )
    ]


def gold_rank(ranking: Sequence[str], gold: str) -> int | None:
    for i, candidate in enumerate(ranking, start=1):
        if candidate == gold:
            return i
    return None


def candidate_texts_top5(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("personal_candidate_texts_top5")
    if isinstance(values, list):
        return tuple(dict.fromkeys(str(v) for v in values[:5]))
    values = row.get("personal_candidates_top5")
    if not isinstance(values, list):
        raise RuntimeError(f"{row.get('row_id')}: no frozen personal K5 field")
    out: list[str] = []
    for item in values[:5]:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Mapping):
            text = item.get("text") or item.get("candidate") or item.get("target")
            if text is None:
                raise RuntimeError(f"{row.get('row_id')}: candidate dict has no text")
            out.append(str(text))
        else:
            raise RuntimeError(f"{row.get('row_id')}: unsupported candidate value {item!r}")
    return tuple(dict.fromkeys(out))


def generic_texts(row: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("top10_candidates", "generic_candidates"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        out: list[str] = []
        for item in values:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("candidate") or item.get("target")
                if text is not None:
                    out.append(str(text))
        if out:
            return tuple(dict.fromkeys(out))
    return ()

# ---------------------------------------------------------------------------
# Inputs and history
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    fit_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    surface_rows: list[dict[str, Any]]
    generic_rows: list[dict[str, Any]]
    fit: dict[str, dict[str, Any]]
    val: dict[str, dict[str, Any]]
    surface: dict[str, dict[str, Any]]
    generic: dict[str, dict[str, Any]]


def load_and_verify(args: argparse.Namespace) -> Inputs:
    checks = (
        (args.fit, EXPECTED_FIT_SHA256, EXPECTED_FIT_ROWS, "Train-Fit"),
        (args.val, EXPECTED_VAL_SHA256, EXPECTED_VAL_ROWS, "Train-Val"),
        (args.candidate_surface, EXPECTED_SURFACE_SHA256, EXPECTED_SURFACE_ROWS, "candidate surface"),
        (args.generic_predictions, EXPECTED_GENERIC_SHA256, EXPECTED_SURFACE_ROWS, "Generic predictions"),
    )
    loaded: dict[str, list[dict[str, Any]]] = {}
    for path, expected_sha, expected_rows, label in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_sha:
            raise RuntimeError(
                f"Frozen {label} SHA mismatch:\nexpected={expected_sha}\nactual={actual}\npath={path}"
            )
        rows = read_jsonl(path)
        if len(rows) != expected_rows:
            raise RuntimeError(f"Unexpected {label} rows: expected={expected_rows} actual={len(rows)}")
        loaded[label] = rows

    fit_rows = loaded["Train-Fit"]
    val_rows = loaded["Train-Val"]
    surface_rows = loaded["candidate surface"]
    generic_rows = loaded["Generic predictions"]
    fit = index_rows(fit_rows, label="fit")
    val = index_rows(val_rows, label="val")
    surface = index_rows(surface_rows, label="surface")
    generic = index_rows(generic_rows, label="generic")
    if set(val) != set(surface) or set(surface) != set(generic):
        raise RuntimeError("Train-Val / candidate surface / Generic row IDs differ")

    frozen_pairs = sum(len(candidate_texts_top5(row)) for row in surface_rows)
    frozen_eligible = sum(bool(candidate_texts_top5(row)) for row in surface_rows)
    if frozen_pairs != EXPECTED_FROZEN_K5_PAIRS or frozen_eligible != EXPECTED_FROZEN_K5_ELIGIBLE:
        raise RuntimeError(
            f"Frozen K5 workload changed: eligible={frozen_eligible} pairs={frozen_pairs}"
        )

    for row_id in val:
        pv = tuple(str(x) for x in val[row_id]["pinyin_segments"])
        ps = tuple(str(x) for x in surface[row_id]["pinyin_segments"])
        pg = tuple(str(x) for x in generic[row_id]["pinyin_segments"])
        if not (pv == ps == pg):
            raise RuntimeError(f"Pinyin mismatch at {row_id}")
        sg = set(generic_texts(surface[row_id]))
        gg = set(generic_texts(generic[row_id]))
        if sg and gg and sg != gg:
            raise RuntimeError(f"Generic Top10 mismatch between surface and predictions at {row_id}")

    return Inputs(
        fit_rows=fit_rows,
        val_rows=val_rows,
        surface_rows=surface_rows,
        generic_rows=generic_rows,
        fit=fit,
        val=val,
        surface=surface,
        generic=generic,
    )


@dataclass(frozen=True)
class HistoryRecord:
    row_id: str
    author: str
    position: int
    pinyin: tuple[str, ...]
    target: str
    context: str


@dataclass(frozen=True)
class VisibleHistory:
    record: HistoryRecord
    age: int


class CausalHistoryIndex:
    """Strictly-prior same-user H5000 selected before exact-Pinyin filtering."""

    def __init__(self, records: Sequence[HistoryRecord]) -> None:
        grouped: dict[str, list[HistoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.author].append(record)
        self.positions: dict[str, tuple[int, ...]] = {}
        self.pinyin_records: dict[tuple[str, tuple[str, ...]], tuple[HistoryRecord, ...]] = {}
        self.pinyin_ordinals: dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}
        for author, values in grouped.items():
            ordered = tuple(sorted(values, key=lambda r: (r.position, r.row_id)))
            self.positions[author] = tuple(r.position for r in ordered)
            by_pinyin: dict[tuple[str, ...], list[tuple[int, HistoryRecord]]] = defaultdict(list)
            for ordinal, record in enumerate(ordered):
                by_pinyin[record.pinyin].append((ordinal, record))
            for pinyin, pairs in by_pinyin.items():
                key = (author, pinyin)
                self.pinyin_ordinals[key] = tuple(o for o, _ in pairs)
                self.pinyin_records[key] = tuple(r for _, r in pairs)

    def visible(self, *, author: str, position: int, pinyin: tuple[str, ...]) -> tuple[VisibleHistory, ...]:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, position)
        start = max(0, stop - HISTORY_BUDGET)
        key = (author, pinyin)
        ordinals = self.pinyin_ordinals.get(key, ())
        matching = self.pinyin_records.get(key, ())
        left = bisect.bisect_left(ordinals, start)
        right = bisect.bisect_left(ordinals, stop)
        return tuple(
            VisibleHistory(record=record, age=stop - 1 - ordinal)
            for ordinal, record in zip(ordinals[left:right], matching[left:right])
        )


def to_history_record(row: Mapping[str, Any]) -> HistoryRecord:
    target = row.get("target", row.get("gold"))
    if target is None:
        raise RuntimeError(f"No target/gold in history row {row.get('row_id')}")
    return HistoryRecord(
        row_id=str(row["row_id"]),
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=tuple(str(x) for x in row["pinyin_segments"]),
        target=str(target),
        context=str(row.get("context", "")),
    )


def build_history_index(inputs: Inputs) -> CausalHistoryIndex:
    records = [to_history_record(row) for row in inputs.fit_rows]
    records.extend(to_history_record(row) for row in inputs.val_rows)
    return CausalHistoryIndex(records)


def query_history(inputs: Inputs, index: CausalHistoryIndex, row_id: str) -> tuple[VisibleHistory, ...]:
    row = inputs.val[row_id]
    return index.visible(
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=tuple(str(x) for x in row["pinyin_segments"]),
    )

# ---------------------------------------------------------------------------
# Faithful K10 surface reconstruction
# ---------------------------------------------------------------------------


def compatible(backend: Any, candidate: str, pinyin: tuple[str, ...]) -> bool:
    characters = list(candidate)
    if len(characters) != len(pinyin):
        return False
    ids = backend.tokenizer.convert_tokens_to_ids(characters)
    for token_id, segment in zip(ids, pinyin):
        if token_id not in backend.allowed_token_ids.get(segment, ()):
            return False
    return True


def ranked_personal_only_targets(
    *, visible: Sequence[VisibleHistory], generic: set[str]
) -> tuple[str, ...]:
    counts = Counter(item.record.target for item in visible)
    ranked = sorted(counts, key=lambda target: (-counts[target], target))
    return tuple(target for target in ranked if target not in generic)


def build_surfaces(args: argparse.Namespace, inputs: Inputs, history_index: CausalHistoryIndex) -> dict[str, dict[str, Any]]:
    output_dir = args.output_root / "adaptive_ngram_top10_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = output_dir / "candidate_surface_k5_k10.jsonl"
    audit_path = output_dir / "surface_audit.json"

    # Always reconstruct from source data rather than trusting a previous exploratory K10 artifact.
    from src.reference_backend_pinyingpt.backend import PinyinGPTConcatBackend

    print("Loading Frozen PinyinGPT on CPU for tokenizer/Pinyin compatibility only...", flush=True)
    backend = PinyinGPTConcatBackend(args.pinyin_checkpoint, device="cpu")

    rows: dict[str, dict[str, Any]] = {}
    mismatch_examples: list[dict[str, Any]] = []
    frozen_k5_match = 0
    k10_eligible = 0
    k10_pairs = 0
    k5_gold_in = 0
    k10_gold_in = 0
    generic_missing = 0
    generic_missing_k5_gold = 0
    generic_missing_k10_gold = 0
    candidate_count_k10: Counter[int] = Counter()

    with surface_path.open("w", encoding="utf-8", newline="\n") as sink:
        for number, row_id in enumerate(sorted(inputs.surface), start=1):
            frozen_k5 = candidate_texts_top5(inputs.surface[row_id])
            visible = query_history(inputs, history_index, row_id)
            generic = set(generic_texts(inputs.generic[row_id]))
            if not generic:
                generic = set(generic_texts(inputs.surface[row_id]))
            pinyin = tuple(str(x) for x in inputs.val[row_id]["pinyin_segments"])
            raw_personal = ranked_personal_only_targets(visible=visible, generic=generic)

            compatible_targets: list[str] = []
            for target in raw_personal:
                if compatible(backend, target, pinyin):
                    compatible_targets.append(target)
                if len(compatible_targets) >= 10:
                    break
            k10 = tuple(compatible_targets[:10])
            reconstructed_k5 = k10[:5]

            if reconstructed_k5 != frozen_k5:
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "row_id": row_id,
                            "frozen_k5": list(frozen_k5),
                            "reconstructed_k5": list(reconstructed_k5),
                            "raw_personal_prefix": list(raw_personal[:15]),
                        }
                    )
            else:
                frozen_k5_match += 1

            gold = str(inputs.val[row_id].get("gold", inputs.val[row_id].get("target", "")))
            generic_missing_here = gold not in generic
            if generic_missing_here:
                generic_missing += 1
            if gold in frozen_k5:
                k5_gold_in += 1
                if generic_missing_here:
                    generic_missing_k5_gold += 1
            if gold in k10:
                k10_gold_in += 1
                if generic_missing_here:
                    generic_missing_k10_gold += 1

            candidate_count_k10[len(k10)] += 1
            if k10:
                k10_eligible += 1
                k10_pairs += len(k10)

            row_out = {
                "schema_version": 1,
                "row_id": row_id,
                "author": str(inputs.val[row_id]["author"]),
                "chronological_position": int(inputs.val[row_id]["chronological_position"]),
                "pinyin_segments": list(pinyin),
                "personal_k5": list(frozen_k5),
                "personal_k10": list(k10),
                "visible_same_pinyin_history": len(visible),
                "raw_personal_only_count": len(raw_personal),
                "candidate_selection_used_gold": False,
            }
            rows[row_id] = row_out
            sink.write(json.dumps(row_out, ensure_ascii=False, separators=(",", ":")) + "\n")

            if number % args.progress_every == 0 or number == len(inputs.surface):
                print(f"SURFACE: {number}/{len(inputs.surface)}", flush=True)

    del backend
    gc.collect()

    if mismatch_examples:
        write_json(
            audit_path,
            {
                "status": "FAILED_K5_RECONSTRUCTION",
                "mismatch_count_at_least": len(mismatch_examples),
                "examples": mismatch_examples,
                "candidate_selection_used_gold": False,
                "dev3000_used": False,
                "test_used": False,
            },
        )
        raise RuntimeError(
            "Exploratory K10 construction failed to reproduce the frozen K5 prefix. "
            f"See {audit_path}. No scorer comparison should be interpreted."
        )

    audit = {
        "schema_version": 1,
        "status": "complete",
        "rows": len(rows),
        "frozen_k5_exact_prefix_matches": frozen_k5_match,
        "frozen_k5_exact_prefix_match_rate": frozen_k5_match / len(rows),
        "k10_eligible_rows": k10_eligible,
        "k10_candidate_pairs": k10_pairs,
        "k10_candidate_count_distribution": dict(sorted(candidate_count_k10.items())),
        "gold_in_personal_k5_all_rows": k5_gold_in,
        "gold_in_personal_k10_all_rows": k10_gold_in,
        "k10_incremental_gold_availability_all_rows": k10_gold_in - k5_gold_in,
        "generic_missing_rows": generic_missing,
        "generic_missing_gold_in_personal_k5": generic_missing_k5_gold,
        "generic_missing_gold_in_personal_k10": generic_missing_k10_gold,
        "generic_missing_incremental_recovery_k10_vs_k5": generic_missing_k10_gold - generic_missing_k5_gold,
        "candidate_surface_k5_k10_sha256": sha256_file(surface_path),
        "candidate_selection_used_gold": False,
        "compatibility_source": "Frozen PinyinGPT tokenizer + allowed_token_ids; CPU only; no inference",
        "ordering": "legal same-Pinyin historical target count descending, target lexical tie-break; Generic Top10 excluded; compatibility filtered",
        "history_semantics": "strict prior same-author H5000 before exact-Pinyin filtering; earlier Train-Val rolls into later Train-Val history",
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(audit_path, audit)
    print("\n=== K5/K10 SURFACE AUDIT COMPLETE ===")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return rows


def load_surface_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out = index_rows(rows, label="exploratory surface")
    if len(out) != EXPECTED_SURFACE_ROWS:
        raise RuntimeError(f"Unexpected exploratory surface rows: {len(out)}")
    return out

# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def suffix_matches(a: str, b: str, n: int) -> bool:
    if n <= 0:
        return True
    return len(a) >= n and len(b) >= n and a[-n:] == b[-n:]


def longest_suffix_match(a: str, b: str, max_n: int) -> int:
    upper = min(max_n, len(a), len(b))
    for n in range(upper, 0, -1):
        if a[-n:] == b[-n:]:
            return n
    return 0


def recency_weight(age: int, tau: float) -> float:
    return math.exp(-float(age) / float(tau))


def frequency_support(candidates: Sequence[str], visible: Sequence[VisibleHistory]) -> list[float]:
    counts = Counter(item.record.target for item in visible)
    return normalize([math.log1p(counts.get(c, 0)) for c in candidates])


def hard_backoff_recency(
    *, candidates: Sequence[str], query_context: str, visible: Sequence[VisibleHistory], max_n: int, tau: float
) -> tuple[list[float], dict[str, Any]]:
    candidate_set = set(candidates)
    effective_n = 0
    for n in range(max_n, 0, -1):
        if any(
            item.record.target in candidate_set
            and suffix_matches(query_context, item.record.context, n)
            for item in visible
        ):
            effective_n = n
            break
    raw = {c: 0.0 for c in candidates}
    matched = 0
    for item in visible:
        if item.record.target not in raw:
            continue
        if not suffix_matches(query_context, item.record.context, effective_n):
            continue
        matched += 1
        raw[item.record.target] += recency_weight(item.age, tau)
    return normalize([raw[c] for c in candidates]), {
        "effective_n": effective_n,
        "matched_history_rows": matched,
    }


def soft_suffix_recency(
    *, candidates: Sequence[str], query_context: str, visible: Sequence[VisibleHistory], max_n: int, beta: float, tau: float
) -> tuple[list[float], dict[str, Any]]:
    raw = {c: 0.0 for c in candidates}
    match_lengths: Counter[int] = Counter()
    used = 0
    for item in visible:
        target = item.record.target
        if target not in raw:
            continue
        m = longest_suffix_match(query_context, item.record.context, max_n)
        match_lengths[m] += 1
        used += 1
        raw[target] += math.exp(beta * float(m)) * recency_weight(item.age, tau)
    return normalize([raw[c] for c in candidates]), {
        "used_history_rows": used,
        "suffix_match_length_distribution": dict(sorted(match_lengths.items())),
    }


def interpolated_ngram_recency(
    *, candidates: Sequence[str], query_context: str, visible: Sequence[VisibleHistory], max_n: int, kappa: float, tau: float
) -> tuple[list[float], dict[str, Any]]:
    # k=0 recency-weighted baseline over candidate-target history.
    base_raw = {c: 0.0 for c in candidates}
    for item in visible:
        if item.record.target in base_raw:
            base_raw[item.record.target] += recency_weight(item.age, tau)
    p = normalize([base_raw[c] for c in candidates])

    layer_info: dict[str, Any] = {}
    for n in range(1, max_n + 1):
        raw = {c: 0.0 for c in candidates}
        mass = 0.0
        matched = 0
        for item in visible:
            target = item.record.target
            if target not in raw:
                continue
            if not suffix_matches(query_context, item.record.context, n):
                continue
            w = recency_weight(item.age, tau)
            raw[target] += w
            mass += w
            matched += 1
        if mass > 0:
            p_hat = normalize([raw[c] for c in candidates])
            lam = mass / (mass + kappa)
            p = [lam * h + (1.0 - lam) * prior for h, prior in zip(p_hat, p)]
            p = normalize(p)
        else:
            lam = 0.0
        layer_info[str(n)] = {
            "matched_history_rows": matched,
            "recency_mass": mass,
            "lambda": lam,
        }
    return p, {"layers": layer_info}

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def hard_key(k: int, max_n: int, tau: float) -> str:
    return f"K{k}|HardBackoff|maxN={max_n}|tau={tau:g}"


def soft_key(k: int, max_n: int, beta: float, tau: float) -> str:
    return f"K{k}|SoftSuffix|maxN={max_n}|beta={beta:g}|tau={tau:g}"


def interp_key(k: int, max_n: int, kappa: float, tau: float) -> str:
    return f"K{k}|Interpolated|maxN={max_n}|kappa={kappa:g}|tau={tau:g}"


def expected_method_keys(args: argparse.Namespace) -> set[str]:
    keys: set[str] = set()
    for k in args.candidate_ks:
        keys.add(f"K{k}|F")
        for max_n in args.max_ns:
            for tau in args.taus:
                keys.add(hard_key(k, max_n, tau))
                for beta in args.betas:
                    keys.add(soft_key(k, max_n, beta, tau))
                for kappa in args.kappas:
                    keys.add(interp_key(k, max_n, kappa, tau))
    return keys

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def load_completed(path: Path, surface_rows: Mapping[str, Mapping[str, Any]], expected_keys: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["row_id"])
            if row_id in completed:
                raise RuntimeError(f"Duplicate cached row {row_id} at line {line_number}")
            if row_id not in surface_rows:
                raise RuntimeError(f"Stale cached row {row_id}")
            if set(row.get("methods", {})) != expected_keys:
                raise RuntimeError(f"Method grid mismatch in cached row {row_id}")
            completed[row_id] = row
    return completed


def run_score(args: argparse.Namespace, inputs: Inputs, history_index: CausalHistoryIndex, surfaces: Mapping[str, Mapping[str, Any]]) -> None:
    output_dir = args.output_root / "adaptive_ngram_top10_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "scores.jsonl"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "score_summary.json"
    keys = expected_method_keys(args)
    completed = load_completed(score_path, surfaces, keys)
    pending = [row_id for row_id in sorted(surfaces) if row_id not in completed]

    print("\n=== ADAPTIVE N-GRAM K5/K10 SCORE ===")
    print(f"rows: {len(surfaces)}")
    print(f"already completed: {len(completed)}")
    print(f"pending: {len(pending)}")
    print(f"candidate K values: {tuple(args.candidate_ks)}")
    print(f"max N grid: {tuple(args.max_ns)}")
    print(f"tau grid: {tuple(args.taus)}")
    print(f"beta grid: {tuple(args.betas)}")
    print(f"kappa grid: {tuple(args.kappas)}")
    print("device: CPU (PinyinGPT is NOT used in this phase)")

    row_latencies: list[float] = []
    method_latencies: dict[str, list[float]] = defaultdict(list)
    started_all = time.perf_counter()
    mode = "a" if score_path.exists() else "w"
    with score_path.open(mode, encoding="utf-8", newline="\n") as sink:
        for number, row_id in enumerate(pending, start=1):
            visible = query_history(inputs, history_index, row_id)
            query_context = str(inputs.generic[row_id]["model_used_context"])
            methods: dict[str, Any] = {}
            row_started = time.perf_counter()

            for k in args.candidate_ks:
                candidates = tuple(str(x) for x in surfaces[row_id][f"personal_k{k}"])
                if not candidates:
                    # Keep a complete method grid for resumability/provenance.
                    methods[f"K{k}|F"] = {"candidates": [], "support": [], "latency_ms": 0.0}
                    for max_n in args.max_ns:
                        for tau in args.taus:
                            methods[hard_key(k, max_n, tau)] = {"candidates": [], "support": [], "latency_ms": 0.0}
                            for beta in args.betas:
                                methods[soft_key(k, max_n, beta, tau)] = {"candidates": [], "support": [], "latency_ms": 0.0}
                            for kappa in args.kappas:
                                methods[interp_key(k, max_n, kappa, tau)] = {"candidates": [], "support": [], "latency_ms": 0.0}
                    continue

                key = f"K{k}|F"
                t0 = time.perf_counter()
                support = frequency_support(candidates, visible)
                elapsed = (time.perf_counter() - t0) * 1000.0
                method_latencies[key].append(elapsed)
                methods[key] = {"candidates": list(candidates), "support": support, "latency_ms": elapsed}

                for max_n in args.max_ns:
                    for tau in args.taus:
                        key = hard_key(k, max_n, tau)
                        t0 = time.perf_counter()
                        support, diag = hard_backoff_recency(
                            candidates=candidates,
                            query_context=query_context,
                            visible=visible,
                            max_n=max_n,
                            tau=tau,
                        )
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        method_latencies[key].append(elapsed)
                        methods[key] = {"candidates": list(candidates), "support": support, "latency_ms": elapsed, **diag}

                        for beta in args.betas:
                            key = soft_key(k, max_n, beta, tau)
                            t0 = time.perf_counter()
                            support, diag = soft_suffix_recency(
                                candidates=candidates,
                                query_context=query_context,
                                visible=visible,
                                max_n=max_n,
                                beta=beta,
                                tau=tau,
                            )
                            elapsed = (time.perf_counter() - t0) * 1000.0
                            method_latencies[key].append(elapsed)
                            methods[key] = {"candidates": list(candidates), "support": support, "latency_ms": elapsed, **diag}

                        for kappa in args.kappas:
                            key = interp_key(k, max_n, kappa, tau)
                            t0 = time.perf_counter()
                            support, diag = interpolated_ngram_recency(
                                candidates=candidates,
                                query_context=query_context,
                                visible=visible,
                                max_n=max_n,
                                kappa=kappa,
                                tau=tau,
                            )
                            elapsed = (time.perf_counter() - t0) * 1000.0
                            method_latencies[key].append(elapsed)
                            methods[key] = {"candidates": list(candidates), "support": support, "latency_ms": elapsed, **diag}

            row_ms = (time.perf_counter() - row_started) * 1000.0
            row_latencies.append(row_ms)
            row_out = {
                "schema_version": 1,
                "row_id": row_id,
                "author": str(inputs.val[row_id]["author"]),
                "visible_same_pinyin_history": len(visible),
                "methods": methods,
                "grid_score_ms": row_ms,
                "gold_used_for_scoring": False,
            }
            sink.write(json.dumps(row_out, ensure_ascii=False, separators=(",", ":")) + "\n")

            if number % args.progress_every == 0 or number == len(pending):
                elapsed = time.perf_counter() - started_all
                rate = number / elapsed if elapsed > 0 else 0.0
                mean_ms = statistics.fmean(row_latencies) if row_latencies else 0.0
                print(
                    f"SCORE: {number}/{len(pending)} new; overall={len(completed)+number}/{len(surfaces)}; "
                    f"rate={rate:.1f} rows/s; full_grid_mean={mean_ms:.3f} ms/row",
                    flush=True,
                )
                write_json(
                    progress_path,
                    {
                        "status": "running",
                        "completed_rows": len(completed) + number,
                        "total_rows": len(surfaces),
                        "gold_used_for_scoring": False,
                        "dev3000_used": False,
                        "test_used": False,
                    },
                )

    final = load_completed(score_path, surfaces, keys)
    if len(final) != len(surfaces):
        print(f"Partial/resumable: {len(final)}/{len(surfaces)}")
        return

    # Re-read per-method latency from durable rows so resumed runs have exact summaries.
    durable_method_latency: dict[str, list[float]] = defaultdict(list)
    durable_grid_latency: list[float] = []
    for row in final.values():
        durable_grid_latency.append(float(row["grid_score_ms"]))
        for key, value in row["methods"].items():
            if value.get("candidates"):
                durable_method_latency[key].append(float(value["latency_ms"]))

    summary = {
        "schema_version": 1,
        "status": "complete",
        "rows": len(final),
        "full_grid_latency": timing_summary(durable_grid_latency),
        "per_method_latency": {key: timing_summary(values) for key, values in sorted(durable_method_latency.items())},
        "scores_sha256": sha256_file(score_path),
        "gold_used_for_scoring": False,
        "dev3000_used": False,
        "test_used": False,
    }
    write_json(summary_path, summary)
    write_json(progress_path, {**summary, "completed_rows": len(final), "total_rows": len(final)})
    print("\n=== ADAPTIVE N-GRAM SCORE COMPLETE ===")
    print(json.dumps(summary["full_grid_latency"], indent=2))

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def metric_summary(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    by_author: dict[str, list[int]] = defaultdict(list)
    ranks: list[int] = []
    for row in rows:
        rank = row.get("rank")
        if rank is None:
            continue
        rank_i = int(rank)
        ranks.append(rank_i)
        by_author[str(row["author"])].append(rank_i)
    per_author = {
        author: sum(r == 1 for r in values) / len(values)
        for author, values in by_author.items()
        if values
    }
    n = len(rows)
    return {
        "method": method,
        "n": n,
        "rank_observed_n": len(ranks),
        "authors": len(per_author),
        "macro_author_top1": statistics.fmean(per_author.values()) if per_author else None,
        "micro_top1": sum(r == 1 for r in ranks) / n if n else None,
        "top3": sum(r <= 3 for r in ranks) / n if n else None,
        "top5": sum(r <= 5 for r in ranks) / n if n else None,
        "mrr_at_10": sum((1.0 / r) if r <= 10 else 0.0 for r in ranks) / n if n else None,
        "mean_gold_rank": statistics.fmean(ranks) if ranks else None,
        "per_author_top1": per_author,
    }


def choose_best(results: Mapping[str, Mapping[str, Any]], prefix: str) -> str:
    candidates = [(key, value) for key, value in results.items() if key.startswith(prefix)]
    if not candidates:
        raise RuntimeError(f"No methods for prefix {prefix}")
    # Primary: Macro-author Top1. Secondary: MRR@10, then lower mean latency,
    # then lexical key for deterministic selection.
    def selection_key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, str]:
        key, value = item
        macro = float(value["macro_author_top1"] or -1.0)
        mrr = float(value["mrr_at_10"] or -1.0)
        latency = float(value.get("mean_latency_ms") or 1e30)
        return (macro, mrr, -latency, key)
    return max(candidates, key=selection_key)[0]


def disagreement(a_rows: Mapping[str, Mapping[str, Any]], b_rows: Mapping[str, Mapping[str, Any]], label_a: str, label_b: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row_id in sorted(set(a_rows) & set(b_rows)):
        a = a_rows[row_id]
        b = b_rows[row_id]
        if a["winner"] == b["winner"]:
            counts["same_winner"] += 1
        elif a["winner"] == a["gold"]:
            counts[f"{label_a}_only_correct"] += 1
        elif b["winner"] == b["gold"]:
            counts[f"{label_b}_only_correct"] += 1
        else:
            counts["disagree_neither_correct"] += 1
    return dict(counts)


def run_evaluate(args: argparse.Namespace, inputs: Inputs, surfaces: Mapping[str, Mapping[str, Any]]) -> None:
    output_dir = args.output_root / "adaptive_ngram_top10_v1"
    score_path = output_dir / "scores.jsonl"
    summary_path = output_dir / "score_summary.json"
    comparison_path = output_dir / "comparison.json"
    if not score_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Score phase is incomplete")
    score_rows = index_rows(read_jsonl(score_path), label="score")
    if set(score_rows) != set(surfaces):
        raise RuntimeError("Score/surface row IDs differ")
    score_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    method_latency = score_summary["per_method_latency"]

    all_method_keys = sorted(expected_method_keys(args))
    metrics_by_k: dict[str, Any] = {}
    selected_by_k: dict[str, Any] = {}
    row_eval_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for k in args.candidate_ks:
        k_label = f"K{k}"
        candidate_field = f"personal_k{k}"
        method_keys = [key for key in all_method_keys if key.startswith(f"K{k}|")]
        eval_by_method: dict[str, list[dict[str, Any]]] = {key: [] for key in method_keys}
        shared_k5_by_method: dict[str, list[dict[str, Any]]] = {key: [] for key in method_keys}
        row_eval_cache[k_label] = {}

        for row_id in sorted(surfaces):
            candidates = tuple(str(x) for x in surfaces[row_id][candidate_field])
            gold = str(inputs.val[row_id].get("gold", inputs.val[row_id].get("target", "")))
            author = str(inputs.val[row_id]["author"])
            if len(candidates) < 2 or gold not in candidates:
                continue
            frozen_k5 = tuple(str(x) for x in surfaces[row_id]["personal_k5"])
            shared_k5_eligible = len(frozen_k5) >= 2 and gold in frozen_k5

            for key in method_keys:
                method = score_rows[row_id]["methods"][key]
                method_candidates = tuple(str(x) for x in method["candidates"])
                if method_candidates != candidates:
                    raise RuntimeError(f"Candidate mismatch at {row_id} {key}")
                ranking = rank_candidates(candidates, method["support"])
                rank = gold_rank(ranking, gold)
                record = {
                    "row_id": row_id,
                    "author": author,
                    "gold": gold,
                    "rank": rank,
                    "winner": ranking[0] if ranking else None,
                }
                eval_by_method[key].append(record)
                if shared_k5_eligible:
                    shared_k5_by_method[key].append(record)

        metrics: dict[str, Any] = {}
        shared_metrics: dict[str, Any] = {}
        for key in method_keys:
            value = metric_summary(eval_by_method[key], key)
            latency = method_latency.get(key, {})
            value["mean_latency_ms"] = latency.get("mean_ms")
            value["p95_latency_ms"] = latency.get("p95_ms")
            metrics[key] = value
            shared = metric_summary(shared_k5_by_method[key], key)
            shared["mean_latency_ms"] = latency.get("mean_ms")
            shared["p95_latency_ms"] = latency.get("p95_ms")
            shared_metrics[key] = shared

        best_hard = choose_best(metrics, f"K{k}|HardBackoff|")
        best_soft = choose_best(metrics, f"K{k}|SoftSuffix|")
        best_interp = choose_best(metrics, f"K{k}|Interpolated|")
        frequency_key = f"K{k}|F"
        selected_keys = [frequency_key, best_hard, best_soft, best_interp]

        # Row-level cache only for selected methods, enabling transparent disagreement analysis.
        selected_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for key in selected_keys:
            selected_rows[key] = {str(row["row_id"]): row for row in eval_by_method[key]}
        row_eval_cache[k_label] = selected_rows

        metrics_by_k[k_label] = {
            "k_specific_gold_in_personal_and_k_ge_2": metrics,
            "shared_frozen_k5_gold_in_personal_and_k_ge_2": shared_metrics,
        }
        selected_by_k[k_label] = {
            "frequency": frequency_key,
            "best_hard_backoff": best_hard,
            "best_soft_suffix": best_soft,
            "best_interpolated": best_interp,
            "selected_metrics_k_specific": {key: metrics[key] for key in selected_keys},
            "selected_metrics_shared_k5": {key: shared_metrics[key] for key in selected_keys},
            "disagreement_vs_frequency": {
                best_hard: disagreement(selected_rows[best_hard], selected_rows[frequency_key], "hard", "frequency"),
                best_soft: disagreement(selected_rows[best_soft], selected_rows[frequency_key], "soft", "frequency"),
                best_interp: disagreement(selected_rows[best_interp], selected_rows[frequency_key], "interpolated", "frequency"),
            },
            "pairwise_context_disagreement": {
                "hard_vs_soft": disagreement(selected_rows[best_hard], selected_rows[best_soft], "hard", "soft"),
                "hard_vs_interpolated": disagreement(selected_rows[best_hard], selected_rows[best_interp], "hard", "interpolated"),
                "soft_vs_interpolated": disagreement(selected_rows[best_soft], selected_rows[best_interp], "soft", "interpolated"),
            },
        }

    # Availability / K10 expansion diagnostics.
    availability: dict[str, Any] = {}
    counts = {
        "all_rows": len(surfaces),
        "generic_missing": 0,
        "gold_in_k5": 0,
        "gold_in_k10": 0,
        "gold_newly_in_k10_not_k5": 0,
        "generic_missing_gold_in_k5": 0,
        "generic_missing_gold_in_k10": 0,
        "generic_missing_gold_newly_in_k10_not_k5": 0,
    }
    for row_id in sorted(surfaces):
        gold = str(inputs.val[row_id].get("gold", inputs.val[row_id].get("target", "")))
        generic = set(generic_texts(inputs.generic[row_id])) or set(generic_texts(inputs.surface[row_id]))
        k5 = set(str(x) for x in surfaces[row_id]["personal_k5"])
        k10 = set(str(x) for x in surfaces[row_id]["personal_k10"])
        miss = gold not in generic
        in5 = gold in k5
        in10 = gold in k10
        if miss:
            counts["generic_missing"] += 1
        if in5:
            counts["gold_in_k5"] += 1
        if in10:
            counts["gold_in_k10"] += 1
        if in10 and not in5:
            counts["gold_newly_in_k10_not_k5"] += 1
        if miss and in5:
            counts["generic_missing_gold_in_k5"] += 1
        if miss and in10:
            counts["generic_missing_gold_in_k10"] += 1
        if miss and in10 and not in5:
            counts["generic_missing_gold_newly_in_k10_not_k5"] += 1
    availability.update(counts)
    availability["generic_missing_recoverable_rate_k5"] = counts["generic_missing_gold_in_k5"] / counts["generic_missing"] if counts["generic_missing"] else None
    availability["generic_missing_recoverable_rate_k10"] = counts["generic_missing_gold_in_k10"] / counts["generic_missing"] if counts["generic_missing"] else None

    comparison = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_candidate_scoring_adaptive_ngram_top10_v1",
        "method_definitions": {
            "HardBackoffNGramRecency": "largest exact suffix order <= maxN with candidate-target evidence; recency-weighted counts at that one order",
            "SoftSuffixRecency": "all candidate-target history retained; exp(beta * capped longest suffix match) * exp(-age/tau)",
            "InterpolatedNGramRecency": "recency-weighted distributions at k=0..maxN recursively interpolated with lambda_k=mass_k/(mass_k+kappa)",
            "F": "normalized log(1+count) on legal same-Pinyin history",
        },
        "selection_population": "Train-Val Gold-in-current-Personal-K and K>=2; diagnostic selection/evaluation on same Train-Val population",
        "selection_metric": "Macro-author Top1; tie-break MRR@10, then lower online latency",
        "candidate_surface": {
            "K5": "frozen compatible Personal K5",
            "K10": "exploratory extension reconstructed without Gold; must reproduce frozen K5 prefix exactly",
        },
        "availability": availability,
        "metrics_by_k": metrics_by_k,
        "selected_by_k": selected_by_k,
        "provenance": {
            "fit_sha256": EXPECTED_FIT_SHA256,
            "val_sha256": EXPECTED_VAL_SHA256,
            "frozen_candidate_surface_sha256": EXPECTED_SURFACE_SHA256,
            "generic_predictions_sha256": EXPECTED_GENERIC_SHA256,
            "scores_sha256": sha256_file(score_path),
            "exploratory_surface_sha256": sha256_file(output_dir / "candidate_surface_k5_k10.jsonl"),
            "history_budget": HISTORY_BUDGET,
            "candidate_selection_used_gold": False,
            "gold_used_for_scoring": False,
            "gold_used_for_final_train_val_selection_evaluation": True,
            "dev3000_used": False,
            "test_used": False,
        },
    }
    write_json(comparison_path, comparison)

    print("\n=== ADAPTIVE N-GRAM K5/K10 COMPARISON COMPLETE ===")
    print("availability:")
    print(json.dumps(availability, indent=2))
    for k_label, selected in selected_by_k.items():
        print(f"\n{k_label} selected methods:")
        for label in ("frequency", "best_hard_backoff", "best_soft_suffix", "best_interpolated"):
            key = selected[label]
            m = selected["selected_metrics_k_specific"][key]
            print(
                f"  {label:24s} {key:55s} "
                f"MacroTop1={m['macro_author_top1']:.6f} "
                f"MicroTop1={m['micro_top1']:.6f} Top3={m['top3']:.6f} "
                f"MRR10={m['mrr_at_10']:.6f} mean_ms={m['mean_latency_ms']:.4f}"
            )
    print(f"\nsaved: {comparison_path}")
    print("Gold used for scoring/candidate construction: false")
    print("Gold used for Train-Val diagnostic selection/evaluation only: true")
    print("Dev3000 used: false")
    print("Test used: false")

# ---------------------------------------------------------------------------
# Audit / orchestration
# ---------------------------------------------------------------------------


def normalize_args(args: argparse.Namespace) -> None:
    args.candidate_ks = tuple(sorted(set(int(v) for v in args.candidate_ks)))
    args.max_ns = tuple(sorted(set(int(v) for v in args.max_ns)))
    args.taus = tuple(sorted(set(float(v) for v in args.taus)))
    args.betas = tuple(sorted(set(float(v) for v in args.betas)))
    args.kappas = tuple(sorted(set(float(v) for v in args.kappas)))
    if args.candidate_ks != (5, 10):
        raise ValueError("This version intentionally supports exactly --candidate-ks 5 10")
    if any(v <= 0 for v in args.max_ns):
        raise ValueError("max N must be positive")
    if any(v <= 0 for v in args.taus + args.betas + args.kappas):
        raise ValueError("tau, beta and kappa values must be positive")


def run_audit(args: argparse.Namespace, inputs: Inputs) -> None:
    print("\n=== ADAPTIVE N-GRAM TOP10 PRE-AUDIT ===")
    print(f"Train-Fit rows: {len(inputs.fit_rows)}")
    print(f"Train-Val rows: {len(inputs.val_rows)}")
    print(f"frozen K5 eligible rows: {sum(bool(candidate_texts_top5(r)) for r in inputs.surface_rows)}")
    print(f"frozen K5 pairs: {sum(len(candidate_texts_top5(r)) for r in inputs.surface_rows)}")
    print(f"candidate K values: {args.candidate_ks}")
    print(f"max N: {args.max_ns}")
    print(f"tau: {args.taus}")
    print(f"beta: {args.betas}")
    print(f"kappa: {args.kappas}")
    print("K10 will be accepted only if its first five compatible candidates exactly reproduce frozen K5 on every row.")
    print("H5000-before-Pinyin rolling causal history: true")
    print("Gold used for candidate construction/scoring: false")
    print("Dev3000 used: false")
    print("Test used: false")


def get_or_build_surfaces(args: argparse.Namespace, inputs: Inputs, history_index: CausalHistoryIndex, *, rebuild: bool) -> dict[str, dict[str, Any]]:
    surface_path = args.output_root / "adaptive_ngram_top10_v1" / "candidate_surface_k5_k10.jsonl"
    audit_path = args.output_root / "adaptive_ngram_top10_v1" / "surface_audit.json"
    if not rebuild and surface_path.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "complete" and audit.get("frozen_k5_exact_prefix_match_rate") == 1.0:
            print(f"Reusing audited K5/K10 surface: {surface_path}")
            return load_surface_rows(surface_path)
    return build_surfaces(args, inputs, history_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "audit", "surface", "score", "evaluate"), default="all")
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--candidate-surface", type=Path, required=True)
    parser.add_argument("--generic-predictions", type=Path, required=True)
    parser.add_argument("--pinyin-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-ks", type=int, nargs="+", default=list(DEFAULT_CANDIDATE_KS))
    parser.add_argument("--max-ns", type=int, nargs="+", default=list(DEFAULT_MAX_NS))
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    parser.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    parser.add_argument("--kappas", type=float, nargs="+", default=list(DEFAULT_KAPPAS))
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--rebuild-surface", action="store_true")
    args = parser.parse_args()
    normalize_args(args)

    inputs = load_and_verify(args)
    history_index = build_history_index(inputs)

    if args.phase in ("all", "audit"):
        run_audit(args, inputs)
        if args.phase == "audit":
            return

    if args.phase in ("all", "surface"):
        surfaces = get_or_build_surfaces(args, inputs, history_index, rebuild=args.rebuild_surface)
        if args.phase == "surface":
            return
    else:
        surfaces = get_or_build_surfaces(args, inputs, history_index, rebuild=False)

    if args.phase in ("all", "score"):
        run_score(args, inputs, history_index, surfaces)
        if args.phase == "score":
            return

    if args.phase in ("all", "evaluate"):
        run_evaluate(args, inputs, surfaces)


if __name__ == "__main__":
    main()
