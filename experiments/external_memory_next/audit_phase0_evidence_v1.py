"""Read-only Full+Short Train-Fit/Train-Val evidence audit.

Historical inputs are hash-checked. Dev3000 and Test are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED = {
    "fit": (144526, "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"),
    "val": (34416, "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"),
    "generic": (34416, "cf4ae382fa23e5ec1154bf28320d13ac1d6ca9600e9dcf8a6aa599600bc28eab"),
    "stage1": (34416, "e311261cb0c5ea252ce593fdaa43ff87826d19eef440f39e2efc67ddc4310b13"),
    "stage2": (34416, "d413d02650a759c0a759e1845212e68b2d948c1a1d20fc88abfd89ea7973bc64"),
    "predictions": (34416, "f3e902e5a9e7d25e62799b9abb719026c336381eacc42999d1e7edccf2731b22"),
    "config": (None, "3dc3fb908aeeaa853526ad71cf85de7400f47d261ed7c09acdd8197446f5fa3d"),
}

N_BINS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
          ("6-10", 6, 10), ("11-20", 11, 20), ("21-50", 21, 50),
          ("51-100", 51, 100), (">100", 101, None))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source_split", "")).lower() == "test" or bool(row.get("used_test", False)):
                raise RuntimeError(f"Test row in {path}:{number}")
            yield row


def quantiles(values: Sequence[float | int]) -> dict[str, float | None]:
    names = ("min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean")
    if not values:
        return dict.fromkeys(names)
    ordered = sorted(float(value) for value in values)

    def at(q: float) -> float:
        position = q * (len(ordered) - 1)
        left, right = math.floor(position), math.ceil(position)
        return ordered[left] if left == right else ordered[left] * (right - position) + ordered[right] * (position - left)

    return {"min": ordered[0], "p10": at(.1), "p25": at(.25), "p50": at(.5),
            "p75": at(.75), "p90": at(.9), "p95": at(.95), "p99": at(.99),
            "max": ordered[-1], "mean": statistics.fmean(ordered)}


def n_bucket(value: int) -> str:
    for name, lower, upper in N_BINS:
        if value >= lower and (upper is None or value <= upper):
            return name
    raise AssertionError(value)


def score_margin(values: Mapping[str, Any]) -> float:
    scores = sorted((float(value) for value in values.values()), reverse=True)
    return 0.0 if not scores else scores[0] if len(scores) == 1 else scores[0] - scores[1]


def rank_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    by_author: dict[str, list[int | None]] = defaultdict(list)
    ranks: list[int | None] = []
    for row in rows:
        rank = None if row.get(key) is None else int(row[key])
        ranks.append(rank)
        by_author[str(row["author"])].append(rank)

    def top(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    per_author = {author: top(values, 1) for author, values in sorted(by_author.items())}
    return {"n": len(ranks), "macro_author_top1": statistics.fmean(per_author.values()),
            "micro_top1": top(ranks, 1), "top3": top(ranks, 3), "top5": top(ranks, 5),
            "mrr_at_10": sum(0 if rank is None else 1 / rank for rank in ranks) / len(ranks),
            "missing_at_10": sum(rank is None for rank in ranks) / len(ranks),
            "per_author_top1": per_author}


def verify(label: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != EXPECTED[label][1]:
        raise RuntimeError(f"{label} SHA mismatch: {actual}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def scan_split(path: Path, partition: str, collect_prior: bool = False) -> tuple[dict[str, Any], Counter, Counter]:
    authors: Counter[str] = Counter()
    counts: list[int] = []
    distinct: list[int] = []
    bins: Counter[str] = Counter()
    pinyins: set[str] = set()
    prior: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for row in iter_jsonl(path):
        if row.get("standardized_partition") != partition:
            raise RuntimeError(f"Wrong partition in {path}")
        author = str(row["author"])
        pinyin = " ".join(map(str, row["pinyin_segments"]))
        n = int(row["same_pinyin_history_count"])
        authors[author] += 1
        counts.append(n)
        distinct.append(int(row["distinct_history_targets"]))
        bins[n_bucket(n)] += 1
        pinyins.add(pinyin)
        if collect_prior:
            prior[(author, pinyin, str(row["target"]))] += 1
            totals[(author, pinyin)] += 1
    label = "fit" if partition == "train_fit" else "val"
    if len(counts) != EXPECTED[label][0]:
        raise RuntimeError(f"{partition} count changed: {len(counts)}")
    return ({"rows": len(counts), "per_author": dict(sorted(authors.items())),
             "same_pinyin_history_count": quantiles(counts),
             "same_pinyin_history_bins": {name: bins[name] for name, _, _ in N_BINS},
             "distinct_history_targets": quantiles(distinct), "unique_pinyin": len(pinyins),
             "history_available_n": sum(value > 0 for value in counts)}, prior, totals)


def other_author_prior(author: str, pinyin: str, candidate: str, prior: Mapping, totals: Mapping, authors: Sequence[str]) -> tuple[int, int, float | None]:
    count = sum(prior.get((other, pinyin, candidate), 0) for other in authors if other != author)
    total = sum(totals.get((other, pinyin), 0) for other in authors if other != author)
    return count, total, count / total if total else None


def all_author_prior(pinyin: str, candidate: str, prior: Mapping, totals: Mapping, authors: Sequence[str]) -> tuple[int, int, float | None]:
    count = sum(prior.get((author, pinyin, candidate), 0) for author in authors)
    total = sum(totals.get((author, pinyin), 0) for author in authors)
    return count, total, count / total if total else None


def as_float_map(value: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(item) for key, item in value.items()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "val", "generic", "stage1", "stage2", "predictions", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in EXPECTED}
    provenance = {name: verify(name, path) for name, path in paths.items()}

    fit, prior, prior_totals = scan_split(args.fit, "train_fit", True)
    val, _, _ = scan_split(args.val, "train_val")
    authors = tuple(fit["per_author"])
    val_rows = {str(row["row_id"]): row for row in iter_jsonl(args.val)}

    generic_counts = [len(row.get("top10_candidates", [])) for row in iter_jsonl(args.generic)]
    if len(generic_counts) != EXPECTED["generic"][0]:
        raise RuntimeError("Generic count changed")

    features: dict[str, dict[str, Any]] = {}
    choice_values: list[float] = []
    p_ng_values: list[float] = []
    entropy_values: list[float] = []
    personal_counts: list[int] = []
    all_prior_values: list[float] = []
    other_prior_values: list[float] = []
    all_prior_seen = other_prior_seen = prior_total = 0
    all_prior_rows_seen = other_prior_rows_seen = 0
    injected = missing = recoverable = 0
    author_missing: Counter[str] = Counter()
    author_recoverable: Counter[str] = Counter()
    for row in iter_jsonl(args.stage1):
        row_id = str(row["row_id"])
        if row_id in features or row_id not in val_rows:
            raise RuntimeError(f"Stage1 identity mismatch: {row_id}")
        source = val_rows[row_id]
        if int(source["same_pinyin_history_count"]) != int(row["same_pinyin_history_count"]):
            raise RuntimeError(f"History count mismatch: {row_id}")
        features[row_id] = row
        personal = list(map(str, row["personal_k5"]))
        choice = as_float_map(row["choice_share"])
        p_ng = as_float_map(row["p_ng"])
        if set(personal) != set(choice) or set(personal) != set(p_ng):
            raise RuntimeError(f"Personal key mismatch: {row_id}")
        personal_counts.append(len(personal))
        choice_values.extend(choice.values())
        p_ng_values.extend(p_ng.values())
        entropy_values.append(float(row["entropy_concentration"]))
        injected += bool(personal)
        author = str(row["author"])
        if bool(row["generic_missing"]):
            missing += 1
            author_missing[author] += 1
        if bool(row["generic_missing"]) and bool(row["gold_in_personal_k5"]):
            recoverable += 1
            author_recoverable[author] += 1
        pinyin = " ".join(map(str, source["pinyin_segments"]))
        any_all_seen = False
        any_other_seen = False
        for candidate in personal:
            all_count, _, all_value = all_author_prior(pinyin, candidate, prior, prior_totals, authors)
            other_count, _, other_value = other_author_prior(author, pinyin, candidate, prior, prior_totals, authors)
            prior_total += 1
            all_prior_seen += all_count > 0
            other_prior_seen += other_count > 0
            any_all_seen |= all_count > 0
            any_other_seen |= other_count > 0
            if all_value is not None:
                all_prior_values.append(all_value)
            if other_value is not None:
                other_prior_values.append(other_value)
        all_prior_rows_seen += any_all_seen
        other_prior_rows_seen += any_other_seen
    if len(features) != EXPECTED["stage1"][0]:
        raise RuntimeError("Stage1 count changed")

    source_counts: Counter[str] = Counter()
    stage1_counts: list[int] = []
    n_effective: list[int] = []
    n_matched: list[int] = []
    n_margins: list[float] = []
    b_margins: list[float] = []
    b_counts: list[int] = []
    support_n = 0
    for row in iter_jsonl(args.stage2):
        support_n += 1
        row_id = str(row["row_id"])
        if row_id not in features:
            raise RuntimeError(f"Stage2 identity mismatch: {row_id}")
        candidates = row["retuned_stage1_candidates"]
        names = {str(item["candidate"]) for item in candidates}
        ngram = as_float_map(row["retuned_ngram_support"])
        bge = as_float_map(row["retuned_bge_support"])
        histories = {str(key): int(value) for key, value in row["bge_history_counts"].items()}
        if names != set(ngram) or names != set(bge) or names != set(histories):
            raise RuntimeError(f"Support key mismatch: {row_id}")
        stage1_counts.append(len(candidates))
        source_counts.update(str(item["source"]) for item in candidates)
        n_effective.append(int(row["ngram_effective_n"]))
        n_matched.append(int(row["ngram_matched_history_rows"]))
        n_margins.append(score_margin(ngram))
        b_margins.append(score_margin(bge))
        b_counts.append(sum(histories.values()))
    if support_n != EXPECTED["stage2"][0]:
        raise RuntimeError("Stage2 count changed")

    predictions = list(iter_jsonl(args.predictions))
    if len(predictions) != EXPECTED["predictions"][0]:
        raise RuntimeError("Prediction count changed")
    baseline = rank_metrics(predictions, "RetunedFinal_rank")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_metrics = config["selected_train_val_metrics"]
    aliases = {"missing_at_10": "missing10"}
    for key in ("macro_author_top1", "micro_top1", "top3", "top5", "mrr_at_10", "missing_at_10"):
        if not math.isclose(float(baseline[key]), float(expected_metrics[aliases.get(key, key)]), abs_tol=1e-15, rel_tol=0):
            raise RuntimeError(f"Baseline mismatch: {key}")
    if config.get("used_test") is not False or config.get("used_dev3000_for_selection") is not False:
        raise RuntimeError("Selection boundary changed")

    audit = {
        "schema_version": 1, "status": "complete", "fit": fit, "val": val,
        "candidate_surface": {"generic_candidate_count": quantiles(generic_counts),
            "retuned_stage1_candidate_count": quantiles(stage1_counts),
            "personal_k5_count": quantiles(personal_counts),
            "source_composition": dict(sorted(source_counts.items())), "rows_with_personal_injection": injected},
        "choice_share": {"candidate_values": quantiles(choice_values),
            "entropy_concentration": quantiles(entropy_values), "candidate_count": len(choice_values)},
        "personal_support": {"p_ng_candidate_values": quantiles(p_ng_values),
            "ngram_effective_order": dict(sorted(Counter(n_effective).items())),
            "ngram_matched_history_rows": quantiles(n_matched), "ngram_top_margin": quantiles(n_margins),
            "bge_history_count_total": quantiles(b_counts), "bge_top_margin": quantiles(b_margins)},
        "population_prior_feasibility": {
            "personal_candidates": prior_total, "rows_with_personal_k5": injected,
            "all_author_train_fit": {"definition": "all-author Train-Fit P(target|segmented Pinyin)",
                "candidate_seen_n": all_prior_seen,
                "candidate_seen_rate": all_prior_seen / prior_total if prior_total else None,
                "rows_with_any_seen_candidate": all_prior_rows_seen,
                "candidate_prior_values_when_pinyin_seen": quantiles(all_prior_values),
                "unseen_policy_required": all_prior_seen < prior_total},
            "other_author_train_fit": {"definition": "other-author Train-Fit P(target|segmented Pinyin)",
                "candidate_seen_n": other_prior_seen,
                "candidate_seen_rate": other_prior_seen / prior_total if prior_total else None,
                "rows_with_any_seen_candidate": other_prior_rows_seen,
                "candidate_prior_values_when_pinyin_seen": quantiles(other_prior_values),
                "unseen_policy_required": other_prior_seen < prior_total}},
        "missing_and_recovery": {"generic_missing_n": missing, "generic_missing_rate": missing / len(features),
            "generic_missing_gold_in_personal_k5_n": recoverable,
            "recoverable_rate_within_missing": recoverable / missing if missing else None,
            "per_author_missing": dict(sorted(author_missing.items())),
            "per_author_recoverable": dict(sorted(author_recoverable.items()))},
        "baseline_reproduction_gate": {"selected_stage1": config["selected_stage1"],
            "selected_stage2": config["selected_stage2"], "metrics": baseline, "exact_match": True},
        "learned_fusion_readiness": {"train_val_candidate_features_complete": True,
            "train_fit_generic_candidate_surface_found": False,
            "conclusion": "Do not fit a learned ranker until a frozen Train-Fit Generic surface and causal feature table are generated."},
        "provenance": provenance, "used_dev3000": False, "used_test": False,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = args.output_root / "audit.json"
    write_json(result, audit)
    write_json(args.output_root / "artifact_checksums.json",
               {"runner": sha256_file(Path(__file__)), "audit.json": sha256_file(result),
                "used_dev3000": False, "used_test": False})
    print(json.dumps({"status": "complete", "output": str(result), "baseline": baseline,
                      "used_dev3000": False, "used_test": False}, indent=2))


if __name__ == "__main__":
    main()
