from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_ROWS = 94_590
EXPECTED_FAMILIES = 18_918
TOP_K_RECOVERY = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out = {}

    for row in read_jsonl(path):
        row_id = str(row["row_id"])

        if row_id in out:
            raise RuntimeError(
                f"Duplicate row_id in {path}: {row_id}"
            )

        out[row_id] = row

    return out


def find_composition_schema(
    path: Path,
) -> tuple[str, str]:
    """
    Find:
      - candidate list field
      - OOF probability field

    We intentionally consume R3's serialized OOF probabilities
    rather than re-applying the all-Val final calibrator.
    """
    candidate_list_key = None
    probability_key = None

    checked = 0

    for row in read_jsonl(path):
        checked += 1

        list_keys = [
            k
            for k, v in row.items()
            if isinstance(v, list)
            and v
            and isinstance(v[0], dict)
            and "text" in v[0]
        ]

        if not list_keys:
            if checked >= 10000:
                break
            continue

        preferred = [
            k
            for k in list_keys
            if "candidate" in k.lower()
        ]

        if len(preferred) == 1:
            candidate_list_key = preferred[0]
        elif len(list_keys) == 1:
            candidate_list_key = list_keys[0]
        else:
            raise RuntimeError(
                "Ambiguous Composition candidate list keys: "
                f"{list_keys}"
            )

        sample = row[candidate_list_key][0]

        preferred_probability_names = (
            "oof_calibrated_probability",
            "c1_oof_probability",
            "oof_probability",
            "calibrated_oof_probability",
            "composition_oof_probability",
            "calibrated_probability",
            "probability",
        )

        for key in preferred_probability_names:
            if key in sample:
                value = sample[key]

                if isinstance(
                    value,
                    (int, float),
                ):
                    probability_key = key
                    break

        if probability_key is None:
            candidates = []

            for key, value in sample.items():
                low = key.lower()

                if not isinstance(
                    value,
                    (int, float),
                ):
                    continue

                if (
                    "prob" in low
                    and 0.0
                    <= float(value)
                    <= 1.0
                ):
                    candidates.append(key)

            if len(candidates) == 1:
                probability_key = candidates[0]

        if probability_key is None:
            raise RuntimeError(
                "Could not uniquely identify Composition "
                f"OOF probability. Candidate keys={sorted(sample)}"
            )

        print(
            "COMPOSITION_CANDIDATE_LIST_KEY =",
            candidate_list_key,
        )

        print(
            "COMPOSITION_OOF_PROBABILITY_KEY =",
            probability_key,
        )

        print(
            "COMPOSITION_SAMPLE_CANDIDATE =",
            sample,
        )

        return (
            candidate_list_key,
            probability_key,
        )

    raise RuntimeError(
        "Could not find a non-empty Composition row"
    )


def generic_candidates(
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []

    for x in row["top10_distinct_candidates"]:
        result.append(
            {
                "text": str(x["text"]),
                "rank": int(x["rank"]),
                "log_probability": float(
                    x["log_probability"]
                ),
            }
        )

    return result


def latency_summary(
    values: list[float],
) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    x = sorted(values)
    n = len(x)

    def percentile(q: float) -> float:
        if n == 1:
            return x[0]

        pos = (n - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))

        if lo == hi:
            return x[lo]

        frac = pos - lo

        return (
            x[lo] * (1.0 - frac)
            + x[hi] * frac
        )

    return {
        "n": n,
        "mean": float(sum(x) / n),
        "median": float(percentile(0.50)),
        "p95": float(percentile(0.95)),
        "p99": float(percentile(0.99)),
        "max": float(x[-1]),
    }


def new_metric() -> dict[str, Any]:
    return {
        "n": 0,
        "generic_top10_hit": 0,
        "exact": {
            1: 0,
            3: 0,
            5: 0,
            10: 0,
            "any": 0,
        },
        "composition": {
            1: 0,
            3: 0,
            5: 0,
            10: 0,
            "any": 0,
        },
        "fusion": {
            1: 0,
            3: 0,
            5: 0,
            10: 0,
            "any": 0,
        },
        "final_pool_hit": 0,
    }


def update_metric(
    metric: dict[str, Any],
    *,
    gold: str,
    generic_texts: list[str],
    exact_ranked: list[dict[str, Any]],
    comp_ranked: list[dict[str, Any]],
    fused_ranked: list[dict[str, Any]],
) -> None:
    metric["n"] += 1

    generic_hit = gold in generic_texts

    if generic_hit:
        metric["generic_top10_hit"] += 1

    branches = (
        ("exact", exact_ranked),
        ("composition", comp_ranked),
        ("fusion", fused_ranked),
    )

    for name, ranked in branches:
        texts = [
            str(x["text"])
            for x in ranked
        ]

        if gold in texts:
            metric[name]["any"] += 1

        for k in (1, 3, 5, 10):
            if gold in texts[:k]:
                metric[name][k] += 1

    if (
        generic_hit
        or gold
        in [
            str(x["text"])
            for x in fused_ranked[:TOP_K_RECOVERY]
        ]
    ):
        metric["final_pool_hit"] += 1


def finalize_metric(
    metric: dict[str, Any],
) -> dict[str, Any]:
    n = int(metric["n"])

    def rate(x: int):
        return (
            float(x / n)
            if n
            else None
        )

    result = {
        "n": n,
        "generic_top10_recall": rate(
            metric["generic_top10_hit"]
        ),
        "final_pool_recall_generic10_plus_personal5": rate(
            metric["final_pool_hit"]
        ),
    }

    for branch in (
        "exact",
        "composition",
        "fusion",
    ):
        result[branch] = {
            "recall_at_1": rate(
                metric[branch][1]
            ),
            "recall_at_3": rate(
                metric[branch][3]
            ),
            "recall_at_5": rate(
                metric[branch][5]
            ),
            "recall_at_10": rate(
                metric[branch][10]
            ),
            "recall_any": rate(
                metric[branch]["any"]
            ),
            "hits_at_5": int(
                metric[branch][5]
            ),
            "hits_any": int(
                metric[branch]["any"]
            ),
        }

    result[
        "absolute_final_pool_recall_gain_over_generic_top10"
    ] = (
        result[
            "final_pool_recall_generic10_plus_personal5"
        ]
        - result["generic_top10_recall"]
        if n
        else None
    )

    return result


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--m0",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--composition",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--composition-summary",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--exact",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--exact-summary",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    if (
        args.output_root.exists()
        and any(args.output_root.iterdir())
    ):
        raise RuntimeError(
            "Refusing to overwrite non-empty R5 output"
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "===== M4-R5 CALIBRATED DUAL-BRANCH FUSION ====="
    )

    print(
        "Fusion: same-text max(p_exact, p_composition)"
    )

    print(
        "Recovery: remove Generic Top10 overlap, "
        "then Personal-only Top5"
    )

    print(
        "Train-Val uses branch-specific OOF probabilities"
    )

    print(
        "DEV3000=CLOSED TEST=CLOSED"
    )

    candidate_key, comp_prob_key = (
        find_composition_schema(
            args.composition
        )
    )

    if comp_prob_key != "oof_calibrated_probability":
        raise RuntimeError(
            "Train-Val R5 must use "
            "oof_calibrated_probability; got "
            + str(comp_prob_key)
        )

    print(
        "COMPOSITION_OOF_FIELD_GATE=PASS"
    )

    print()
    print("Loading M0 and Exact rows ...")

    m0_by_id = load_by_id(
        args.m0
    )

    exact_by_id = load_by_id(
        args.exact
    )

    if len(m0_by_id) != EXPECTED_ROWS:
        raise RuntimeError(
            f"M0 count mismatch: {len(m0_by_id)}"
        )

    if len(exact_by_id) != EXPECTED_ROWS:
        raise RuntimeError(
            "Exact count mismatch: "
            f"{len(exact_by_id)}"
        )

    if set(m0_by_id) != set(exact_by_id):
        raise RuntimeError(
            "M0/Exact row IDs differ"
        )

    comp_summary = json.loads(
        args.composition_summary.read_text(
            encoding="utf-8-sig"
        )
    )

    exact_summary = json.loads(
        args.exact_summary.read_text(
            encoding="utf-8-sig"
        )
    )

    overall = new_metric()
    gp = new_metric()

    by_length = {
        length: new_metric()
        for length in range(1, 6)
    }

    provenance_all = Counter()
    provenance_after_generic = Counter()
    provenance_top5 = Counter()
    gold_top5_provenance = Counter()

    winner_top5 = Counter()

    exact_generic_overlap = 0
    comp_generic_overlap = 0
    merged_generic_overlap = 0

    exact_candidate_total = 0
    comp_candidate_total = 0

    merged_unique_before_generic = 0
    merged_unique_after_generic = 0

    personal_slots_total = 0

    active_exact_queries = 0
    active_comp_queries = 0
    active_either_queries = 0
    active_both_queries = 0

    gold_exact_any = 0
    gold_comp_any = 0
    gold_either_any = 0
    gold_both_any = 0

    fusion_ms = []

    output_rows_path = (
        args.output_root
        / "fusion_rows.jsonl"
    )

    comp_row_count = 0
    family_ids = set()

    with output_rows_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as sink:

        for comp_row in read_jsonl(
            args.composition
        ):
            comp_row_count += 1

            row_id = str(
                comp_row["row_id"]
            )

            if row_id not in m0_by_id:
                raise RuntimeError(
                    f"Composition row missing from M0: {row_id}"
                )

            if row_id not in exact_by_id:
                raise RuntimeError(
                    f"Composition row missing from Exact: {row_id}"
                )

            m0 = m0_by_id[row_id]
            exact = exact_by_id[row_id]

            family_id = str(
                m0["family_id"]
            )

            family_ids.add(
                family_id
            )

            gold = str(
                m0["gold"]
            )

            if str(comp_row["gold"]) != gold:
                raise RuntimeError(
                    f"Composition gold mismatch: {row_id}"
                )

            if str(exact["gold_posthoc"]) != gold:
                raise RuntimeError(
                    f"Exact gold mismatch: {row_id}"
                )

            length = int(
                m0["multi_token_length"]
            )

            if int(
                comp_row["multi_token_length"]
            ) != length:
                raise RuntimeError(
                    f"Composition length mismatch: {row_id}"
                )

            if int(
                exact["multi_token_length"]
            ) != length:
                raise RuntimeError(
                    f"Exact length mismatch: {row_id}"
                )

            if int(
                comp_row["calibration_fold"]
            ) != int(
                exact["calibration_fold"]
            ):
                raise RuntimeError(
                    f"OOF fold mismatch: {row_id}"
                )

            generic_pruned = not bool(
                m0["gold_survived_final_beam"]
            )

            if bool(
                comp_row["generic_beam_pruned"]
            ) != generic_pruned:
                raise RuntimeError(
                    f"Composition GP mismatch: {row_id}"
                )

            if bool(
                exact["generic_beam_pruned"]
            ) != generic_pruned:
                raise RuntimeError(
                    f"Exact GP mismatch: {row_id}"
                )

            generic = generic_candidates(
                m0
            )

            generic_texts = [
                x["text"]
                for x in generic
            ]

            generic_set = set(
                generic_texts
            )

            exact_raw = []

            for cand in exact[
                "exact_candidates"
            ]:
                p = float(
                    cand[
                        "exact_oof_probability"
                    ]
                )

                if not 0.0 <= p <= 1.0:
                    raise RuntimeError(
                        "Invalid Exact probability"
                    )

                exact_raw.append(
                    {
                        "text": str(
                            cand["text"]
                        ),
                        "probability": p,
                        "raw_score": float(
                            cand[
                                "exact_raw_score"
                            ]
                        ),
                        "source": "exact",
                    }
                )

            comp_raw = []

            for cand in comp_row.get(
                candidate_key,
                [],
            ):
                p = float(
                    cand[
                        comp_prob_key
                    ]
                )

                if not 0.0 <= p <= 1.0:
                    raise RuntimeError(
                        "Invalid Composition probability"
                    )

                comp_raw.append(
                    {
                        "text": str(
                            cand["text"]
                        ),
                        "probability": p,
                        "raw_score": (
                            float(
                                cand["c1_raw_score"]
                            )
                            if "c1_raw_score"
                            in cand
                            else None
                        ),
                        "piece_count": (
                            int(
                                cand["piece_count"]
                            )
                            if "piece_count"
                            in cand
                            else None
                        ),
                        "fragmentation_ratio": (
                            float(
                                cand[
                                    "fragmentation_ratio"
                                ]
                            )
                            if "fragmentation_ratio"
                            in cand
                            else None
                        ),
                        "longest_span_ratio": (
                            float(
                                cand[
                                    "longest_span_ratio"
                                ]
                            )
                            if "longest_span_ratio"
                            in cand
                            else None
                        ),
                        "source": "composition",
                    }
                )

            exact_candidate_total += len(
                exact_raw
            )

            comp_candidate_total += len(
                comp_raw
            )

            if exact_raw:
                active_exact_queries += 1

            if comp_raw:
                active_comp_queries += 1

            if exact_raw or comp_raw:
                active_either_queries += 1

            if exact_raw and comp_raw:
                active_both_queries += 1

            exact_gold = any(
                x["text"] == gold
                for x in exact_raw
            )

            comp_gold = any(
                x["text"] == gold
                for x in comp_raw
            )

            gold_exact_any += int(
                exact_gold
            )

            gold_comp_any += int(
                comp_gold
            )

            gold_either_any += int(
                exact_gold or comp_gold
            )

            gold_both_any += int(
                exact_gold and comp_gold
            )

            started = time.perf_counter_ns()

            merged = {}

            for cand in exact_raw:
                text = cand["text"]

                item = merged.setdefault(
                    text,
                    {
                        "text": text,
                        "p_exact": None,
                        "p_composition": None,
                        "exact_raw_score": None,
                        "composition_raw_score": None,
                        "piece_count": None,
                        "fragmentation_ratio": None,
                        "longest_span_ratio": None,
                    },
                )

                p = float(
                    cand["probability"]
                )

                if (
                    item["p_exact"] is None
                    or p > item["p_exact"]
                ):
                    item["p_exact"] = p
                    item["exact_raw_score"] = (
                        cand["raw_score"]
                    )

            for cand in comp_raw:
                text = cand["text"]

                item = merged.setdefault(
                    text,
                    {
                        "text": text,
                        "p_exact": None,
                        "p_composition": None,
                        "exact_raw_score": None,
                        "composition_raw_score": None,
                        "piece_count": None,
                        "fragmentation_ratio": None,
                        "longest_span_ratio": None,
                    },
                )

                p = float(
                    cand["probability"]
                )

                if (
                    item["p_composition"]
                    is None
                    or p
                    > item["p_composition"]
                ):
                    item[
                        "p_composition"
                    ] = p

                    item[
                        "composition_raw_score"
                    ] = cand["raw_score"]

                    item[
                        "piece_count"
                    ] = cand[
                        "piece_count"
                    ]

                    item[
                        "fragmentation_ratio"
                    ] = cand[
                        "fragmentation_ratio"
                    ]

                    item[
                        "longest_span_ratio"
                    ] = cand[
                        "longest_span_ratio"
                    ]

            merged_unique_before_generic += len(
                merged
            )

            for item in merged.values():
                has_e = (
                    item["p_exact"]
                    is not None
                )

                has_c = (
                    item["p_composition"]
                    is not None
                )

                if has_e and has_c:
                    provenance = "both"
                elif has_e:
                    provenance = "exact_only"
                elif has_c:
                    provenance = (
                        "composition_only"
                    )
                else:
                    raise RuntimeError(
                        "Invalid provenance"
                    )

                item["provenance"] = provenance

                provenance_all[
                    provenance
                ] += 1

                if item["text"] in generic_set:
                    merged_generic_overlap += 1

                    if has_e:
                        exact_generic_overlap += 1

                    if has_c:
                        comp_generic_overlap += 1

            personal = [
                item
                for item in merged.values()
                if item["text"]
                not in generic_set
            ]

            merged_unique_after_generic += len(
                personal
            )

            for item in personal:
                provenance_after_generic[
                    item["provenance"]
                ] += 1

                pe = item["p_exact"]
                pc = item["p_composition"]

                if pe is None:
                    confidence = float(pc)
                    winner = "composition"

                elif pc is None:
                    confidence = float(pe)
                    winner = "exact"

                elif pe > pc:
                    confidence = float(pe)
                    winner = "exact"

                elif pc > pe:
                    confidence = float(pc)
                    winner = "composition"

                else:
                    confidence = float(pe)
                    winner = "tie"

                item[
                    "recovery_confidence"
                ] = confidence

                item[
                    "confidence_winner"
                ] = winner

            # Main frozen fusion:
            # descending max calibrated probability,
            # then text asc for deterministic ties.
            personal.sort(
                key=lambda x: (
                    -float(
                        x[
                            "recovery_confidence"
                        ]
                    ),
                    str(x["text"]),
                )
            )

            top5 = personal[
                :TOP_K_RECOVERY
            ]

            elapsed_ms = (
                time.perf_counter_ns()
                - started
            ) / 1_000_000.0

            fusion_ms.append(
                elapsed_ms
            )

            personal_slots_total += len(
                top5
            )

            for item in top5:
                provenance_top5[
                    item["provenance"]
                ] += 1

                winner_top5[
                    item[
                        "confidence_winner"
                    ]
                ] += 1

                if item["text"] == gold:
                    gold_top5_provenance[
                        item["provenance"]
                    ] += 1

            exact_ranked = sorted(
                [
                    x
                    for x in exact_raw
                    if x["text"]
                    not in generic_set
                ],
                key=lambda x: (
                    -float(
                        x["probability"]
                    ),
                    str(x["text"]),
                ),
            )

            comp_ranked = sorted(
                [
                    x
                    for x in comp_raw
                    if x["text"]
                    not in generic_set
                ],
                key=lambda x: (
                    -float(
                        x["probability"]
                    ),
                    str(x["text"]),
                ),
            )

            update_metric(
                overall,
                gold=gold,
                generic_texts=generic_texts,
                exact_ranked=exact_ranked,
                comp_ranked=comp_ranked,
                fused_ranked=personal,
            )

            update_metric(
                by_length[length],
                gold=gold,
                generic_texts=generic_texts,
                exact_ranked=exact_ranked,
                comp_ranked=comp_ranked,
                fused_ranked=personal,
            )

            if generic_pruned:
                update_metric(
                    gp,
                    gold=gold,
                    generic_texts=generic_texts,
                    exact_ranked=exact_ranked,
                    comp_ranked=comp_ranked,
                    fused_ranked=personal,
                )

            out_top5 = []

            for rank, item in enumerate(
                top5,
                start=1,
            ):
                out_top5.append(
                    {
                        "rank": rank,
                        "text": item["text"],
                        "recovery_confidence": (
                            item[
                                "recovery_confidence"
                            ]
                        ),
                        "provenance": (
                            item[
                                "provenance"
                            ]
                        ),
                        "confidence_winner": (
                            item[
                                "confidence_winner"
                            ]
                        ),
                        "p_exact": (
                            item["p_exact"]
                        ),
                        "p_composition": (
                            item[
                                "p_composition"
                            ]
                        ),
                        "exact_raw_score": (
                            item[
                                "exact_raw_score"
                            ]
                        ),
                        "composition_raw_score": (
                            item[
                                "composition_raw_score"
                            ]
                        ),
                        "piece_count": (
                            item[
                                "piece_count"
                            ]
                        ),
                        "fragmentation_ratio": (
                            item[
                                "fragmentation_ratio"
                            ]
                        ),
                        "longest_span_ratio": (
                            item[
                                "longest_span_ratio"
                            ]
                        ),
                    }
                )

            final_pool = (
                [
                    {
                        "text": x["text"],
                        "source": "generic",
                        "generic_rank": x["rank"],
                    }
                    for x in generic
                ]
                + [
                    {
                        "text": x["text"],
                        "source": (
                            "personal_recovery"
                        ),
                        "personal_rank": (
                            x["rank"]
                        ),
                        "provenance": (
                            x["provenance"]
                        ),
                        "recovery_confidence": (
                            x[
                                "recovery_confidence"
                            ]
                        ),
                    }
                    for x in out_top5
                ]
            )

            if len(
                {
                    x["text"]
                    for x in final_pool
                }
            ) != len(final_pool):
                raise RuntimeError(
                    f"Final pool duplicate: {row_id}"
                )

            if len(final_pool) > 15:
                raise RuntimeError(
                    f"Final pool >15: {row_id}"
                )

            output_row = {
                "schema_version": 1,
                "row_id": row_id,
                "family_id": family_id,
                "multi_token_length": length,
                "calibration_fold": int(
                    comp_row[
                        "calibration_fold"
                    ]
                ),
                "generic_beam_pruned": (
                    generic_pruned
                ),
                "generic_top10": generic_texts,
                "personal_recovery_top5": (
                    out_top5
                ),
                "final_pool": final_pool,
                "final_pool_size": len(
                    final_pool
                ),
                "gold_posthoc": gold,
                "gold_in_generic_top10_posthoc": (
                    gold in generic_texts
                ),
                "gold_in_personal_top5_posthoc": (
                    gold
                    in [
                        x["text"]
                        for x in out_top5
                    ]
                ),
                "gold_in_final_pool_posthoc": (
                    gold
                    in [
                        x["text"]
                        for x in final_pool
                    ]
                ),
                "used_dev3000": False,
                "used_test": False,
            }

            sink.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

            if (
                comp_row_count % 10000
                == 0
            ):
                print(
                    f"fusion "
                    f"{comp_row_count}/"
                    f"{EXPECTED_ROWS}",
                    flush=True,
                )

    if comp_row_count != EXPECTED_ROWS:
        raise RuntimeError(
            "Composition row count mismatch: "
            f"{comp_row_count}"
        )

    if len(family_ids) != EXPECTED_FAMILIES:
        raise RuntimeError(
            "Family count mismatch: "
            f"{len(family_ids)}"
        )

    overall_result = finalize_metric(
        overall
    )

    gp_result = finalize_metric(
        gp
    )

    by_length_result = {
        f"M{length}": finalize_metric(
            by_length[length]
        )
        for length in range(1, 6)
    }

    summary = {
        "schema_version": 1,
        "status": "complete",
        "experiment": (
            "m4_r5_calibrated_dual_branch_"
            "personal_recovery_fusion_v1"
        ),
        "fusion_rule": (
            "same-text max(p_exact, "
            "p_composition)"
        ),
        "recovery_slot_rule": (
            "remove Generic Top10 overlap, "
            "sort by calibrated recovery "
            "confidence descending, text ascending "
            "for deterministic ties, retain "
            "Personal-only Top5"
        ),
        "evaluation_probability_protocol": (
            "Train-Val branch-specific "
            "family-level OOF probabilities only"
        ),
        "query_count": comp_row_count,
        "family_count": len(
            family_ids
        ),
        "overall": overall_result,
        "generic_beam_pruned": (
            gp_result
        ),
        "by_multi_length": (
            by_length_result
        ),
        "candidate_structure": {
            "exact_candidate_total": (
                exact_candidate_total
            ),
            "composition_candidate_total": (
                comp_candidate_total
            ),
            "merged_unique_before_generic_removal": (
                merged_unique_before_generic
            ),
            "merged_unique_after_generic_removal": (
                merged_unique_after_generic
            ),
            "personal_top5_slots_total": (
                personal_slots_total
            ),
            "active_exact_queries": (
                active_exact_queries
            ),
            "active_composition_queries": (
                active_comp_queries
            ),
            "active_either_queries": (
                active_either_queries
            ),
            "active_both_queries": (
                active_both_queries
            ),
            "exact_generic_overlap_events": (
                exact_generic_overlap
            ),
            "composition_generic_overlap_events": (
                comp_generic_overlap
            ),
            "merged_unique_generic_overlap": (
                merged_generic_overlap
            ),
        },
        "gold_recoverability_any_rank": {
            "exact": gold_exact_any,
            "composition": gold_comp_any,
            "either": gold_either_any,
            "both": gold_both_any,
            "exact_rate": (
                gold_exact_any
                / comp_row_count
            ),
            "composition_rate": (
                gold_comp_any
                / comp_row_count
            ),
            "either_rate": (
                gold_either_any
                / comp_row_count
            ),
            "both_rate": (
                gold_both_any
                / comp_row_count
            ),
        },
        "provenance": {
            "all_merged_unique": dict(
                provenance_all
            ),
            "after_generic_removal": dict(
                provenance_after_generic
            ),
            "personal_top5_slots": dict(
                provenance_top5
            ),
            "gold_hits_in_personal_top5": dict(
                gold_top5_provenance
            ),
            "top5_confidence_winner": dict(
                winner_top5
            ),
        },
        "latency_ms_per_query": {
            "fusion_only": latency_summary(
                fusion_ms
            ),
            "composition_r3_inherited": (
                comp_summary.get(
                    "latency_ms_per_query"
                )
            ),
            "exact_calibration_inherited": (
                exact_summary.get(
                    "latency_ms_per_query"
                )
            ),
            "note": (
                "Fusion latency is directly measured. "
                "R3 Composition latency is inherited "
                "from its frozen experiment. Exact "
                "full runtime was not isolated per "
                "query in the frozen R4 scoring run, "
                "so no fabricated end-to-end total "
                "is reported here."
            ),
        },
        "hashes": {
            "m0": sha256_file(
                args.m0
            ),
            "composition_oof": (
                sha256_file(
                    args.composition
                )
            ),
            "exact_oof": sha256_file(
                args.exact
            ),
            "fusion_rows": (
                sha256_file(
                    output_rows_path
                )
            ),
            "runner": sha256_file(
                Path(__file__)
            ),
        },
        "gold_used_for_fusion_scoring": False,
        "gold_used_for_fusion_selection": False,
        "gold_used_for_posthoc_metrics_only": True,
        "used_dev3000": False,
        "used_test": False,
    }

    summary_path = (
        args.output_root
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== R5 HEADLINE =====")

    print(
        "OVERALL_GENERIC_TOP10_RECALL =",
        overall_result[
            "generic_top10_recall"
        ],
    )

    print(
        "OVERALL_PERSONAL_FUSION_R1 =",
        overall_result[
            "fusion"
        ]["recall_at_1"],
    )

    print(
        "OVERALL_PERSONAL_FUSION_R3 =",
        overall_result[
            "fusion"
        ]["recall_at_3"],
    )

    print(
        "OVERALL_PERSONAL_FUSION_R5 =",
        overall_result[
            "fusion"
        ]["recall_at_5"],
    )

    print(
        "OVERALL_PERSONAL_FUSION_R10 =",
        overall_result[
            "fusion"
        ]["recall_at_10"],
    )

    print(
        "OVERALL_FINAL_POOL_RECALL =",
        overall_result[
            "final_pool_recall_generic10_plus_personal5"
        ],
    )

    print(
        "OVERALL_FINAL_POOL_GAIN_PP =",
        100.0
        * overall_result[
            "absolute_final_pool_recall_gain_over_generic_top10"
        ],
    )

    print()
    print(
        "GP_N =",
        gp_result["n"],
    )

    print(
        "GP_EXACT_R5 =",
        gp_result[
            "exact"
        ]["recall_at_5"],
    )

    print(
        "GP_COMPOSITION_R5 =",
        gp_result[
            "composition"
        ]["recall_at_5"],
    )

    print(
        "GP_FUSION_R1 =",
        gp_result[
            "fusion"
        ]["recall_at_1"],
    )

    print(
        "GP_FUSION_R3 =",
        gp_result[
            "fusion"
        ]["recall_at_3"],
    )

    print(
        "GP_FUSION_R5 =",
        gp_result[
            "fusion"
        ]["recall_at_5"],
    )

    print(
        "GP_FUSION_R10 =",
        gp_result[
            "fusion"
        ]["recall_at_10"],
    )

    print(
        "GP_RECOVERED_TOP5_COUNT =",
        gp_result[
            "fusion"
        ]["hits_at_5"],
    )

    print()
    print(
        "TOP5_PROVENANCE =",
        dict(
            provenance_top5
        ),
    )

    print(
        "GOLD_TOP5_PROVENANCE =",
        dict(
            gold_top5_provenance
        ),
    )

    print(
        "FUSION_LATENCY_MS =",
        summary[
            "latency_ms_per_query"
        ]["fusion_only"],
    )

    print()
    print(
        "FUSION_ROWS_SHA =",
        summary[
            "hashes"
        ]["fusion_rows"],
    )

    print(
        "RUNNER_SHA =",
        summary[
            "hashes"
        ]["runner"],
    )

    print()
    print(
        "TRAIN_VAL_OOF_FUSION=PASS"
    )

    print(
        "PERSONAL_ONLY_TOP5=PASS"
    )

    print(
        "DEV3000_TEST_CLOSED=PASS"
    )

    print(
        "M4_R5_FUSION_GATE=PASS"
    )


if __name__ == "__main__":
    main()