"""Decompose Choice Share scale suppression from population-prior information."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.external_memory_next import run_choice_share_smoothing_v1 as smooth
from experiments.external_memory_next.reproduce_full_retuned_baseline_v1 import candidate_text, final_rerank


MODES = ("raw_wcs6", "raw_wcs2", "zero_prior_alpha128",
         "all_author_alpha128", "other_author_alpha128")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in smooth.EXPECTED:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in smooth.EXPECTED}
    provenance = {}
    for name, path in paths.items():
        actual = smooth.sha256_file(path)
        if actual != smooth.EXPECTED[name]:
            raise RuntimeError(f"{name} SHA mismatch: {actual}")
        provenance[name] = {"path": str(path.resolve()), "sha256": actual}
    config = json.loads(args.config.read_text(encoding="utf-8"))
    weights = {**{key: float(config["selected_stage1"][key]) for key in ("w_p", "w_cs", "w_e")},
               **{key: float(config["selected_stage2"][key]) for key in ("lambda_n", "lambda_b")}}
    all_counts, all_totals = smooth.build_prior(smooth.iter_jsonl(args.fit))
    author_counts: Counter[tuple[str, str, str]] = Counter()
    author_totals: Counter[tuple[str, str]] = Counter()
    for row in smooth.iter_jsonl(args.fit):
        author, pinyin, target = str(row["author"]), smooth.pinyin_of(row), str(row["target"])
        author_counts[(author, pinyin, target)] += 1
        author_totals[(author, pinyin)] += 1

    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    iterators = (smooth.iter_jsonl(args.val), smooth.iter_jsonl(args.stage1),
                 smooth.iter_jsonl(args.stage2), smooth.iter_jsonl(args.predictions))
    for number, group in enumerate(itertools.zip_longest(*iterators), start=1):
        if any(row is None for row in group):
            raise RuntimeError("Input row counts differ")
        val, feature, support, prediction = group
        if len({str(row["row_id"]) for row in group}) != 1:
            raise RuntimeError(f"Input row order differs at {number}")
        pinyin, author = smooth.pinyin_of(val), str(feature["author"])
        other_total = all_totals.get(pinyin, 0) - author_totals.get((author, pinyin), 0)
        other_counts = {(pinyin, candidate): all_counts.get((pinyin, candidate), 0)
                        - author_counts.get((author, pinyin, candidate), 0)
                        for candidate in map(str, feature["personal_k5"])}
        settings = {
            "raw_wcs6": (0.0, 6.0, all_counts, all_totals),
            "raw_wcs2": (0.0, 2.0, all_counts, all_totals),
            "zero_prior_alpha128": (128.0, 6.0, {}, all_totals),
            "all_author_alpha128": (128.0, 6.0, all_counts, all_totals),
            "other_author_alpha128": (128.0, 6.0, other_counts, {pinyin: other_total}),
        }
        gold = str(feature["gold"])
        for mode, (alpha, w_cs, counts, totals) in settings.items():
            mode_weights = {**weights, "w_cs": w_cs}
            stage1 = smooth.fixed_surface_stage1(feature, support, counts, totals,
                                                  pinyin, alpha, mode_weights)
            final = final_rerank(stage1, support["retuned_ngram_support"],
                                 support["retuned_bge_support"], mode_weights)
            top10 = [candidate_text(row) for row in final]
            rank = next((index for index, candidate in enumerate(top10, start=1) if candidate == gold), None)
            if mode == "raw_wcs6" and (top10 != list(map(str, prediction["RetunedFinal_top10"]))
                                        or rank != prediction.get("RetunedFinal_rank")):
                raise RuntimeError(f"Frozen baseline mismatch: {feature['row_id']}")
            by_mode[mode].append({"row_id": feature["row_id"], "author": author,
                                  "ambiguous": bool(feature["ambiguous"]),
                                  "conflict": bool(feature["conflict"]), "rank": rank})
    baseline = by_mode["raw_wcs6"]
    results = []
    for mode in MODES:
        rows = by_mode[mode]
        results.append({"mode": mode,
                        "metrics": {name: smooth.metrics([row for row in rows if name == "overall" or row[name]])
                                    for name in ("overall", "ambiguous", "conflict")},
                        "transition_from_raw_wcs6": smooth.transitions(
                            [{**row, "baseline_rank": baseline[index]["rank"]}
                             for index, row in enumerate(rows)], "baseline_rank", "rank")})
    result = {"schema_version": 1, "status": "complete",
              "experiment": "choice_share_prior_decomposition_fixed_surface_v1",
              "modes": results, "selection_performed": False,
              "provenance": provenance, "used_dev3000": False, "used_test": False}
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "result.json"
    smooth.write_json(output, result)
    smooth.write_json(args.output_root / "artifact_checksums.json", {
        "runner": smooth.sha256_file(Path(__file__)),
        "smoothing_core": smooth.sha256_file(Path(smooth.__file__)),
        "result.json": smooth.sha256_file(output), "used_dev3000": False, "used_test": False})
    print(json.dumps({row["mode"]: {"metrics": row["metrics"]["overall"],
                                         "transition": row["transition_from_raw_wcs6"]}
                      for row in results}, indent=2))


if __name__ == "__main__":
    main()
