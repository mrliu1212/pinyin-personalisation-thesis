"""Prepare causal Full RetunedFinal Train-Fit candidate/support features.

Stage ``stage1`` uses frozen Generic predictions and strictly-prior H5000
history to build the frozen RetunedFinal Top10 surface. Stage ``supports``
adds the existing NGramRecency and BGERecency evidence for that surface.
Dev3000 and Test are never accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.context_comparison import run_full_retune_final_trainval_dev_v1 as retune
from experiments.context_comparison import run_full_transfer_initial_final_v1 as base


EXPECTED_FIT_SHA256 = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
EXPECTED_BASE_NORMALIZED_SHA256 = "f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0"
EXPECTED_ROWS = 144526
EXPECTED_REVISIONS = {
    "checkpoint_revision": "76dd20dc92d8236a350fb732e99dde6fa15e2263",
    "official_code_revision": "8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1",
}
WEIGHTS = {"w_p": 2.0, "w_cs": 6.0, "w_e": 4.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source_split", "")).lower() == "test" or bool(row.get("used_test", False)):
                raise RuntimeError(f"Test row in {path}:{number}")
            yield row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def load_fit(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != EXPECTED_FIT_SHA256:
        raise RuntimeError("Train-Fit SHA changed")
    rows = list(iter_jsonl(path))
    if len(rows) != EXPECTED_ROWS or any(row.get("standardized_partition") != "train_fit" for row in rows):
        raise RuntimeError("Train-Fit population changed")
    return rows


def visible_rows(history: Any, row: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    visible = history.visible_same_pinyin(
        author=str(row["author"]), position=int(row["chronological_position"]), pinyin=base.pinyin_of(row)
    )
    plain = [{"row_id": item.record.row_id, "author": item.record.author,
              "chronological_position": item.record.position,
              "pinyin_segments": list(item.record.pinyin), "target": item.record.target,
              "context": item.record.context} for item in visible]
    return visible, plain


def frozen_frequency_rows(base_module: Any, *, query: Any,
                          generic_candidates: list[Any],
                          history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not generic_candidates:
        return []
    return base_module.frequency_rows(
        query=query, generic_candidates=generic_candidates, history_rows=history_rows)


def run_stage1(args: argparse.Namespace) -> None:
    fit_rows = load_fit(args.fit)
    history = base.CausalHistoryIndex(fit_rows)
    if normalized_source_sha256(Path(base.__file__)) != EXPECTED_BASE_NORMALIZED_SHA256:
        raise RuntimeError("Frozen Full-transfer source semantics changed")
    print("Loading PinyinGPT compatibility backend on CPU ...", flush=True)
    backend = base.PinyinGPTConcatBackend(args.checkpoint, device="cpu")
    output = args.output_root / "train_fit_stage1_features.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    started = time.perf_counter()
    generic_count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for number, pair in enumerate(itertools.zip_longest(fit_rows, iter_jsonl(args.generic)), start=1):
            fit_row, grow = pair
            if fit_row is None or grow is None:
                raise RuntimeError("Train-Fit and Generic row counts differ")
            if str(fit_row["row_id"]) != str(grow["row_id"]):
                raise RuntimeError(f"Generic row order differs at {number}")
            for key, expected in EXPECTED_REVISIONS.items():
                if str(grow.get(key)) != expected:
                    raise RuntimeError(f"Generic {key} changed at {grow['row_id']}")
            generic_count += 1
            query = base.query_of(fit_row)
            generic_candidates = base.candidates_of(grow)
            visible, plain_history = visible_rows(history, fit_row)
            flags = base.subset_membership(query, base.target_of(fit_row), plain_history)
            frequency = frozen_frequency_rows(
                base, query=query, generic_candidates=generic_candidates,
                history_rows=plain_history)
            generic_texts = {candidate.text for candidate in generic_candidates}
            personal = base.build_personal_k5(visible=visible, generic_texts=generic_texts,
                                              pinyin=base.pinyin_of(fit_row), backend=backend)
            counts = Counter(item.record.target for item in visible)
            total = sum(counts.values())
            choice = {candidate: counts.get(candidate, 0) / total if total else 0.0 for candidate in personal}
            p_ng = base.interpolated_ngram_recency(
                candidates=personal, query_context=base.scoring_context(fit_row, grow), visible=visible
            )
            entropy = base.entropy_concentration(counts)
            surface = retune.merge_stage1(
                base, generic_rows=frequency, personal_k5=personal, p_ng=p_ng,
                choice_share=choice, entropy=entropy, **WEIGHTS,
            )
            record = {"schema_version": 1, "row_id": str(fit_row["row_id"]),
                      "author": str(fit_row["author"]), "gold": base.target_of(fit_row),
                      "ambiguous": bool(flags["ambiguous"]), "conflict": bool(flags["conflict"]),
                      "raw_history_count": history.raw_visible_count(
                          author=str(fit_row["author"]), position=int(fit_row["chronological_position"])),
                      "same_pinyin_history_count": len(visible), "personal_k5": list(personal),
                      "choice_share": choice, "p_ng": p_ng, "entropy_concentration": entropy,
                      "generic_frequency_candidates": frequency,
                      "retuned_stage1_candidates": surface,
                      "gold_used_for_candidate_construction_or_scoring": False,
                      "gold_used_as_supervised_label_only": True,
                      "used_dev3000": False, "used_test": False}
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if args.progress_every and (number % args.progress_every == 0 or number == len(fit_rows)):
                print(f"Train-Fit Stage1 {number}/{len(fit_rows)} rate={number/max(time.perf_counter()-started, 1e-9):.1f}/s", flush=True)
    if generic_count != EXPECTED_ROWS:
        raise RuntimeError(f"Generic row count changed: {generic_count}")
    temporary.replace(output)
    write_json(args.output_root / "stage1_manifest.json", {
        "status": "complete", "rows": generic_count, "fit_sha256": EXPECTED_FIT_SHA256,
        "generic_sha256": sha256_file(args.generic),
        "normalized_base_source_sha256": EXPECTED_BASE_NORMALIZED_SHA256,
        "output": str(output.resolve()), "output_sha256": sha256_file(output),
        "weights": WEIGHTS, "used_dev3000": False, "used_test": False})
    print(f"Stage1 complete: {output}", flush=True)


def run_supports(args: argparse.Namespace) -> None:
    fit_rows = load_fit(args.fit)
    features = list(iter_jsonl(args.output_root / "train_fit_stage1_features.jsonl"))
    if len(features) != EXPECTED_ROWS:
        raise RuntimeError("Train-Fit Stage1 feature population changed")
    by_id = {str(row["row_id"]): row for row in fit_rows}
    if len(by_id) != EXPECTED_ROWS:
        raise RuntimeError("Duplicate Train-Fit row IDs")
    history = base.CausalHistoryIndex(fit_rows)
    required_contexts: set[str] = set()
    for feature in features:
        row = by_id[str(feature["row_id"])]
        candidates = {base.candidate_text(item) for item in feature["retuned_stage1_candidates"]}
        if not candidates:
            continue
        required_contexts.add(base.context_of(row)[-base.BGE_CONTEXT_CHARS:])
        visible, _ = visible_rows(history, row)
        required_contexts.update(item.record.context[-base.BGE_CONTEXT_CHARS:]
                                 for item in visible if item.record.target in candidates)
    vectors, cache_info = retune.fill_bge_vectors(
        base, contexts=required_contexts, cache_path=args.output_root / "bge_context_cache.sqlite3",
        seed_cache=args.seed_bge_cache, bge_model=args.bge_model,
        cuda_path=args.cuda_path, progress_every=args.progress_every,
    )
    write_json(args.output_root / "bge_cache_info.json", cache_info)
    output = args.output_root / "train_fit_candidate_supports.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    started = time.perf_counter()
    with temporary.open("w", encoding="utf-8", newline="\n") as sink:
        for number, feature in enumerate(features, start=1):
            row = by_id[str(feature["row_id"])]
            surface = feature["retuned_stage1_candidates"]
            candidates = [base.candidate_text(item) for item in surface]
            if candidates:
                visible, _ = visible_rows(history, row)
                ngram, effective_n, matched = base.ngram_recency_support(
                    query_context=base.context_of(row), candidates=candidates, visible=visible)
                query_context = base.context_of(row)[-base.BGE_CONTEXT_CHARS:]
                bge, bge_counts = base.bge_recency_support(
                    query_vector=vectors[query_context], candidates=candidates,
                    visible=visible, vectors=vectors)
            else:
                ngram, bge, effective_n, matched, bge_counts = {}, {}, 0, 0, {}
            record = {**feature, "retuned_ngram_support": ngram,
                      "retuned_bge_support": bge, "ngram_effective_n": effective_n,
                      "ngram_matched_history_rows": matched,
                      "bge_history_counts": bge_counts}
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if args.progress_every and (number % args.progress_every == 0 or number == len(features)):
                print(f"Train-Fit supports {number}/{len(features)} rate={number/max(time.perf_counter()-started, 1e-9):.1f}/s", flush=True)
    temporary.replace(output)
    write_json(args.output_root / "supports_manifest.json", {
        "status": "complete", "rows": len(features), "stage1_sha256": sha256_file(args.output_root / "train_fit_stage1_features.jsonl"),
        "output": str(output.resolve()), "output_sha256": sha256_file(output),
        "bge_cache": cache_info, "used_dev3000": False, "used_test": False})
    print(f"Supports complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("stage1", "supports"), required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--generic", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bge-model", type=Path)
    parser.add_argument("--seed-bge-cache", type=Path)
    parser.add_argument("--cuda-path", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.phase == "stage1":
        if args.generic is None or args.checkpoint is None:
            parser.error("stage1 requires --generic and --checkpoint")
        run_stage1(args)
    else:
        if args.bge_model is None:
            parser.error("supports requires --bge-model")
        run_supports(args)


if __name__ == "__main__":
    main()
