#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import statistics
import time
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import lightgbm as lgb

from experiments.context_comparison import run_full_transfer_initial_final_v1 as base
from experiments.context_comparison import run_full_retune_final_trainval_dev_v1 as retune
from experiments.external_memory_next import audit_learned_fusion_inputs_v1 as audit
from src.reference_backend_pinyingpt import PinyinGPTConcatBackend as ReferenceBackend
from src.evaluation.multi_full_path_trace import decode_batch_with_gold_trace

FIT_SHA = "547a4f8179f5d664a8621888236599938a2f967f055ef0c262be658b3500c8a6"
VAL_SHA = "d7ae1cc21ee029dde8458189b9dc7a0989b2b3a372627e079c3e2699307f2220"
M0_SHA = "59cb59170e348c60747522fef2c0af94deaaa0cc9cd022db273063249618915e"
MANIFEST_SHA = "9789aaa3d5f5a2276f31b18b8047c43c210061a13fd7a2383f0b04947c2d7c9b"
FIT_MULTI_SHA = "cb9f02ababbb7e4272a3dd39314425f56c0263ce35d48dd075c2969a1e13beb1"
VAL_MULTI_SHA = "9789aaa3d5f5a2276f31b18b8047c43c210061a13fd7a2383f0b04947c2d7c9b"
EXPECTED_FIT_MULTI_ROWS = 411_105
EXPECTED_VAL_MULTI_ROWS = 94_590
MODEL_SHA = "406b1693e5b8bb10b0af92c6bb31f494f8a78a13590d47ec5bf138fdba18df4e"
BGE_SHA = "5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039"
BASE_SOURCE_NORMALIZED_SHA = "f75d40f381e966f85cd4b20647ba7dc6a95df9116ad8657ca9a07505949a37b0"
EXPECTED_ROWS = 94_590
EXPECTED_FAMILIES = 18_918
WEIGHTS = {"w_p": 2.0, "w_cs": 6.0, "w_e": 4.0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("source_split", "")).lower() == "test" or bool(row.get("used_test", False)):
                raise RuntimeError(f"Test row encountered: {path}:{number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def candidate_rows(m0: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("top10_distinct_candidates", "top10_candidates", "top_candidates", "candidates", "final_beam_candidates"):
        value = m0.get(key)
        if not isinstance(value, list):
            continue
        rows, seen = [], set()
        for item in value:
            if not isinstance(item, Mapping) or "text" not in item:
                continue
            text = str(item["text"])
            if text in seen:
                continue
            score = item.get("log_probability", item.get("cumulative_log_probability", item.get("score")))
            if score is None:
                continue
            seen.add(text)
            rows.append({"text": text, "rank": len(rows) + 1, "log_probability": float(score)})
            if len(rows) == 10:
                break
        if rows:
            return rows

    trace = m0.get("step_trace")
    if isinstance(trace, list) and trace:
        last = trace[-1]
        texts = last.get("beam_prefixes_in_rank_order")
        scores = last.get("beam_scores_in_rank_order")
        if isinstance(texts, list) and isinstance(scores, list) and len(texts) == len(scores):
            rows, seen = [], set()
            for text, score in zip(texts, scores):
                text = str(text)
                if text in seen:
                    continue
                seen.add(text)
                rows.append({"text": text, "rank": len(rows) + 1, "log_probability": float(score)})
                if len(rows) == 10:
                    break
            if rows:
                return rows
    raise RuntimeError(f"No reusable Generic candidate surface in M0 row {m0.get('row_id')}; keys={sorted(m0)}")



def regenerate_generic_surfaces(
    manifest_rows: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    *,
    batch_size: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Regenerate the exact frozen M0 Generic Top10 surface when rows omitted it.

    Uses the same accepted reference backend, truncation, beam=16, TopK=10 and
    path-trace batch decoder as formal M0. Gold is observed only after beam
    selection by the trace function and never affects candidates or scores.
    """
    if batch_size < 1:
        raise ValueError("generic batch_size must be positive")

    backend = ReferenceBackend(checkpoint)
    if not str(backend.device).startswith("cuda"):
        raise RuntimeError(f"Generic fallback requires CUDA; got {backend.device}")

    buckets: dict[tuple[int, int], list[tuple[Mapping[str, Any], dict[str, Any]]]] = defaultdict(list)
    surfaces: dict[str, list[dict[str, Any]]] = {}
    completed = 0
    started = time.perf_counter()

    def flush(key: tuple[int, int]) -> None:
        nonlocal completed
        items = buckets[key]
        if not items:
            return
        requests = [req for _, req in items]
        results = decode_batch_with_gold_trace(
            backend, requests, top_k=10, beam_size=16, include_beam_surface=False
        )
        if len(results) != len(items):
            raise RuntimeError("Generic fallback result count mismatch")
        for (row, _), result in zip(items, results):
            rid = str(row["row_id"])
            surface = [
                {
                    "text": str(item["text"]),
                    "rank": int(item["rank"]),
                    "log_probability": float(item["log_probability"]),
                }
                for item in result["top10_distinct_candidates"]
            ]
            # Strong regression gates against the already-accepted M0 diagnostics
            # when those fields are serialized in rows.jsonl.
            expected_top1 = row.get("_accepted_m0_top1_text")
            if expected_top1 is not None:
                got_top1 = surface[0]["text"] if surface else None
                if got_top1 != expected_top1:
                    raise RuntimeError(
                        f"Generic regeneration Top1 mismatch for {rid}: {got_top1!r} != {expected_top1!r}"
                    )
            expected_rank = row.get("_accepted_m0_gold_rank")
            if expected_rank is not None:
                got_rank = rank_of(str(row["gold"]), [x["text"] for x in surface])
                if got_rank != expected_rank:
                    raise RuntimeError(
                        f"Generic regeneration Gold-rank mismatch for {rid}: {got_rank!r} != {expected_rank!r}"
                    )
            surfaces[rid] = surface
            completed += 1
        buckets[key] = []
        if completed % 500 == 0 or completed == len(manifest_rows):
            print(
                f"generic-regeneration {completed}/{len(manifest_rows)} "
                f"rate={completed/max(time.perf_counter()-started,1e-9):.1f}/s",
                flush=True,
            )

    for row in manifest_rows:
        segments = list(map(str, row.get("pinyin_segments") or str(row["pinyin_input"]).split()))
        used_context, _, _, _ = backend.truncate_context_for_generation(
            str(row["context"]), segments
        )
        prompt_ids, _ = backend._prompt(used_context, segments)
        key = (len(segments), len(prompt_ids))
        request = {
            "context": used_context,
            "pinyin_segments": segments,
            "gold": str(row["gold"]),
        }
        buckets[key].append((row, request))
        if len(buckets[key]) >= batch_size:
            flush(key)

    for key in list(buckets):
        flush(key)

    if len(surfaces) != len(manifest_rows):
        raise RuntimeError(
            f"Generic fallback rows differ: {len(surfaces)} != {len(manifest_rows)}"
        )

    del backend
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return surfaces

def rank_of(gold: str, names: Sequence[str]) -> int | None:
    try:
        return list(names).index(gold) + 1
    except ValueError:
        return None


def m0_gold_survived(row: Mapping[str, Any]) -> bool:
    if "gold_survived_final_beam" in row:
        return bool(row["gold_survived_final_beam"])
    trace = row.get("step_trace")
    if isinstance(trace, list) and trace:
        return bool(trace[-1].get("gold_prefix_in_beam"))
    if "gold_rank_in_final_beam" in row:
        return row.get("gold_rank_in_final_beam") is not None
    # Conservative fallback: Top10 presence is weaker than final-beam presence,
    # but accepted M0 v5 normally serializes one of the fields above.
    return row.get(
        "gold_rank_in_top10_distinct_candidates",
        row.get("gold_rank_in_top10"),
    ) is not None



class ExpandedCausalHistoryIndex:
    """Frozen Full history with same-event Multi1--5 PV expansion.

    The H5000 budget is applied to raw interactions exactly as in
    base.CausalHistoryIndex. Derived Multi entries never create additional
    events and therefore never alter raw history count or raw-stream age.
    """

    def __init__(
        self,
        raw_rows: Sequence[Mapping[str, Any]],
        multi_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        raw_by_id = {str(row["row_id"]): row for row in raw_rows}
        if len(raw_by_id) != len(raw_rows):
            raise RuntimeError("duplicate raw history row_id")

        derived_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

        for row in multi_rows:
            source_id = str(row["source_standard_row_id"])
            if source_id not in raw_by_id:
                raise RuntimeError(
                    f"Multi history row has missing source raw event: {source_id}"
                )
            derived_by_source[source_id].append(row)

        grouped: dict[
            str,
            list[tuple[base.HistoryRecord, tuple[base.HistoryRecord, ...]]],
        ] = defaultdict(list)

        for row in raw_rows:
            source_id = str(row["row_id"])
            author = str(row["author"])
            position = int(row["chronological_position"])
            context = base.context_of(row)

            original = base.HistoryRecord(
                row_id=source_id,
                author=author,
                position=position,
                pinyin=base.pinyin_of(row),
                target=base.target_of(row),
                context=context,
            )

            # One raw event can contribute at most one record for a query
            # pinyin. Original Short is inserted first, so identical Multi1
            # is automatically deduplicated.
            by_pinyin: dict[tuple[str, ...], base.HistoryRecord] = {
                original.pinyin: original
            }

            derived = sorted(
                derived_by_source.get(source_id, ()),
                key=lambda x: (
                    int(x["multi_token_length"]),
                    str(x["row_id"]),
                ),
            )

            for multi in derived:
                pinyin = tuple(
                    map(
                        str,
                        multi.get("pinyin_segments")
                        or str(multi["pinyin_input"]).split(),
                    )
                )
                target = str(multi["gold"])

                prior = by_pinyin.get(pinyin)
                if prior is not None:
                    if prior.target != target:
                        raise RuntimeError(
                            "same raw event has same pinyin with different targets: "
                            f"source={source_id} pinyin={pinyin} "
                            f"{prior.target!r} != {target!r}"
                        )
                    continue

                if str(multi["author"]) != author:
                    raise RuntimeError(
                        f"derived/raw author mismatch: {source_id}"
                    )

                by_pinyin[pinyin] = base.HistoryRecord(
                    row_id=str(multi["row_id"]),
                    author=author,
                    position=position,
                    pinyin=pinyin,
                    target=target,
                    # Preserve original raw-event context semantics.
                    context=context,
                )

            grouped[author].append(
                (
                    original,
                    tuple(by_pinyin.values()),
                )
            )

        self.events: dict[
            str,
            tuple[tuple[base.HistoryRecord, tuple[base.HistoryRecord, ...]], ...],
        ] = {}
        self.positions: dict[str, tuple[int, ...]] = {}

        for author, events in grouped.items():
            ordered = tuple(
                sorted(
                    events,
                    key=lambda item: (
                        item[0].position,
                        item[0].row_id,
                    ),
                )
            )
            self.events[author] = ordered
            self.positions[author] = tuple(
                item[0].position for item in ordered
            )

    def visible_same_pinyin(
        self,
        *,
        author: str,
        position: int,
        pinyin: tuple[str, ...],
    ) -> tuple[base.VisibleHistory, ...]:
        events = self.events.get(author, ())
        positions = self.positions.get(author, ())

        stop = bisect.bisect_left(positions, int(position))
        start = max(0, stop - base.HISTORY_BUDGET)

        result: list[base.VisibleHistory] = []

        for ordinal in range(start, stop):
            _, entries = events[ordinal]

            matched = [
                record
                for record in entries
                if record.pinyin == pinyin
            ]

            if len(matched) > 1:
                raise RuntimeError(
                    "one raw event contributed more than one exact-pinyin "
                    f"observation: author={author} position={position}"
                )

            if matched:
                result.append(
                    base.VisibleHistory(
                        record=matched[0],
                        age=stop - 1 - ordinal,
                    )
                )

        return tuple(result)

    def raw_visible_count(self, *, author: str, position: int) -> int:
        positions = self.positions.get(author, ())
        stop = bisect.bisect_left(positions, int(position))
        return min(stop, base.HISTORY_BUDGET)


def visible_rows(history: Any, row: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    visible = history.visible_same_pinyin(
        author=str(row["author"]),
        position=int(row["chronological_position"]),
        pinyin=base.pinyin_of(row),
    )
    plain = [{
        "row_id": item.record.row_id,
        "author": item.record.author,
        "chronological_position": item.record.position,
        "pinyin_segments": list(item.record.pinyin),
        "target": item.record.target,
        "context": item.record.context,
    } for item in visible]
    return visible, plain


def make_query_row(multi: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(anchor)
    pinyin = list(map(str, multi.get("pinyin_segments") or str(multi["pinyin_input"]).split()))
    gold = str(multi["gold"])
    row.update({
        "row_id": str(multi["row_id"]),
        "author": str(multi["author"]),
        "work_id": str(multi["work_id"]),
        "context": str(multi["context"]),
        "gold": gold,
        "target": gold,
        "pinyin_input": " ".join(pinyin),
        "full_pinyin": " ".join(pinyin),
        "pinyin_segments": pinyin,
        "source_position_start": int(multi["source_position_start"]),
        "source_position_end": int(multi["source_position_end"]),
        "standardized_partition": "train_val",
        "source_split": "history",
    })
    return row


def fake_generic(row_id: str, context: str, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "top10_candidates": [dict(x) for x in candidates],
        "used_context": context,
        "context": context,
        "checkpoint_revision": "76dd20dc92d8236a350fb732e99dde6fa15e2263",
        "official_code_revision": "8f1573ed0bd4d1f3d8d3f10a05f7e870725646f1",
        "used_test": False,
    }


def metric_values(ranks: Sequence[int | None]) -> dict[str, float | int]:
    n = len(ranks)
    if not n:
        return {"n": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "mrr_at_10": 0.0, "missing_at_10": 0.0}
    return {
        "n": n,
        "top1": sum(r == 1 for r in ranks) / n,
        "top3": sum(r is not None and r <= 3 for r in ranks) / n,
        "top5": sum(r is not None and r <= 5 for r in ranks) / n,
        "top10": sum(r is not None and r <= 10 for r in ranks) / n,
        "mrr_at_10": sum(0.0 if r is None else 1.0 / r for r in ranks) / n,
        "missing_at_10": sum(r is None for r in ranks) / n,
    }


def aggregate(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    per_author: dict[str, list[int | None]] = defaultdict(list)
    for row in rows:
        per_author[str(row["author"])].append(row.get(key))
    pa = {a: metric_values(v) for a, v in sorted(per_author.items())}
    fields = ("top1", "top3", "top5", "top10", "mrr_at_10", "missing_at_10")
    return {
        "micro": metric_values([r.get(key) for r in rows]),
        "macro_author": {f: statistics.fmean(float(v[f]) for v in pa.values()) for f in fields},
        "per_author": pa,
    }


def transition(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out = {"n": len(rows), "rescue": 0, "harm": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for row in rows:
        g, z = row.get("generic_rank") == 1, row.get("zero_shot_rank") == 1
        if not g and z:
            out["rescue"] += 1
        elif g and not z:
            out["harm"] += 1
        elif g and z:
            out["unchanged_correct"] += 1
        else:
            out["unchanged_wrong"] += 1
    out["net"] = out["rescue"] - out["harm"]
    return out


def subset_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "generic": aggregate(rows, "generic_rank"),
        "zero_shot": aggregate(rows, "zero_shot_rank"),
        "transition": transition(rows),
        "personalized_gold_availability": sum(bool(r["gold_in_personalized_surface"]) for r in rows) / len(rows),
        "same_pinyin_history_nonempty": sum(int(r["same_pinyin_history_count"]) > 0 for r in rows) / len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("fit", "val", "manifest", "m0_rows", "checkpoint", "bge_model", "model", "output_root"):
        ap.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--fit-multi", type=Path, required=True)
    ap.add_argument("--val-multi", type=Path, required=True)
    ap.add_argument("--cuda-path", type=Path)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--generic-batch-size", type=int, default=8)
    args = ap.parse_args()

    for path, expected in (
        (args.fit, FIT_SHA), (args.val, VAL_SHA), (args.manifest, MANIFEST_SHA),
        (args.fit_multi, FIT_MULTI_SHA), (args.val_multi, VAL_MULTI_SHA),
        (args.m0_rows, M0_SHA), (args.model, MODEL_SHA), (args.bge_model, BGE_SHA),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"SHA mismatch: {path}\nactual={actual}\nexpected={expected}")

    if normalized_source_sha256(Path(base.__file__)) != BASE_SOURCE_NORMALIZED_SHA:
        raise RuntimeError("Frozen Full-transfer source semantics changed")

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    model_feature_line = next(
        (x for x in args.model.read_text(encoding="utf-8").splitlines() if x.startswith("feature_names=")), None
    )
    if model_feature_line and tuple(model_feature_line.split("=", 1)[1].split()) != tuple(audit.FEATURE_NAMES):
        raise RuntimeError("Frozen LambdaMART feature order differs")

    fit_rows, val_rows = read_jsonl(args.fit), read_jsonl(args.val)
    fit_multi_rows = read_jsonl(args.fit_multi)
    val_multi_rows = read_jsonl(args.val_multi)
    manifest_rows, m0_rows = read_jsonl(args.manifest), read_jsonl(args.m0_rows)
    if (len(fit_rows), len(val_rows), len(manifest_rows), len(m0_rows)) != (144526, 34416, EXPECTED_ROWS, EXPECTED_ROWS):
        raise RuntimeError("Frozen population row count changed")
    if (len(fit_multi_rows), len(val_multi_rows)) != (EXPECTED_FIT_MULTI_ROWS, EXPECTED_VAL_MULTI_ROWS):
        raise RuntimeError("Frozen historical Multi-v2 population row count changed")

    manifest_by_id = {str(r["row_id"]): r for r in manifest_rows}
    m0_by_id = {str(r["row_id"]): r for r in m0_rows}
    if set(manifest_by_id) != set(m0_by_id):
        raise RuntimeError("Manifest/M0 row IDs differ")

    families: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in manifest_rows:
        families[str(r["family_id"])][int(r["multi_token_length"])] = r
    if len(families) != EXPECTED_FAMILIES or any(set(x) != {1,2,3,4,5} for x in families.values()):
        raise RuntimeError("Multi family structure changed")

    val_by_key = {
        (str(r["work_id"]), int(r["source_position_start"]), int(r["source_position_end"])): r
        for r in val_rows
    }
    anchors = {}
    for family, by_len in families.items():
        one = by_len[1]
        key = (str(one["work_id"]), int(one["source_position_start"]), int(one["source_position_end"]))
        anchor = val_by_key.get(key)
        if anchor is None or str(anchor["gold"]) != str(one["gold"]):
            raise RuntimeError(f"Multi1/Short anchor mismatch: {family}")
        anchors[family] = anchor

    query_rows = {
        str(r["row_id"]): make_query_row(r, anchors[str(r["family_id"])])
        for r in manifest_rows
    }

    # Prefer the accepted M0 candidate scores if they were serialized. Formal M0
    # deliberately used include_beam_surface=False, so some accepted rows may
    # contain only diagnostics. In that case regenerate the exact Generic surface
    # with the same frozen reference decoder, then gate it against accepted Top1/
    # Gold-rank diagnostics.
    generic_surfaces: dict[str, list[dict[str, Any]]] = {}
    direct_ok = True
    for m0 in m0_rows:
        try:
            generic_surfaces[str(m0["row_id"])] = candidate_rows(m0)
        except RuntimeError:
            direct_ok = False
            break

    if direct_ok and len(generic_surfaces) == EXPECTED_ROWS:
        print("Generic surface source: accepted M0 serialized candidates", flush=True)
    else:
        print(
            "Generic surface not fully serialized in accepted M0 rows; "
            "regenerating exact beam16/Top10 surface on CUDA.",
            flush=True,
        )
        regen_rows = []
        for multi in manifest_rows:
            rid = str(multi["row_id"])
            accepted = m0_by_id[rid]
            augmented = dict(multi)
            augmented["_accepted_m0_top1_text"] = accepted.get("top1_text")
            rank = accepted.get(
                "gold_rank_in_top10_distinct_candidates",
                accepted.get("gold_rank_in_top10"),
            )
            augmented["_accepted_m0_gold_rank"] = rank
            regen_rows.append(augmented)
        generic_surfaces = regenerate_generic_surfaces(
            regen_rows, args.checkpoint, batch_size=args.generic_batch_size
        )

    # Exact aggregate sanity: regenerated/reused Generic ranks must reproduce
    # the accepted M0 headline Top1 counts length-by-length.
    expected_top1 = {1: 14999, 2: 11392, 3: 8426, 4: 6219, 5: 4695}
    for length, expected_correct in expected_top1.items():
        correct = 0
        for multi in manifest_rows:
            if int(multi["multi_token_length"]) != length:
                continue
            names = [x["text"] for x in generic_surfaces[str(multi["row_id"])]]
            correct += int(rank_of(str(multi["gold"]), names) == 1)
        if correct != expected_correct:
            raise RuntimeError(
                f"Generic Top1 sanity failed Multi{length}: {correct} != {expected_correct}"
            )
    print("Generic M0 headline reproduction: PASS", flush=True)

    history = ExpandedCausalHistoryIndex(
        [*fit_rows, *val_rows],
        [*fit_multi_rows, *val_multi_rows],
    )
    print(
        "M1-C history: raw H5000 -> Original PV + same-event Multi1-5 PV -> exact Pinyin",
        flush=True,
    )

    print("Loading frozen PinyinGPT compatibility backend on CPU ...", flush=True)
    backend = base.PinyinGPTConcatBackend(args.checkpoint, device="cpu")

    required_contexts: set[str] = set()
    features: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, m0 in enumerate(m0_rows, 1):
        row_id = str(m0["row_id"])
        multi, qrow = manifest_by_id[row_id], query_rows[row_id]
        raw = generic_surfaces[row_id]
        grow = fake_generic(row_id, str(qrow["context"]), raw)
        generic = base.candidates_of(grow)
        query = base.query_of(qrow)
        visible, plain = visible_rows(history, qrow)
        frequency = base.frequency_rows(query=query, generic_candidates=generic, history_rows=plain) if generic else []
        personal = base.build_personal_k5(
            visible=visible, generic_texts={x.text for x in generic},
            pinyin=base.pinyin_of(qrow), backend=backend,
        )
        counts = Counter(x.record.target for x in visible)
        total = sum(counts.values())
        choice = {c: counts.get(c, 0) / total if total else 0.0 for c in personal}
        p_ng = base.interpolated_ngram_recency(
            candidates=personal, query_context=base.scoring_context(qrow, grow), visible=visible
        )
        entropy = base.entropy_concentration(counts)
        surface = retune.merge_stage1(
            base, generic_rows=frequency, personal_k5=personal, p_ng=p_ng,
            choice_share=choice, entropy=entropy, **WEIGHTS,
        )
        names = [base.candidate_text(x) for x in surface]
        if names:
            name_set = set(names)
            required_contexts.add(base.context_of(qrow)[-base.BGE_CONTEXT_CHARS:])
            required_contexts.update(
                x.record.context[-base.BGE_CONTEXT_CHARS:] for x in visible if x.record.target in name_set
            )
        features.append({
            "schema_version": 1,
            "row_id": row_id,
            "family_id": str(multi["family_id"]),
            "multi_token_length": int(multi["multi_token_length"]),
            "author": str(qrow["author"]),
            "gold": str(qrow["gold"]),
            "raw_history_count": history.raw_visible_count(
                author=str(qrow["author"]), position=int(qrow["chronological_position"])
            ),
            "same_pinyin_history_count": len(visible),
            "personal_k5": list(personal),
            "choice_share": choice,
            "p_ng": p_ng,
            "entropy_concentration": entropy,
            "generic_frequency_candidates": frequency,
            "retuned_stage1_candidates": surface,
            "used_dev3000": False,
            "used_test": False,
        })
        if args.progress_every and (number % args.progress_every == 0 or number == EXPECTED_ROWS):
            print(f"stage1 {number}/{EXPECTED_ROWS} rate={number/max(time.perf_counter()-started,1e-9):.1f}/s", flush=True)

    print(f"BGE unique contexts required: {len(required_contexts)}", flush=True)
    vectors, cache_info = retune.fill_bge_vectors(
        base,
        contexts=required_contexts,
        cache_path=args.output_root / "bge_context_cache.sqlite3",
        seed_cache=Path(__import__("os").environ["M4_BGE_SEED_CACHE"]),
        bge_model=args.bge_model,
        cuda_path=args.cuda_path,
        progress_every=args.progress_every,
    )
    write_json(args.output_root / "bge_cache_info.json", cache_info)

    booster = lgb.Booster(model_file=str(args.model))
    result_rows = []
    out_rows = args.output_root / "rows.jsonl"
    started = time.perf_counter()
    with out_rows.open("w", encoding="utf-8", newline="\n") as sink:
        for number, feature in enumerate(features, 1):
            row_id, qrow, m0 = str(feature["row_id"]), query_rows[str(feature["row_id"])], m0_by_id[str(feature["row_id"])]
            surface = feature["retuned_stage1_candidates"]
            names = [base.candidate_text(x) for x in surface]
            if names:
                visible, _ = visible_rows(history, qrow)
                ngram, effective_n, matched = base.ngram_recency_support(
                    query_context=base.context_of(qrow), candidates=names, visible=visible
                )
                qctx = base.context_of(qrow)[-base.BGE_CONTEXT_CHARS:]
                bge, bge_counts = base.bge_recency_support(
                    query_vector=vectors[qctx], candidates=names, visible=visible, vectors=vectors
                )
            else:
                ngram, bge, effective_n, matched, bge_counts = {}, {}, 0, 0, {}

            support = {
                **feature,
                "retuned_ngram_support": ngram,
                "retuned_bge_support": bge,
                "ngram_effective_n": effective_n,
                "ngram_matched_history_rows": matched,
                "bge_history_counts": bge_counts,
            }
            matrix, _, extracted = audit.extract_group(feature, support)
            if extracted != names:
                raise RuntimeError(f"Candidate order changed: {row_id}")
            scores = np.asarray(
                booster.predict(np.asarray(matrix, dtype=np.float64)),
                dtype=np.float64,
            ) if matrix else np.asarray([])

            # The frozen LambdaMART runner resolves equal tree scores by the
            # frozen RetunedFinal order, then candidate text. Reconstruct that
            # baseline order exactly from Stage1 + 6*NGram + 6*BGE.
            baseline_indices = sorted(
                range(len(names)),
                key=lambda i: (
                    -(
                        float(surface[i]["final_score"])
                        + 6.0 * float(ngram[names[i]])
                        + 6.0 * float(bge[names[i]])
                    ),
                    int(surface[i].get("base_rank", surface[i].get("rank", i + 1))),
                    names[i],
                ),
            )
            baseline_order = {
                names[i]: rank
                for rank, i in enumerate(baseline_indices, start=1)
            }
            order = sorted(
                range(len(names)),
                key=lambda i: (
                    -float(scores[i]),
                    baseline_order[names[i]],
                    names[i],
                ),
            )
            ranked = [names[i] for i in order]
            generic_names = [x["text"] for x in generic_surfaces[row_id]]
            zero_rank, generic_rank = rank_of(str(qrow["gold"]), ranked), rank_of(str(qrow["gold"]), generic_names)

            out = {
                "row_id": row_id,
                "family_id": str(feature["family_id"]),
                "author": str(feature["author"]),
                "multi_token_length": int(feature["multi_token_length"]),
                "gold": str(qrow["gold"]),
                "generic_rank": generic_rank,
                "zero_shot_rank": zero_rank,
                "generic_gold_survived_final_beam": m0_gold_survived(m0),
                "gold_in_personalized_surface": str(qrow["gold"]) in names,
                "same_pinyin_history_count": int(feature["same_pinyin_history_count"]),
                "raw_history_count": int(feature["raw_history_count"]),
                "personal_k5_count": len(feature["personal_k5"]),
                "personal_k5": feature["personal_k5"],
                "retuned_final_top10": [names[i] for i in baseline_indices],
                "zero_shot_candidates": [
                    {"rank": rank, "text": names[i], "lambdamart_score": float(scores[i]), "source": surface[i].get("source")}
                    for rank, i in enumerate(order, 1)
                ],
                "used_dev3000": False,
                "used_test": False,
            }
            sink.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
            result_rows.append(out)
            if args.progress_every and (number % args.progress_every == 0 or number == EXPECTED_ROWS):
                print(f"score {number}/{EXPECTED_ROWS} rate={number/max(time.perf_counter()-started,1e-9):.1f}/s", flush=True)

    per_length = {}
    for length in range(1, 6):
        rows = [r for r in result_rows if r["multi_token_length"] == length]
        per_length[f"Multi{length}"] = {
            "n": len(rows),
            "generic": aggregate(rows, "generic_rank"),
            "zero_shot": aggregate(rows, "zero_shot_rank"),
            "transition": transition(rows),
            "generic_beam_survived": subset_summary([r for r in rows if r["generic_gold_survived_final_beam"]]),
            "generic_beam_pruned": subset_summary([r for r in rows if not r["generic_gold_survived_final_beam"]]),
            "same_pinyin_history_nonempty_n": sum(r["same_pinyin_history_count"] > 0 for r in rows),
            "personal_k5_nonempty_n": sum(r["personal_k5_count"] > 0 for r in rows),
        }

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "multi_full_m1c_pv_expansion_frozen_full_lambdamart_v2",
        "scientific_role": "M1-C minimal personal-vocabulary expansion; frozen Full mechanism with no Multi refit or tuning",
        "candidate_surface": "unchanged frozen Full RetunedFinal construction and frozen Full-trained LambdaMART; only personal-history PV evidence is expanded",
        "personal_history": "same author -> strictly prior -> latest H5000 raw interactions -> Original Short PV plus same-event Multi1-5 v2 PV expansion -> within-event deduplication -> exact segmented-Pinyin",
        "model_fit_domain": "frozen Full+Short Train-Fit LambdaMART; no Multi fitting",
        "evaluation_population": "Multi1-5 standardized Train-Val v2",
        "rows": len(result_rows),
        "families": EXPECTED_FAMILIES,
        "feature_names": list(audit.FEATURE_NAMES),
        "weights": {"w_p": 2.0, "w_cs": 6.0, "w_e": 4.0, "lambda_ngram": 6.0, "lambda_bge": 6.0},
        "hashes": {
            "fit": FIT_SHA, "val": VAL_SHA, "multi_manifest": MANIFEST_SHA,
            "fit_multi_history": FIT_MULTI_SHA, "val_multi_history": VAL_MULTI_SHA,
            "m0_rows": M0_SHA, "lambdamart_model": MODEL_SHA, "bge_model": BGE_SHA,
            "runner": sha256_file(Path(__file__)), "rows": sha256_file(out_rows),
        },
        "per_length": per_length,
        "used_dev3000": False,
        "dev3000_rows_opened": 0,
        "used_test": False,
        "test_rows_opened": 0,
        "history_budget_unit": "raw interaction",
        "derived_entries_count_as_raw_events": False,
        "within_raw_event_exact_pinyin_deduplicated": True,
        "multi_training": False,
        "multi_refit": False,
        "multi_hyperparameter_tuning": False,
        "gold_used_for_runtime_candidate_construction_or_scoring": False,
    }
    write_json(args.output_root / "result.json", result)
    print(json.dumps(per_length, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("RESULT:", args.output_root / "result.json", flush=True)


if __name__ == "__main__":
    main()
