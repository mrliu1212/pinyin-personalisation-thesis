from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from src.model_level.pinyingpt_adapter import (
    SerialAdapterConfig,
    install_adapters,
    load_adapter_bundle,
    set_adapter_training_mode,
)
from src.reference_backend_pinyingpt.backend import (
    CHECKPOINT_REVISION,
    PinyinGPTConcatBackend,
)

EXPECTED_FULL_MANIFEST_SHA256 = (
    "e05b07d5020ebf0ccf090e98b34e4c66"
    "db7dab9b565541d39feb2d6f1d1bb20b"
)

EXPECTED_ROWS = 40_000
PER_AUTHOR = 8_000
TOP_K = 10
BEAM_SIZE = 16

AUTHORS = [
    "Re_spectators",
    "Etinjat",
    "Agent Phage",
    "QBLevi",
    "breaddddd",
]

ADAPTERS = {
    "Re_spectators":
        "re_spectators/re_spectators_adapter_v1.safetensors",
    "Etinjat":
        "etinjat/etinjat_adapter_v1.safetensors",
    "Agent Phage":
        "agent_phage/agent_phage_adapter_v1.safetensors",
    "QBLevi":
        "qblevi/qblevi_adapter_v1.safetensors",
    "breaddddd":
        "breaddddd/breaddddd_adapter_v1.safetensors",
}

UNSUPPORTED = (
    "no tokenizer candidates for Pinyin ",
    "unsupported Pinyin input: ",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_test(path: Path):
    got = sha256_file(path)
    print("FULL_MANIFEST_SHA256 =", got, flush=True)

    if got != EXPECTED_FULL_MANIFEST_SHA256:
        raise RuntimeError("Frozen full-manifest SHA mismatch")

    rows = []
    counts = Counter()
    seen = set()

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            if str(r["split"]).lower() != "test":
                continue

            rid = str(r["row_id"])
            if rid in seen:
                raise RuntimeError(f"Duplicate Test row_id: {rid}")
            seen.add(rid)

            author = str(r["author_name"])
            if author not in AUTHORS:
                raise RuntimeError(f"Unexpected author: {author}")

            parts = [
                x for x in str(r["pinyin_input"]).split()
                if x
            ]

            rows.append({
                "row_id": rid,
                "author_name": author,
                "split": "test",
                "gold": str(r["gold"]),
                "context": str(r.get("context") or ""),
                "pinyin_input": str(r["pinyin_input"]),
                "pinyin_segments": parts,
                "M": int(r["M"]),
                "typing_mode": str(r["typing_mode"]),
                "effective_typing_mode":
                    str(r["effective_typing_mode"]),
            })
            counts[author] += 1

    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"TEST_ROWS={len(rows)} != {EXPECTED_ROWS}"
        )

    for author in AUTHORS:
        if counts[author] != PER_AUTHOR:
            raise RuntimeError(
                f"{author}: {counts[author]} != {PER_AUTHOR}"
            )

    print("TEST_MANIFEST_GATE=PASS", flush=True)
    return rows


def load_existing(path: Path, expected):
    if not path.exists():
        return set()

    expected_ids = {r["row_id"] for r in expected}
    seen = set()

    with path.open("rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        last_good = 0

        while True:
            raw = f.readline()
            if not raw:
                break

            end = f.tell()

            if not raw.strip():
                last_good = end
                continue

            try:
                r = json.loads(raw.decode("utf-8"))
            except Exception:
                if end == size:
                    with path.open("r+b") as out:
                        out.truncate(last_good)
                    break
                raise

            rid = str(r["row_id"])

            if rid not in expected_ids:
                raise RuntimeError(
                    f"Unknown resume row: {rid}"
                )

            if rid in seen:
                raise RuntimeError(
                    f"Duplicate resume row: {rid}"
                )

            seen.add(rid)
            last_good = end

    return seen


def predict(backend, row):
    started = time.perf_counter()
    generated = None
    reason = None

    with torch.inference_mode():
        try:
            generated = backend.generate(
                row["context"],
                row["pinyin_segments"],
                top_k=TOP_K,
                beam_size=BEAM_SIZE,
            )
        except ValueError as exc:
            msg = str(exc)

            if not any(
                msg.startswith(prefix)
                for prefix in UNSUPPORTED
            ):
                raise

            reason = msg

    elapsed = time.perf_counter() - started

    if generated is None:
        texts = []
        scores = []
        device = str(
            getattr(backend, "device", "unknown")
        )
    else:
        texts = [
            x.text for x in generated.candidates
        ]
        scores = [
            float(x.log_probability)
            for x in generated.candidates
        ]
        device = str(generated.runtime_device)

    gold = row["gold"]

    rank = (
        texts.index(gold) + 1
        if gold in texts
        else None
    )

    return {
        "schema_version": 1,
        "experiment":
            "final_fiveauthor_adapter_test_candidates_v1",

        **{
            k: row[k]
            for k in (
                "row_id",
                "author_name",
                "split",
                "gold",
                "context",
                "pinyin_input",
                "M",
                "typing_mode",
                "effective_typing_mode",
            )
        },

        "segmented_pinyin":
            row["pinyin_segments"],

        "top10_candidates": texts,
        "top10_candidate_scores": scores,
        "gold_top10_rank": rank,

        "unsupported_input":
            reason is not None,
        "unsupported_reason": reason,

        "inference_seconds": elapsed,
        "beam_size": BEAM_SIZE,
        "top_k": TOP_K,
        "runtime_device": device,

        "used_test": True,
        "used_h5000": False,
        "used_rich30": False,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--full-manifest",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--adapter-root",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )

    ap.add_argument(
        "--log-every",
        type=int,
        default=250,
    )

    args = ap.parse_args()

    rows = load_test(args.full_manifest)

    print("TEST_ROWS =", len(rows), flush=True)
    print("TEST_EVALUATION_OPEN=TRUE", flush=True)

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for author in AUTHORS:
        print()
        print(
            "===== AUTHOR",
            author,
            "=====",
            flush=True,
        )

        author_rows = [
            r for r in rows
            if r["author_name"] == author
        ]

        adapter_path = (
            args.adapter_root
            / ADAPTERS[author]
        )

        if not adapter_path.is_file():
            raise FileNotFoundError(adapter_path)

        safe = (
            author.lower()
            .replace(" ", "_")
        )

        outpath = (
            args.output_root
            / f"{safe}_test_top10_v1.jsonl"
        )

        done = load_existing(
            outpath,
            author_rows,
        )

        print(
            f"RESUME={len(done)}/{PER_AUTHOR}",
            flush=True,
        )

        backend = PinyinGPTConcatBackend(
            args.checkpoint,
            device=args.device,
        )

        install_adapters(
            backend.model,
            SerialAdapterConfig(),
        )

        metadata = load_adapter_bundle(
            backend.model,
            adapter_path,
        )

        set_adapter_training_mode(
            backend.model,
            enabled=False,
        )

        if metadata.get("author") != author:
            raise RuntimeError(
                f"Adapter author mismatch: {author}"
            )

        if (
            metadata.get("checkpoint_revision")
            != CHECKPOINT_REVISION
        ):
            raise RuntimeError(
                "Adapter checkpoint revision mismatch"
            )

        completed = len(done)

        with outpath.open(
            "a",
            encoding="utf-8",
        ) as f:

            for row in author_rows:
                if row["row_id"] in done:
                    continue

                result = predict(
                    backend,
                    row,
                )

                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                f.flush()

                completed += 1

                if (
                    completed == 1
                    or completed % args.log_every == 0
                    or completed == PER_AUTHOR
                ):
                    print(
                        f"{author}: "
                        f"{completed}/{PER_AUTHOR}",
                        flush=True,
                    )

        del backend
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n = sum(
            1 for line in outpath.open(
                encoding="utf-8"
            )
            if line.strip()
        )

        if n != PER_AUTHOR:
            raise RuntimeError(
                f"{author}: final rows={n}"
            )

        print(
            author,
            "TEST_ADAPTER_COMPLETE",
            flush=True,
        )

    print()
    print(
        "FINAL_TEST_ADAPTER_CANDIDATES=PASS",
        flush=True,
    )
    print(
        "TOTAL_ROWS=40000",
        flush=True,
    )


if __name__ == "__main__":
    main()
