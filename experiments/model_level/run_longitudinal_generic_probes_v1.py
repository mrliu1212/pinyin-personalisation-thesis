from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from experiments.model_level import run_adapter_dev_evaluation_v1 as base
from experiments.model_level import run_adapter_longitudinal_warm_v1 as warm_runner


EXPERIMENT = "model_level_longitudinal_generic_probes_v1"

EXPECTED_PROTOCOL_SHA256 = (
    "dfaf14e0e47af27bb173edc67632e902b5030e31b275d9e4821d76c7f553bc3d"
)

DEFAULT_PROTOCOL = Path(
    "results/model_level/long_term_qualification_v1/"
    "alternating_six_train_six_probe_v1/frozen_protocol.json"
)

DEFAULT_WARM_RESULT = Path(
    "results/model_level/longitudinal_warm_v1/"
    "longitudinal_result.json"
)

DEFAULT_OUTPUT_ROOT = Path(
    "results/model_level/longitudinal_generic_probes_v1"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_probe_row_ids(path: Path) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            return obj

        if all(isinstance(x, dict) and "row_id" in x for x in obj):
            return [str(x["row_id"]) for x in obj]

    if isinstance(obj, dict):
        for key in (
            "row_ids",
            "selected_row_ids",
            "probe_row_ids",
            "rows",
        ):
            value = obj.get(key)

            if isinstance(value, list):
                if all(isinstance(x, str) for x in value):
                    return value

                if all(
                    isinstance(x, dict) and "row_id" in x
                    for x in value
                ):
                    return [str(x["row_id"]) for x in value]

    raise RuntimeError(
        f"Unsupported probe manifest structure: {path}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-fit", type=Path, required=True)
    p.add_argument("--train-val", type=Path, required=True)

    p.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )

    p.add_argument(
        "--warm-result",
        type=Path,
        default=DEFAULT_WARM_RESULT,
    )

    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--beam-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--probe",
        type=int,
        choices=range(1, 7),
        default=None,
        help="Run only one frozen probe P1-P6",
    )

    p.add_argument(
        "--validate-only",
        action="store_true",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.protocol.is_file():
        raise FileNotFoundError(args.protocol)

    protocol_sha = sha256_file(args.protocol)

    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(
            "Frozen protocol SHA mismatch:\n"
            f"expected={EXPECTED_PROTOCOL_SHA256}\n"
            f"actual={protocol_sha}"
        )

    protocol = json.loads(
        args.protocol.read_text(encoding="utf-8")
    )

    if protocol.get("status") != "FROZEN":
        raise RuntimeError("Protocol status is not FROZEN")

    population = protocol["population"]

    if population.get("test_used") is not False:
        raise RuntimeError("Protocol unexpectedly uses Test")

    if population.get("primary_probe_rows") != 3000:
        raise RuntimeError(
            "Expected exactly 3000 primary probe rows"
        )

    audits = protocol["probe_selection_audit"]

    if len(audits) != 6:
        raise RuntimeError(
            f"Expected 6 probes, got {len(audits)}"
        )

    author = protocol["author"]

    probe_specs: list[dict[str, Any]] = []

    seen_ids: set[str] = set()

    for expected_cycle, audit in enumerate(audits, start=1):
        cycle = int(audit["cycle"])

        if cycle != expected_cycle:
            raise RuntimeError(
                f"Unexpected cycle ordering: {cycle}"
            )

        if int(audit["probe_rows"]) != 500:
            raise RuntimeError(
                f"P{cycle}: expected 500 probe rows"
            )

        manifest = Path(audit["probe_manifest"])

        if not manifest.is_file():
            raise FileNotFoundError(manifest)

        actual_sha = sha256_file(manifest)
        expected_sha = str(audit["probe_sha256"])

        if actual_sha != expected_sha:
            raise RuntimeError(
                f"P{cycle} manifest SHA mismatch:\n"
                f"expected={expected_sha}\n"
                f"actual={actual_sha}"
            )

        row_ids = load_probe_row_ids(manifest)

        if len(row_ids) != 500:
            raise RuntimeError(
                f"P{cycle}: manifest contains "
                f"{len(row_ids)} IDs, expected 500"
            )

        if len(set(row_ids)) != 500:
            raise RuntimeError(
                f"P{cycle}: duplicate row IDs"
            )

        overlap = seen_ids.intersection(row_ids)

        if overlap:
            raise RuntimeError(
                f"P{cycle}: overlaps previous probes"
            )

        seen_ids.update(row_ids)

        probe_specs.append(
            {
                "cycle": cycle,
                "manifest": manifest,
                "manifest_sha256": actual_sha,
                "row_ids": row_ids,
            }
        )

    if len(seen_ids) != 3000:
        raise RuntimeError(
            f"Expected 3000 unique probe rows, "
            f"got {len(seen_ids)}"
        )

    print("===== GENERIC MATCHED PROBE GATE =====")
    print(f"protocol_sha256 = {protocol_sha}")
    print(f"author          = {author}")
    print(f"probes          = {len(probe_specs)}")
    print(f"rows/probe      = 500")
    print(f"total_rows      = {len(seen_ids)}")
    print("adapter_loaded  = false")
    print("training        = false")
    print("test_used       = false")

    for spec in probe_specs:
        print(
            f"P{spec['cycle']} "
            f"manifest_sha256={spec['manifest_sha256']}"
        )

    if args.validate_only:
        if author != warm_runner.AUTHOR:
            raise RuntimeError(
                f"Protocol author mismatch: {author!r} != {warm_runner.AUTHOR!r}"
            )

        validation_population = warm_runner.load_population(
            args.train_fit,
            args.train_val,
        )

        validation_missing = sorted(
            seen_ids.difference(validation_population)
        )

        if validation_missing:
            raise RuntimeError(
                f"{len(validation_missing)} frozen probe IDs "
                "not found in combined Train-Fit + Train-Val"
            )

        print(
            "combined_population = "
            f"{len(validation_population)}"
        )
        print("frozen_probe_ids_found = 3000/3000")
        print("STATUS = VALIDATED_ONLY")
        return

    if args.top_k != 10:
        raise RuntimeError("top_k must remain 10")

    if args.beam_size != 16:
        raise RuntimeError("beam_size must remain 16")

    args.output_root.mkdir(parents=True, exist_ok=True)

    if author != warm_runner.AUTHOR:
        raise RuntimeError(
            f"Protocol author mismatch: {author!r} != {warm_runner.AUTHOR!r}"
        )

    by_row_id = warm_runner.load_population(
        args.train_fit,
        args.train_val,
    )

    population_audit = {
        "author": author,
        "combined_effective_rows": len(by_row_id),
        "sources": ["train_fit", "train_val"],
        "test_used": False,
    }

    missing = sorted(seen_ids.difference(by_row_id))

    if missing:
        raise RuntimeError(
            f"{len(missing)} frozen probe IDs "
            "not found in Train-Val"
        )

    warm_result = json.loads(
        args.warm_result.read_text(encoding="utf-8")
    )

    warm_by_cycle: dict[int, dict[str, Any]] = {}

    for item in warm_result["cycles"]:
        cycle = int(item["cycle"])
        warm_by_cycle[cycle] = item["probe_metrics"]["micro"]

    if args.probe is not None:
        probe_specs = [
            spec for spec in probe_specs
            if int(spec["cycle"]) == args.probe
        ]
        if len(probe_specs) != 1:
            raise RuntimeError(
                f"Expected exactly one P{args.probe} spec"
            )

    backend = base.PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    results: list[dict[str, Any]] = []

    try:
        for spec in probe_specs:
            cycle = int(spec["cycle"])

            rows = [
                by_row_id[row_id]
                for row_id in spec["row_ids"]
            ]

            print()
            print("=" * 60, flush=True)
            print(f" GENERIC P{cycle}", flush=True)
            print("=" * 60, flush=True)

            predictions_path = (
                args.output_root
                / f"p{cycle}_generic_predictions.jsonl"
            )

            predictions = base.evaluate_backend(
                backend,
                rows,
                condition=f"generic_p{cycle}",
                output_path=predictions_path,
                top_k=args.top_k,
                beam_size=args.beam_size,
                log_every=args.log_every,
            )

            metrics = base.compute_metrics(predictions)
            micro = metrics["micro"]

            warm = warm_by_cycle[cycle]

            delta = {
                "top1": (
                    warm["top1"]
                    - micro["top1"]
                ),
                "top3": (
                    warm["top3"]
                    - micro["top3"]
                ),
                "mrr_at_10": (
                    warm["mrr_at_10"]
                    - micro["mrr_at_10"]
                ),
                "missing_at_10_rate": (
                    warm["missing_at_10_rate"]
                    - micro["missing_at_10_rate"]
                ),
            }

            item = {
                "cycle": cycle,
                "probe": f"P{cycle}",
                "rows": len(rows),
                "probe_manifest": str(spec["manifest"]),
                "probe_manifest_sha256": (
                    spec["manifest_sha256"]
                ),
                "generic_metrics": metrics,
                "warm_micro": warm,
                "warm_minus_generic": delta,
            }

            results.append(item)

            write_json(
                args.output_root
                / f"p{cycle}_generic_metrics.json",
                item,
            )

            print(
                f"P{cycle} Generic "
                f"Top1={micro['top1']:.6f} "
                f"Top3={micro['top3']:.6f} "
                f"MRR={micro['mrr_at_10']:.6f} "
                f"Missing="
                f"{micro['missing_at_10_rate']:.6f}",
                flush=True,
            )

            print(
                f"P{cycle} Warm-Gain "
                f"Top1={delta['top1']:+.6f} "
                f"Top3={delta['top3']:+.6f} "
                f"MRR={delta['mrr_at_10']:+.6f}",
                flush=True,
            )

    finally:
        base.release_backend(backend)

    mean_generic_top1 = sum(
        x["generic_metrics"]["micro"]["top1"]
        for x in results
    ) / len(results)

    mean_warm_top1 = sum(
        x["warm_micro"]["top1"]
        for x in results
    ) / len(results)

    mean_gain_top1 = sum(
        x["warm_minus_generic"]["top1"]
        for x in results
    ) / len(results)

    final = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "complete",
        "author": author,
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_sha,
        "checkpoint": str(args.checkpoint),
        "train_val": str(args.train_val),
        "warm_result": str(args.warm_result),
        "population": population_audit,
        "probe_count": len(results),
        "rows_per_probe": 500,
        "total_probe_rows": sum(x["rows"] for x in results),
        "adapter_loaded": False,
        "training": False,
        "used_test": False,
        "results": results,
        "summary": {
            "mean_generic_top1": mean_generic_top1,
            "mean_warm_top1": mean_warm_top1,
            "mean_warm_minus_generic_top1": (
                mean_gain_top1
            ),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "device_name": (
                torch.cuda.get_device_name(0)
                if (
                    torch.cuda.is_available()
                    and args.device.startswith("cuda")
                )
                else "CPU"
            ),
            "visible_cuda_devices": (
                torch.cuda.device_count()
            ),
        },
    }

    write_json(
        args.output_root / "generic_probe_result.json",
        final,
    )

    print()
    print("===== GENERIC MATCHED PROBES COMPLETE =====")

    for item in results:
        cycle = item["cycle"]
        g = item["generic_metrics"]["micro"]
        w = item["warm_micro"]
        d = item["warm_minus_generic"]

        print(
            f"P{cycle} "
            f"Generic={g['top1']:.6f} "
            f"Warm={w['top1']:.6f} "
            f"Gain={d['top1']:+.6f}"
        )

    print()
    print(
        f"Mean Generic Top1 = "
        f"{mean_generic_top1:.6f}"
    )
    print(
        f"Mean Warm Top1    = "
        f"{mean_warm_top1:.6f}"
    )
    print(
        f"Mean Warm Gain    = "
        f"{mean_gain_top1:+.6f}"
    )
    print("TEST USED = false")
    print("STATUS = COMPLETE")


if __name__ == "__main__":
    main()
