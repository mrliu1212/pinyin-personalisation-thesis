"""Freeze deterministic 1,000-row development manifests per author."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.model_level import run_adapter_dev_evaluation_v1 as base


AUTHORS = (
    "Agent Phage",
    "Etinjat",
    "breaddddd",
)


def slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def row_id_sha(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["row_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-val", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "name": "fixed_dev1000_v1",
        "seed": args.seed,
        "rows_per_author": args.rows,
        "authors": {},
        "used_test": False,
    }

    for author in AUTHORS:
        author_rows, audit = base.load_val_rows(
            args.train_val,
            author=author,
        )

        selected = base.deterministic_subset(
            author_rows,
            seed=args.seed,
            max_rows=args.rows,
        )

        path = args.output_root / f"{slug(author)}_dev1000.jsonl"

        with path.open("w", encoding="utf-8") as destination:
            for row in selected:
                destination.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        manifest["authors"][author] = {
            "author_train_val_rows": len(author_rows),
            "selected_rows": len(selected),
            "file": str(path),
            "file_sha256": base.sha256_file(path),
            "row_id_manifest_sha256": row_id_sha(selected),
            "population_audit": audit,
        }

        print(
            f"{author}: "
            f"{len(selected)} rows "
            f"row_manifest={row_id_sha(selected)}"
        )

    base.write_json(
        args.output_root / "manifest.json",
        manifest,
    )


if __name__ == "__main__":
    main()
