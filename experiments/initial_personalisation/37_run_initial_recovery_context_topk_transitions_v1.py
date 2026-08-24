from __future__ import annotations

"""Post-hoc Top-k rescue/harm diagnostics for frozen Initial-Pinyin context models.

Reads only already-frozen Train-Val artifacts. No scoring, no tuning, no BGE
embedding, and no Dev3000/Test access.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_VAL_SHA256 = "d908d4dbd534e921f0bfd5e7a39b03037690073e8e567cfffecf61466ec0f0e4"
EXPECTED_ROWS = 34_416
EXPECTED_GENERIC_MISSING = 12_565
EXPECTED_RECOVERABLE_K5 = 4_910

BASE_ORDER = (
    "K5+Entropy",
    "4P+4CS+2E",
    "6P+2CS+.25E",
)

TRANSITIONS = (
    ("Recovery->NG-R", "recovery_rank", "ng_rank"),
    ("NG-R->Full", "ng_rank", "full_rank"),
    ("Recovery->Full", "recovery_rank", "full_rank"),
)

SUBSETS = (
    "overall",
    "generic_covered",
    "generic_missing",
    "recoverable_R",
    "ambiguous",
    "conflict",
)

TOP_K = (1, 3, 5)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "row_id" not in row:
                raise RuntimeError(f"Missing row_id at {path}:{line_no}")
            out.append(row)
    return out


def index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in out:
            raise RuntimeError(f"Duplicate row_id in {label}: {row_id}")
        out[row_id] = dict(row)
    return out


def index_selected(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {base: {} for base in BASE_ORDER}
    for row in rows:
        base = str(row.get("base", ""))
        if base not in out:
            raise RuntimeError(f"Unexpected base in {label}: {base!r}")
        row_id = str(row["row_id"])
        if row_id in out[base]:
            raise RuntimeError(f"Duplicate row in {label}: {base} {row_id}")
        out[base][row_id] = dict(row)
    for base in BASE_ORDER:
        if len(out[base]) != EXPECTED_ROWS:
            raise RuntimeError(f"Unexpected {label} rows for {base}: {len(out[base])}/{EXPECTED_ROWS}")
    return out


def rank_is_topk(value: Any, k: int) -> bool:
    return value is not None and int(value) <= k


def transition_counts(rows: Sequence[Mapping[str, Any]], before_key: str, after_key: str, k: int) -> dict[str, Any]:
    rescue = harm = unchanged_in = unchanged_out = 0
    for row in rows:
        before = rank_is_topk(row.get(before_key), k)
        after = rank_is_topk(row.get(after_key), k)
        if not before and after:
            rescue += 1
        elif before and not after:
            harm += 1
        elif before and after:
            unchanged_in += 1
        else:
            unchanged_out += 1
    n = len(rows)
    return {
        "n": n,
        "k": k,
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "rescue_rate": rescue / n if n else None,
        "harm_rate": harm / n if n else None,
        "net_rate": (rescue - harm) / n if n else None,
        "unchanged_in_topk": unchanged_in,
        "unchanged_outside_topk": unchanged_out,
    }


def memberships(row: Mapping[str, Any]) -> set[str]:
    out = {"overall"}
    if bool(row["generic_missing"]):
        out.add("generic_missing")
        if bool(row["gold_in_personal_k5"]):
            out.add("recoverable_R")
    else:
        out.add("generic_covered")
    if bool(row.get("ambiguous", False)):
        out.add("ambiguous")
    if bool(row.get("conflict", False)):
        out.add("conflict")
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_new_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--base-ngram-root", type=Path, required=True)
    parser.add_argument("--base-v3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    ensure_new_output_root(args.output_root)
    if sha256_file(args.val) != EXPECTED_VAL_SHA256:
        raise RuntimeError("Train-Val SHA256 mismatch")

    stage1_path = args.base_ngram_root / "stage1_frozen.jsonl"
    ng_path = args.base_ngram_root / "selected_predictions.jsonl"
    full_path = args.base_v3_root / "selected_predictions.jsonl"
    for path in (stage1_path, ng_path, full_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    val = index_rows(read_jsonl(args.val), "Train-Val")
    stage1 = index_rows(read_jsonl(stage1_path), "Stage1")
    ng = index_selected(read_jsonl(ng_path), "V1 NGram predictions")
    full = index_selected(read_jsonl(full_path), "V3 full predictions")

    if len(val) != EXPECTED_ROWS or len(stage1) != EXPECTED_ROWS or set(val) != set(stage1):
        raise RuntimeError("Train-Val / Stage1 row alignment failed")

    rows_by_base: dict[str, list[dict[str, Any]]] = {base: [] for base in BASE_ORDER}
    for base in BASE_ORDER:
        for row_id in sorted(val):
            vrow = val[row_id]
            srow = stage1[row_id]
            npred = ng[base][row_id]
            fpred = full[base][row_id]
            s_top10 = [str(x) for x in srow["bases"][base]["top10"]]
            n_top10 = [str(x) for x in npred["top10"]]
            f_top10 = [str(x) for x in fpred["top10"]]
            if set(s_top10) != set(n_top10) or set(s_top10) != set(f_top10):
                raise RuntimeError(f"Candidate-set invariant failed: {base} {row_id}")
            rows_by_base[base].append({
                "row_id": row_id,
                "generic_missing": bool(srow["generic_missing"]),
                "gold_in_personal_k5": bool(srow["gold_in_personal_k5"]),
                "ambiguous": bool(vrow.get("ambiguous", False)),
                "conflict": bool(vrow.get("conflict", False)),
                "recovery_rank": srow["bases"][base]["gold_rank"],
                "ng_rank": npred.get("gold_rank"),
                "full_rank": fpred.get("gold_rank"),
            })

    ref_rows = rows_by_base[BASE_ORDER[0]]
    generic_missing_n = sum(r["generic_missing"] for r in ref_rows)
    recoverable_n = sum(r["generic_missing"] and r["gold_in_personal_k5"] for r in ref_rows)
    if generic_missing_n != EXPECTED_GENERIC_MISSING:
        raise RuntimeError(f"Generic Missing mismatch: {generic_missing_n}")
    if recoverable_n != EXPECTED_RECOVERABLE_K5:
        raise RuntimeError(f"Recoverable R mismatch: {recoverable_n}")

    output_rows: list[dict[str, Any]] = []
    for base in BASE_ORDER:
        rows = rows_by_base[base]
        for subset in SUBSETS:
            chosen = rows if subset == "overall" else [r for r in rows if subset in memberships(r)]
            for transition, before, after in TRANSITIONS:
                for k in TOP_K:
                    output_rows.append({
                        "base": base,
                        "subset": subset,
                        "transition": transition,
                        **transition_counts(chosen, before, after, k),
                    })

    csv_path = args.output_root / "topk_transitions.csv"
    json_path = args.output_root / "topk_transitions.json"
    write_csv(csv_path, output_rows)
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "complete",
        "experiment": "initial_recovery_context_topk_transitions_v1",
        "definition": {
            "rescue_at_k": "before rank > k or missing, after rank <= k",
            "harm_at_k": "before rank <= k, after rank > k or missing",
            "net_at_k": "rescue - harm",
        },
        "rows": EXPECTED_ROWS,
        "generic_missing_n": generic_missing_n,
        "recoverable_R_n": recoverable_n,
        "gold_used_for_posthoc_diagnosis_only": True,
        "tuning_performed": False,
        "dev3000_used": False,
        "test_used": False,
        "results": output_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== TOP-K RESCUE/HARM DIAGNOSTICS COMPLETE ===")
    print("Definition: rescue@k = outside Top-k -> inside Top-k; harm@k = inside Top-k -> outside Top-k")
    print(f"Rows: {EXPECTED_ROWS}  Generic Missing: {generic_missing_n}  Recoverable R: {recoverable_n}")

    for subset in ("overall", "generic_covered", "recoverable_R"):
        print(f"\n=== TOP3 RESCUE/HARM — {subset} ===")
        for base in BASE_ORDER:
            for transition, _, _ in TRANSITIONS:
                row = next(
                    r for r in output_rows
                    if r["base"] == base and r["subset"] == subset
                    and r["transition"] == transition and r["k"] == 3
                )
                print(
                    f"{base:<18} {transition:<16} "
                    f"rescue={row['rescue']:>5} harm={row['harm']:>5} net={row['net']:+5d}"
                )

    print("\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print("Dev3000 used: false")
    print("Test used: false")
    print("Diagnosis only: no tuning performed")


if __name__ == "__main__":
    main()
