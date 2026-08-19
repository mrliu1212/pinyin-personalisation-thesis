"""EM-2A: PinyinGPT hidden-state extraction engineering gate.

Purpose:
Verify that the final-layer hidden state at the final prompt token is
exactly the representation whose LM-head logits drive prediction of the
first target Chinese character.

This is an engineering gate only.
No retrieval metric or Test data is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.reference_backend_pinyingpt.backend import (
    PinyinGPTConcatBackend,
)


AUTHORS = (
    "Etinjat",
    "Re_spectators",
    "breaddddd",
)

SAMPLES_PER_AUTHOR = 3
TOLERANCE = 1e-4

EXPECTED_DEV_CACHE_SHA256 = (
    "588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    actual_hash = sha256_file(path)

    if actual_hash != EXPECTED_DEV_CACHE_SHA256:
        raise RuntimeError(
            "Frozen Dev Generic cache SHA mismatch:\n"
            f"expected={EXPECTED_DEV_CACHE_SHA256}\n"
            f"actual={actual_hash}"
        )

    rows = []

    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)

            if str(row.get("author")) not in AUTHORS:
                continue

            if (
                "pilot_partition" in row
                and row["pilot_partition"] != "tune"
            ):
                continue

            if not row.get("pinyin_segments"):
                continue

            if not row.get("top10_candidates"):
                continue

            rows.append(row)

    return rows


def choose_samples(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []

    for author in AUTHORS:
        author_rows = [
            row
            for row in rows
            if str(row["author"]) == author
        ]

        author_rows.sort(
            key=lambda row: str(
                row.get(
                    "anchor_id",
                    row.get("condition_id", ""),
                )
            )
        )

        if len(author_rows) < SAMPLES_PER_AUTHOR:
            raise RuntimeError(
                f"Not enough Dev rows for {author}: "
                f"{len(author_rows)}"
            )

        selected.extend(
            author_rows[:SAMPLES_PER_AUTHOR]
        )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dev-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    rows = load_rows(args.dev_cache)
    samples = choose_samples(rows)

    print(
        f"Loading Frozen PinyinGPT: {args.checkpoint}"
    )

    backend = PinyinGPTConcatBackend(
        args.checkpoint,
        device=args.device,
    )

    torch = backend.torch

    results = []

    for index, row in enumerate(samples, start=1):
        author = str(row["author"])

        context = str(
            row["model_used_context"]
        )

        pinyin = tuple(
            str(value)
            for value in row["pinyin_segments"]
        )

        prompt_ids, prompt_positions = (
            backend._prompt(
                context,
                pinyin,
            )
        )

        if prompt_ids[-1] != backend.tokenizer.sep_token_id:
            raise RuntimeError(
                "Prompt does not end in [SEP]"
            )

        input_ids = torch.tensor(
            [prompt_ids],
            device=backend.device,
        )

        position_ids = torch.tensor(
            [prompt_positions],
            device=backend.device,
        )

        with torch.inference_mode():
            output = backend.model(
                input_ids=input_ids,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        if output.hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden states"
            )

        final_hidden = (
            output.hidden_states[-1][0, -1]
        )

        direct_logits = (
            output.logits[0, -1].float()
        )

        with torch.inference_mode():
            reconstructed_logits = (
                backend.model.lm_head(
                    final_hidden
                ).float()
            )

        hidden_to_logits_diff = float(
            (
                direct_logits
                - reconstructed_logits
            )
            .abs()
            .max()
            .item()
        )

        first_segment = pinyin[0]

        allowed = torch.tensor(
            backend.allowed_token_ids[
                first_segment
            ],
            device=backend.device,
        )

        direct_distribution = (
            torch.log_softmax(
                direct_logits,
                dim=-1,
            )
        )

        reconstructed_distribution = (
            torch.log_softmax(
                reconstructed_logits,
                dim=-1,
            )
        )

        allowed_direct = (
            direct_distribution.index_select(
                0,
                allowed,
            )
        )

        allowed_reconstructed = (
            reconstructed_distribution.index_select(
                0,
                allowed,
            )
        )

        allowed_distribution_diff = float(
            (
                allowed_direct
                - allowed_reconstructed
            )
            .abs()
            .max()
            .item()
        )

        direct_best_index = int(
            allowed_direct.argmax().item()
        )

        reconstructed_best_index = int(
            allowed_reconstructed.argmax().item()
        )

        direct_best_token_id = int(
            allowed[
                direct_best_index
            ].item()
        )

        reconstructed_best_token_id = int(
            allowed[
                reconstructed_best_index
            ].item()
        )

        best_token_agrees = (
            direct_best_token_id
            == reconstructed_best_token_id
        )

        top_candidate = (
            row["top10_candidates"][0]
        )

        candidate = str(
            top_candidate["text"]
        )

        characters = list(candidate)

        if len(characters) != len(pinyin):
            raise RuntimeError(
                f"Candidate length mismatch: "
                f"{candidate!r}"
            )

        candidate_ids = (
            backend.tokenizer.convert_tokens_to_ids(
                characters
            )
        )

        # Reproduce score_candidates position logic.
        output_positions = [
            prompt_positions[
                -len(pinyin) - 1
            ]
            + offset
            for offset in range(
                len(candidate_ids)
            )
        ]

        full_sequence = (
            prompt_ids
            + candidate_ids
        )

        full_positions = (
            prompt_positions
            + output_positions
        )

        with torch.inference_mode():
            teacher_output = backend.model(
                input_ids=torch.tensor(
                    [full_sequence],
                    device=backend.device,
                ),
                position_ids=torch.tensor(
                    [full_positions],
                    device=backend.device,
                ),
            ).logits[0]

        teacher_first_distribution = (
            torch.log_softmax(
                teacher_output[
                    len(prompt_ids) - 1
                ].float(),
                dim=-1,
            )
        )

        first_candidate_id = int(
            candidate_ids[0]
        )

        direct_first_score = float(
            direct_distribution[
                first_candidate_id
            ].item()
        )

        teacher_first_score = float(
            teacher_first_distribution[
                first_candidate_id
            ].item()
        )

        first_step_diff = abs(
            direct_first_score
            - teacher_first_score
        )

        # Independent existing backend path.
        exact_score = backend.score_candidates(
            context=context,
            typed_pinyin=pinyin,
            candidates=(candidate,),
        )[0].log_probability

        cached_beam_score = float(
            top_candidate[
                "log_probability"
            ]
        )

        cached_score_diff = abs(
            float(exact_score)
            - cached_beam_score
        )

        prompt_final_token = (
            backend.tokenizer.convert_ids_to_tokens(
                prompt_ids[-1]
            )
        )

        row_pass = (
            hidden_to_logits_diff
            <= TOLERANCE
            and allowed_distribution_diff
            <= TOLERANCE
            and first_step_diff
            <= TOLERANCE
            and cached_score_diff
            <= TOLERANCE
            and best_token_agrees
            and final_hidden.shape[-1]
            == backend.model.config.n_embd
        )

        result = {
            "sample": index,
            "author": author,
            "anchor_id": str(
                row.get("anchor_id", "")
            ),
            "pinyin_segments": list(pinyin),
            "prompt_length": len(
                prompt_ids
            ),
            "prompt_final_index": (
                len(prompt_ids) - 1
            ),
            "prompt_final_token": (
                prompt_final_token
            ),
            "hidden_layer": "final",
            "hidden_size": int(
                final_hidden.shape[-1]
            ),
            "expected_hidden_size": int(
                backend.model.config.n_embd
            ),
            "hidden_to_logits_max_abs_diff": (
                hidden_to_logits_diff
            ),
            "allowed_distribution_max_abs_diff": (
                allowed_distribution_diff
            ),
            "first_candidate": candidate,
            "first_step_logprob_direct": (
                direct_first_score
            ),
            "first_step_logprob_teacher": (
                teacher_first_score
            ),
            "first_step_abs_diff": (
                first_step_diff
            ),
            "cached_beam_log_probability": (
                cached_beam_score
            ),
            "fixed_candidate_log_probability": (
                float(exact_score)
            ),
            "cached_vs_fixed_abs_diff": (
                cached_score_diff
            ),
            "best_allowed_next_token_agrees": (
                best_token_agrees
            ),
            "pass": row_pass,
        }

        results.append(result)

        print(
            f"[{index}/{len(samples)}] "
            f"{author} "
            f"pinyin={' '.join(pinyin)} "
            f"hidden->logits={hidden_to_logits_diff:.3e} "
            f"first-step={first_step_diff:.3e} "
            f"cached-fixed={cached_score_diff:.3e} "
            f"{'PASS' if row_pass else 'FAIL'}"
        )

    maximum_hidden_diff = max(
        row[
            "hidden_to_logits_max_abs_diff"
        ]
        for row in results
    )

    maximum_distribution_diff = max(
        row[
            "allowed_distribution_max_abs_diff"
        ]
        for row in results
    )

    maximum_first_step_diff = max(
        row[
            "first_step_abs_diff"
        ]
        for row in results
    )

    maximum_cached_diff = max(
        row[
            "cached_vs_fixed_abs_diff"
        ]
        for row in results
    )

    all_pass = all(
        row["pass"]
        for row in results
    )

    summary = {
        "experiment": (
            "em2a_hidden_state_extraction_gate"
        ),
        "partition": "dev_tune_only",
        "authors": list(AUTHORS),
        "samples": len(results),
        "samples_per_author": (
            SAMPLES_PER_AUTHOR
        ),
        "tolerance": TOLERANCE,
        "representation": (
            "final-layer hidden state at "
            "final prompt token"
        ),
        "prompt_final_token_expected": (
            "[SEP]"
        ),
        "hidden_size": int(
            backend.model.config.n_embd
        ),
        "maximum_hidden_to_logits_abs_diff": (
            maximum_hidden_diff
        ),
        "maximum_allowed_distribution_abs_diff": (
            maximum_distribution_diff
        ),
        "maximum_first_step_abs_diff": (
            maximum_first_step_diff
        ),
        "maximum_cached_vs_fixed_abs_diff": (
            maximum_cached_diff
        ),
        "best_allowed_token_agreement": (
            sum(
                row[
                    "best_allowed_next_token_agrees"
                ]
                for row in results
            )
        ),
        "passed_samples": sum(
            row["pass"]
            for row in results
        ),
        "gate_pass": all_pass,
        "used_gold": False,
        "inspected_retrieval_metrics": False,
        "dev_cache_sha256": (
            sha256_file(
                args.dev_cache
            )
        ),
    }

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        args.output_root
        / "rows.jsonl"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for row in results:
            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    with (
        args.output_root
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        json.dump(
            summary,
            destination,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        destination.write("\n")

    print()
    print(
        "=== EM-2A Hidden-State "
        "Extraction Gate ==="
    )
    print(
        f"Samples: {summary['samples']}"
    )
    print(
        f"Passed: "
        f"{summary['passed_samples']}"
        f"/{summary['samples']}"
    )
    print(
        "Representation: "
        f"{summary['representation']}"
    )
    print(
        f"Hidden size: "
        f"{summary['hidden_size']}"
    )
    print(
        "Max hidden -> logits diff: "
        f"{maximum_hidden_diff:.12g}"
    )
    print(
        "Max allowed distribution diff: "
        f"{maximum_distribution_diff:.12g}"
    )
    print(
        "Max first-step score diff: "
        f"{maximum_first_step_diff:.12g}"
    )
    print(
        "Max cached beam vs fixed score diff: "
        f"{maximum_cached_diff:.12g}"
    )
    print(
        "Best allowed next-token agreement: "
        f"{summary['best_allowed_token_agreement']}"
        f"/{summary['samples']}"
    )
    print(
        f"Gold used: {summary['used_gold']}"
    )
    print(
        "Retrieval metrics inspected: "
        f"{summary['inspected_retrieval_metrics']}"
    )
    print()

    if not all_pass:
        raise SystemExit(
            "STOP: EM-2A extraction gate FAILED."
        )

    print(
        "PASS: final prompt hidden state is "
        "engineering-aligned with the Frozen "
        "PinyinGPT first-character prediction path."
    )


if __name__ == "__main__":
    main()
