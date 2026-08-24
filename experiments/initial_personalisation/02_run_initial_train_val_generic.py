from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_ROWS = 34416
BEAM_SIZE = 16
TOP_K = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if 'row_id' not in row:
                raise RuntimeError(f'Missing row_id at line {line_no}')
            rows.append(row)
    return rows


def load_existing(path: Path, expected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = str(row.get('row_id', ''))
            if rid in done or rid not in expected:
                raise RuntimeError(f'Invalid/duplicate cached row_id at line {line_no}: {rid}')
            src = expected[rid]
            for key in ('author','work_id','chronological_position','context','pinyin_input','pinyin_segments','gold'):
                if row.get(key) != src.get(key):
                    raise RuntimeError(f'Cached row differs from source for {rid}: {key}')
            candidates = row.get('top10_candidates')
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= TOP_K:
                raise RuntimeError(f'Invalid candidate list for {rid}')
            ranks = [int(c['rank']) for c in candidates]
            if ranks != list(range(1, len(candidates) + 1)):
                raise RuntimeError(f'Invalid ranks for {rid}')
            if len({str(c['text']) for c in candidates}) != len(candidates):
                raise RuntimeError(f'Duplicate candidate text for {rid}')
            done[rid] = row
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description='Resumable standardized Initial+Short Train-Val Generic Top10 runner.')
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--window-size', type=int, default=16)
    ap.add_argument('--micro-batch', type=int, default=2)
    args = ap.parse_args()

    if args.window_size <= 0 or args.micro_batch <= 0:
        raise ValueError('window-size and micro-batch must be positive')

    rows = load_jsonl(args.input)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f'Expected {EXPECTED_ROWS} rows, found {len(rows)}')
    if any(r.get('condition') != 'initial_short' for r in rows):
        raise RuntimeError('Input contains non-initial_short rows')
    ids = [str(r['row_id']) for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError('Duplicate row_id in input')

    expected = {str(r['row_id']): r for r in rows}
    args.output_root.mkdir(parents=True, exist_ok=True)
    partial = args.output_root / 'predictions.partial.jsonl'
    final = args.output_root / 'predictions.jsonl'
    runtime_path = args.output_root / 'runtime_summary.json'

    # Prefer final cache if it exists; otherwise resume partial.
    existing_path = final if final.exists() else partial
    completed = load_existing(existing_path, expected) if existing_path.exists() else {}
    pending = [r for r in rows if str(r['row_id']) not in completed]

    print('=== A1 STANDARDIZED INITIAL GENERIC ===')
    print('Input rows:', len(rows))
    print('Already cached:', len(completed))
    print('Pending:', len(pending))
    print('Beam:', BEAM_SIZE, 'Top-K:', TOP_K, 'Device:', args.device)

    if not pending:
        print('Cache already complete; no inference needed.')
        return

    from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

    load_started = time.perf_counter()
    backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)
    load_seconds = time.perf_counter() - load_started

    latencies_ms: list[float] = []
    started = time.perf_counter()
    mode = 'a' if partial.exists() and partial.stat().st_size else 'w'

    with partial.open(mode, encoding='utf-8', newline='\n') as out:
        for window_start in range(0, len(pending), args.window_size):
            window = pending[window_start:window_start + args.window_size]
            prepared = []
            for row in window:
                segments = list(row['pinyin_segments'])
                used_context, original_tokens, used_tokens, truncated = backend.truncate_context_for_generation(
                    str(row['context']), segments
                )
                prompt, _ = backend._prompt(used_context, segments)
                prepared.append((row, segments, used_context, original_tokens, used_tokens, truncated, len(prompt)))

            groups: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
            for item in prepared:
                groups[(len(item[1]), item[6])].append(item)

            for equal_shape_group in groups.values():
                for group_start in range(0, len(equal_shape_group), args.micro_batch):
                    group = equal_shape_group[group_start:group_start + args.micro_batch]
                    t0 = time.perf_counter()
                    results = backend.generate_batch(
                        [(item[2], item[1]) for item in group], top_k=TOP_K, beam_size=BEAM_SIZE
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.extend([elapsed_ms / len(group)] * len(group))

                    for item, result in zip(group, results):
                        row, _, used_context, original_tokens, used_tokens, truncated, _ = item
                        candidates = [c.to_dict() for c in result.candidates]
                        gold = str(row.get('gold', row.get('target')))
                        gold_rank = next((int(c['rank']) for c in candidates if str(c['text']) == gold), None)
                        output = {
                            **row,
                            'model_used_context': used_context,
                            'original_stored_context_tokens': original_tokens,
                            'model_used_context_tokens': used_tokens,
                            'context_truncated': truncated,
                            'top10_candidates': candidates,
                            'gold_rank': gold_rank,
                            'beam_size': BEAM_SIZE,
                            'top_k': TOP_K,
                            'runtime_device': result.runtime_device,
                            'checkpoint_path': str(args.checkpoint.resolve()),
                            'generic_condition': 'initial_short',
                            'generic_stage': 'standardized_train_val_v1',
                        }
                        out.write(canonical_json(output) + '\n')
                        completed[str(row['row_id'])] = output
                    out.flush()

            done = len(completed)
            elapsed = time.perf_counter() - started
            rate = (done - (EXPECTED_ROWS - len(pending))) / elapsed if elapsed else 0.0
            print(f'Generic: {done}/{EXPECTED_ROWS} current_run_rate={rate:.2f} rows/s', flush=True)

    completed = load_existing(partial, expected)
    if set(completed) != set(expected):
        raise RuntimeError(f'Partial cache incomplete after run: {len(completed)}/{EXPECTED_ROWS}')

    # Canonical final file in input order.
    with final.open('w', encoding='utf-8', newline='\n') as out:
        for row in rows:
            out.write(canonical_json(completed[str(row['row_id'])]) + '\n')

    summary = {
        'schema_version': 1,
        'experiment': 'initial_short_standardized_train_val_generic_v1',
        'status': 'complete',
        'rows': EXPECTED_ROWS,
        'beam_size': BEAM_SIZE,
        'top_k': TOP_K,
        'device': args.device,
        'input_path': str(args.input.resolve()),
        'input_sha256': sha256_file(args.input),
        'checkpoint_path': str(args.checkpoint.resolve()),
        'model_load_seconds': load_seconds,
        'run_seconds': time.perf_counter() - started,
        'mean_inference_ms_per_row': statistics.fmean(latencies_ms) if latencies_ms else None,
        'median_inference_ms_per_row': statistics.median(latencies_ms) if latencies_ms else None,
        'prediction_path': str(final.resolve()),
        'prediction_sha256': sha256_file(final),
        'dev3000_used': False,
        'test_used': False,
    }
    runtime_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('COMPLETE:', final)
    print('SHA256:', summary['prediction_sha256'])


if __name__ == '__main__':
    main()
