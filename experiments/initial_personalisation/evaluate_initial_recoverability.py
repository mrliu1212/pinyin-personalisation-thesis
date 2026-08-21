from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_ROWS = 34416
K_VALUES = (1, 3, 5)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_of(row: dict[str, Any]) -> str:
    return str(row.get('gold', row.get('target')))


def candidate_texts(row: dict[str, Any]) -> list[str]:
    return [str(c['text']) for c in row['top10_candidates']]


def rank_of_generic(row: dict[str, Any], gold: str) -> int | None:
    for c in row['top10_candidates']:
        if str(c['text']) == gold:
            return int(c['rank'])
    return None


def compatibility(backend: Any, target: str, pinyin: tuple[str, ...]) -> tuple[bool, str]:
    chars = list(target)
    if len(chars) != len(pinyin):
        return False, 'character_count_mismatch'
    token_ids = backend.tokenizer.convert_tokens_to_ids(chars)
    for idx, (token_id, segment) in enumerate(zip(token_ids, pinyin)):
        if token_id == backend.tokenizer.unk_token_id:
            return False, f'tokenizer_unknown_at_{idx}'
        if token_id not in backend.allowed_token_ids.get(segment, ()):
            return False, f'pinyin_incompatible_at_{idx}'
    return True, 'compatible'


def metrics_from_ranks(ranks: list[int | None]) -> dict[str, Any]:
    n = len(ranks)
    if not n:
        return {'n': 0, 'top1': None, 'top3': None, 'mrr_at_10': None, 'missing_at_10': None, 'recall_at_10': None}
    missing = sum(r is None for r in ranks)
    return {
        'n': n,
        'top1': sum(r == 1 for r in ranks) / n,
        'top3': sum(r is not None and r <= 3 for r in ranks) / n,
        'mrr_at_10': sum(0.0 if r is None else 1.0 / r for r in ranks) / n,
        'missing_at_10': missing / n,
        'recall_at_10': 1.0 - missing / n,
    }


def safe_rate(n: int, d: int) -> float | None:
    return n / d if d else None


def main() -> None:
    ap = argparse.ArgumentParser(description='A2 Initial+Short standardized Train-Val candidate recoverability evaluator.')
    ap.add_argument('--initial-train-fit', type=Path, required=True)
    ap.add_argument('--initial-train-val', type=Path, required=True)
    ap.add_argument('--generic-predictions', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    fit = load_jsonl(args.initial_train_fit)
    val = load_jsonl(args.initial_train_val)
    generic = load_jsonl(args.generic_predictions)
    if len(val) != EXPECTED_ROWS or len(generic) != EXPECTED_ROWS:
        raise RuntimeError(f'Expected {EXPECTED_ROWS} Val and Generic rows; found {len(val)} and {len(generic)}')

    val_by_id = {str(r['row_id']): r for r in val}
    gen_by_id = {str(r['row_id']): r for r in generic}
    if set(val_by_id) != set(gen_by_id):
        raise RuntimeError('Train-Val and Generic row_id surfaces differ')

    from src.personalisation.pilot_a import HistoryIndex
    from src.personalisation.context_memory import PredictionQuery
    from src.reference_backend_pinyingpt import PinyinGPTConcatBackend

    combined = fit + val
    index = HistoryIndex(combined, 5000)
    backend = PinyinGPTConcatBackend(args.checkpoint, device=args.device)

    out_rows = []
    incompat = Counter()
    authors = sorted({str(r['author']) for r in val})

    for number, row in enumerate(val, 1):
        rid = str(row['row_id'])
        pred = gen_by_id[rid]
        gold = gold_of(row)
        pinyin = tuple(str(x) for x in row['pinyin_segments'])
        query = PredictionQuery(
            row_id=rid,
            author=str(row['author']),
            work_id=str(row['work_id']),
            chronological_position=int(row['chronological_position']),
            context=str(row['context']),
            pinyin=pinyin,
        )
        visible = index.visible(query)

        counts = Counter(str(h.get('target', h.get('gold'))) for h in visible)
        # Historical PV/EM1 ordering: descending frequency, then lexical tie-break.
        lexicon = sorted(counts, key=lambda t: (-counts[t], t))
        generic_text = set(candidate_texts(pred))
        personal_only = [t for t in lexicon if t not in generic_text]

        compatible_personal = []
        incompatible_personal = []
        for target in personal_only:
            ok, reason = compatibility(backend, target, pinyin)
            if ok:
                compatible_personal.append(target)
            else:
                incompatible_personal.append(target)
                incompat[reason] += 1

        generic_rank = rank_of_generic(pred, gold)
        missing = generic_rank is None
        gold_in_history = gold in counts
        raw_any = bool(missing and gold in personal_only)
        compatible_any = bool(missing and gold in compatible_personal)

        raw_at = {k: bool(missing and gold in personal_only[:k]) for k in K_VALUES}
        comp_at = {k: bool(missing and gold in compatible_personal[:k]) for k in K_VALUES}

        # Oracle coverage only: if Gold is inserted by PersonalTopK, it is considered covered.
        oracle_covered = {k: (generic_rank is not None or raw_at[k]) for k in K_VALUES}
        oracle_compatible_covered = {k: (generic_rank is not None or comp_at[k]) for k in K_VALUES}

        out_rows.append({
            'row_id': rid,
            'author': str(row['author']),
            'gold': gold,
            'generic_rank': generic_rank,
            'generic_missing': missing,
            'history_available': bool(visible),
            'ambiguous': bool(row.get('ambiguous', len(counts) >= 2)),
            'conflict': bool(row.get('conflict', False)),
            'same_initial_history_count': len(visible),
            'distinct_initial_targets': len(counts),
            'raw_personal_only_count': len(personal_only),
            'compatible_personal_only_count': len(compatible_personal),
            'incompatible_personal_only_count': len(incompatible_personal),
            'gold_in_history': gold_in_history,
            'raw_recoverable_any': raw_any,
            'compatible_recoverable_any': compatible_any,
            **{f'raw_recoverable_at_{k}': raw_at[k] for k in K_VALUES},
            **{f'compatible_recoverable_at_{k}': comp_at[k] for k in K_VALUES},
            **{f'oracle_raw_covered_at_{k}': oracle_covered[k] for k in K_VALUES},
            **{f'oracle_compatible_covered_at_{k}': oracle_compatible_covered[k] for k in K_VALUES},
            'personal_only_targets_top5': personal_only[:5],
            'compatible_personal_only_targets_top5': compatible_personal[:5],
        })
        if number % 1000 == 0 or number == len(val):
            print(f'A2: {number}/{len(val)}', flush=True)

    subsets = {
        'Overall': lambda r: True,
        'History Available': lambda r: r['history_available'],
        'Ambiguous': lambda r: r['ambiguous'],
        'Conflict': lambda r: r['conflict'],
        'Generic Missing': lambda r: r['generic_missing'],
        'Raw Recoverable Missing': lambda r: r['raw_recoverable_any'],
        'Compatible Recoverable Missing': lambda r: r['compatible_recoverable_any'],
    }

    summary_subsets: dict[str, Any] = {}
    for name, pred_fn in subsets.items():
        selected = [r for r in out_rows if pred_fn(r)]
        ranks = [r['generic_rank'] for r in selected]
        base = metrics_from_ranks(ranks)
        missing_n = sum(r['generic_missing'] for r in selected)
        raw_any_n = sum(r['raw_recoverable_any'] for r in selected)
        comp_any_n = sum(r['compatible_recoverable_any'] for r in selected)
        base.update({
            'history_available_rate': safe_rate(sum(r['history_available'] for r in selected), len(selected)),
            'ambiguous_rate': safe_rate(sum(r['ambiguous'] for r in selected), len(selected)),
            'conflict_rate': safe_rate(sum(r['conflict'] for r in selected), len(selected)),
            'generic_missing_n': missing_n,
            'raw_recoverable_any_n': raw_any_n,
            'raw_recoverable_given_missing': safe_rate(raw_any_n, missing_n),
            'compatible_recoverable_any_n': comp_any_n,
            'compatible_recoverable_given_missing': safe_rate(comp_any_n, missing_n),
        })
        for k in K_VALUES:
            rawk = sum(r[f'raw_recoverable_at_{k}'] for r in selected)
            compk = sum(r[f'compatible_recoverable_at_{k}'] for r in selected)
            raw_covered = sum(r[f'oracle_raw_covered_at_{k}'] for r in selected)
            comp_covered = sum(r[f'oracle_compatible_covered_at_{k}'] for r in selected)
            base[f'raw_recoverable_at_{k}_given_missing'] = safe_rate(rawk, missing_n)
            base[f'compatible_recoverable_at_{k}_given_missing'] = safe_rate(compk, missing_n)
            base[f'oracle_raw_recall_at_10_plus_personal_{k}'] = safe_rate(raw_covered, len(selected))
            base[f'oracle_compatible_recall_at_10_plus_personal_{k}'] = safe_rate(comp_covered, len(selected))
        summary_subsets[name] = base

    per_author = {}
    for author in authors:
        selected = [r for r in out_rows if r['author'] == author]
        missing_n = sum(r['generic_missing'] for r in selected)
        ranks = [r['generic_rank'] for r in selected]
        m = metrics_from_ranks(ranks)
        m['raw_recoverable_given_missing'] = safe_rate(sum(r['raw_recoverable_any'] for r in selected), missing_n)
        m['compatible_recoverable_given_missing'] = safe_rate(sum(r['compatible_recoverable_any'] for r in selected), missing_n)
        per_author[author] = m

    macro_top1 = statistics.fmean(v['top1'] for v in per_author.values()) if per_author else None

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_root / 'recoverability_rows.jsonl'
    with rows_path.open('w', encoding='utf-8', newline='\n') as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')

    summary = {
        'schema_version': 1,
        'experiment': 'initial_short_standardized_train_val_recoverability_v1',
        'status': 'complete',
        'history_budget': 5000,
        'history_semantics': 'same author -> strictly prior -> latest 5000 raw interactions -> exact Initial-segment match',
        'personal_candidate_order': 'frequency descending, lexical tie-break',
        'k_values': list(K_VALUES),
        'rows': len(out_rows),
        'authors': authors,
        'generic_macro_author_top1': macro_top1,
        'subsets': summary_subsets,
        'per_author': per_author,
        'backend_incompatibility_reasons': dict(incompat),
        'provenance': {
            'train_fit_sha256': sha256_file(args.initial_train_fit),
            'train_val_sha256': sha256_file(args.initial_train_val),
            'generic_predictions_sha256': sha256_file(args.generic_predictions),
            'checkpoint_path': str(args.checkpoint.resolve()),
            'rows_output_sha256': sha256_file(rows_path),
        },
        'dev3000_used': False,
        'test_used': False,
        'note': 'Oracle recovery coverage measures candidate-surface coverage only; it is not a realized reranking accuracy.',
    }
    (args.output_root / 'recoverability_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )

    # Flat headline table for quick inspection.
    csv_path = args.output_root / 'recoverability_metrics.csv'
    fields = ['subset','n','top1','top3','mrr_at_10','missing_at_10','recall_at_10',
              'raw_recoverable_given_missing','compatible_recoverable_given_missing']
    for k in K_VALUES:
        fields += [f'raw_recoverable_at_{k}_given_missing', f'compatible_recoverable_at_{k}_given_missing',
                   f'oracle_raw_recall_at_10_plus_personal_{k}', f'oracle_compatible_recall_at_10_plus_personal_{k}']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, values in summary_subsets.items():
            w.writerow({'subset': name, **{k: values.get(k) for k in fields if k != 'subset'}})

    print('\n=== A2 INITIAL RECOVERABILITY COMPLETE ===')
    overall = summary_subsets['Overall']
    print('Rows:', overall['n'])
    print('Generic Top1:', overall['top1'])
    print('Generic Missing@10:', overall['missing_at_10'])
    print('Raw recoverable | missing:', overall['raw_recoverable_given_missing'])
    print('Compatible recoverable | missing:', overall['compatible_recoverable_given_missing'])
    for k in K_VALUES:
        print(f'Compatible recoverable@{k} | missing:', overall[f'compatible_recoverable_at_{k}_given_missing'])
    print('Summary:', args.output_root / 'recoverability_summary.json')


if __name__ == '__main__':
    main()