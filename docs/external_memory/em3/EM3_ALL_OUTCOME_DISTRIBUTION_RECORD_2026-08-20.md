# EM3 Full G/F/H Outcome Distribution Record — 2026-08-20

**Condition:** Full+Short / H5000 / old exploratory 3-author Dev diagnostic surface

**Authors:** Etinjat, Re_spectators, breaddddd

**Rows:** 5608

**Test used:** No

> Gold-based fields are analysis-only oracle diagnostics; they are not available at inference time.

## 1. Complete 8-way outcome table

| G | F | H | N | Share | Gold in history | Raw winner = Gold | Gold in candidate set |
|---|---|---|---:|---:|---:|---:|---:|
| ✓ | ✓ | ✓ | 3361 | 59.93% | 68.8% | 66.3% | 100.0% |
| ✓ | ✓ | ✗ | 24 | 0.43% | 100.0% | 66.7% | 100.0% |
| ✓ | ✗ | ✓ | 42 | 0.75% | 78.6% | 0.0% | 100.0% |
| ✓ | ✗ | ✗ | 100 | 1.78% | 39.0% | 0.0% | 100.0% |
| ✗ | ✓ | ✓ | 403 | 7.19% | 100.0% | 92.8% | 100.0% |
| ✗ | ✓ | ✗ | 35 | 0.62% | 100.0% | 82.9% | 100.0% |
| ✗ | ✗ | ✓ | 45 | 0.80% | 100.0% | 20.0% | 100.0% |
| ✗ | ✗ | ✗ | 1598 | 28.50% | 20.2% | 7.3% | 55.6% |

## 2. Aggregate method accuracy and Context net effect

| Metric | Count | Share / accuracy |
|---|---:|---:|
| Generic correct | 3527 | 62.89% |
| Frequency correct | 3823 | 68.17% |
| Hidden-M1 correct | 3851 | 68.67% |
| F wrong → H correct (Context rescue) | 87 | 1.55% |
| F correct → H wrong (Context harm) | 59 | 1.05% |
| Net H over F | +28 | 0.50% |

## 3. Focused subset summary

| Subset | N | Authors | Gold in history | Gold in candidates | Raw winner = Gold | Top-3 doc share |
|---|---:|---|---:|---:|---:|---:|
| G✓ H✗ | 124 | Etinjat 79; Re_spectators 4; breaddddd 41 | 50.8% | 100% | 12.9% | 66.9% |
| F✓ H✗ | 59 | Etinjat 45; Re_spectators 1; breaddddd 13 | 100% | 100% | 76.3% | 74.6% |
| G✓ F✓ H✗ | 24 | Etinjat 14; Re_spectators 1; breaddddd 9 | 100% | 100% | 66.7% | 75.0% |
| G✓ F✗ H✗ | 100 | Etinjat 65; Re_spectators 3; breaddddd 32 | 39.0% | 100% | 0.0% | 65.0% |
| G✗ F✓ H✗ | 35 | Etinjat 31; breaddddd 4 | 100% | 100% | 82.9% | 77.1% |
| G✗ F✓ H✓ | 403 | Etinjat 237; Re_spectators 8; breaddddd 158 | 100% | 100% | 92.8% | 69.7% |
| G✗ F✗ H✗ + Gold in history | 323 | Etinjat 239; Re_spectators 5; breaddddd 79 | 100% | 53.9% | 36.2% | 76.5% |
| G✗ F✗ H✗ + Gold not in history | 1275 | Etinjat 951; Re_spectators 7; breaddddd 317 | 0% | 56.1% | 0% | 74.2% |
| G✗ F✗ H✓ + Gold in history | 45 | Etinjat 33; breaddddd 12 | 100% | 100% | 20.0% | 84.4% |

## 4. History-distribution statistics

| Subset | Same-Pinyin history | Raw winner share | Raw margin | Distinct targets | Entropy (bits) |
|---|---|---|---|---|---|
| G✓ H✗ | mean 24.70; med 7.5; P25 2; P75 36.25; P90 74.7; max 128 | mean .755; med .750; P25 .570; P75 1.000; P90 1.000 | mean 11.86; med 3; P25 1; P75 14.25; P90 37.5; max 107 | mean 3.35; med 2; P25 1; P75 4; P90 7; max 13 | mean .865; med .892; P25 0; P75 1.393; P90 1.953; max 3.386 |
| F✓ H✗ | mean 40.37; med 26; P25 8; P75 69; P90 93; max 128 | mean .587; med .571; P25 .447; P75 .701; P90 .836 | mean 15.95; med 7; P25 2; P75 22; P90 39; max 107 | mean 4.97; med 4; P25 3; P75 6; P90 10; max 13 | mean 1.462; med 1.500; P25 1.074; P75 1.875; P90 2.057; max 2.896 |
| G✓ F✓ H✗ | mean 57.83; med 69; P25 18.25; P75 89.25; P90 105.2; max 128 | mean .587; med .519; P25 .444; P75 .705; P90 .824 | mean 21.75; med 15.5; P25 2; P75 34.5; P90 47.4; max 107 | mean 5.33; med 3; P25 3; P75 7.25; P90 11.4; max 13 | mean 1.420; med 1.326; P25 1.126; P75 1.930; P90 2.081; max 2.518 |
| G✓ F✗ H✗ | mean 16.75; med 5; P25 2; P75 22; P90 59.1; max 113 | mean .795; med .825; P25 .656; P75 1.000; P90 1.000 | mean 9.49; med 2; P25 1; P75 8.25; P90 33; max 103 | mean 2.87; med 2; P25 1; P75 3.25; P90 6; max 13 | mean .732; med .701; P25 0; P75 1.196; P90 1.874; max 3.386 |
| G✗ F✓ H✗ | mean 28.40; med 16; P25 6; P75 46.5; P90 67.2; max 109 | mean .587; med .571; P25 .485; P75 .695; P90 .822 | mean 11.97; med 5; P25 1.5; P75 21; P90 32; max 54 | mean 4.71; med 4; P25 3; P75 5.5; P90 7.6; max 12 | mean 1.490; med 1.684; P25 1.074; P75 1.862; P90 1.990; max 2.896 |
| G✗ F✓ H✓ | mean 23.51; med 7; P25 2; P75 30; P90 77; max 276 | mean .824; med .889; P25 .667; P75 1.000; P90 1.000 | mean 16.04; med 4; P25 1; P75 20; P90 54; max 269 | mean 2.72; med 2; P25 1; P75 3; P90 6; max 13 | mean .654; med .531; P25 0; P75 1.057; P90 1.763; max 3.169 |
| G✗ F✗ H✗ + Gold in history | mean 27.33; med 14; P25 3; P75 43; P90 76.8; max 163 | mean .665; med .638; P25 .449; P75 .940; P90 1.000 | mean 10.49; med 3; P25 1; P75 14.5; P90 29.8; max 157 | mean 4.35; med 3; P25 2; P75 6; P90 9.8; max 13 | mean 1.192; med 1.228; P25 .357; P75 1.914; P90 2.368; max 3.499 |
| G✗ F✗ H✗ + Gold not in history | mean 3.07; med 0; P25 0; P75 1; P90 5; max 146 | mean .199; med 0; P25 0; P75 .250; P90 1.000 | mean 1.64; med 0; P25 0; P75 0; P90 2; max 138 | mean .74; med 0; P25 0; P75 1; P90 2; max 13 | mean .197; med 0; P25 0; P75 0; P90 .918; max 3.482 |
| G✗ F✗ H✓ + Gold in history | mean 46.87; med 48; P25 7; P75 75; P90 93; max 126 | mean .578; med .571; P25 .452; P75 .707; P90 .782; max .967 | mean 16.89; med 13; P25 2; P75 22; P90 44.6; max 70 | mean 5.24; med 4; P25 3; P75 7; P90 10.6; max 13 | mean 1.497; med 1.349; P25 1.000; P75 2.043; P90 2.269; max 3.486 |

## 5. Pattern and document concentration

### G✓ H✗

- Top patterns: Etinjat you→又 ×4; Etinjat yi→亦 ×4; breaddddd you→有 ×4
- Top documents: Etinjat 1278079946 ×34; breaddddd 1306372902 ×32; Etinjat 1306311171 ×17
- Top-3 document share: 66.9%

### F✓ H✗

- Top patterns: Etinjat yi→亦 ×5; breaddddd you→有 ×4; Etinjat you→有 ×3
- Top documents: Etinjat 1306311171 ×18; Etinjat 1278079946 ×16; breaddddd 1306372902 ×10
- Top-3 document share: 74.6%

### G✓ F✓ H✗

- Top patterns: Etinjat yi→亦 ×4; breaddddd you→有 ×4
- Top documents: breaddddd 1306372902 ×8; Etinjat 1306311171 ×6; Etinjat 1278079946 ×4
- Top-3 document share: 75.0%

### G✓ F✗ H✗

- Top patterns: Etinjat you→又 ×3; Etinjat dao→倒 ×3; Etinjat dao→到 ×2
- Top documents: Etinjat 1278079946 ×30; breaddddd 1306372902 ×24; Etinjat 1306311171 ×11
- Top-3 document share: 65.0%

### G✗ F✓ H✗

- Top patterns: Etinjat wei→为 ×3; Etinjat zai→在 ×2; Etinjat di→地 ×2
- Top documents: Etinjat 1278079946 ×12; Etinjat 1306311171 ×12; Etinjat 1304974711 ×3
- Top-3 document share: 77.1%

### G✗ F✓ H✓

- Top patterns: breaddddd wei→未 ×16; Etinjat zai→在 ×13; breaddddd zai→在 ×10
- Top documents: breaddddd 1306372902 ×126; Etinjat 1306311171 ×82; Etinjat 1278079946 ×73
- Top-3 document share: 69.7%

### G✗ F✗ H✗ + Gold in history

- Top patterns: Etinjat yi→以 ×12; breaddddd ji jin hui→基金会 ×9; Etinjat yu→欲 ×8
- Top documents: Etinjat 1306311171 ×131; breaddddd 1306372902 ×74; Etinjat 1278079946 ×42
- Top-3 document share: 76.5%

### G✗ F✗ H✗ + Gold not in history

- Top patterns: mostly singletons
- Top documents: Etinjat 1306311171 ×366; Etinjat 1278079946 ×300; breaddddd 1306372902 ×280
- Top-3 document share: 74.2%

### G✗ F✗ H✓ + Gold in history

- Top patterns: breaddddd you→由 ×6; Etinjat you→有 ×4; Etinjat yi→一 ×3
- Top documents: Etinjat 1306311171 ×21; breaddddd 1306372902 ×11; Etinjat 1278079946 ×6
- Top-3 document share: 84.4%

## 6. All-wrong decomposition

| Subset | N | Share of G✗F✗H✗ | Gold in candidate set | Interpretation |
|---|---:|---:|---:|---|
| Gold not in same-Pinyin history | 1275 | 79.8% | 56.1% | No direct personal precedent in history; many are not solvable by history reranking alone. |
| Gold in same-Pinyin history | 323 | 20.2% | 53.9% | Potentially personalisable; must separate recovery from ranking failures. |
| Gold in history + candidate set | ≈174 | ≈10.9% | 100% | Main EM3 ranking/fusion opportunity. |
| Gold in history but not candidate set | ≈149 | ≈9.3% | 0% | Candidate-recovery / EM1-type failure. |

## 7. Key paired distributions

| Comparison | Failure group | Success group | Main difference |
|---|---|---|---|
| F successfully rescues G, then H outcome | G✗F✓H✗ = 35 | G✗F✓H✓ = 403 | Failure median raw-winner share 57.1% vs 88.9%; entropy 1.684 vs 0.531; distinct targets 4 vs 2. |
| G/F both fail but Gold is usable | ≈174 G✗F✗H✗ with Gold in history+candidates | 45 G✗F✗H✓ | Main EM3-v2 solvable-failure vs Context-rescue comparison. |
| Pure Context regression | G✓F✓H✗ = 24 | — | Context alone changes a correct G/F result into an error. |
| Frequency/history harms correct Generic | G✓F✗H✗ = 100 | G✓F✗H✓ = 42 | In the failure group only 39% of Golds exist in same-Pinyin history. |

## 8. Current conclusions

- The largest raw error block is **G✗F✗H✗ (1598, 28.50%)**.
- Most all-wrong rows (**1275/1598 = 79.8%**) have no Gold in same-Pinyin history, so they are not all pure EM3 ranking failures.
- The main EM3 ranking opportunity is the subset where Gold is already in history and the current candidate set (approximately **174 rows**).
- Context has genuine rescue ability: it rescues **87** rows relative to F, but harms **59**, leaving only **+28** net.
- **G✗F✗H✓ = 45** is especially important: only **20%** of these Golds are raw history winners, so Context often succeeds by selecting a minority historical target.
- **G✓F✓H✗ = 24** is the cleanest harmful Context override set and should be retained as hard-negative / ranking supervision.
- Ambiguous same-Pinyin distributions are much harder for Context: failed F-rescue cases have lower winner dominance and higher entropy than successful ones.
- Raw user frequency should not automatically be treated as personal preference; user-vs-global frequency lift should be investigated.
- EM3-v2 should learn **candidate-specific historical utility**, not just semantic similarity or a single global Context weight.

## 9. Reproducibility / recovery

**Runner**

```text
experiments\external_memory\em3_all_outcome_audit.py
```

**Output root**

```text
results\personalisation\external_memory\em3_all_outcome_audit\
```

Expected files: `summary.json`, `provenance.json`, `report.txt`, `all_rows.jsonl`, `groups\`, `focused_subsets\`.

### Input hashes

| Input | Path | SHA256 |
|---|---|---|
| history_manifest | `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\history_manifest.jsonl` | `7c85c38728d03985856d742f452992b3b3072af5f1c07845e099d9d07854da68` |
| dev_manifest | `C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\dev_manifest.jsonl` | `cf072d9323328b77e3d47d8a0c1beed8c40edc8767e075fb58593d6b72120606` |
| four_way_rows | `C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_four_way_dev_compare\rows.jsonl` | `7bc20cddc5a772e7c1f9fb3fdd60ec17e8c2813667b7c32ec835b4cbc15d87d7` |
| surface_rows | `C:\Users\chiar\Desktop\LBH\thesis-context-lab\results\personalisation\external_memory\em2_fixed_gfc_dev\selected_rows.jsonl` | `6e4007b2ba7cd0bffea4c869a7860cc08c3671bf078c22e957ad09d6ce18ea25` |

### Reproduce

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
& $python -m experiments.external_memory.em3_all_outcome_audit
```

No Test data used.