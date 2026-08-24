# Initial-Pinyin Personalisation — Recovery / Controllability File Index

**Date:** 2026-08-21
**Scope:** compact file index for the current Initial-Pinyin recovery and controllability phase.

---

## 1. Core frozen inputs

Base root:

```text
results\personalisation\initial_recovery_comparison_v1
```

| Path | Role | Frozen identity / note |
|---|---|---|
| `initial_train_fit_v1.jsonl` | causal prior history | SHA `162f5c98...bdb4` |
| `initial_train_val_v1.jsonl` | Train-Val development/evaluation | SHA `d908d4db...f0e4` |
| `candidate_surface\train_val_candidate_surface.jsonl` | frozen Personal-only K5 | SHA `205c0ba0...4b2` |
| `frequency_pv1\predictions.jsonl` | frozen F/PV1 rankings/support | SHA `7fd8aa15...ea7` |
| `candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\scores.jsonl` | cached adaptive NGram scores | reused by latest runner |
| `candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\comparison.json` | frozen selected NGram parameters/provenance | K5 Interpolated maxN=2,kappa=1,tau=2048 |

---

## 2. Important experiment runners

Canonical experiment code root:

```text
experiments\initial_personalisation
```

| File | Purpose | Known SHA / status |
|---|---|---|
| `13_run_initial_candidate_scoring_q8_bge64_v1.py` | candidate-only Q8/BGE64 comparison | historical completed |
| `14_run_initial_candidate_scoring_ngram_recency_v1.py` | NGram / NGramRecency scoring | historical completed |
| `15_run_initial_candidate_scoring_adaptive_ngram_top10_v1.py` | Hard/Soft/Interpolated NGram, K5/K10 exploration | historical completed; K10 now out of main scope |
| `30_run_initial_pv1_ngram_selector_k5_v1.py` | K1 selector prototype | SHA `7f4efcfaa7aa39fdf13aba1e50bab40032dd38a5748d282e40b30e7ee079e83d` |
| `32_run_initial_pv1_ngram_selector_k135_v1.py` | K1/K3/K5 selector ablation | completed; freeze local hash |
| `17_run_initial_pv1_ngram_k5_additive_concentration_v1.py` | fixed-gamma K5 + concentration | completed; freeze local hash |
| `18_run_initial_pv1_ngram_k5_joint_frequency_concentration_v1.py` | joint gamma/lambda K5 concentration | completed; freeze local hash |
| `20_run_initial_ngram_cs_entropy_two_anchors_v1.py` | final two-anchor Context–Preference–Confidence sweep | SHA `e5460a81435a76375619bdac809d464f567a9cdc8ae4f9c37115388d3b25d9cc` |
| `19_summarize_initial_all_results_v1.py` | read-only unified result summarizer | SHA `2624b43d0c3e21e50cbac9062d0f53d6890c4d7669832591146ec6f4bb98f4df` |

---

## 3. Important result directories

| Directory / file | Purpose |
|---|---|
| `candidate_scoring_q8_bge64_v1\candidate_scoring_comparison.json` | candidate scoring: F, Q8, BGE64, Q8+F |
| `candidate_scoring_ngram_recency_v1\ngram_recency\candidate_scoring_comparison.json` | NGram / NGramRecency candidate-only comparison |
| `candidate_scoring_adaptive_ngram_top10_v1\adaptive_ngram_top10_v1\comparison.json` | Hard/Soft/Interpolated selected parameters |
| `candidate_scoring_ngram_frequency_fusion_v1\ngram_frequency_fusion_comparison.json` | NGram + F candidate-scoring fusion |
| `pv1_ngram_selector_k135_v1\comparison.json` | K1/K3/K5 selector metrics |
| `pv1_ngram_selector_k135_v1\predictions.jsonl` | selector row-level predictions/ranks |
| `recovery_ngram_cs_interpolation_k5_v1\comparison.json` | NGram-CS alpha interpolation |
| `pv1_ngram_k5_additive_concentration_v1\comparison.json` | fixed-gamma concentration experiment |
| `pv1_ngram_k5_joint_frequency_concentration_v1\comparison.json` | joint gamma + concentration experiment |
| `ngram_cs_entropy_two_anchors_v1\comparison.json` | latest main comparison / full two-anchor grid |
| `ngram_cs_entropy_two_anchors_v1\grid_results.csv` | flat grid table |
| `ngram_cs_entropy_two_anchors_v1\features.jsonl` | row-level runtime-visible feature record plus analysis metadata |
| `ngram_cs_entropy_two_anchors_v1\predictions.jsonl` | row-level selected/ranked outputs |
| `ngram_cs_entropy_two_anchors_v1\feature_summary.json` | feature/protocol summary |
| `ngram_cs_entropy_two_anchors_v1\artifact_checksums.json` | exact local hashes for latest outputs |

---

## 4. Unified summarizer outputs

Generated under the chosen summary output directory:

```text
overall_end_to_end.csv
recovery_metrics.csv
candidate_only.csv
missing_metrics_report.txt
all_results_summary.json
discovered_artifacts.txt
```

The summarizer is read-only and does not rerun model inference.

---

## 5. Documentation generated for the current phase

Suggested repo documentation destination:

```text
docs\initial_personalisation\
```

| Documentation file | Purpose | SHA256 |
|---|---|---|
| `11_INITIAL_PERSONALISATION_EVALUATION_METRICS_AND_DESIGN.md` | metric definitions/calculation and evaluation design | `534b5d5a091671a3d1e955fde951de613da4d0a1aded8e0a82d146be5299761f` |
| `12_INITIAL_PERSONALISATION_METRIC_PURPOSE_AND_PRIORITIES.md` | metric purpose, priority, key vs diagnostic metrics | `9827d461c0ed81fe1ede3b9f35592986fc8be24f8b7d3a39fd2cd8da5fc0d38e` |
| `13_INITIAL_PERSONALISATION_CURRENT_CONCLUSIONS_AND_CONTROLLABILITY.md` | current conclusions, all key data, controllability interpretation | `3a73881188de740a6e8022d07931fdce8013d9f81f0b4529ffa1c2021230de78` |
| `15_INITIAL_PERSONALISATION_RECOVERY_REPRODUCIBILITY_2026-08-21.md` | compact reproducibility for the recovery/controllability phase | generated in this closeout |
| `14_INITIAL_PERSONALISATION_FILE_INDEX_2026-08-21.md` | this compact index | generated in this closeout |

---

## 6. Current main model checkpoints

```text
Overall balanced:
B + 4 P_NG + 4 CS + 2 E
Macro=.404807, Micro=.429364, Top3=.614801, Top5=.685815, MRR=.537433, Missing=.243172

Top3-oriented:
B + 6 P_NG + 2 CS + .25 E
Top3=.615876, Rec@3=.5737, Rec@10=.9283

Top5-oriented:
B + 6 P_NG + 2 CS + 1 E
Top5=.686715

Selector alternative:
NGramSelector@K3
Macro=.403964, Rec@3=.6051, Rec@10=.9051
```

---

## 7. Local hash freeze command

Run from repo root before closeout:

```powershell
$root = '.\results\personalisation\initial_recovery_comparison_v1'

Get-FileHash '.\experiments\initial_personalisation\20_run_initial_ngram_cs_entropy_two_anchors_v1.py' -Algorithm SHA256
Get-FileHash "$root\ngram_cs_entropy_two_anchors_v1\comparison.json" -Algorithm SHA256
Get-FileHash "$root\ngram_cs_entropy_two_anchors_v1\grid_results.csv" -Algorithm SHA256
Get-FileHash "$root\ngram_cs_entropy_two_anchors_v1\features.jsonl" -Algorithm SHA256
Get-FileHash "$root\ngram_cs_entropy_two_anchors_v1\predictions.jsonl" -Algorithm SHA256
Get-FileHash "$root\ngram_cs_entropy_two_anchors_v1\feature_summary.json" -Algorithm SHA256
Get-FileHash "$root\ngram_cs_entropy_two_anchors_v1\artifact_checksums.json" -Algorithm SHA256
```
