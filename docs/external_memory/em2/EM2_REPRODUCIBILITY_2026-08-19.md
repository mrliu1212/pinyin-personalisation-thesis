# EM-2 Reproducibility Record -2026-08-19

**Status:** CLOSED / FROZEN DEV-STAGE
**Canonical result report:** `docs/external_memory/em2/EM2_FINAL_REPORT_2026-08-19.md`

This file records the canonical runners, commands, dependencies, and expected result checkpoints for EM-2.

---

## Environment

Repository working tree:

```text
C:\Users\chiar\Desktop\LBH\thesis-context-lab
```

Python:

```text
C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe
```

Frozen PinyinGPT:

```text
C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat
```

Pilot / Dev manifests and Generic cache:

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\pilot_a_context_memory
```

Frozen generic reranker:

```text
C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base
```

Original M2 pair cache:

```text
C:\Users\chiar\Desktop\LBH\thesis-personalisation\
results\personalisation\m2_h5000\cache\pair_scores.sqlite3
```

---

## Frozen hashes

Generic Dev cache:

```text
588aa84c6397e8cb1a13576c0d5dfecd9dd2c4305b45be351328dd83ef62007d
```

EM-2 hidden cache:

```text
9a80a3314c184ccf3f0540916203c651474fad162dc3dab1fc97f7451f441df1
```

---

## EM-2B hidden cache

Runner:

```text
experiments/external_memory/em2_cache_hidden_dev.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_cache_hidden_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --checkpoint 'C:\Users\chiar\Desktop\LBH\thesis\.build\pinyingpt2-concat' `
  --output-root 'results\personalisation\external_memory\em2_hidden_dev' `
  --device cuda `
  --batch-size 64
```

Expected checkpoint:

```text
Cached rows: 11475
Hidden size: 768
Context-truncated rows: 0
```

---

## EM-2E1 Hidden-M1

Runner:

```text
experiments/external_memory/em2_hidden_m1_dev.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_hidden_m1_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --hidden-cache 'results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3' `
  --output-root 'results\personalisation\external_memory\em2_hidden_m1_dev'
```

Frozen selection:

```text
Top-N = 3
lambda_hidden = 4
```

Boundary wrapper:

```text
experiments/external_memory/em2_hidden_m1_dev_boundary8.py
```

Expected selected Macro Overall Top1:

```text
0.768748
```

---

## Four-way G/F/Original-M1/Hidden-M1 comparison

Runner:

```text
experiments/external_memory/em2_four_way_dev_compare.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_four_way_dev_compare
```

Expected Overall Macro Top1:

```text
G             0.722948
F             0.765240
Original-M1   0.768888
Hidden-M1     0.768748
```

---

## Original M2 same-surface control

Runner:

```text
experiments/external_memory/em2_original_m2_same_surface_dev.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_original_m2_same_surface_dev
```

Expected:

```text
Required pair uses: 39415
Missing pair scores: 0
Overall Macro Top1: 0.766869
```

---

## Hidden-M2

Runner:

```text
experiments/external_memory/em2_hidden_m2_dev.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_hidden_m2_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --hidden-cache 'results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3' `
  --reranker-model 'C:\Users\chiar\Desktop\LBH\thesis\.build\bge-reranker-base' `
  --output-root 'results\personalisation\external_memory\em2_hidden_m2_dev' `
  --batch-size 32
```

Expected Dev selection:

```text
K = 10
lambda_m2 = 4
Overall Macro Top1 = 0.768372
```

Strict K20/lambda4 representation control:

```text
Hidden-M2 = 0.767776
Original M2 = 0.766869
```

---

## Fixed G+F+C

Runner:

```text
experiments/external_memory/em2_fixed_gfc_dev.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_fixed_gfc_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --generic-cache 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory\cache\generic_predictions.jsonl' `
  --hidden-cache 'results\personalisation\external_memory\em2_hidden_dev\hidden_states.sqlite3' `
  --output-root 'results\personalisation\external_memory\em2_fixed_gfc_dev'
```

Expected:

```text
lambda_F = 0.5
lambda_C = 4
Overall Macro Top1 = 0.768825
```

---

## Adaptive G+F+C

Runner:

```text
experiments/external_memory/em2_adaptive_gfc_dev.py
```

Command:

```powershell
& 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe' `
  -m experiments.external_memory.em2_adaptive_gfc_dev `
  --pilot-root 'C:\Users\chiar\Desktop\LBH\thesis-personalisation\results\personalisation\pilot_a_context_memory' `
  --fixed-gfc-root 'results\personalisation\external_memory\em2_fixed_gfc_dev' `
  --output-root 'results\personalisation\external_memory\em2_adaptive_gfc_dev'
```

Expected:

```text
Count-aware selected L = 16
Count-aware Overall = 0.765894

No-count selected L = 4
No-count Overall = 0.768591
```

---

## EM-2A and EM-2C

Canonical runners:

```text
experiments/external_memory/em2_hidden_state_gate.py
experiments/external_memory/em2_hidden_knn_dev.py
```

The stage-close script attempts to recover their exact original command lines from the current PowerShell history and appends them below before commit.

If the commands cannot be recovered, the closeout script stops before commit rather than silently inventing an invocation.

<!-- EM2-RECOVERED-CLI-2026-08-19 -->
## Recovered canonical command lines

### EM-2A hidden-state engineering gate

``powershell
experiments/external_memory/em2_hidden_state_gate.py
``

### EM-2C hidden-state kNN Dev diagnostic

``powershell
experiments/external_memory/em2_hidden_knn_dev.py
``


---

## Generated outputs

Generated outputs are under:

```text
results/personalisation/external_memory/
```

Important namespaces:

```text
em2_hidden_dev/
em2_hidden_m1_dev/
em2_hidden_m1_dev_boundary8/
em2_original_m2_dev/
em2_hidden_m2_dev/
em2_fixed_gfc_dev/
em2_adaptive_gfc_dev/
```

These result trees, logs, and SQLite caches are generated/local evidence and are not normal Git source files.

---

## Reproduction status

When the stage-close script succeeds:

```text
Method documentation        COMPLETE
Runner inventory            COMPLETE
Known exact commands        COMPLETE
Recovered EM-2A/EM-2C CLI   COMPLETE
Frozen hashes               COMPLETE
Generated result paths      COMPLETE
Test status                 NOT OPENED
Git checkpoint              external-memory-em2-closed-20260819
```
