# M4 Research Index

Latest frozen checkpoint: **2026-08-26**

## Status

| Stage | Status | Role |
|---|---|---|
| R0 | Frozen | compositional retrieval feasibility |
| R1 | Frozen | bounded Composition Beam16 |
| R2 | Frozen | Composition LambdaMART |
| R3 | Frozen | Composition calibration |
| R4 | Frozen | Frozen Full / Exact reconstruction |
| R4B | Frozen | Exact calibration |
| R5 | Frozen | calibrated dual-branch recovery |
| R6-v2 | Frozen | final <=15-candidate reranking |
| R7 | Frozen | explicit post-model controllability |

## Headline

- Frozen Generic Top1: **48.346548%**
- Final M4 default Top1: **59.800190%**
- Absolute improvement: **+11.453642 pp**
- R5 final candidate-pool recall: **78.620362%**

## Documents

- [Final architecture](M4_FINAL_ARCHITECTURE_2026-08-26.md)
- [Reproducibility record](M4_REPRODUCIBILITY_2026-08-26.md)
- [Checkpoint manifest](M4_CHECKPOINT_MANIFEST_2026-08-26.json)

## Frozen evidence

- `freeze/R5_FREEZE.json`
- `freeze/R6_V2_FREEZE.json`
- `freeze/R7_FREEZE.json`
- `evidence/R5_summary.json`
- `evidence/R6_V2_summary.json`
- `evidence/R7A_summary.json`
- `evidence/R7B_summary.json`
- `evidence/R7B_control_curves.csv`

## Next stage

**End-to-end runtime and final closed-split evaluation.**

M4 R0-R7 architecture/accuracy tuning is closed.
