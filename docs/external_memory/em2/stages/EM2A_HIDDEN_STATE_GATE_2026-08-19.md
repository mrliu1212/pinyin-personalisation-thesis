# EM-2A Hidden-State Extraction Gate

Status: PASSED / FROZEN

Date: 2026-08-19

## Purpose

Verify the engineering definition of the PinyinGPT representation before
any retrieval metric is inspected.

## Frozen representation

EM-2 uses:

- Frozen PinyinGPT2-Concat
- final Transformer layer
- hidden state at the final prompt token
- final prompt token = [SEP]
- hidden dimension = 768

This position is selected because the Frozen backend uses the logits at this
same final prompt position to predict the first Chinese target character.

The representation was selected from backend inference semantics, not from
retrieval performance.

## Gate setup

Development data only.

Authors:
- Etinjat
- Re_spectators
- breaddddd

Samples:
- 3 per author
- 9 total

Tolerance:
- 1e-4

Gold was not used.

No retrieval metric was inspected.

## Results

Samples passed:
- 9 / 9

Maximum hidden-state -> LM-head logits absolute difference:
- 2.67028808594e-05

Maximum allowed-token distribution absolute difference:
- 1.14440917969e-05

Maximum direct vs teacher-forced first-step absolute difference:
- 3.81469726562e-06

Maximum cached beam vs fixed-candidate score absolute difference:
- 9.17911529541e-06

Best allowed next-token agreement:
- 9 / 9

## Conclusion

PASS.

The final-layer hidden state at the final prompt [SEP] token is
engineering-aligned with the Frozen PinyinGPT first-character prediction
path.

This representation definition is now frozen for EM-2.

Poor downstream retrieval performance is not, by itself, a justification
for changing the layer, pooling method, or extraction position.

Any future change would require independent evidence of an engineering or
methodological error.

## Next stage

EM-2B:
Build the Dev hidden-state memory cache using the frozen H5000
HistoryIndex semantics.
