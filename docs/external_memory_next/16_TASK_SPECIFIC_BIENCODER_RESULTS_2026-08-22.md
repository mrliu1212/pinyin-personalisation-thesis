# 16 - Task-Specific Bi-Encoder Results

Date: 2026-08-22

Status: **COMPLETE / RETRIEVAL-POSITIVE, FINAL-RANKING-NEGATIVE**

## 1. Result in one sentence

Task-specific contrastive training substantially improved local historical
retrieval, but the improvement did not survive the frozen candidate-support
aggregation and linear decision layer. The predeclared nonlinear-refit gate
therefore failed and no task-specific LambdaMART model was fitted.

Dev3000 and Test remained closed. The candidate surface, Generic predictions,
NGramRecency support, fixed coefficients, history semantics, Gold policy, and
completed generic-BGE LambdaMART result were unchanged.

### Research question

Can a full-precision BGE encoder trained only on strictly causal, query-local
same-Pinyin Train-Fit supervision improve historical retrieval and, without
changing the candidate surface or ranking formula, improve final Train-Val IME
ranking over the frozen generic-BGE RetunedFinal route?

### Frozen protocol and necessary clarification

Record 15 was frozen before training: the model revision and serialization,
query-local listwise loss, chronological inner split, two-epoch selection rule,
fresh all-Train-Fit refit, one-shot Train-Val evaluation, and conditional
nonlinear-refit gate were not changed after results were observed. The only
clarification was data-mechanical and was documented before neural training:
32,999 sampled rounds contained a positive but no legal wrong-target negative.
They remained in the population audit but were excluded from optimization
because their one-class cross-entropy is identically zero. No replacement,
padding, borrowing, or post-hoc resampling was introduced.

## 2. Data and leakage audit

The frozen Clean3 Train-Fit pair registry was reconstructed against the source
manifest row by row:

| Item | Count |
|---|---:|
| Eligible causal queries | 35,290 |
| Sampled positive rounds | 99,671 |
| Positive pairs | 99,671 |
| Query-local wrong-target pairs | 169,400 |
| Total pairs | 269,071 |
| Trainable groups with at least one negative | 66,672 |

There were zero unknown rows, duplicate query/history pairs, cross-author
pairs, cross-Pinyin pairs, non-prior pairs, position mismatches, target/label
mismatches, or context mismatches. The 32,999 positive-only late rounds were
retained in the audit and excluded from optimization because a one-class
contrastive loss is identically zero. No negative was resampled or borrowed.

The whole-position chronological split yielded 59,686 trainable inner-fit
groups and 6,986 trainable inner-gate groups. For every author,
`max(inner_fit_position) < min(inner_gate_position)`.

Compact group registry SHA256:
`9b9eda5629842ec2b57428a53c0e2b6e273c533d24dd918ed1914afbfb4c4441`.

## 3. Model and training provenance

- Base: full-precision `BAAI/bge-small-zh-v1.5`.
- Revision: `7999e1d3359715c523056ef9478215996d62a620`.
- Base semantic asset-tree SHA256:
  `4d71fdf52d2c78025befad48d042d2bafa9199e19cdcf2b635c678a1e436b252`.
- Base `model.safetensors` SHA256:
  `354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026`.
- Four BERT layers, 512-dimensional embeddings, mean pooling and L2
  normalization, last 64 Unicode context characters.
- PyTorch `2.11.0+cu128`, Transformers `4.57.6`, CUDA runtime `12.8`, NVIDIA
  GeForce RTX 4060 Laptop GPU.
- No author, Pinyin, target, Gold, correctness, or future-row field was
  serialized.

The eight-group CUDA smoke passed. Saving and reloading produced exact
embedding identity (`max_abs_difference=0.0`). The full run used only the two
frozen epoch choices:

| Epoch | Inner-gate Macro R@1 | Micro R@1 | MRR | Mean training loss |
|---:|---:|---:|---:|---:|
| 1 | .571341975 | .572287432 | .745467125 | .921397338 |
| 2 | **.594849963** | **.595906098** | **.760711900** | .596252349 |

Epoch 2 was selected by the predeclared rule. A fresh model was then refitted
on all 66,672 trainable Train-Fit groups for two epochs. Final mean loss was
`.583126731`. Training plus gate/refit took `948.51` seconds. Final checkpoint
save/reload difference was `0.0`.

Final checkpoint tree SHA256:
`f9b87af11fcff692ad7c25fb6330f44f9f23ffedb480af9aec36af0e7cd08a8e`.
Final `model.safetensors` SHA256:
`81f1bc54ec80567cc15c3f986b4acc88033b0a9d268b78fa6c1a893360e63364`.

## 4. Intrinsic Train-Val retrieval

The exact comparison population contains 9,325 Train-Val queries with both a
Gold-target and wrong-target legal history among the frozen Stage-1
candidate-conditioned histories. Both encoders scored identical histories.

| Metric | Generic BGE | Task-specific | Delta |
|---|---:|---:|---:|
| Macro-author Recall@1 | .778943777 | **.810971191** | **+.032027414** |
| Micro Recall@1 | .788203753 | **.821233244** | **+.033029491** |
| Recall@5 | .959785523 | **.962252011** | +.002466488 |
| Recall@10 | **.981340483** | .980375335 | -.000965147 |
| MRR@10 | .863742033 | **.883514537** | **+.019772504** |
| Target-support Top1 | **.803860590** | .781233244 | -.022627346 |
| Mean Gold support margin | **.354740655** | .336470761 | -.018269893 |

Per-author task-specific Recall@1 was `.858741682` (Agent Phage),
`.699135899` (Etinjat), and `.875035992` (breaddddd), exceeding generic BGE
for every author. However, target-level aggregation became worse. The training
objective improved individual positive-history placement but did not optimize
the deployed Top5-per-candidate, recency-weighted, candidate-normalized support
decision.

## 5. Final IME ranking

The generic-BGE route was reconstructed exactly: maximum support difference
from the frozen artifact was `0.0`, and all 34,416 candidate orders/ranks
matched. The task model then replaced only the BGE vectors.

### Overall

| Method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen RetunedFinal, generic BGE | **.796004927** | **.824994189** | **.912075779** | .930758949 | **.871377873** | .051981636 |
| Task bi-encoder, frozen linear fusion | .795711724 | .824645514 | .912046722 | .930758949 | .871191140 | .051981636 |
| Completed LambdaMART, generic BGE | **.798839063** | **.827783589** | .911988610 | .930613668 | **.873043096** | .051981636 |

Task-specific fixed fusion versus frozen RetunedFinal: 34 rescues, 46 harms,
net `-12`. Macro Top1 changed by `-.000293202` (-.0293 percentage points).
Missing@10 is necessarily unchanged because the candidate surface is fixed.

Per-author task-specific Top1 was `.886471145` (Agent Phage), `.601494396`
(Etinjat), and `.899169632` (breaddddd), versus frozen `.887271669`,
`.601494396`, and `.899248715`.

### Ambiguous and Conflict

| Population/method | Macro Top1 | Micro Top1 | Top3 | Top5 | MRR@10 | Missing@10 |
|---|---:|---:|---:|---:|---:|---:|
| Ambiguous frozen BGE | .803894152 | .808514871 | .940117378 | .958420372 | .874789726 | .029344474 |
| Ambiguous task | .802710612 | .807321198 | .940017905 | .958420372 | .874150455 | .029344474 |
| Conflict frozen BGE | .252588201 | .235924933 | .765147453 | .838069705 | .500956849 | .108847185 |
| Conflict task | .247313280 | .232171582 | .764611260 | .838069705 | .498610154 | .108847185 |

## 6. Attribution and decision

- **Improved retrieval:** attributable to the task-specific bi-encoder. The
  intrinsic same-population Macro Recall@1 gain is +3.203 points.
- **Improved final fixed ranking:** absent. The frozen linear route became
  slightly worse, so retrieval gain alone was insufficient.
- **Improvement after nonlinear fusion:** the completed LambdaMART comparator
  remains the best overall result, but it uses the original generic-BGE
  support. Its gain cannot be attributed to the task-specific encoder.
- **Task-specific nonlinear fusion:** not run. The frozen gate required both
  intrinsic and fixed-fusion Macro improvements; only the first passed.

This is a valid negative end-to-end result. The protocol was not extended or
retuned after Train-Val was observed.

## 7. Limitations

- The task objective ranks individual histories within query-local groups,
  whereas deployment aggregates the Top5 histories independently for every
  candidate with recency weighting and candidate-wise normalization.
- The experiment isolates representation replacement under the frozen linear
  fusion. It does not establish that the learned representation could not help
  under another predeclared support aggregation or nonlinear model.
- Only the predeclared two checkpoints and one Train-Val evaluation were used;
  no open-ended hyperparameter search was performed.
- This is Train-Val development evidence only. Dev3000 and Test were not used,
  so no held-out final-evaluation claim is made.

## 8. Runtime and artifact hashes

- Preparation audit: `10.48` seconds.
- Gate training plus full refit: `948.51` seconds.
- Train-Val embedding and evaluation: `68.45` seconds.
- Task vector cache: 42,278 rows, 182,607,872 bytes, SHA256
  `e3e92d827a51daa3bbe2e75cc0855a633d6eb8334a0d084db4288a9b817e46a9`.
- Training result SHA256:
  `5ca5f5702bc2c2427fb5be3653c2de3a24bd9540f8b1a1b6a5781604327391e6`.
- Evaluation result SHA256:
  `493d7901e7295ae58e2dcfc7d267bfe44ea797bd638357f47ca8fcf1791da0ad`.
- Prediction SHA256:
  `cbc5ce85605d2ddd035254e3a02a623512319976e0c931c5f4767996cf368ddf`.
- Preparation runner SHA256:
  `8aac7061aafa20c1f38719111737a67e67e752b369e0a3c15455c982e4bc2383`.
- Training runner SHA256:
  `bce0c90bf7f7df738cdb4204c5eaa750b569f711687d6f9b10588e5d60a10661`.
- Evaluation runner SHA256:
  `7ada1c54a9a10514668136c085a7fd3b9502912c08c07700e74c2bb08d15b693`.
- Shared helper SHA256:
  `afb0414180fba5ee863cecb874552e8f4d54988ec573608849947216d2ef06ea`.

All checkpoints, vectors, registries, predictions, results, and logs are
generated local-only artifacts under
`results/personalisation/external_memory_next/task_specific_biencoder_v1/` or
`.build/external_memory_next_biencoder/`.

## 9. Exact Windows reproduction

From `C:\Users\chiar\Desktop\LBH\thesis-external-memory-next`:

```powershell
$python = 'C:\Users\chiar\Desktop\LBH\thesis\.venv\Scripts\python.exe'
$compare = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_v2'
$follow = 'C:\Users\chiar\Desktop\LBH\thesis-context-compare\results\personalisation\context_comparison_followup_v1\full_retune_final_trainval_dev_v1\tune'
$root = '.\results\personalisation\external_memory_next\task_specific_biencoder_v1'
$revision = '7999e1d3359715c523056ef9478215996d62a620'
$model = ".\.build\external_memory_next_biencoder\base\$revision"

& $python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-small-zh-v1.5', revision='$revision', local_dir=r'$model', allow_patterns=['config.json','model.safetensors','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','modules.json','sentence_bert_config.json','1_Pooling/*'])"

& $python -m experiments.external_memory_next.prepare_task_specific_biencoder_v1 `
  --fit "$compare\clean3_train_fit_v1.jsonl" `
  --pairs "$compare\em3_train_pairs_v1\train_pairs.jsonl" `
  --output-root "$root\preparation"

& $python -m experiments.external_memory_next.run_task_specific_biencoder_v1 `
  --phase smoke `
  --groups "$root\preparation\groups.jsonl" `
  --audit "$root\preparation\audit.json" `
  --base-model $model `
  --output-root "$root\training"

& $python -m experiments.external_memory_next.run_task_specific_biencoder_v1 `
  --phase full `
  --groups "$root\preparation\groups.jsonl" `
  --audit "$root\preparation\audit.json" `
  --base-model $model `
  --output-root "$root\training"

& $python -m experiments.external_memory_next.evaluate_task_specific_biencoder_v1 `
  --fit "$compare\clean3_train_fit_v1.jsonl" `
  --val "$compare\clean3_train_val_v1.jsonl" `
  --stage2 "$follow\train_val_stage2_supports.jsonl" `
  --frozen-predictions "$follow\train_val_selected_predictions.jsonl" `
  --generic-bge-cache "$follow\bge_context_cache.sqlite3" `
  --training-result "$root\training\training_result.json" `
  --lambdamart-result '.\results\personalisation\external_memory_next\lambdamart_fusion_v1\result.json' `
  --checkpoint "$root\training\final_refit\epoch_2" `
  --output-root "$root\evaluation"
```

Every runner rejects paths containing Dev3000 or Test and records
`used_dev3000=false`, `used_test=false`.
