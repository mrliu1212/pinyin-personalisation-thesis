# Standardized Full+Short comparison — sealed Dev3000

This is the one post-freeze evaluation on the unchanged balanced Clean3 Dev3000. Test remains closed.

- Dev manifest SHA256: `9181f895eb19d0c36852e511263bfaefb34459dcd44efa6f45a44252e6b03f93`
- PRE_DEV_FREEZE SHA256: `7c0fcf69823f0b4b7d8b914a81ea54a097e12c03cb61c515c2400be46df46824`
- Predictions SHA256: `dd219bfcb28fcad6a65f31eb14ddb16fc03c80f54a8b62a1cfe2504113c84233`
- Selection population: chronological whole-work Clean3 Train-Val only
- Candidate surface: identical frozen Generic Top10 for all methods; no injection/removal/fusion
- History: same author → strictly prior → latest H5000 raw → exact segmented-Pinyin
- `used_test=false`

## Frozen Train-Val selections

| Method | Frozen configuration |
|---|---|
| Generic | `{"beam": 16, "checkpoint_revision": "76dd20dc92d8236a350fb732e99dde6fa15e2263", "n_positions": 1024, "top_k": 10}` |
| Frequency | `{"lambda": 4.0}` |
| M1 | `{"bge_model_sha256": "5a88d266870fbd27c6f329df60de80e2d4cf3bbd5e6f080bd5c1b2e5abb12039", "lambda": 4.0, "retriever": "BGE Full cosine", "top_n": 5}` |
| M2 | `{"lambda": 4.0, "metrics": {"macro_author": {"missing_at_10": 0.08781858060264291, "mrr_at_10": 0.8261062032411733, "top1": 0.7769184346882935, "top3": 0.8661508098220336}, "micro": {"mean_rank_given_top10": 1.3137915964287945, "missing_at_10": 0.06921199442119944, "mrr_at_10": 0.8543217832421982, "n": 34416, "top1": 0.8088970246397025, "top3": 0.8924337517433751}, "per_author": {"Agent Phage": {"mean_rank_given_top10": 1.1868486166304266, "missing_at_10": 0.029401062513645295, "mrr_at_10": 0.9149882231717615, "n": 13741, "top1": 0.8767193071828834, "top3": 0.9509497125391165}, "Etinjat": {"mean_rank_given_top10": 1.8413476747864599, "missing_at_10": 0.21270236612702367, "mrr_at_10": 0.6367990076894187, "n": 8030, "top1": 0.5621419676214197, "top3": 0.6902864259028643}, "breaddddd": {"mean_rank_given_top10": 1.1810909090909092, "missing_at_10": 0.021352313167259787, "mrr_at_10": 0.9265313788623397, "n": 12645, "top1": 0.8918940292605773, "top3": 0.9572162910241202}}}, "reranker_model_sha256": "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd", "reranker_revision": "2cfc18c9415c912f9d8155881c133215df768a70", "reranker_tokenizer_sha256": "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e", "retrieval_k": 10, "retriever": "BGE Full cosine"}` |
| Hidden-M1 | `{"lambda": 4.0, "metrics": {"macro_author": {"missing_at_10": 0.08781858060264291, "mrr_at_10": 0.8270339079999426, "top1": 0.7785269413665622, "top3": 0.8669251571325299}, "micro": {"mean_rank_given_top10": 1.311887369669726, "missing_at_10": 0.06921199442119944, "mrr_at_10": 0.8550806933652122, "n": 34416, "top1": 0.81017549976755, "top3": 0.8931601580660158}, "per_author": {"Agent Phage": {"mean_rank_given_top10": 1.1844492764489765, "missing_at_10": 0.029401062513645295, "mrr_at_10": 0.9155826382174537, "n": 13741, "top1": 0.8774470562550033, "top3": 0.9516774616112365}, "Etinjat": {"mean_rank_given_top10": 1.8367605188231573, "missing_at_10": 0.21270236612702367, "mrr_at_10": 0.6389063333926348, "n": 8030, "top1": 0.566002490660025, "top3": 0.6914072229140722}, "breaddddd": {"mean_rank_given_top10": 1.1810909090909092, "missing_at_10": 0.021352313167259787, "mrr_at_10": 0.9266127523897393, "n": 12645, "top1": 0.892131277184658, "top3": 0.9576907868722815}}}, "retrieval_k": 5, "retriever": "PinyinGPT hidden cosine"}` |
| Hidden-M2 | `{"lambda": 4.0, "metrics": {"macro_author": {"missing_at_10": 0.08781858060264291, "mrr_at_10": 0.8257993517084636, "top1": 0.7762991774482791, "top3": 0.8666944827392266}, "micro": {"mean_rank_given_top10": 1.3142286320784167, "missing_at_10": 0.06921199442119944, "mrr_at_10": 0.8540647045855378, "n": 34416, "top1": 0.8083740120874012, "top3": 0.8929277080427708}, "per_author": {"Agent Phage": {"mean_rank_given_top10": 1.1862487815850642, "missing_at_10": 0.029401062513645295, "mrr_at_10": 0.9151635760434247, "n": 13741, "top1": 0.8770104068117314, "top3": 0.9513135870751764}, "Etinjat": {"mean_rank_given_top10": 1.8430876304966783, "missing_at_10": 0.21270236612702367, "mrr_at_10": 0.6362354958587836, "n": 8030, "top1": 0.5610211706102117, "top3": 0.6911581569115816}, "breaddddd": {"mean_rank_given_top10": 1.181979797979798, "missing_at_10": 0.021352313167259787, "mrr_at_10": 0.9259989832231825, "n": 12645, "top1": 0.8908659549228944, "top3": 0.9576117042309213}}}, "reranker_model_sha256": "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd", "reranker_revision": "2cfc18c9415c912f9d8155881c133215df768a70", "reranker_tokenizer_sha256": "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e", "retrieval_k": 10, "retriever": "PinyinGPT hidden cosine"}` |
| EM3 | `{"checkpoint": "results\\personalisation\\context_comparison_v2\\em3_clean3\\train\\final", "checkpoint_files_sha256": {"config.json": "b654d6598b95be4656a4eefd389695542aa4bf30c7eb24378f2e9da8abcfcaa5", "model.safetensors": "0e846deeeaf06c3e5c61dc39bfae1c1f986d37a82668146dd030a1ccca793dfa", "special_tokens_map.json": "152f77ffe3cc174ee82dfc88eed567a04152a7ed214818ec74c99548fbf347c8", "tokenizer.json": "4a8d0b7573869188be52cca17a27a84f3cfbc0a5536c28ee1eca82903e8c68c6", "tokenizer_config.json": "410d349ad6778e60273579081da479cf72b4a979de111d597bdb805b9afc6bab"}, "lambda": 4.0, "metrics": {"macro_author": {"missing_at_10": 0.08781858060264291, "mrr_at_10": 0.8263020690756983, "top1": 0.7771533899623565, "top3": 0.8666487665827125}, "micro": {"mean_rank_given_top10": 1.3129799587937816, "missing_at_10": 0.06921199442119944, "mrr_at_10": 0.8544228113723407, "n": 34416, "top1": 0.8089551371455137, "top3": 0.8928986517898652}, "per_author": {"Agent Phage": {"mean_rank_given_top10": 1.1851990702556796, "missing_at_10": 0.029401062513645295, "mrr_at_10": 0.9153649199533779, "n": 13741, "top1": 0.8772287315333673, "top3": 0.9514591368896005}, "Etinjat": {"mean_rank_given_top10": 1.8373932299905094, "missing_at_10": 0.21270236612702367, "mrr_at_10": 0.637745063156022, "n": 8030, "top1": 0.563760896637609, "top3": 0.6910336239103363}, "breaddddd": {"mean_rank_given_top10": 1.1827878787878787, "missing_at_10": 0.021352313167259787, "mrr_at_10": 0.9257962241176951, "n": 12645, "top1": 0.8904705417160933, "top3": 0.9574535389482008}}}, "retrieval_k": 10, "retriever": "PinyinGPT hidden cosine", "training_optimizer_steps": 8409, "training_seed": 42}` |

## Overall results

| Method | Macro-author Top1 | Micro Top1 | Micro Top3 | Micro MRR@10 | Micro Missing@10 |
|---|---:|---:|---:|---:|---:|
| Generic | 72.267% | 72.267% | 87.667% | 0.806844 | 5.133% |
| Frequency | 82.500% | 82.500% | 91.533% | 0.872403 | 5.133% |
| M1 | 82.833% | 82.833% | 91.333% | 0.874323 | 5.133% |
| M2 | 82.433% | 82.433% | 91.333% | 0.872074 | 5.133% |
| Hidden-M1 | 82.633% | 82.633% | 91.200% | 0.872864 | 5.133% |
| Hidden-M2 | 82.367% | 82.367% | 91.300% | 0.871268 | 5.133% |
| EM3 | 82.733% | 82.733% | 91.200% | 0.873047 | 5.133% |

## Per-author Top1

| Method | Agent Phage | Etinjat | breaddddd |
|---|---:|---:|---:|
| Generic | 76.200% | 56.800% | 83.800% |
| Frequency | 89.600% | 65.200% | 92.700% |
| M1 | 90.300% | 65.300% | 92.900% |
| M2 | 89.600% | 65.000% | 92.700% |
| Hidden-M1 | 90.100% | 64.900% | 92.900% |
| Hidden-M2 | 89.800% | 64.500% | 92.800% |
| EM3 | 89.800% | 65.500% | 92.900% |

## Diagnostic subsets (Macro-author Top1)

| Method | Ambiguous | Conflict | Mature-H5000 |
|---|---:|---:|---:|
| Generic | 63.736% | 28.062% | 72.267% |
| Frequency | 77.537% | 14.425% | 82.500% |
| M1 | 78.412% | 24.756% | 82.833% |
| M2 | 77.433% | 17.887% | 82.433% |
| Hidden-M1 | 78.033% | 22.234% | 82.633% |
| Hidden-M2 | 77.457% | 19.395% | 82.367% |
| EM3 | 78.087% | 18.035% | 82.733% |

## Paired rescue/harm/net

| Method | vs Frequency rescue/harm/net | vs Generic rescue/harm/net |
|---|---|---|
| M1 | 40 / 30 / 10 | 396 / 79 / 317 |
| M2 | 39 / 41 / -2 | 388 / 83 / 305 |
| Hidden-M1 | 38 / 34 / 4 | 391 / 80 / 311 |
| Hidden-M2 | 34 / 38 / -4 | 382 / 79 / 303 |
| EM3 | 34 / 27 / 7 | 389 / 75 / 314 |

## Provenance

Machine result: `results\personalisation\context_comparison_v2\dev3000\standardized_dev3000_result.json`
Checksums: `results\personalisation\context_comparison_v2\dev3000\checksums.json`

Dev3000 was not used for training or selection. No Test inference or Test metric was run.
