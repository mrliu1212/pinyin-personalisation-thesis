\# Historical History-Depth Result Provenance



Date: 2026-08-20



Status: \*\*HISTORICAL EVIDENCE ONLY\*\*



This note recovers and preserves the historical Full+Short personalisation

history-depth experiment.



These results come from the historical frozen T1 Test benchmark and must not

be used for current standardized-reset model or hyperparameter selection.



They are retained for provenance, interpretation, and possible future

history-depth analysis.



\---



\## 1. Research question



Historical question:



> How does personalisation performance change as the amount of available

> same-user history increases?



History budgets:



\- H500

\- H5000

\- HFull



Methods:



\- Frequency (F)

\- M1

\- M2



Input condition:



\- Full + Short



Primary metric:



\- Macro-author Top-1



Historical benchmark:



\- frozen T1 Test

\- six authors

\- 1,000 anchors per author

\- 6,000 rows total



\---



\## 2. Historical history semantics



The historical reranking matrix used causal same-user history.



For query q\_t:



1\. select the same user;

2\. retain only strictly prior interactions;

3\. apply the author-level history budget;

4\. then apply exact segmented-Pinyin matching.



Therefore:



H500(q\_t)

=

latest up to 500 strictly-prior same-user interactions



H5000(q\_t)

=

latest up to 5,000 strictly-prior same-user interactions



HFull(q\_t)

=

all valid strictly-prior same-user interactions



The history budget is applied BEFORE exact segmented-Pinyin filtering.



This means HFull is not future-aware and is not an unrestricted global

history pool. It is the complete legal same-user history available before the

current query.



\---



\## 3. Historical Full+Short headline results



| History | Frequency | M1 | M2 |

|---|---:|---:|---:|

| H500 | 74.0000% | 74.0333% | 73.6500% |

| H5000 | 77.1833% | 76.7500% | 76.5000% |

| HFull | 80.3500% | 80.6500% | 80.4000% |



Exact machine values:



\### H500



Frequency:



0.7399999999999999



M1:



0.7403333333333334



M2:



0.7364999999999999



\### H5000



Frequency:



0.7718333333333334



M1:



0.7675000000000001



M2:



0.765



\### HFull



Frequency:



0.8035



M1:



0.8065000000000001



M2:



0.8039999999999999



\---



\## 4. Change with history depth



\### Frequency



H500 -> H5000:



+3.18 percentage points



H5000 -> HFull:



+3.17 percentage points



H500 -> HFull:



+6.35 percentage points



\### M1



H500 -> H5000:



+2.72 percentage points



H5000 -> HFull:



+3.90 percentage points



H500 -> HFull:



+6.62 percentage points



\### M2



H500 -> H5000:



+2.85 percentage points



H5000 -> HFull:



+3.90 percentage points



H500 -> HFull:



+6.75 percentage points



\---



\## 5. Hyperparameter provenance



The historical learning curve is NOT a fully fixed-hyperparameter ablation.



Each history-budget condition used its corresponding selected configuration.



| History | Frequency | M1 | M2 |

|---|---|---|---|

| H500 | lambda\_frequency=4 | lambda\_memory=4, top\_n=5 | lambda\_m2=4, retrieval\_k=10 |

| H5000 | lambda\_frequency=4 | lambda\_memory=4, top\_n=5 | lambda\_m2=4, retrieval\_k=20 |

| HFull | lambda\_frequency=4 | lambda\_memory=4, top\_n=20 | lambda\_m2=4, retrieval\_k=10 |



Important differences:



M1 retrieval depth:



H500: Top-N = 5

H5000: Top-N = 5

HFull: Top-N = 20



M2 retrieval depth:



H500: K = 10

H5000: K = 20

HFull: K = 10



Therefore this historical result should be interpreted as:



> performance under each history budget with its corresponding

> Dev-selected configuration



and NOT as:



> a pure fixed-parameter ablation where only the number of history rows changed.



\---



\## 6. Test-selection safety



For the directly recovered H500 and HFull cells:



test\_gold\_used\_for\_tuning = false



generic\_test\_inference\_rows = 0



status = complete



The frozen Generic Test predictions were reused rather than regenerated.



The historical H5000 M2 selection record also states:



\- selection metric: Macro-author Top-1

\- selection population: chronologically earlier whole-work Dev tune partition

\- Test Gold used for selection: false

\- Test rows seen during selection: 0



Therefore the historical Test results were evaluation outputs rather than the

source of the recorded hyperparameter selection.



These are still historical Test results and must not be reused for current

standardized-reset model selection.



\---



\## 7. Authoritative H500 artifacts



\### Frequency



Path:



`results/personalisation/reranking\_matrix/cells/full\_short/H500/F/result.json`



Macro-author Top1:



0.7399999999999999



Selected parameters:



`{"lambda\_frequency":4.0}`



Rows:



6000



Status:



complete



Test Gold used for tuning:



false



Generic Test inference rows:



0



SHA256:



`96D70E7EECED4FE78DC5B637146A4CDA5F268B087FD0C8BBD72D6BB9AD88A8B2`



\### M1



Path:



`results/personalisation/reranking\_matrix/cells/full\_short/H500/M1/result.json`



Macro-author Top1:



0.7403333333333334



Selected parameters:



`{"lambda\_memory":4.0,"top\_n":5}`



Rows:



6000



Status:



complete



Test Gold used for tuning:



false



Generic Test inference rows:



0



SHA256:



`1026A877ECB995F676700444C659C8DE09BB4105866D3AFB2D3F832BE050CEFE`



\### M2



Path:



`results/personalisation/reranking\_matrix/cells/full\_short/H500/M2/result.json`



Macro-author Top1:



0.7364999999999999



Selected parameters:



`{"lambda\_m2":4.0,"retrieval\_k":10}`



Rows:



6000



Status:



complete



Test Gold used for tuning:



false



Generic Test inference rows:



0



SHA256:



`D79161EB586E8780FC46E6487E9E2CCC2B09B4FBE2ABAEF8A0A7F4CD3A2BFB6C`



\---



\## 8. Authoritative H5000 artifacts



The reranking matrix marked Full+Short H5000 F/M1/M2 as reused-complete

rather than creating duplicate cell directories.



Historical H5000 outputs were inherited from the earlier completed

personalisation runs.



Headline results:



Frequency:



0.7718333333333334



M1:



0.7675000000000001



M2:



0.765



Known M2 selected-hyperparameter artifact:



`results/personalisation/m2\_h5000/selected\_hyperparameters.json`



M2 selected parameters:



`{"lambda\_m2":4.0,"retrieval\_k":20}`



M2 selection metric:



Macro-author Top-1



M2 selection population:



chronologically earlier whole-work Dev tune partition



M2 Test Gold used for selection:



false



M2 Test rows seen during selection:



0



M2 selected-hyperparameter SHA256:



`E47E765B950804CEAED2D2FFF5A4D2D1DBA0DDEB652AC9BF20CCD89A42A182F4`



Historical H5000 F/M1 configuration:



Frequency:



lambda\_frequency = 4



M1:



lambda\_memory = 4

top\_n = 5



The exact SHA256 values for the reused H5000 F/M1 result artifacts should be

added if/when their authoritative files are explicitly re-audited.



\---



\## 9. Authoritative HFull artifacts



\### Frequency



Path:



`results/personalisation/reranking\_matrix/cells/full\_short/HFull/F/result.json`



Macro-author Top1:



0.8035



Selected parameters:



`{"lambda\_frequency":4.0}`



Rows:



6000



Status:



complete



Test Gold used for tuning:



false



Generic Test inference rows:



0



SHA256:



`291E26E96808F976F8D0FE87FF326C2670E3EFDA1FF9D6B75E14958A104ECB2F`



\### M1



Path:



`results/personalisation/reranking\_matrix/cells/full\_short/HFull/M1/result.json`



Macro-author Top1:



0.8065000000000001



Selected parameters:



`{"lambda\_memory":4.0,"top\_n":20}`



Rows:



6000



Status:



complete



Test Gold used for tuning:



false



Generic Test inference rows:



0



SHA256:



`9A265CB0CF5D4038E9B5CB7299637C637AEE9DC4A638E973FDB50926CC1F5BAB`



\### M2



Path:



`results/personalisation/reranking\_matrix/cells/full\_short/HFull/M2/result.json`



Macro-author Top1:



0.8039999999999999



Selected parameters:



`{"lambda\_m2":4.0,"retrieval\_k":10}`



Rows:



6000



Status:



complete



Test Gold used for tuning:



false



Generic Test inference rows:



0



SHA256:



`EFDF038ABDA114A76EC1B8271E822E02CE3FE49302ECBDFB3A0BF1C326FCD637`



\---



\## 10. Historical interpretation



The historical Full+Short benchmark shows a clear descriptive pattern:



H500

<

H5000

<

HFull



for all three methods.



Frequency:



74.00 -> 77.18 -> 80.35



M1:



74.03 -> 76.75 -> 80.65



M2:



73.65 -> 76.50 -> 80.40



Therefore the historical experiment provides evidence that, on this

benchmark, useful personal information remained available beyond the most

recent 5,000 interactions.



In particular, H5000 did not behave like an empirically demonstrated

saturation point.



However, this result does NOT establish that more history will always improve

performance.



Possible effects of larger history include:



\- more useful user-specific evidence;

\- more repeated personal vocabulary;

\- more context examples;

\- stale preferences;

\- conflicting historical targets;

\- topic/domain shifts.



The historical result is therefore evidence for a history-depth effect on

this benchmark, not a universal monotonicity claim.



\---



\## 11. Interpretation of H5000



H5000 should be described as:



> a controlled bounded-memory setting



rather than:



> an empirically established saturation point.



The main reason to use H5000 in the standardized comparison is experimental

control and a common bounded recency budget across methods.



Historical Full+Short evidence shows that HFull can provide additional gains.



\---



\## 12. Relationship to the current standardized reset



This historical evidence does NOT modify the current standardized-reset

protocol.



Current standardized pipeline remains:



Clean3 Train

\-> Train-Fit / Train-Val

\-> model retraining / re-tuning

\-> PRE-DEV FREEZE

\-> frozen Dev3000 evaluation under rolling H5000

\-> method freeze

\-> Test later



Current H5000 is used to provide a common, controlled memory boundary.



Historical HFull results must not be used to retune the current methods or to

change the current Dev3000 protocol after seeing evaluation outcomes.



A future separately versioned secondary experiment may examine history depth.



Suggested research question:



> How does personalisation performance change as the amount of available

> personal history grows, and does performance saturate under bounded memory?



\---



\## 13. Important limitation for the historical learning curve



Because the historical selected retrieval configuration changes across

history budgets, the observed curve combines:



1\. increased history availability; and

2\. the corresponding Dev-selected retrieval configuration.



Therefore the result supports:



> History-budget-specific optimized performance increased from H500 to H5000

> to HFull.



It does not by itself isolate the pure causal effect of increasing history

while holding all retrieval hyperparameters fixed.



A future controlled history-depth ablation could freeze one common

configuration and vary only:



H500

H5000

HFull



if that distinction becomes important.



\---



\## 14. Reproducibility status



Recovered directly:



\- H500 F/M1/M2 result artifacts

\- H500 F/M1/M2 Macro-author Top1

\- H500 F/M1/M2 selected parameters

\- H500 result SHA256s

\- H500 Test-selection flags

\- HFull F/M1/M2 result artifacts

\- HFull F/M1/M2 Macro-author Top1

\- HFull F/M1/M2 selected parameters

\- HFull result SHA256s

\- HFull Test-selection flags

\- H5000 headline results

\- H5000 M2 selection provenance

\- historical history-budget semantics



Still optional to recover:



\- authoritative SHA256 for reused H5000 F result artifact

\- authoritative SHA256 for reused H5000 M1 result artifact

\- exact source commit/tag of every completed matrix cell



These missing items do not invalidate the recovered headline historical

result, but should be added if a fully archival provenance package is needed.



\---



\## 15. Scientific status



Historical evidence only.



Do not use these Test results for current:



\- model selection;

\- hyperparameter selection;

\- EM3 training decisions;

\- standardized Dev3000 decisions.



The current standardized protocol remains independent of this recovered

historical Test evidence.
