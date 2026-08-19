# EM-2 Adaptive G+F+C Fusion Design

Status: DESIGN FROZEN BEFORE DEV RESULT
Date: 2026-08-19

## Question

Can prediction-visible confidence signals determine, per query, whether
Frequency or Hidden Context should receive more trust?

This experiment follows the negative/neutral Fixed G+F+C result.

The Adaptive Fusion hypothesis itself was registered before the Fixed Fusion
result was observed.

## Scope

- Full+Short
- H5000
- Dev tune only
- Etinjat
- Re_spectators
- breaddddd
- 5,608 queries
- Frozen Generic candidate surface
- Frozen Hidden representation
- Frozen Hidden retrieval Top-N = 3
- no Recovery
- no Test

## Signals

G(c):
Frozen Generic normalized score.

F(c):
Frozen same-Pinyin frequency support.

C(c):
Frozen Hidden-M1 Top-3 target support.

## Prediction-visible confidence

No Gold-derived information may be used.

### History confidence

For n visible legal same-Pinyin historical interactions:

    H(q) = n / (n + 5)

The constant 5 is fixed before results and is not tuned.

### Frequency confidence

    CF(q) = H(q) * frequency_margin(q)

frequency_margin is the normalized difference between the first and second
historical target counts.

### Context confidence

    CC(q) =
        H(q)
        * clamp(top1_hidden_cosine, 0, 1)
        * retrieved_target_agreement

retrieved_target_agreement is the fraction of the frozen Hidden Top-3
retrieved histories supporting the most common retrieved target.

## Dynamic personalisation budget

For global personalisation scale L:

    strength(q) = L * max(CF(q), CC(q))

If CF + CC > 0:

    alpha_F(q) = CF / (CF + CC)
    alpha_C(q) = CC / (CF + CC)

    lambda_F(q) = strength * alpha_F
    lambda_C(q) = strength * alpha_C

Otherwise:

    lambda_F(q) = 0
    lambda_C(q) = 0

Final score:

    score(c) =
        G(c)
        + lambda_F(q) * F(c)
        + lambda_C(q) * C(c)

This gives Frequency and Context one shared personalisation budget instead
of independently adding two full-strength personal signals.

## Global scale grid

    L in {1, 2, 4, 8, 16}

Primary selection:

    Macro-author Overall Top1

Tie break:

    lower L

The grid is bounded at 16. No further boundary expansion is allowed.

## Count-ablation control

A diagnostic no-history-shrinkage version is also evaluated:

    H(q) = 1 when history exists

All other confidence logic remains identical.

This control tests whether using history quantity itself contributes beyond
Frequency/Context confidence.

It is diagnostic and is not used to redefine the primary Adaptive method.

## Baselines

Report on the same surface:

- G
- F
- Hidden-M1
- Fixed GFC (lambda_F=0.5, lambda_C=4)
- Adaptive Fusion
- Adaptive no-count control

## Reporting

Report:

- Overall
- History Available
- Ambiguous
- Conflict

Also report:

- F -> Adaptive rescue/harm/net
- Hidden-M1 -> Adaptive rescue/harm/net
- Fixed GFC -> Adaptive rescue/harm/net

And descriptive gate statistics:

- mean lambda_F(q)
- mean lambda_C(q)
- median lambda_F(q)
- median lambda_C(q)
- queries where lambda_F > lambda_C
- queries where lambda_C > lambda_F
- equal / zero-personalisation queries

## Information boundary

Forbidden gate inputs:

- Gold
- correctness
- Conflict label
- future history
- author identity

Conflict may be used only after prediction for evaluation.

## Stop rule

This is the final transparent Adaptive Fusion experiment in EM-2.

Do not add new gating features after seeing this Dev result.

After this experiment:

- update EM-2 documentation;
- freeze the Dev method state;
- then decide on Test opening.
