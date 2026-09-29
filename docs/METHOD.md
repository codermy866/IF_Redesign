# Current Method: Stagewise Multimodal Information Innovation

## Scientific target

The current method is designed for settings in which evidence arrives in stages. Let `C` denote the evidence already available, `O` a newly acquired modality, and `Y` the prediction target. The central question is not only whether `(C,O)` can predict `Y`, but whether `O` contributes task-relevant information beyond the belief already established from `C`.

We represent the evidence order by nested information states `F0 = sigma(C)` and `F1 = sigma(C,O)`.

## 1. Prior-anchored update

The pre-evidence belief is `q0 = P(Y=1|C)`. A multimodal encoder produces a scalar evidence score `r(C,O)`. The post-evidence prediction is constrained to update the previous belief:

```
logit(q1) = logit(q0) + beta * r,    beta >= 0
```

This is intentionally different from passing `q0` as an ordinary feature to an unconstrained joint predictor.

## 2. Stagewise Proper-Score Innovation

For a strictly proper scoring loss `ell`, define

```
I_ell(F1:F0) = E[ ell(Y,p0) - ell(Y,p1) ].
```

This quantity measures outcome-aligned predictive-risk reduction rather than posterior displacement.

Two realizations are used:

- Brier innovation `J`: at the Bayes predictors, `J = E[(p1-p0)^2]`.
- Log-score innovation `V`: at the Bayes predictors, `V = I(Y;O|C)`.

## 3. Cross-fitted estimation

SOPI is estimated only from held-out/cross-fitted predictions:

```
I_hat = mean_i [ ell(Y_i, q0_i) - ell(Y_i, q1_i) ].
```

The outcome is used for retrospective attribution, not for forward inference.

## 4. Orthogonality

At the true stage-specific predictive distributions, the first directional derivative of the proper-score risk difference vanishes. Under standard smoothness conditions, local nuisance-prediction error enters the SOPI estimand through second-order terms. This is a bias-robustness property, not a claim of uniformly lower finite-sample variance.

## 5. Stagewise accounting

For nested evidence states `F0 subset F1 subset ... subset FT`, proper-score innovation telescopes across stages. This allows a long diagnostic process to be decomposed into the predictive information added at each evidence transition.

## Current cervical instantiation

The current study uses:

```
age / HPV / cytology -> pre-OCT belief q0
OCT evidence         -> anchored update q1
pathology            -> retrospective SOPI evaluation
```

The MLLM/VLM backbone is treated as an evidence encoder. The primary methodological contribution is the stage-ordered fusion formulation and the information estimand, not a new Transformer block.
