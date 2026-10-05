# Experimental Status

This document records the aggregate experimental evidence supporting the current IF_Redesign method. No patient-level data are exposed.

## Stage102: patient-specific OCT information

Patient-matched versus mismatched/correspondence analyses showed a positive pooled patient-specific OCT signal. This supports the premise that OCT contains diagnostic information beyond a fixed clinical prior.

Boundary: fixed correspondence objectives were not uniformly robust across centres and are not retained as the final method.

## Stage103: discrete Evidence-State routing

The discrete patient state formulation `{clinical, OCT, fusion}` was tested.

Result: state-utility relationships were centre-dependent and the learned routing policy produced unstable regret/negative-fusion behaviour.

Conclusion: discrete Evidence-State classification is not retained as the final core.

## Stage104: absolute action-risk / JEV-like decision formulation

The model estimated absolute conditional losses for clinical, OCT, and fusion actions and formed a relative value from those estimates.

Result: the apparent reduction in negative fusion was largely explained by policy collapse toward the clinical-only action (approximately 97-100% clinical selection in the decisive folds).

Conclusion: estimating several noisy absolute risks and subtracting them is rejected as the final formulation.

## Stage105: Direct Evidence Advantage

Stage105 directly targets the relative predictive consequence of accepting the OCT update:

```
A_i = loss_i(clinical) - loss_i(clinical + OCT update)
```

### Decisive two-centre probe

The initial frozen-backbone probe produced positive advantage ranking/discrimination without clinical-only collapse. This provided the GO decision for Direct Evidence Advantage.

### Four-centre x two-seed stability study

A lower-variance benefit-sign estimator was evaluated over 8 fold-seed cells.

The independent calibration -> threshold-selection -> validation protocol currently yields:

- benefit AUC > 0.5 in **8/8** cells;
- mean benefit AUC **0.723**;
- benefit AUC range **0.638-0.801**;
- positive realized gain over clinical in **7/8** cells;
- selected-update coverage **0.264-0.969**;
- decision regret lower than always-updating in **4/8** cells.

### Interpretation

The principal positive result is that the **patient-specific sign of OCT benefit is predictably non-random across every tested fold-seed cell**.

The remaining weakness is threshold transfer: selecting one operating threshold on a source selection split does not yet yield uniformly lower regret on validation.

Therefore the current method status is:

```
Evidence Advantage mathematical object         FROZEN / RETAIN
Direct benefit prediction                      SUPPORTED
Clinical-only policy collapse                  RESOLVED at score-learning level
Universal selective threshold superiority      NOT YET SUPPORTED
Finite-sample risk guarantee                    NOT CLAIMED
```

## Main paper implication

The paper should claim that multimodal evidence has patient-specific decision consequences and that these consequences can be learned directly. It should not yet claim a universally dominating risk-controlled selective policy.

The final experiment package should continue to report:

- diagnostic AUROC/AUPRC/NLL/Brier;
- benefit AUC;
- update coverage;
- decision regret;
- harmful/negative update rate;
- bootstrap confidence intervals;
- comparison with DIRECT, GATE, state routing, absolute-risk decision models, and recent reliability-aware fusion baselines.

## Clinical boundary

The current IF study uses clinical information plus OCT and pathology reference. Colposcopy images are not part of the core IF input. Whether OCT can reduce the incremental need for subsequent colposcopy is reserved for Discussion/exploratory analysis and is not a current replacement claim.

## Public-repository policy

Patient identifiers, raw images, private clinical records, model weights, private prediction tables, and manuscript files are excluded. Only reusable methodology, protocol definitions, aggregate results, and tests are published.
