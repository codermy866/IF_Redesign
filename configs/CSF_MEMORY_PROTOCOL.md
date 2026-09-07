# Independent structural branch: source-trained risk-gain memory

Locked before CSF validation readout, 2026-09-06. Same 5 folds and 3 frozen
checkpoint seeds. Train a small Ridge(alpha=10) action-value model on source-fit
patients; retain the identical source-calibration partition used by CSF.
Targets are Brier(S,y)-Brier(S+true acquired OCT_j,y) from frozen predictors.
The classifier is never optimized to make deleted evidence predictions worse.
Training states follow seeded random acquisition paths at budgets 0-3, enumerate
all remaining actions, and use labels only from source-fit patients.

Action features contain clinical inputs, masked observed visual means projected
with a fixed Gaussian matrix (16 dimensions each for colpo and acquired OCT),
current probability, acquired fraction, and action-position interactions. There
is no hidden recipient OCT feature, recipient outcome, or future-model output in
the action-value inputs. True next observations construct TRAINING targets only.

Compare: one center-balanced pooled value model (gain_mean); the minimum prediction
of four leave-one-source-center-out value models (gain_transport). This minimum
is a conservative expert heuristic, NOT a statistical confidence bound or a
proof of cross-center transportability. Donor/feature memory is not a clinical
knowledge base, and no language rationale is generated.

Evaluate fixed budgets 0-4 on the existing source validation; finite-view regret
and frontier budget choice follow CSF_SCREEN_PROTOCOL.md. Training targets come
from cases previously seen by the frozen classifier, so optimism is possible.
This is a feasibility branch; any positive gate requires nested out-of-fold
predictor training next, not direct clinical/causal claims.

Use the same gate and full/random/attention controls as CSF; retain all results.
This branch changes the learning target, not validation-selected hyperparameters.
