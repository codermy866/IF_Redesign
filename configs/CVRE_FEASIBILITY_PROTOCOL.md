# CVRE mechanism feasibility screen, 2026-09-06

Exploratory revision after ICES failure. No new claims of clinical causality or
methodological priority. Existing frozen source-trained cluster_m1_m2 predictors
are used to isolate action selection from backbone retraining. Five existing
source-validation folds; seed 20260905 only for this first-stage screen.

Initial evidence is clinical and all acquired colposcopy views. Actions reveal
one of 12 currently hidden OCT C/S clusters. Before selection, recipient hidden
OCT is zeroed. Completion donors come exclusively from the source training set,
ranked on observed clinical and observed visual features. Three donors are
sampled from the nearest 16. No recipient outcome or hidden OCT is used in
ranking. This is a finite empirical completion model, not a medical causal q.

G(j|S) = mean_d KL(p(Y|S,z_dj) || p(Y|S)).
U(j|S) = mean_d symmetric_KL(p(Y|S,frame1_dj),p(Y|S,frame10_dj)).
V(j|S) = G(j|S) - U(j|S) - 0.005.

lambda=1, normalized incremental OCT cost=1, beta=0.005 fixed before outcomes.
Stop at nonpositive maximal V, or the operational cap of four OCT units.
The cap is logged separately from value-based stopping. No thresholds tuned.
Frame changes may alter diagnostic content: U is repeat-view sensitivity, NOT
identified scanner/center nuisance sensitivity. Report-template interventions
and CIN3+ evaluation are absent. CoE is a factual trace of acquired slot,
expected gain, instability and observed posterior before/after, not fabricated
morphology or a lesion report.

Ablations: information only (lambda=0); robust; robust without cost; robust
with unmatched donors; random four-step chain; expected donor-attention chain.
Random and expected-attention prefixes are also evaluated at each case's robust
budget to provide matched-cost descriptive controls.

Primary outcomes: paired source-validation macro-AUPRC and case-level log-loss
differences; case bootstrap stratified by center and outcome, descriptive 95%
intervals. Folds overlap and will not be pooled as independent patient samples.
No clinical noninferiority or confirmatory p-values. Full-evidence reference.
Feasibility continuation rule: robust-information AP point delta positive in
>=4/5 folds, median paired AP delta versus full >= -0.02, mean robust OCT cost
below four and positive median AP delta versus matched-cost random. These are
screening rules, not proof of safety or causality. If they fail, report the
failure and propose a bounded next design; do not launch a parameter search.

Mathematical limitation: expected KL equals conditional mutual information only
with coherent conditional predictive distributions and completion q. Here the
frozen discriminative predictor and empirical q need not be coherent. Expected
KL is therefore called posterior-update magnitude, not verified information
gain. A future robust risk-reduction action value and observation-conditioned
uncertainty bound require additional estimation and validation.
