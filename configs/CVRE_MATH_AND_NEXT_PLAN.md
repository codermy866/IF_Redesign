# CoE acquisition with counterfactual re-inspection: mathematical roadmap

## Executed screen

See CVRE_FEASIBILITY_PROTOCOL.md and results/cvre_feasibility_v1/summary.json.
Source-only frozen-predictor screen, five folds and one seed. All clinical and
colposcopy evidence is initially present; actions reveal OCT position clusters.
CoE is an auditable trajectory, not a generated pathology explanation. No new
selector training, report-template intervention, center causal intervention,
CIN3+ safety study, or external center evaluation has been completed here.

## Why expected KL alone is not diagnostic utility

Write p0=p_theta(Y|S), pz=p_theta(Y|S,z), and pbar=E_q[pz]. Exactly,

E_q KL(pz || p0) = E_q KL(pz || pbar) + KL(pbar || p0).

The first term is disagreement resolved by an observation under the completion
mixture. The second measures a mismatch between the mixture and the present
predictor. Under coherent predictive conditionals it vanishes; with an empirical
donor completion and discriminative subset predictor it need not vanish.
Derivation: insert log(pz/pbar)+log(pbar/p0) into the KL integrand, then use
E_q[pz]=pbar in the second term. This is an algebraic identity, not a claim of
a novel information-theoretic theorem. Numerical or empirical contributions
must be demonstrated against prior active-feature-acquisition work.

## Candidate next action value, not yet validated

Define J_j(S)=E_q KL(pz || pbar). Let U_j(S) measure sensitivity to an explicitly
validated diagnostic-content-preserving observation transformation. Let
W_j(S) quantify uncertainty from finite completion support. A candidate is

A_j(S) = J_j(S) - lambda U_j(S) - eta W_j(S) - beta c_j.

Compare this to raw expected KL at identical budgets, predictor and q. A high
J is still not proof of correctness. A complementary supervised action target
is the source-only observed proper-score improvement R_y(S)-R_y(S+e_j), learned
from source training trajectories and evaluated out of sample. The predictor
should be frozen while estimating deletion/reinspection targets so it cannot
improve the necessity score by deliberately worsening deletion predictions.

Do not use opposite-label recipients to choose inference donors: recipient
outcomes are unavailable at decision time. Opposite-label matching is permissible
only as a labeled source-training or post-hoc diagnostic audit with its own
interpretation; it does not identify a same-patient causal counterfactual.

## Evidence-chain record

Record observed slots -> estimated action value -> revealed slot -> posterior
update -> continue/stop rationale. Reports may verbalize these recorded actions
but cannot invent lesions, reviewed ROIs or morphological conclusions. If
computing an unobserved feature is necessary for a score, count it as acquired;
do not report the eventual retained subset as all the information accessed.

## Ordered next experiments

1. Check calibration and source-center/outcome composition. Add the initial
   clinical+colposcopy baseline and OCT-only/full references. Separate marginal
   OCT utility from selector quality. Use source-training calibration splits.
2. At common budgets 1-4, compare expected KL, centered J, attention and random.
   Quantify KL(pbar||p0) separately. No new risk penalty until a gain is shown.
3. Validate nuisance transformations with evidence. First/last frames can change
   diagnostic content; they are repeat-view sensitivity controls only. Center
   feature translation is also a proxy unless content preservation is verified.
4. Add one risk term at a time, with q support/uncertainty and same-budget
   comparisons. Report action quality and correct-to-wrong as well as
   wrong-to-correct posterior transitions.
5. After fixed-budget improvement, learn a stopping policy on source train and
   calibrate its error/cost constraint on a separate source calibration split.
   Low expected value should trigger uncertainty/referral when confidence is
   unsupported. Use a cap as an operational limit, not a sufficiency certificate.
6. Repeat predeclared finalists with three seeds and patient-level paired
   statistics; preserve validation reuse as exploratory. Only then consider
   new-center assessment. Existing five centers do not provide a fresh
   confirmatory cohort after repeated development.

## Observations from the completed one-seed screen

All five folds completed. Robust-minus-information macro-AUPRC point deltas
were negative in four folds and zero in one. Across-fold medians: robust .5870,
information .6744, cost-free robust .6089, four-step random .6767, and
robust-budget-matched random .6744. Mean fold robust OCT count was .5493.
These medians are descriptive: differences of medians are not paired effects.
The current observed-state completion + repeat-view penalty does not support
advancement to a large training campaign under the locked screening rule.

Every fold contained a source-validation center with only one positive case;
the enshi fold had two such strata (15 cases/1 positive and 14 cases/1 positive).
Macro-AUPRC may move sharply after a small rank change there. Report per-center
support and paired intervals; do not treat the five overlapping source folds
as independent external validations. The initial state already contains all
colposcopy and clinical features, so a short OCT chain alone does not establish
overall diagnostic efficiency.

## Relation to prior work

Active feature acquisition, information-gain selection and cost tradeoffs have
substantial precedent; adding their names does not establish novelty.
https://proceedings.mlr.press/v267/norcliffe25a.html
https://proceedings.mlr.press/v235/valancius24a.html
https://www.jmlr.org/beta/papers/v26/23-1635.html

The defensible potential contribution is an observable, counterfactually audited
evidence-acquisition value with validated intervention assumptions and useful
finite-sample stopping behavior. This remains a research hypothesis.
