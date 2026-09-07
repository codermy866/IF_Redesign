# CSF v1: locked development screen

2026-09-06. Execute seven acquisition rules, budgets 0/1/2/3/4 OCT positions,
five existing source folds and all three existing checkpoint seeds (20260905-07).
No outer-test evaluation, new backbone fitting, or test-driven threshold search.
Preserve all historical runs. A position is ten cached frames, not one B-scan;
clinical and all colposcopy evidence are acquired initially. Costs count additional
OCT positions only, not runtime, clinical cost or total acquired modalities.

## Modules and contrasts

M1: observed-state completion and exact information/coherence decomposition.
For binary predictive distributions p_z and p_0, let p_bar = E_q p_z.
E KL(p_z||p_0) = E KL(p_z||p_bar) + KL(p_bar||p_0).
This is a standard identity, not a new theorem or a causal identification result.
Compare total KL, centered disagreement J, and coherence gap B at identical budgets.

M2: completion portability and sensitivity controls. Compare J to J minus
first/last-frame symmetric KL (unit coefficient); compare matched donors with
cross-source-center matched donors. Frame changes can alter disease content:
these are sensitivity proxies, NOT medically validated nuisance interventions.
Random and expected-attention acquisition are concurrent equal-budget controls.
Eight nearest-pool donors per state; labels and hidden recipient evidence are
excluded from matching. Cross-center matching excludes the recipient center.

M3: finite perturbation / worst-source-center / cost frontier. For each fixed
budget and policy, record worst-center Brier risk and mean positive excess Brier
under joint selected-OCT first-frame / last-frame substitutions, including identity.
This is an empirical finite-perturbation frontier, not a supremum over all clinical
counterfactuals, and population risk is of a policy, not of an individual patient.
Select the cheapest budget on a deterministic 20% stratified subset of source
training cases, excluded from donors, with worst-center risk <= full + .02 and
mean finite-perturbation regret <= .02. Otherwise return full with explicit
constraint-failure flag; full is not automatically certified. The frozen predictor
has seen these calibration cases during fitting: calibration is development-only,
not independent risk certification. Validation never selects budgets.

## Evidence gate

Primary diagnostic contrast: centered versus total KL at budget 4. Secondary:
centered versus random and attention; stable and cross-center versus centered.
Report all budgets, seeds, folds and negative findings. A candidate warrants new
training only if its seed-averaged paired macro-AP gain over random AND attention
is positive in >=4/5 folds, with median gain >=.01 over each, and median paired
gain relative to full >=-.02. This is an exploratory resource-allocation gate,
not a significance claim. Seed ensembles are probability averages, not independent
patients. Validation patients overlap folds; never bootstrap fold appearances as
independent cases. Bootstrap per fold at patient level, stratified center/outcome;
intervals descriptive without multiplicity claims. Report center class counts.

No CIN3+ claims with binary labels. HPV16 replacement is a diagnostic-content
intervention, not an invariance target. Opposite-pathology donor matching is
prohibited for acquisition. No hospital-style or report-corruption results without
corresponding data. No clinical safety, causal or transportability guarantees.

Stop this finite screen after verified completion and pivot on mechanism evidence,
not by searching validation thresholds until a favorable result appears.
