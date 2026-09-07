# Cross-fitted diagnostic gain: structural pivot, 2026-09-06

## Scientific question and scope

Can a predictor of out-of-fold, context-conditioned diagnostic loss reduction
rank evidence more usefully than attention, generic uncertainty or OCT abnormality?
Prior failures apply to the implementations/cohort tested; they do not disprove
all information-theoretic or robustness-based acquisition. Point labels improved
point localization but not the old diagnostic reference. Do not hide this result.

Use all existing eligible cases in their locked SOURCE roles; do not merge train,
validation and held-out hospital for the sake of claiming use of complete data.
984 total cases, 981 point-label matches. No case is dropped from diagnosis for
missing point labels. No LLM, stopping-rule optimization, new architecture or
unreviewed nuisance-invariance training in this stage.

## Cross-fitting and leakage barrier

Five existing outer source splits, one fixed seed 20260905 for feasibility. Within
each source-training set construct five center/outcome-stratified patient folds.
For each held fold H, train a fresh ICES diagnostic model on source minus H,
with a separate 15% internal calibration split from that complement. H never
enters model fitting, clinical age scaling or checkpoint selection. Feature
extractor is the existing frozen ImageNet encoder, not cohort-trained weights.
Same architecture, lr3e-4, wd1e-4, batch48, max50 epochs, patience10, full plus
.5 random-mask class-weighted BCE. Pick internal-calibration macro-AP. Archive
fit/cal/gain membership and scaling per teacher. No old checkpoint initialization.

Train a sixth independent diagnostic reference per outer split, with the same
internal-calibration procedure and only source-training data. Freeze it for all
selector evaluations. No selector gradient updates diagnosis weights.

## Targets and controlled ablations

For each source patient i, seeded random evidence states t=0..5 include clinical,
all colpo and t OCT positions. Enumerate every remaining OCT position j.
G_ij(S)=CE(y_i,f_-H(S))-CE(y_i,f_-H(S union j)). Negative gains are retained.
This is an out-of-fold MODEL loss difference, not true causal/clinical utility.

Compare gain_oof with gain_seen: identical patient/state/action rows, same Ridge
alpha10 and inputs; the seen teacher is a different equal-architecture inner
teacher whose gradient-fit set contains patient i (not just calibration). Both
teacher sizes/fit membership are recorded. This reduces capacity/cohort confounding
but different fitted teachers remain a limitation. Do not substitute the larger
full-source teacher and call it a perfectly matched control.

Input: observed state features, clinical variables, current probability, C slot,
and fixed-projection candidate OCT features. The candidate has already been
imaged. Hence this is OFFLINE visual re-inspection/retention, not prospective
image acquisition. No label, donor pathology or true gain at validation inference.
Scale selector inputs only on source training action rows. Train center-balanced
Ridge with identical capacity for gain_oof/gain_seen and a point-label ranking
control. Point labels never define a patient's CIN2+ outcome.

Baselines: random, full-context attention, entropy of candidate-updated prediction,
KL change, five-dropout-pass variance, predicted OCT point abnormality; unselected
clinical+colpo and full12 diagnostic references. Evaluate all fixed budgets0..6,
plus full12, no selected stopping threshold. Compute-greedy baselines inspect all
candidates; retained image count is not total runtime/acquisition cost.

## Attention audit and endpoints

On source validation only AFTER trajectories are fixed, report within-patient
attention-versus-realized-gain rank agreement, site-positive versus negative gain,
site recall@4, CIN2+ case macro-AP and AUROC, source-center and HPV16/18-group
breakdowns. Do not call OCT reread sites pathology regions; no CIN3 endpoint is
inferred from a binary cache. Low-count subgroups are descriptive only.

Primary contrast gain_oof - gain_seen at4; other comparisons are secondary and
all retained. Patient is the resampling unit, not action/site/seed; outer-source
validation cases overlap. No cross-fold significance by treating folds as iid.
A positive pilot is not a top-journal certificate: freeze hypotheses/settings,
then repeat seeds and perform locked exploratory held-hospital validation. Select
on development only; report the complete search ledger and negative controls.

## Nuisance and novelty boundaries

Brightness/noise/style changes are NOT automatically structure-preserving in OCT.
Before adding invariance loss, create an intensity-range/saturation and matched
position audit, then require clinical plausibility review. No raw-frame endpoint
swap is called a pure nuisance intervention. Scanner identity cannot be changed
causally by translating an embedding without validation.

Cross-fitting and expected loss reduction alone are not new inventions. Relevant
primary work includes https://proceedings.mlr.press/v235/valancius24a.html and
https://proceedings.mlr.press/v139/li21p/li21p.pdf. The candidate contribution is
the tested distinction between reread abnormality and out-of-fold conditional
diagnostic utility, with an auditable evidence chain—not a new name for Ridge.
