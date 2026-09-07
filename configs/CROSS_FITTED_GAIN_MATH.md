# Scientific interpretation: abnormality, information and diagnostic utility

## What the completed exploration supports

Point reread supervision improved within-patient localization in all five pilot
source splits, but shared-network fine-tuning reduced diagnosis relative to the
unmodified checkpoint. Thus detection of OCT reread abnormalities is not a
sufficient surrogate for preserving this model's CIN2+ diagnostic performance.
This does not show that the abnormalities are clinically irrelevant, nor that
every stability/information criterion fails in medicine.

The original frozen full-context attention audit (phase1_attention_summary.json)
shows label-dependent association with single-position loss reduction. Averaging
within patient across overlapping source folds, mean within-case Spearman rho is
0.3133 among 53 CIN2+ patients and -0.2514 among 144 CIN2-negative patients.
This is descriptive and retrospective, not an inference-time use of outcomes.
It motivates separating sensitivity to abnormality from conditional diagnostic
benefit. It does not identify inflammation or a particular microscopic mechanism.

This audit's full-context attention is not the donor-completion expected-attention
policy in the older CSF screen; historical labels named 'attention' are not enough
to establish identical controls across protocols.

## A necessary correction to the proposed theoretical story

For a fixed observed state s, fixed candidate position j, true distribution P and
log loss, write p0=P(Y|s), pj=P(Y|s,Xj), and model predictions f0 and fj. Then

E[G_f | s] = I_P(Y;Xj | s)
              + KL(p0 || f0)
              - E_{Xj|s} KL(pj || fj).

Derivation: expected cross entropy is conditional entropy plus the model's KL
approximation error; subtract before/after terms. This is a standard decomposition,
not a new theorem. Under the true Bayes posterior, both approximation errors vanish
and expected diagnostic log-loss reduction equals conditional mutual information.
Hence 'information is never diagnostic value in medicine' is mathematically too
strong. A learned model, finite donor approximation, distribution shift, or a
realized individual outcome can break the intended correspondence of a plug-in
information score with observed diagnostic benefit.

The fixed-candidate expectation above is not a theorem about an offline selector
that inspects all candidate values: selection itself may convey information.
The current experiment is explicitly offline re-inspection/retention.

Related primary results and prior work:
- Covert et al., ICML 2023, propositions1–2 connect greedy conditional information
  with Bayes-optimal one-step prediction loss:
  https://proceedings.mlr.press/v202/covert23a.html
- Acquisition Conditioned Oracle, ICML 2024, studies active feature acquisition:
  https://proceedings.mlr.press/v235/valancius24a.html

## Testable candidate, not three pre-certified innovations

1. Cross-fitted conditional diagnostic gain: train g(s,e) on signed loss changes
   from teachers that never saw the target patient, including clinical scaling
   and checkpoint selection. Compare with equal-capacity seen-patient teachers.
2. Decouple abnormality from utility: keep diagnosis frozen and compare identical
   value models trained against reread point labels versus OOF diagnostic gain.
   Require both localization and diagnostic endpoints, not one favorable metric.
3. Nuisance-validated evidence verification: proposed next stage, NOT implemented
   as a successful module. Require matched raw-image structure/quality audits
   before intensity/noise/style invariance. Neither frame replacement nor latent
   center translation establishes a pure nuisance intervention.

OOF gains remain model-based, noisy targets. Cross-fitting avoids self-training
memorization in target generation; it does not identify clinical causal effects,
guarantee cross-hospital transport, or make loss-difference regression novel by
itself. Claims need complete controls and external replication.

## Proposed paper-level question

Can out-of-fold conditional diagnostic utility distinguish visually abnormal
evidence from evidence that actually improves the decision of an independently
trained diagnostic model, without sacrificing diagnostic performance?

This question survives positive or negative results. The manuscript should retain
the full search history and designate any best development result as selected,
not present the best cell from each incompatible protocol as a single model.
