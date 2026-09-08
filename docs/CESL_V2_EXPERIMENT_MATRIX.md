# CESL-v2: Three-Module Innovation and Ablation Matrix

Status: exploratory follow-on to the locked Formal v1a experiment. It must be
reported separately from Formal v1a, because the method and experimental
question were introduced after the v1a training plan was fixed.

## Higher-level method narrative

Conventional multimodal systems optimize a static predictor over all available
inputs. CESL-v2 reframes diagnosis as a sequential evidential decision: given
a patient's partially observed multimodal record, the model asks which next
piece of evidence is worth acquiring, whether the present record is already
sufficient, and whether that decision remains stable under clinically
constrained evidence replacement and centre shift.

The target is therefore a **centre-stable sufficient evidence frontier**:
the smallest evidence set that retains discriminative performance, calibrated
uncertainty, and conditional counterfactual robustness on a previously unseen
hospital. This is a candidate mechanism, not an established causal finding.

## Module M1: Conditional Counterfactual Evidence Valuation (CEV)

For a selected atom k, CEV estimates the degradation in supervised diagnostic
risk after replacement with a source-centre-matched donor atom. Matching
variables and every donor edge are retained in the audit. CEV asks whether an
atom is decision-relevant under the declared replacement distribution rather
than merely salient in an attribution map.

**Ablation M1-off:** replace CEV ranking by the same backbone's attention or
gradient ranking, with the selected atom count held fixed.

## Module M2: Budgeted Sufficient Evidence Policy (BSEP)

BSEP uses predicted acquisition value per relative evidence-cost unit to choose
the next atom and stops only when margin and residual acquisition-value
criteria are met. It turns a fixed fusion model into a selective-acquisition
model; costs are initially abstract evidence units, not clinical time/cost
claims.

**Ablation M2-off:** acquire all available atoms, or use a fixed-cardinality
top-k policy matched to the full policy's mean atom count.

## Module M3: Centre-Conditional Sufficiency Stability (CSS)

CSS regularizes the source-centre policy so that conditional evidence-value
ranks and stopping failures do not depend on centre identity after conditioning
on declared observable strata. It minimizes a worst-source-centre criterion
and penalizes cross-centre disagreement in atom rankings; it does not force
identical raw probabilities across centres, because prevalence and true case
mix differ.

**Ablation M3-off:** pooled empirical-risk training with identical atoms,
counterfactual samples, and budget, but no stability penalty or worst-centre
selection criterion.

## Required experiment families

| ID | M1 CEV | M2 BSEP | M3 CSS | Purpose |
|---|---|---|---|---|
| C0 | no | no | no | Full-evidence hierarchical reference. |
| C1 | attention/gradient | fixed top-k | no | Attribution selection control. |
| C2 | yes | fixed top-k | no | Isolate counterfactual valuation. |
| C3 | yes | adaptive stop | no | Isolate selective acquisition. |
| C4 | yes | adaptive stop | yes | Full CESL-v2. |

Each condition must use the same backbone, source-centre data, atom dictionary,
and outer LOCO folds. C1--C4 must be matched on expected evidence cost where
applicable. The evaluation unit is the patient/case, not individual frames.

## Falsification checks

1. C4 must not lose pre-specified held-out macro-centre AUPRC or sensitivity
   beyond the declared tolerance at matched evidence cost.
2. CEV-ranked atoms must produce a larger matched replacement loss than
   attention/gradient and random atoms of the same type and cardinality.
3. C4's rank/stopping stability across held-out centres must exceed C3's;
   otherwise CSS has no demonstrated role.
4. If results depend strongly on unconditioned donors, the method is sensitive
   to q and cannot support the intended robustness story.

## Data admission gate

Formal v1 contact sheets are insufficient for a B-scan-level claim. CESL-v2
starts only after the raw-frame feasibility audit confirms case-level frame
availability and writes a leakage-safe raw-atom manifest. No reviewed ROI is
required for whole-atom replacement, but unreviewed ROIs remain forbidden from
lesion-causal, negative-label, or GRPO claims.
