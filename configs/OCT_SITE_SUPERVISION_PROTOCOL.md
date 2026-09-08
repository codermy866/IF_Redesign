# Point-annotated M1/M2 factorial pilot

User-confirmed on 2026-09-06: every number listed in OCT二次判读 is a positive
OCT site; unlisted sites are negative. This overrides the historical script's
incorrect-for-this-task conversion of low-grade annotations to all-negative.
These are OCT reread findings, NOT site pathology, individual-frame labels or masks.
981/984 current patients align; ambiguous/unmatched cases retain case supervision
but are masked from site losses. Mapping uses C=1..12, not S or string ordering.

Primary development design: 5 existing source folds x seed 20260905 x 2x2 factors.
All four arms start from the identical cluster_m1_m2 checkpoint per fold, same
12 fine-tuning epochs, minibatch32, AdamW lr1e-4/wd1e-4 and seeded batches/masks.
No validation-based epoch selection, no outer evaluation. Existing checkpoint
selection used this validation cohort; report development-only conclusions.

- A00: full-case BCE + .5 randomly retained OCT-subset BCE; no site losses.
- A10 / M1: add .25 balanced site BCE on OCT importance logits, using site labels
  only from source-training patients. The ranking is thus explicitly grounded in
  reread findings rather than interpreting arbitrary attention as lesions.
- A01 / M2: add .1 directional site-intervention ranking hinge, margin .5 logit.
  For an annotated positive anchor position, select an annotated negative donor
  from a different SOURCE center at the same C position; replace only that token
  and require its site score to fall. Identity/context are held fixed in model
  input, but this synthetic replacement is not an identified clinical counterfactual.
  Both compared passes use eval-mode dropout for the intervention loss, with
  gradients enabled. No constraint forces the whole patient's CIN2+ label to flip.
- A11: both M1 and M2. Add an unmodified checkpoint reference at evaluation.

M2 donor selection uses source training labels only, never recipient validation or
outer labels. M1/M2 training is paired in model initialization, patient batches and
random subset masks. Primary mechanisms: within-case site AP (mixed-label cases),
positive-site recall at4, matched-count random recall and case CIN2+ macro-AP at4.
Also report full evidence, random4, zero OCT. Report all main-effect comparisons
and interaction A11-A10-A01+A00. A single-seed pilot establishes feasibility only.

Important: evaluation ranks already extracted features from ALL OCT positions.
This is offline evidence retention/re-inspection prioritization, NOT prospective
acquisition cost reduction. Site labels score selections AFTER inference; they
never prefilter evaluation frames. No pixel localization or clinical necessity
claim. Point-level metrics weight cases, not all 12 sites as independent patients.

Keep CSF/M3 unchanged until the new M1/M2 mechanism pilot is validated. Differences
from earlier CSF strategies include task regime and supervision, so compare the
four concurrent arms, not unpaired historical medians.
