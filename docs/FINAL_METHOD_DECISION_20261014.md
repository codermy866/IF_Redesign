# Manuscript method decision (2026-10-14)

This document supersedes the previously planned Kernel-RIME primary method, based on user-reported 984-patient five-centre development results.

**Empirical model to write up:** PALE-JEV (Patient-Anchored Local Evidence for JEV Decisions), implemented as `RIMEHead(pooling="mlp", readout="laplace")` using the existing full clinical+colposcopy+OCT decision anchor. This is a manuscript label, not evidence of fundamental algorithmic novelty.

**Code-matched steps:** released frozen JEV frame contrasts; source-only patient-balanced 8D PCA; shared residual frame MLP `u=tanh(h+W2*tanh(W1*h))`; uniform masked frame mean of 56D signed positive/negative exponential responses with t=0.5,2,8; shared site attention; final patient logit `z=anchor+beta.T @ site_evidence`. Beta initializes to zero; no OCT -> anchor. Source-centre equal BCE, within-centre pairwise ranking (.1), L2, source-only selection. Do not refer to the selected MLP as patient-conditioned frame selection or replication-invariant kernel measure.

**Reported developer results:** pooled AUROC MLP/Laplace 0.8074; kernel/moment 0.7815; original anchor 0.7374; kernel/Laplace 0.5046, all on reused-cohort internal development. MLP centre-macro AUROC 0.7872 < original anchor 0.7922, and MLP NLL 0.5749 > anchor 0.5707; improvements are not uniform and independent confirmation is absent. Individual MLP seed AUROCs 0.8065, 0.7662, 0.7854; 0.8074 is probability-ensemble performance. Do not attribute MLP's performance to Laplace without matched MLP+Moment.

**Removed from main Method:** differentiable kernel QP, frame-level patient-conditioned q-score, CFV supervision, replication-invariance claim. Retain failed kernel experiments transparently as negative baselines and CFV as optional full-pipeline site deletion analysis.

**Critical minimum before claims:** matched MLP/Moment versus MLP/Laplace; same-patient paired comparison against anchor, centre-level AUROC, uncertainty, NLL/calibration, CIN3+ misses with missing-label repair, histopathology-only sensitivity, missing-view robustness. Diagnose pooled-versus-centre-macro AUC mismatch using patient-level score distributions without target-set tuning.

**Manuscript:** 3.1 patient-level three-modal JEV decision; 3.2 frozen frame evidence and local residual adaptation; 3.3 signed multiscale readout and site aggregation; 3.4 anchor-preserving risk update and source-only learning. Be explicit that this empirically selected design uses established MLP/pooling/features and has not yet established independent top-journal-level novelty.

No new method search or outer-held-out outcome optimization.