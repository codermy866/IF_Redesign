# RIME-JEV — manuscript METHOD FREEZE (2026-10-13)

**FINAL DESIGN DECISION FOR WRITING; NOT A CLAIM OF CLINICAL VALIDATION.**

## One problem, one method

Patient-level cervical CIN2+ prediction from age/HPV/TCT, colposcopy and multi-site OCT requires decisions that do not treat correlated repeat frames as independent support and do not erase sparse evidence. The study proposes patient-context-conditioned evidence measure learning. Frozen, released JEV-27B is a **representation/decision backbone**, not a new pretrained foundation model.

### Inputs and anchor — fixed

Each patient has clinical covariates, colposcopy and up to 12 OCT positions with up to 10 frames per site. A complete **source-trained** clinical/native/26D semantic multimodal anchor `a_i` uses clinical, colposcopy and an OCT montage; the new OCT evidence branch encodes each raw frame independently with the frozen JEV-27B released backbone. Frame features are gray-referenced, source-only patient-balanced PCA-projected to **8D**, zero-origin preserved. All transforms fitted within the source fold. Never conflate patient montage and raw-frame protocols.

### Main new contribution: conditional kernel evidence measure

Within each position collapse **exact** duplicate 8D vectors to unique atoms. Define Gaussian kernel `K_{jk}=exp(-||h_j-h_k||²/(2 sigma²))` with source-only bandwidth and context-conditioned scores `q_j=tanh(a^T h_j + <U^T h_j,V^T b_i>)` (rank 2). Solve `w*=argmin_{w>=0,1^Tw=1} .5 w^T K w - tau q^T w`. A source-fitted differentiable active-set solver implements this conditional non-redundant evidence measure. Claim exact replication invariance only **conditional on fixed feature mapping/anchor/context**, and distinguish the role of atom merging from that of the conditional QP. Do NOT imply near-duplicate invariance without experiment.

### Evidence readout: 56D signed multiscale Laplace responses

Use `phi(h)=[h, 1-exp(-t[h]+), 1-exp(-t[-h]+)]` for `t in {0.5,2,8}` (56 coordinates for 8D input). A site descriptor is `v_is=sum_j w_isj phi(h_isj)/s_source`. Source-only scales are estimated before deduplication and remain frozen. Shared site attention pools valid sites into `r_i`; the patient prediction is `logit(p_i)=a_i + beta^T r_i` with `beta=0` initially, ensuring the original anchor is the initial decision and exact no-OCT fallback. This is **not** an estimated Laplace probability law or calibrated uncertainty.

### Learning and counterfactual analysis — fixed scope

Fit source-centre equal BCE, 0.1 within-centre ranking loss and L2 with source-only selection. Keep published `configs/rime.json` as the base experimental menu; do not enlarge the search. The two-stage selector warmup is an **optimizer ablation**, not part of the frozen primary method. Keep trainable CFV-supervision **out of the main Method** because it is not integrated/tested in current RIME. For scientific insight, full-pipeline patient/site deletion analysis is an **Experiments-level diagnostic**: recompute every impacted anchor/context, not merely remove a head input. A full recomputation is a model intervention / predictive evidence audit, not an identified causal effect.

## Freeze boundaries — not allowed before submission

- No new backbone, no additional modalities, no larger PCA dimension for the main model, no new RL/GRPO/agent, no additional attention gate, no moving CFV training into main Method without complete matched evidence.
- Do not swap fixed `K`, `q`, 56D `phi`, or three-modal anchor after looking at held-out centre scores.
- Existing older MOSAIC AUC and other tasks are historical baselines only, **never** results of current RIME.
- Existing five-centre LOCO reused across exploratory studies: report as internal-external CV and avoid any independent confirmation claim.

## Minimum essential Experiments (fixed, do not expand)

1. Anchor vs full RIME. Compare same patients and held-out centre distributions; CIN2+ AUROC/AUPRC primary; CIN3+ misses/sensitivity, NLL, calibration and histopathology-only sensitivity.
2. **A1** Attention + 56D Moment, **A2** Attention + 56D Laplace, **A3** Kernel Measure + Moment, **A4** Kernel Measure + Laplace. Fix same JEV representation, patient selection, source-centre fitting, capacity, default optimizer and valid masks. **Important:** force equal exact deduplication across A1–A4; otherwise the 'measure' effect is mixed with preprocessing. Evaluate attention with/without dedup as an isolated robustness control.
3. **B2:** geometry q=0 vs q(h) vs q(h,b). Source-only tuning and as-close-as-possible equal parameter budget; report held-out paired effects and centre variation.
4. **Single essential mechanism test:** exact-duplicate additions with frozen context, near-copy/noisy-frame stress, signed sparse-response synthetic diagnostics. Separately verify conditioned-vs-unconditioned measure without claiming they are statistically independent.
5. **CFV audit:** for a fixed selected model, do full-pipeline remove-one-site and compare change in decision/loss and correlation with fitted site weights. Label as diagnostic, not a new trainable contribution.
6. Three seeds 17/29/43 and five outer LOCO centres; source-internal tuning; 95% paired patient-bootstrap CI. Preserve source/histology label provenance. No cherry-picking of successful centres or seeds.

### Integrity gates

- The earlier Legendre Moment zero-origin recurrence bug has been patched on this experimental branch. Run `pytest tests/test_rime_moment_origin.py` before ANY A1–A4 results; add protocol-change footnote where comparing with older moment numbers. The patch/test are committed but no GPU/real-patient execution is reported in this doc.
- Keep synchronized per-patient/patient-id prediction files and frozen source settings for all cells.
- Do not claim Laplace has independent benefit until A1–A4 compares validate it. If it fails, paper can present it as a hypothesis-informed readout with null/negative ablation, but cannot call it a validated contribution.
- No claim that this guarantees publication or AUROC>0.8.

## Submit-ready narrative and writing structure

**Scientific motivation:** Patient-level multimodal decisions can be dominated by repeated OCT observations while focal evidence is diluted by aggregate summary statistics.

**One scientific question:** How do we learn a non-redundant evidence measure conditioned on a patient's existing multimodal decision state, and read it out without collapsing sparse signed support?

**Technical novelty:** A mathematically explicit simplex kernel measure with patient-conditioned evidence reward, paired with zero-origin signed multiscale evidence readout and anchor-preserving risk update.

**Clinical/engineering significance:** exact-repeat numerical robustness under fixed context; cautious risk updates; five-centre internal-external evaluation and full-pipeline deletion audit.

Proposed Methods:
- 3.1 Patient-level Multimodal JEV Decision Representation
- 3.2 Patient-Conditioned Replication-Invariant Evidence Measure
- 3.3 Signed Multiscale Evidence Readout and Decision Update
- 3.4 Source-only Optimization and Inference

Proposed Experiments:
- 4.1 Dataset, Endpoints and Evaluation Protocol
- 4.2 Main Results and Centre-wise Assessment
- 4.3 A1–A4 Factorial Ablation and B2 Conditionality
- 4.4 Replication, Sparse Evidence and Site-deletion Audits
- 4.5 Clinical Safety, Calibration and Limitations

**Authoring instruction:** Write complete Methods text immediately from frozen implementation with no made-up patient metrics. Put explicit `TBD:real clinical output` only in Tables. Do not change the architecture after drafting. Remaining evidence can affect *claims/discussion*, not core architecture.
