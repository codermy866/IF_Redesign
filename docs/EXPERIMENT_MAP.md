# Experiment Map: RQ -> Method -> Evidence

| Stage / RQ | Question | Result | Role in final paper |
|---|---|---|---|
| 102 | Does OCT contain patient-specific information? | Positive pooled correspondence/shuffle signal, but not uniformly transportable | Motivation / RQ1 |
| 103 | Can discrete evidence-state routing solve harmful fusion? | NO-GO; centre-dependent utility and unstable regret | Failure analysis / ablation |
| 104 | Can absolute action-risk/value heads select fusion? | NO-GO; policy collapse to clinical-only | Motivation for relative advantage |
| 105 | Is direct patient-specific OCT benefit learnable? | GO; benefit AUC > 0.5 in 8/8 fold-seed cells, mean 0.723 | Core method evidence |
| 106 | Do scalar objective repairs independently improve the method? | NO-GO; CIs cross zero | Boundary / reviewer control |
| 107 | Is naive wrong-patient mismatch a valid correspondence test? | NO-GO for asymmetric gate reuse; full policy must be recomputed | Methodological validity control |
| 109 | Does OCT change the residual value of downstream colposcopy? | GO exploratory; positive OCT advantage and positive OCT-to-colposcopy redundancy | Discussion / mechanism extension |

## Frozen core

```text
Clinical belief
      |
      v
OCT candidate update
      |
      v
Patient-Specific Evidence Advantage
      |
      v
accept / reject OCT update
```

## Stage105 aggregate

- 4 centres x 2 seeds.
- Benefit AUC > 0.5: 8/8.
- Mean benefit AUC: 0.723.
- Positive realized gain over clinical: 7/8.
- Decision regret lower than unconditional updating: 4/8.

Interpretation: benefit prediction is supported; universal threshold dominance is not.

## Stage109 exploratory aligned-cohort result

Held-out test: N=275, CIN2+=32.

- OCT advantage: +0.02543, 95% CI [0.01037, 0.04162].
- Colposcopy advantage before OCT: +0.03726.
- Residual colposcopy advantage after OCT: +0.02415.
- OCT-induced colposcopy redundancy: +0.01311, 95% CI [0.00722, 0.01900].
- Mean reduction in colposcopy incremental advantage: 35.2%.
- Held-out OCT-benefit AUC: 0.891.
- Predicted OCT advantage vs residual colposcopy value:
  Spearman rho=-0.144, 95% CI [-0.267, -0.019].
- Predicted OCT advantage vs evidence redundancy:
  Spearman rho=+0.379, 95% CI [0.256, 0.496].

## Claim boundaries

The repository intentionally does **not** claim:

- that Evidence Advantage is a biological causal effect;
- that OCT replaces colposcopy;
- that wrong-patient correspondence is uniformly established;
- that one threshold yields a universal risk guarantee;
- that source-centre feasibility guarantees arbitrary new-centre safety.