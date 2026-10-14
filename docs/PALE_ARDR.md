# PALE-JEV matched loss implementation

The forward model remains `RIMEHead(pooling="mlp", readout="laplace")`.
Frozen frame contrasts pass through eight source-fitted coordinates, a shared
residual MLP, signed multiscale responses, a masked frame mean, and site
attention. A linear residual updates a caller-provided frozen patient anchor.
The MLP branch does not use the kernel simplex optimizer and has no
replication-invariance guarantee. Kernel candidates remain available separately.

`if_redesign.pale_ardr` exposes six loss controls:

| Name | Training change |
| --- | --- |
| `A0_PALE` | Centre-equal BCE, within-centre ranking (0.1), L2 |
| `A1_STRONG_L2` | Increase only the L2 multiplier |
| `A2_ANCHOR_KL` | Add mean Bernoulli KL from detached anchor to prediction |
| `A3_GROUP_DRO` | Replace the BCE mean by exponential source-group risk weighting |
| `A4_GLOBAL_RISK` | Penalize squared positive patient-mean NLL excess over anchor |
| `A5_ARDR` | Penalize squared positive NLL excess separately in each source centre |

For ARDR the extra loss is `gamma * mean_c relu(R_c(z)-R_c(anchor)-epsilon)^2`.
The anchor has no gradient. This is a soft constraint on empirical source risk,
not a test-centre risk, AUROC, causal-value or clinical-safety guarantee.
Group DRO follows the exponentiated group-weight update in the
[authors' reference implementation](https://github.com/kohpangwei/group_DRO/blob/master/loss.py),
using a log-space softmax for numerical stability.

The portable entry points are `fit`, `penalty`, `objective`, `anchor_kl`,
`centre_risks`, `dro_update`, and `repair_project`. Training and validation
arguments contain `h`, `valid`, `k`, `b`, `anchor`, and `coordinate_scale`,
matching the existing head. Labels and centres are supplied separately;
training labels must correspond only to training rows. Early stopping uses
held-source NLL, then Brier, AUROC and earlier epoch. Defaults are 300 epochs,
patience 30, Adam learning rate 0.003 and seeds supplied by the caller.
Fixed-epoch refitting does not use validation outcomes.

`repair_project(raw, valid, state)` consumes caller-fitted `basis`, `scale`
and `fit_centering`; it never fits on the rows being transformed. It first
centres projected contrasts, clips standardized coordinates at +/-3, subtracts
the identically transformed zero-contrast gray reference, and divides by six.
Invalid frames map to zero. Gray contrast remains exactly zero. This common
repair must be applied to every compared arm and its source-fitted readout
scales. Historical uncentred models remain separately reproducible; their
scores are not matched controls for repaired, NLL-selected models.

Run data-free mathematical, gradient and training checks with:

```bash
python -m pytest -q tests/test_pale_ardr.py
```

ARDR is retained to reproduce the validation comparison, not selected as the
default main method. This release contains implementation and tests; clinical
outcomes, fitted weights, private cohorts and manuscript files are not bundled.
