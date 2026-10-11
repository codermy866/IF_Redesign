# Implementation and evaluation status

The current public implementation is CFV-JEV. It contains the view-replacement
interface, signed empirical evidence encodings, clinical-anchored residual
fitting, and synthetic validation examples. It does not contain patient data,
trained checkpoints, private predictions, or study result tables.

The full-image frozen-protocol evaluation is ongoing. Its remaining work
includes matched residual and representation ablations, modality-content and
training-budget controls, and held-out statistical comparisons. Publishing
the implementation does not establish improved predictive performance.

Current evaluation uses an existing development cohort. A frozen protocol
does not make that cohort an independent confirmation dataset. Independent
confirmation remains pending.

Earlier SOPI, prior-anchoring, and G2-PCGrad modules remain available for
historical reference. Earlier experimental descriptions are not evidence
that the latest CFV-JEV full-image evaluation is complete.

For the reusable equations and implementation boundaries, see
[METHOD.md](METHOD.md). Patient identifiers, clinical records, images, model
weights, private experiment artifacts, and manuscript material are excluded
from this release.
