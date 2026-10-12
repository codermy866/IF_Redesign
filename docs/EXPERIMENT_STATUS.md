# Implementation status

This release adds native Qwen 27B loading, restricted language-LoRA medical
adaptation, raw-frame patient/position prompts, label-free features and six
current 32-coordinate readouts. It also preserves completed semantic/evidence
numerical reference readouts. CFV-JEV, SOPI and G2-PCGrad remain historical
implementations.

The medical adaptation campaign is ongoing. This is a task-adaptation and
mechanism-comparison baseline. Final method selection, predictive benefit and
a differentiated algorithmic contribution are not established by publishing
this code. Synthetic checks verify computation, not clinical performance.

Evaluation reuses an existing development cohort. Source-only fitting and a
frozen protocol do not make it an independent confirmation dataset.
Independent confirmation remains pending.

The public wrapper exposes source data and paths as caller inputs. It does
not include private initialization, manifests, clinical priors, model caches,
multi-fold job scheduling or study reports. Obtain pretrained bundles under
their upstream licenses. Patient records, images, fitted weights, predictions,
result tables and unpublished manuscript materials remain outside this release.

See [METHOD.md](METHOD.md) for equations and implementation boundaries.
