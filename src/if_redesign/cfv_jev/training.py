"""Source-only, center-balanced fitting of the 26-parameter residual."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .evidence import N_FEATURES, REPRESENTATIONS

DEFAULT_L2 = (.003, .03, .3)
SHRINKAGE = .25
SCALE_FLOOR = .005
FEATURE_CLIP = 16.0
NONHARM_TOLERANCE = .001


def _evidence(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != N_FEATURES or not np.isfinite(x).all():
        raise ValueError("evidence must be finite with shape (N, 26)")
    return x


def _vector(values, n, name):
    v = np.asarray(values, dtype=float)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must contain {n} finite values")
    return v


def _groups(values, n):
    g = np.asarray(values)
    if g.shape != (n,) or n == 0:
        raise ValueError("groups must contain one nonmissing source-center identifier per nonempty row")
    if any(v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))
           or (isinstance(v, str) and not v) for v in g.tolist()):
        raise ValueError("Source-center identifiers must be nonmissing")
    try:
        return [np.flatnonzero(g == c) for c in np.unique(g)]
    except (TypeError, ValueError) as exc:
        raise ValueError("Source-center identifiers must have a consistent sortable type") from exc


def center_balanced_nll(logits, labels, groups):
    """Mean of center means, giving small and large source centers equal weight."""
    z = np.asarray(logits, dtype=float)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("logits must be a finite vector")
    y = _vector(labels, len(z), "labels")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    ix = _groups(groups, len(z))
    loss = np.logaddexp(0, z) - y * z
    return float(np.mean([loss[i].mean() for i in ix]))


@dataclass(frozen=True)
class ResidualHead:
    """Bias-free correction: base + shrinkage * clip(x/std) @ weights.

    Standard deviations are fitted on source training only. Coordinates are
    never mean-centered, so zero evidence cannot create a new intercept.
    Shrinkage is used in fitting and prediction; it is not a hard logit bound.
    """
    weights: tuple[float, ...]
    std: tuple[float, ...]
    shrinkage: float = SHRINKAGE
    representation: str = "LAP"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("weights", "std"):
            v = np.asarray(getattr(self, name), dtype=float)
            if v.shape != (N_FEATURES,) or not np.isfinite(v).all():
                raise ValueError(f"{name} must contain 26 finite values")
            object.__setattr__(self, name, tuple(v.tolist()))
        if np.any(np.asarray(self.std) <= 0):
            raise ValueError("std must be positive")
        if not np.isfinite(self.shrinkage) or self.shrinkage <= 0:
            raise ValueError("shrinkage must be positive and finite")
        if self.representation not in REPRESENTATIONS:
            raise ValueError("Unknown evidence representation")
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    def predict_logit(self, base, evidence):
        x = _evidence(evidence)
        b = _vector(base, len(x), "base")
        scaled = np.clip(x / np.asarray(self.std), -FEATURE_CLIP, FEATURE_CLIP)
        z = b + self.shrinkage * (scaled @ np.asarray(self.weights))
        if not np.isfinite(z).all():
            raise ValueError("Residual prediction produced nonfinite logits")
        return z

    def predict_proba(self, base, evidence):
        return expit(self.predict_logit(base, evidence))

    def to_dict(self):
        return dict(weights=list(self.weights), std=list(self.std),
                    shrinkage=float(self.shrinkage), representation=self.representation,
                    metadata=copy.deepcopy(self.metadata))

    @classmethod
    def from_dict(cls, parameters):
        return cls(parameters["weights"], parameters["std"], parameters["shrinkage"],
                   parameters.get("representation", "LAP"), parameters.get("metadata", {}))


def _objective_and_gradient(w, x, base, y, groups, l2, risk_penalty, shrinkage):
    z = base + shrinkage * (x @ w)
    losses = np.logaddexp(0, z) - y * z
    baseline_losses = np.logaddexp(0, base) - y * base
    loss = .5 * l2 * (w @ w)
    grad = l2 * w.copy()
    error = expit(z) - y
    for ix in groups:
        loss += losses[ix].mean() / len(groups)
        multiplier = 1.0
        excess = losses[ix].mean() - baseline_losses[ix].mean() - NONHARM_TOLERANCE
        if risk_penalty and excess > 0:
            loss += excess / len(groups)
            multiplier += 1.0
        grad += multiplier * shrinkage * (x[ix].T @ error[ix]) / (len(ix) * len(groups))
    return float(loss), grad


def fit_residual(
    train_evidence, train_base, train_labels, train_groups,
    selection_evidence, selection_base, selection_labels, selection_groups,
    *, risk_penalty=True, shrinkage=SHRINKAGE, l2_candidates=DEFAULT_L2,
    representation="LAP",
):
    """Fit on source training and select L2 on a disjoint source-selection set.

    This interface accepts no outer evaluation or confirmation set. Callers
    must enforce patient-level separation and source-only provenance before
    passing arrays. CFV-JEV uses LAP and risk_penalty=True; LOOV_LAP disables
    only this soft penalty. All representation controls have 26 parameters.
    """
    x_train, x_selection = _evidence(train_evidence), _evidence(selection_evidence)
    b_train = _vector(train_base, len(x_train), "train_base")
    b_selection = _vector(selection_base, len(x_selection), "selection_base")
    y_train = _vector(train_labels, len(x_train), "train_labels")
    y_selection = _vector(selection_labels, len(x_selection), "selection_labels")
    if not np.isin(y_train, [0, 1]).all() or not np.isin(y_selection, [0, 1]).all():
        raise ValueError("Training and selection labels must be binary 0/1")
    groups = _groups(train_groups, len(x_train))
    _groups(selection_groups, len(x_selection))
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown evidence representation")
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("shrinkage must be positive and finite")
    menu = tuple(float(lam) for lam in l2_candidates)
    if not menu or not all(np.isfinite(lam) and lam > 0 for lam in menu):
        raise ValueError("l2_candidates must be a nonempty list of positive finite values")
    std = np.maximum(x_train.std(axis=0), SCALE_FLOOR)
    if not np.isfinite(std).all():
        raise ValueError("Source training standard deviations must be finite")
    scaled = np.clip(x_train / std, -FEATURE_CLIP, FEATURE_CLIP)
    candidates = []
    for lam in menu:
        result = minimize(
            _objective_and_gradient, np.zeros(N_FEATURES),
            args=(scaled, b_train, y_train, groups, lam, bool(risk_penalty), shrinkage),
            jac=True, method="L-BFGS-B",
            options=dict(maxiter=2000, maxls=100, ftol=1e-12, gtol=1e-7),
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all():
            raise RuntimeError("Nonfinite residual fit")
        max_gradient = float(np.abs(result.jac).max())
        if not result.success and max_gradient > 1e-3:
            raise RuntimeError("Residual did not converge: " + str(result.message))
        metadata = dict(l2=lam, risk_penalty=bool(risk_penalty), parameters=N_FEATURES,
                        optimizer_success=bool(result.success), optimizer_message=str(result.message),
                        max_abs_gradient=max_gradient, iterations=int(result.nit),
                        final_objective=float(result.fun))
        head = ResidualHead(result.x, std, float(shrinkage), representation, metadata)
        nll = center_balanced_nll(head.predict_logit(b_selection, x_selection), y_selection, selection_groups)
        candidates.append((nll, head))
    nll, best = min(candidates, key=lambda entry: entry[0])
    metadata = dict(best.metadata, selection_center_NLL=nll,
                    selection_menu=[dict(l2=head.metadata["l2"], NLL=value) for value, head in candidates])
    return ResidualHead(best.weights, best.std, best.shrinkage, best.representation, metadata)
