"""Source-only fitting of the fixed, neutral-referenced RAW patient readout."""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from .training import _groups, _vector, center_balanced_nll
from .visual_jev import RawPatientReadout, raw_descriptors

RAW_L2 = (.01, .1)
RAW_RHO = (.5, 1.)
RAW_TEMPERATURE = (1., 1.5)
RAW_INITIALIZATION_SEED = 15217
RAW_PARAMETERS = 140


def _option_logits(values, *, n=None, neutral=False):
    logits = np.asarray(values, dtype=float)
    if neutral and logits.shape == (21, 2):
        logits = logits[None]
    if logits.ndim != 3 or logits.shape[1:] != (21, 2) or not np.isfinite(logits).all():
        raise ValueError("option logits must be finite with shape (N, 21, 2)")
    if n is not None:
        if neutral and len(logits) == 1:
            logits = np.broadcast_to(logits, (n, 21, 2))
        elif len(logits) != n:
            raise ValueError("Neutral logits must have one shared row or one row per patient")
    return logits


def _features(logits):
    differences = logits[:, :, 1] - logits[:, :, 0]
    features = np.column_stack([raw_descriptors(expit(differences[:, 1:])), differences[:, 0]])
    if not np.isfinite(features).all():
        raise ValueError("RAW features must be finite")
    return features


def _raw_objective_and_gradient(parameters, features, neutral_features, clinical, labels, groups, l2):
    """Exact float64 GELU, clinical gating, neutral contrast, and autograd."""
    import torch
    from torch.nn import functional as F

    weights = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
    x = torch.as_tensor(features, dtype=torch.float64)
    xn = torch.as_tensor(neutral_features, dtype=torch.float64)
    c = torch.as_tensor(clinical, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)

    def readout(values):
        hidden = F.gelu(values @ weights[:132].reshape(33, 4), approximate="none")
        return torch.cat([hidden, hidden * torch.tanh(c[:, None] / 3)], dim=1) @ weights[132:]

    z = c + readout(x) - readout(xn)
    losses = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    loss = torch.stack([losses[ix].mean() for ix in groups]).mean() + .5 * l2 * weights.square().sum()
    loss.backward()
    return float(loss.detach()), weights.grad.detach().cpu().numpy()


@dataclass(frozen=True)
class FittedRawReadout:
    """Selected RAW predictor plus optimizer and source-selection metadata."""
    readout: RawPatientReadout
    metadata: dict

    def __post_init__(self):
        if not isinstance(self.readout, RawPatientReadout):
            raise TypeError("readout must be a RawPatientReadout")
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    def to_dict(self):
        return dict(readout=self.readout.to_dict(), metadata=copy.deepcopy(self.metadata))

    @classmethod
    def from_dict(cls, parameters):
        return cls(RawPatientReadout.from_dict(parameters["readout"]), parameters["metadata"])


def _positive_menu(values, name):
    menu = tuple(float(value) for value in values)
    if not menu or not all(np.isfinite(value) and value > 0 for value in menu):
        raise ValueError(f"{name} must contain positive finite values")
    return menu


def fit_raw_readout(
    train_logits, train_neutral_logits, train_clinical, train_labels, train_groups,
    selection_logits, selection_neutral_logits, selection_clinical, selection_labels, selection_groups,
    *, l2_candidates=RAW_L2, rho_candidates=RAW_RHO, temperature_candidates=RAW_TEMPERATURE,
):
    """Fit REF_ERM_RAW on source training and select its scales on source selection.

    Real logits have shape (N,21,2). Neutral logits may have shape (21,2),
    (1,21,2), or (N,21,2); sharing is valid only for identical neutral inputs.
    Only the real source-training features fit the 33-coordinate scaler.

    Training uses c+r(X,c)-r(neutral,c), not the unanchored c+r(X,c).
    Selection uses c+(rho/T)*(r(X,c)-r(neutral,c)). The returned readout
    yields raw r; extract_interventions performs the null subtraction and
    clinical anchoring once using readout.effective_scale. Patient-level
    source-training/selection separation must be enforced by the caller.
    No evaluation dataset, outcome labels for inference, or model weights
    are loaded by this function.
    """
    real_train = _option_logits(train_logits)
    real_selection = _option_logits(selection_logits)
    null_train = _option_logits(train_neutral_logits, n=len(real_train), neutral=True)
    null_selection = _option_logits(selection_neutral_logits, n=len(real_selection), neutral=True)
    c_train = _vector(train_clinical, len(real_train), "train_clinical")
    c_selection = _vector(selection_clinical, len(real_selection), "selection_clinical")
    y_train = _vector(train_labels, len(real_train), "train_labels")
    y_selection = _vector(selection_labels, len(real_selection), "selection_labels")
    if not np.isin(y_train, [0, 1]).all() or not np.isin(y_selection, [0, 1]).all():
        raise ValueError("Training and selection labels must be binary 0/1")
    groups = _groups(train_groups, len(real_train))
    _groups(selection_groups, len(real_selection))
    lambdas = _positive_menu(l2_candidates, "l2_candidates")
    rhos = _positive_menu(rho_candidates, "rho_candidates")
    temperatures = _positive_menu(temperature_candidates, "temperature_candidates")

    features = _features(real_train)
    scaler = StandardScaler().fit(features)
    x, xn = scaler.transform(features), scaler.transform(_features(null_train))
    initial = np.random.default_rng(RAW_INITIALIZATION_SEED).normal(0, .01, RAW_PARAMETERS)
    fits, candidates = [], []
    for l2 in lambdas:
        result = minimize(
            _raw_objective_and_gradient, initial.copy(),
            args=(x, xn, c_train, y_train, groups, l2), jac=True, method="L-BFGS-B",
            options=dict(maxiter=1000, maxls=100, ftol=1e-11, gtol=1e-7),
        )
        if not result.success:
            raise RuntimeError("RAW readout did not converge: " + str(result.message))
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all():
            raise RuntimeError("Nonfinite RAW readout fit")
        fit_metadata = dict(
            l2=l2, optimizer_success=True, optimizer_message=str(result.message),
            iterations=int(result.nit), final_objective=float(result.fun),
            max_abs_gradient=float(np.abs(result.jac).max()),
        )
        fits.append(fit_metadata)
        unscaled = RawPatientReadout(result.x, scaler.mean_, scaler.scale_)
        r = unscaled.score_value(real_selection, c_selection)
        rn = unscaled.score_value(null_selection, c_selection)
        for rho in rhos:
            for temperature in temperatures:
                z = c_selection + rho * (r - rn) / temperature
                nll = center_balanced_nll(z, y_selection, selection_groups)
                head = RawPatientReadout(result.x, scaler.mean_, scaler.scale_, rho, temperature)
                candidates.append((nll, head, dict(l2=l2, rho=rho, temperature=temperature, NLL=nll)))
    nll, selected, selected_item = min(candidates, key=lambda item: item[0])
    metadata = dict(
        source_only=True, training_anchor="clinical_plus_real_minus_neutral",
        parameters=RAW_PARAMETERS, feature_dimensions=33,
        initialization_seed=RAW_INITIALIZATION_SEED,
        l2_candidates=list(lambdas), rho_candidates=list(rhos), temperature_candidates=list(temperatures),
        selected=selected_item, selection_center_NLL=nll,
        selection_menu=[item for _, _, item in candidates], optimizer_fits=fits,
    )
    return FittedRawReadout(selected, metadata)
