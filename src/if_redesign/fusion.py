"""Masked OCT pooling and source-fitted clinical/phenotype fusion."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit


def masked_mean_site_probability(site_logits, valid_mask):
    """Average predicted site probabilities over naturally observed sites."""
    if site_logits.shape != valid_mask.shape:
        raise ValueError("site_logits and valid_mask must have the same shape")
    mask = valid_mask.to(dtype=site_logits.dtype)
    count = mask.sum(dim=1)
    if bool((count == 0).any()):
        raise ValueError("Every patient must have at least one observed OCT site")
    return (torch.sigmoid(site_logits) * mask).sum(dim=1) / count


@dataclass(frozen=True)
class FittedFusion:
    feature_mean: tuple[float, float]
    feature_scale: tuple[float, float]
    standardized_weights: tuple[float, float]
    standardized_intercept: float

    @property
    def effective_weights(self):
        return tuple(w / s for w, s in zip(self.standardized_weights, self.feature_scale))

    @property
    def effective_intercept(self):
        return float(self.standardized_intercept - sum(
            w * m / s for w, m, s in zip(
                self.standardized_weights, self.feature_mean, self.feature_scale)))

    def predict_proba(self, features):
        values = np.asarray(features, dtype=float)
        standardized = (values - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale)
        return expit(standardized @ np.asarray(self.standardized_weights) + self.standardized_intercept)

    def to_dict(self):
        return {**asdict(self), "effective_weights": self.effective_weights,
                "effective_intercept": self.effective_intercept}


def fit_nonnegative_fusion(validation_features, validation_labels):
    """Fit the locked L2-regularized fusion using inner validation only."""
    features = np.asarray(validation_features, dtype=float)
    labels = np.asarray(validation_labels, dtype=float)
    if features.ndim != 2 or features.shape[1] != 2 or len(features) != len(labels):
        raise ValueError("Expected two features and one binary label per patient")
    if len(labels) == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("validation_labels must be non-empty and binary")
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    standardized = (features - mean) / scale
    prevalence = np.clip(labels.mean(), 1e-6, 1.0 - 1e-6)
    penalty = 1.0 / len(labels)

    def objective(theta):
        eta = standardized @ theta[:2] + theta[2]
        nll = np.mean(np.logaddexp(0.0, eta) - labels * eta)
        return float(nll + 0.5 * penalty * np.sum(theta[:2] ** 2))

    fitted = minimize(
        objective,
        np.asarray([0.5, 0.5, np.log(prevalence / (1.0 - prevalence))]),
        method="L-BFGS-B",
        bounds=((0.0, 20.0), (0.0, 20.0), (-20.0, 20.0)),
    )
    if not fitted.success:
        raise RuntimeError(f"Constrained fusion failed: {fitted.message}")
    return FittedFusion(
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in fitted.x[:2]),
        float(fitted.x[2]),
    )
