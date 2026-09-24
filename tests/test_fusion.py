import numpy as np
import pytest
import torch

from if_redesign.fusion import fit_nonnegative_fusion, masked_mean_site_probability


def test_masked_pooling_uses_only_observed_sites():
    logits = torch.tensor([[0.0, 10.0, -10.0], [2.0, -2.0, 0.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    pooled = masked_mean_site_probability(logits, mask)
    expected = torch.tensor([
        (torch.sigmoid(torch.tensor(0.0)) + torch.sigmoid(torch.tensor(-10.0))) / 2,
        (torch.sigmoid(torch.tensor(-2.0)) + torch.sigmoid(torch.tensor(0.0))) / 2,
    ])
    assert torch.allclose(pooled, expected)
    with pytest.raises(ValueError):
        masked_mean_site_probability(logits, torch.zeros_like(mask))


def test_inner_validation_fusion_has_nonnegative_feature_weights():
    clinical = np.asarray([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    phenotype = np.asarray([0.1, 0.2, 0.8, 0.3, 0.7, 0.9])
    labels = np.asarray([0, 0, 1, 0, 1, 1])
    fitted = fit_nonnegative_fusion(np.column_stack((clinical, phenotype)), labels)
    assert min(fitted.standardized_weights) >= 0
    probabilities = fitted.predict_proba(np.column_stack((clinical, phenotype)))
    assert probabilities.shape == labels.shape
    assert np.all((probabilities > 0) & (probabilities < 1))
