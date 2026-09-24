"""Patient-bag optimization for granularity-selective phenotype fusion."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .fusion import masked_mean_site_probability
from .pcgrad import gradient_cosine, symmetric_pcgrad


@dataclass(frozen=True)
class StepStatistics:
    patient_loss: float
    site_loss: float
    shared_gradient_cosine: float
    observed_sites: int


class G2PCGradTrainer:
    """Route local and patient losses through the locked G2 update rule."""

    def __init__(self, site_encoder, patient_head, optimizer, lambda_z=0.1, max_grad_norm=1.0):
        if lambda_z <= 0:
            raise ValueError("lambda_z must be positive")
        self.site_encoder = site_encoder
        self.patient_head = patient_head
        self.optimizer = optimizer
        self.lambda_z = float(lambda_z)
        self.max_grad_norm = float(max_grad_norm)

    def step(self, fixed_features, clinical_risk, patient_labels, site_labels, site_valid_mask):
        if fixed_features.ndim < 3:
            raise ValueError("fixed_features must contain patient and site axes")
        patients, sites = fixed_features.shape[:2]
        expected = (patients, sites)
        if site_labels.shape != expected or site_valid_mask.shape != expected:
            raise ValueError("site labels and valid mask must match patient/site axes")
        if clinical_risk.shape != (patients,) or patient_labels.shape != (patients,):
            raise ValueError("clinical risk and labels must contain one value per patient")
        valid = site_valid_mask.bool()
        if not bool(valid.any()) or bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("Each batch patient must have at least one observed OCT site")

        self.site_encoder.train(); self.patient_head.train()
        self.optimizer.zero_grad(set_to_none=True)
        flat = fixed_features.reshape(patients * sites, *fixed_features.shape[2:])
        _, flat_site_logits = self.site_encoder(flat)
        site_logits = flat_site_logits.reshape(patients, sites)
        phenotype = masked_mean_site_probability(site_logits, valid)
        patient_logits = self.patient_head(clinical_risk, phenotype)
        patient_loss = F.binary_cross_entropy_with_logits(patient_logits, patient_labels.float())
        site_loss = F.binary_cross_entropy_with_logits(site_logits[valid], site_labels.float()[valid])

        shared = list(self.site_encoder.shared_parameters())
        site_head = list(self.site_encoder.site_head_parameters())
        fusion_head = list(self.patient_head.parameters())
        site_gradients = torch.autograd.grad(site_loss, shared + site_head, retain_graph=True)
        patient_gradients = torch.autograd.grad(patient_loss, shared + fusion_head)
        n_shared = len(shared)
        gz_shared, gz_head = site_gradients[:n_shared], site_gradients[n_shared:]
        gy_shared, gy_head = patient_gradients[:n_shared], patient_gradients[n_shared:]
        scaled_site = [self.lambda_z * gradient for gradient in gz_shared]
        cosine = gradient_cosine(scaled_site, gy_shared)
        routed = symmetric_pcgrad(scaled_site, gy_shared)

        for parameter, gradient in zip(shared, routed): parameter.grad = gradient
        for parameter, gradient in zip(site_head, gz_head): parameter.grad = self.lambda_z * gradient
        for parameter, gradient in zip(fusion_head, gy_head): parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(shared + site_head + fusion_head, self.max_grad_norm)
        self.optimizer.step()
        return StepStatistics(float(patient_loss.detach()), float(site_loss.detach()),
                              float(cosine.detach()), int(valid.sum()))
