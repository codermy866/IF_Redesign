"""OCT phenotype encoder and patient-level fusion components."""
from __future__ import annotations

import torch
from torch import nn


class ConvNeXtFeatureTail(nn.Module):
    """Trainable final ConvNeXt-Tiny block operating on cached prefix maps."""

    def __init__(self, adapted: nn.Module, pool: nn.Module, norm: nn.Module, channels: int = 768):
        super().__init__()
        self.adapted = adapted
        self.pool = pool
        self.norm = norm
        self.site_head = nn.Linear(int(channels), 1)

    @classmethod
    def from_torchvision(cls, weights="DEFAULT"):
        """Build the locked tail after the frozen prefix and two final-stage blocks."""
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        selected = ConvNeXt_Tiny_Weights.DEFAULT if weights == "DEFAULT" else weights
        pretrained = convnext_tiny(weights=selected)
        return cls(pretrained.features[7][2], pretrained.avgpool,
                   pretrained.classifier[0], channels=768)

    def forward(self, fixed_features):
        hidden = self.norm(self.pool(self.adapted(fixed_features))).flatten(1)
        return hidden, self.site_head(hidden).flatten()

    def shared_parameters(self):
        return list(self.adapted.parameters()) + list(self.norm.parameters())

    def site_head_parameters(self):
        return list(self.site_head.parameters())


class PatientFusionHead(nn.Module):
    """Training-time head coupling frozen clinical risk and OCT phenotype."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.5)
            self.linear.bias.zero_()

    def forward(self, clinical_risk, phenotype_score):
        return self.linear(torch.stack((clinical_risk, phenotype_score), dim=1)).flatten()
