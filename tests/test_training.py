import torch
from torch import nn

from if_redesign.modeling import PatientFusionHead
from if_redesign.training import G2PCGradTrainer


class ToySiteEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(3, 2)
        self.site_head = nn.Linear(2, 1)

    def forward(self, values):
        hidden = torch.tanh(self.shared(values))
        return hidden, self.site_head(hidden).flatten()

    def shared_parameters(self):
        return list(self.shared.parameters())

    def site_head_parameters(self):
        return list(self.site_head.parameters())


def test_g2_step_updates_routed_parameters_with_masked_sites():
    torch.manual_seed(7)
    encoder = ToySiteEncoder()
    patient_head = PatientFusionHead()
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": 1e-2},
        {"params": patient_head.parameters(), "lr": 1e-2},
    ])
    trainer = G2PCGradTrainer(encoder, patient_head, optimizer, lambda_z=0.1)
    before = [p.detach().clone() for p in list(encoder.parameters()) + list(patient_head.parameters())]
    stats = trainer.step(
        fixed_features=torch.randn(3, 4, 3),
        clinical_risk=torch.tensor([-1.0, 0.0, 1.0]),
        patient_labels=torch.tensor([0.0, 1.0, 1.0]),
        site_labels=torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]]),
        site_valid_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
    )
    after = list(encoder.parameters()) + list(patient_head.parameters())
    assert all(not torch.equal(old, new.detach()) for old, new in zip(before, after))
    assert stats.observed_sites == 9
    assert torch.isfinite(torch.tensor(stats.shared_gradient_cosine))
