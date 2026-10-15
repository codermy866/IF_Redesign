"""Data-free smoke example of the actual P4 decision path, not clinical results."""
import torch
from if_redesign.pale_matched import make_p4_head


def main():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    head = make_p4_head().double()
    h = torch.randn(3, 12, 10, 8, dtype=torch.double) * 0.2
    valid = torch.ones(3, 12, 10, dtype=torch.bool)
    valid[-1] = False
    anchor = torch.tensor([-0.5, 0.4, 0.8], dtype=torch.double)
    args = dict(h=h, valid=valid, anchor=anchor,
                coordinate_scale=torch.ones(56, dtype=torch.double),
                b=torch.empty(3, 0, dtype=torch.double), k=None)
    assert torch.equal(head(**args), anchor)  # zero-initialized residual
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        head(**args), torch.tensor([0., 1., 0.], dtype=torch.double))
    loss.backward()
    optimizer = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=.003)
    optimizer.step()
    with torch.no_grad():
        prediction = head(**args)
    assert prediction[-1] == anchor[-1]
    print('P4 trainable head parameters:', sum(p.numel() for p in head.parameters() if p.requires_grad))
    print('Synthetic logits:', prediction.tolist())
    print('Missing-OCT anchor fallback: PASS')


if __name__ == '__main__':
    main()
