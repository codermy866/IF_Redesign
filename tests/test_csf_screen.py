"""Independent algebra, hidden-state and AP implementation checks."""
import unittest
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from run_csf_screen import decomposition, frontier, rollout
from run_csf_memory import features
from summarize_csf_screen import ap


class ToyPredictor(torch.nn.Module):
    def forward(self, x, c, mask):
        values = x.float().sum(-1)*mask
        score = values.sum(1)/mask.sum(1).clamp_min(1)+c.sum(1)
        attention = values.masked_fill(~mask, -1e9).softmax(1)
        return {'logit': score, 'attention': attention}


class Checks(unittest.TestCase):
    def test_identity(self):
        torch.manual_seed(7)
        p = .01+.98*torch.rand(100, 8, dtype=torch.float64)
        total, centered, gap = decomposition(p, torch.tensor(.3, dtype=torch.float64))
        self.assertTrue(torch.allclose(total, centered+gap, atol=1e-12))

    def test_ap_ties(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            y = rng.integers(0, 2, 100); p = np.round(rng.random(100), 1)
            self.assertAlmostEqual(ap(y, p), average_precision_score(y, p), places=12)

    def test_hidden_state(self):
        torch.manual_seed(4)
        x = torch.randn(1, 37, 3)*.1; changed = x.clone(); changed[:, 25:] += 20
        c = torch.zeros(1, 2); available = torch.ones(1, 37, dtype=torch.bool)
        donors = torch.randn(20, 37, 3)*.1; dc = torch.randn(20, 2)*.1
        centers = np.array(['A']*10+['B']*10)
        for mode in ('total_kl', 'centered', 'coherence', 'stable', 'cross_center', 'random', 'attention'):
            def run(xx):
                return rollout(ToyPredictor(), xx, c, available, donors, donors*.9, donors*1.1, dc, centers, 'A', mode, 9, x*.9, x*1.1)
            a, b = run(x), run(changed)
            self.assertEqual(a[2][0], b[2][0])
            self.assertEqual(a[0][0], b[0][0])
            self.assertEqual(len(set(r['slot'] for r in a[2])), 4)
        mask = available.clone(); mask[:, 25:] = False
        projection = torch.randn(3, 16); p = torch.tensor([.4]); slots = torch.arange(25, 37)
        a = features(x, c, mask, p, slots, projection)
        b = features(changed, c, mask, p, slots, projection)
        np.testing.assert_array_equal(a, b)

    def test_identity_regret(self):
        y = np.array([0, 1, 0, 1]); p = np.array([[.2, .1], [.8, .9], [.3, .2], [.7, .8]])
        rows = frontier(y, p, np.repeat(p[:, :, None], 3, axis=2), np.array(['A', 'A', 'B', 'B']))
        self.assertTrue(all(r['regret'] == 0 for r in rows))


if __name__ == '__main__': unittest.main()
