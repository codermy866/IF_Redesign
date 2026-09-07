import unittest
import numpy as np
import torch
from run_missing_modality_fusion import Branch, Fusion, PATTERNS, threshold90, calibration_split


class Tests(unittest.TestCase):
    def test_mask_invariance_and_no_evidence_refusal(self):
        torch.manual_seed(1)
        model = Fusion([Branch(i) for i in range(3)]).eval()
        x = torch.randn(2, 37, 2048)
        c = torch.randn(2, 14)
        av = torch.ones(2, 37, dtype=torch.bool)
        p = torch.tensor([[True, False, True]]).expand(2, -1)
        changed = x.clone()
        changed[:, 1:25] = 100*torch.randn_like(changed[:, 1:25])
        torch.testing.assert_close(model(x, c, av, p), model(changed, c, av, p), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(x, c, av, torch.zeros_like(p))

    def test_empty_visual_branch_finite_and_masks(self):
        b = Branch(2)
        x = torch.randn(2, 37, 2048)
        av = torch.zeros(2, 37, dtype=torch.bool)
        h, z = b(x, torch.randn(2, 14), av)
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue((h == 0).all())
        self.assertEqual(len(PATTERNS), 7)
        self.assertTrue(PATTERNS.any(1).all())

    def test_split_and_sensitivity(self):
        idx = np.arange(80)
        centers = np.array(['a']*40+['b']*40)
        y = np.tile([0, 1], 40)
        fit, cal = calibration_split(idx, centers, y, 5)
        self.assertFalse(set(fit)&set(cal))
        self.assertEqual(set(fit)|set(cal), set(idx))
        p = np.linspace(.01, .99, len(y))
        self.assertGreaterEqual(np.mean(p[y == 1] >= threshold90(y, p)), .9)

    def test_weighted_bootstrap_ap_with_ties(self):
        from summarize_missing_modality_fusion import weighted_ap_draws
        from sklearn.metrics import average_precision_score
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([.7, .7, .3, .2, .1])
        weights = np.array([[1, 2, 0, 3, 1], [0, 1, 3, 0, 1]])
        expected = [average_precision_score(y, p, sample_weight=w) for w in weights]
        np.testing.assert_allclose(weighted_ap_draws(y, p, weights), expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
