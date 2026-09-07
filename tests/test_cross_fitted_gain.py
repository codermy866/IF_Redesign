import unittest
import numpy as np
from run_cross_fitted_gain import inner_folds,calibration_split,ce


class CrossFitting(unittest.TestCase):
    def test_disjoint(self):
        indices=np.arange(103);centers=np.array(['a']*50+['b']*53);y=np.arange(103)%2
        folds=inner_folds(indices,centers,y,42)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)),indices)
        for held in folds:
            fit,cal=calibration_split(np.setdiff1d(indices,held),centers,y,42)
            self.assertFalse(set(held)&set(fit));self.assertFalse(set(held)&set(cal));self.assertFalse(set(fit)&set(cal))
        again=inner_folds(indices,centers,y,42)
        for a,b in zip(folds,again):np.testing.assert_array_equal(a,b)

    def test_signed_gain(self):
        self.assertGreater(ce(1,.3)-ce(1,.8),0)
        self.assertLess(ce(1,.8)-ce(1,.3),0)
        self.assertAlmostEqual(float(ce(0,.4)-ce(0,.4)),0)

    def test_information_error_decomposition(self):
        weights=np.array([.4,.6]);pj=np.array([.2,.8]);p0=float(weights@pj)
        f0=.35;fj=np.array([.5,.6])
        def entropy(p):return -p*np.log(p)-(1-p)*np.log1p(-p)
        def divergence(p,q):return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
        lhs=p0*ce(1,f0)+(1-p0)*ce(0,f0)-weights@(pj*ce(1,fj)+(1-pj)*ce(0,fj))
        rhs=entropy(p0)-weights@entropy(pj)+divergence(p0,f0)-weights@divergence(pj,fj)
        self.assertAlmostEqual(float(lhs),float(rhs),places=12)


if __name__=='__main__':unittest.main()
