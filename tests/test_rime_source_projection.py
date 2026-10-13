"""Holdout perturbations cannot change source-fitted local representations."""
import numpy as np
from if_redesign.rime.preprocessing import fit_projection,project

def test_outer_features_do_not_influence_source_pca_or_scale():
    rng=np.random.default_rng(17);x=rng.normal(size=(12,2,3,16)).astype(np.float32)
    valid=np.ones((12,2,3),bool);train=np.arange(8)
    first=fit_projection(x,valid,train)
    changed=x.copy();changed[8:]=rng.normal(size=changed[8:].shape)*10000
    second=fit_projection(changed,valid,train)
    np.testing.assert_array_equal(first['basis'],second['basis'])
    np.testing.assert_array_equal(first['scale'],second['scale'])
    gray=np.zeros_like(x)
    np.testing.assert_array_equal(project(gray,valid,first),np.zeros((12,2,3,8)))
    for k in range(8):assert first['basis'][np.argmax(abs(first['basis'][:,k])),k]>=0
