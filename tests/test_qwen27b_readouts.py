import numpy as np
import pytest

from if_redesign.qwen27b import readout, semantic_readout, evidence_readout


def data(n=90):
    rng = np.random.default_rng(195)
    groups = np.repeat(['a', 'b', 'c'], n//3)
    y = np.tile(np.repeat([0, 1], n//6), 3)
    d = dict(hidden=rng.normal(size=(n, 64)), site=rng.normal(size=(n, 12, 2)),
             site_mask=np.ones((n, 12), bool), A=rng.normal(size=n),
             K=rng.normal(size=n), any_view=np.ones(n, bool),
             clinical=rng.normal(size=n), delta=rng.normal(size=(n, 20)),
             mask=np.ones((n, 20), bool), I=rng.normal(size=n))
    return d, y, groups


def numerical_gradient(fun, w):
    numeric = []
    for j in range(len(w)):
        step = np.zeros_like(w)
        step[j] = 1e-6
        numeric.append((fun(w + step)[0] - fun(w - step)[0])/2e-6)
    return np.asarray(numeric)


@pytest.mark.parametrize('all_pairs', [False, True])
def test_adaptation_pair_loss_gradient_and_center_blocks(all_pairs):
    d, y, groups = data()
    state = readout.fit_state(d)
    x = readout.features(d, state, 'FOCAL')
    design = readout.pairs(y, groups, all_pairs)
    assert len(design) == (9 if all_pairs else 3)
    w = np.random.default_rng(1).normal(0, .05, 32)
    fun = lambda v: readout.objective(v, x, d['clinical'], y, groups, design, .1)
    np.testing.assert_allclose(fun(w)[1], numerical_gradient(fun, w), atol=1e-6)


def test_focal_geometry_missing_evidence_and_source_state():
    d, y, groups = data()
    state = readout.fit_state(d)
    before = {k: np.asarray(v).copy() for k, v in state.items()}
    perm = np.roll(np.arange(12), 3)
    permuted = {**d, 'site': d['site'][:, perm], 'site_mask': d['site_mask'][:, perm]}
    missing = {**d, 'any_view': np.zeros_like(d['any_view'])}
    identical = {**d, 'site': np.repeat(d['site'][:, :, :1], 2, axis=2)}
    np.testing.assert_allclose(readout.site_geometry(identical, state, 'FOCAL'),
                               readout.site_geometry(identical, state, 'RELIABLE_FOCAL'))
    for geometry in readout.GEOMETRIES:
        np.testing.assert_allclose(readout.site_geometry(d, state, geometry),
                                   readout.site_geometry(permuted, state, geometry), atol=1e-12)
        for mode in ('WITHIN', 'ALL_PAIRS'):
            head = readout.fit_head(d, y, groups, state, geometry+'__'+mode, .1)
            assert np.array_equal(readout.predict(head, missing, state), d['clinical'])
    assert all(np.array_equal(value, state[key]) for key, value in before.items())


@pytest.mark.parametrize('robust', [False, True])
def test_completed_semantic_readout_gradient(robust):
    d, y, groups = data()
    state = semantic_readout.fit_state(d, groups)
    x = semantic_readout.features(d, state, 'CENTERED32')
    design = semantic_readout.pairs(y, groups)
    w = np.random.default_rng(2).normal(0, .05, 32)
    fun = lambda v: semantic_readout.objective(v, x, d['clinical'], y, design, .1, robust)
    np.testing.assert_allclose(fun(w)[1], numerical_gradient(fun, w), atol=1e-6)
    head = semantic_readout.fit_head(x, d['clinical'], y, groups, .1, robust)
    head['family'] = 'CENTERED32__'+('ROBUST' if robust else 'MEAN')
    missing = {**d, 'mask': np.zeros_like(d['mask'])}
    assert np.array_equal(semantic_readout.predict(head, missing, state), d['clinical'])


def test_pca_linear_reference_is_unbounded_and_zero_anchored():
    d, y, groups = data()
    head = evidence_readout.fit_control(d, y, groups, 'PCA_LINEAR', .1)
    assert head['parameters'] == 18 and not head['bounded']
    zero = {**d, 'hidden': np.zeros_like(d['hidden'])}
    assert np.array_equal(evidence_readout.predict_control(head, zero), d['clinical'])


def test_selection_fits_source_basis_once_without_heldout_refit():
    d, y, groups = data()
    tr, se = np.arange(60), np.arange(60, 90)
    state, heads, scores = readout.train_select(
        readout.subset(d, tr), y[tr], groups[tr],
        readout.subset(d, se), y[se], groups[se])
    assert len(heads) == 6 and set(heads) == set(scores)
    before = {key: np.asarray(value).copy() for key, value in state.items()}
    heldout = readout.subset(d, se)
    heldout['hidden'] *= 1e6
    for head in heads.values():
        readout.predict(head, heldout, state)
    assert all(np.array_equal(value, state[key]) for key, value in before.items())
