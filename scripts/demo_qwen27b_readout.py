"""CPU synthetic example of all six readouts; no patient data or checkpoints."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

from if_redesign.qwen27b import readout


def synthetic(rng, n):
    hidden = rng.normal(size=(n, 64))
    clinical = rng.normal(size=n)
    logits = clinical + .8 * hidden[:, 0]
    y = (rng.random(n) < 1/(1 + np.exp(-logits))).astype(int)
    data = dict(hidden=hidden, site=rng.normal(size=(n, 12, 2)),
                site_mask=np.ones((n, 12), dtype=bool), A=rng.normal(size=n),
                K=rng.normal(size=n), any_view=np.ones(n, dtype=bool),
                clinical=clinical)
    return data, y, np.tile(np.arange(3), n//3)


def main():
    rng = np.random.default_rng(195)
    train, y, centers = synthetic(rng, 120)
    selection, sy, sc = synthetic(rng, 60)
    evaluation, ey, ec = synthetic(rng, 60)
    state, heads, scores = readout.train_select(train, y, centers, selection, sy, sc)
    chosen = max(scores, key=lambda name: (scores[name]['AUROC'], -scores[name]['NLL']))
    result = readout.metrics(ey, readout.predict(heads[chosen], evaluation, state))
    zero = {name: np.zeros_like(value) for name, value in evaluation.items()}
    zero['clinical'] = evaluation['clinical'].copy()
    assert np.array_equal(readout.predict(heads[chosen], zero, state), zero['clinical'])
    print('SYNTHETIC DEMONSTRATION ONLY; not study performance')
    print('Source-selected family:', chosen)
    print('Held-out synthetic metrics:', result)
    print('No-evidence clinical fallback: exact')


if __name__ == '__main__':
    main()
