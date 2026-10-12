"""Label-free feature extraction with two disjoint frame bags per OCT position."""
import numpy as np

from .adapter import images


def extract_patient_features(model, item):
    """Extract inference features; ``item`` contains image paths, never labels.

    The patient/position tasks have separate gray-image references. Each
    position response is a model contrast, not a pathological localization.
    """
    null = model.null('diagnosis', hidden=True)
    weak_null = model.null('reread')
    inp = model.prepared(images(item, phase=4), 'diagnosis')
    full = model.forward(*inp, hidden=True)
    z = full['logits'].cpu().numpy()
    q = float(z[1] - z[0])
    n = float(null['logits'][1] - null['logits'][0])
    response = np.zeros((12, 2))
    mask = np.array([bool(item['oct_sites'].get(str(site))) for site in range(1, 13)])
    wn = float(weak_null['logits'][1] - weak_null['logits'][0])
    for site in range(1, 13):
        if not mask[site - 1]:
            continue
        for phase in (0, 1):
            inp = model.prepared(images(item, 'reread', site, phase), 'reread')
            v = model.forward(*inp)['logits'].cpu().numpy()
            response[site - 1, phase] = float(v[1] - v[0]) - wn
    inp = model.prepared(images(item, colpo_only=True), 'diagnosis')
    kv = model.forward(*inp)['logits'].cpu().numpy()
    return dict(
        hidden=full['hidden'].cpu().numpy() - null['hidden'],
        site=response, site_mask=mask,
        A=np.array(q - n), K=np.array(float(kv[1] - kv[0]) - n),
        any_view=np.array(bool(mask.any() or item['colpo_paths'])),
        q=np.array(q), q_null=np.array(n),
    )


def stack_features(rows, clinical_logits):
    """Attach an externally source-fitted clinical prior; no outcome input."""
    clinical_logits = np.asarray(clinical_logits, dtype=float)
    if not rows or clinical_logits.shape != (len(rows),):
        raise ValueError('One clinical logit is required per feature row')
    return dict(
        **{name: np.stack([row[name] for row in rows]) for name in
           ('hidden', 'site', 'site_mask', 'A', 'K', 'any_view')},
        clinical=clinical_logits,
    )
