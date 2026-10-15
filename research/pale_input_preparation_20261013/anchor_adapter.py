"""Same frozen anchor, with the previously audited local/global index correction."""
import numpy as np
import joblib
from rime_dataset import old_run,old_core,OLD

def build_anchor(rows, hidden, mask, A, y, g, train, original_outer=None):
    frame = old_run.clinical_frame(rows)
    if original_outer is not None:
        folds = joblib.load(OLD / 'five_center/private/SOURCE_FROZEN_MODELS.joblib')
        fold = folds[original_outer]
        assert np.array_equal(fold['train'], train)
        prior = fold['prior']
        alpha = fold['base_alpha']
        ascale = fold['A_scale']
        sem = fold['semantic']
    else:
        cv = []
        alpha_grid = [0.0, 0.25, 0.5, 1.0]
        base_oof = {a: np.zeros(len(rows)) for a in alpha_grid}
        for held in sorted(set(g[train])):
            tr = train[g[train] != held]
            va = train[g[train] == held]
            prior = old_run.fit_prior(frame, y, tr)
            ct = prior.decision_function(frame.iloc[tr])
            cvv = prior.decision_function(frame.iloc[va])
            scale = old_run.native_scale(A[tr])
            state = old_run.semantic_state(hidden[tr], mask[tr])
            xt = old_run.semantic_features(hidden[tr], mask[tr], state)
            xv = old_run.semantic_features(hidden[va], mask[va], state)
            for alpha in alpha_grid:
                base_oof[alpha][va] = cvv + alpha * A[va] / scale
            cv.append((tr, va, ct, cvv, scale, state, xt, xv))
        alpha = max(alpha_grid, key=lambda a: (old_core.metrics(y[train], base_oof[a][train])['AUROC'], -old_core.metrics(y[train], base_oof[a][train])['NLL'], -a))
        choices = []
        for lam in [0.003, 0.03, 0.3]:
            oof = np.zeros(len(rows))
            for tr, va, ct, cvv, scale, state, xt, xv in cv:
                head = old_run.fit_head_from_matrix(xt, ct + alpha * A[tr] / scale, y[tr], g[tr], lam)
                oof[va] = cvv + alpha * A[va] / scale + 0.25 * xv @ head['w']
            choices.append((old_core.metrics(y[train], oof[train])['AUROC'], -old_core.metrics(y[train], oof[train])['NLL'], lam))
        lam = max(choices)[2]
        prior = old_run.fit_prior(frame, y, train)
        ascale = old_run.native_scale(A[train])
        st = old_run.semantic_state(hidden[train], mask[train])
        xx = old_run.semantic_features(hidden[train], mask[train], st)
        cl = prior.decision_function(frame)
        sem = dict(state=st, head=old_run.fit_head_from_matrix(xx, cl[train] + alpha * A[train] / ascale, y[train], g[train], lam))
    clinical = prior.decision_function(frame)
    xx = old_run.semantic_features(hidden, mask, sem['state'])
    anchor = clinical + alpha * A / ascale + 0.25 * xx @ sem['head']['w']
    clinical_x = prior.named_steps['preprocess'].transform(frame)
    if hasattr(clinical_x, 'toarray'):
        clinical_x = clinical_x.toarray()
    b = np.column_stack([xx, clinical_x])
    bscale = np.maximum(b[train].std(0), 0.005)
    b = np.clip(b / bscale, -16, 16)
    return dict(prior=prior, alpha=alpha, A_scale=ascale, semantic=sem, context_scale=bscale, anchor=anchor, b=b, train=train, semantic_x=xx, initial_semantic_weight=0.25 * sem['head']['w'])
