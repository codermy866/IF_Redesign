"""Read-only model audit: existing fresh diagnostic references, source val only."""
import json
import numpy as np
import torch
import train_ices_backbone as tr
from cervix_cogalign.cesl import clinical_matrix
from run_missing_modality_fusion import ROOT, OUT, FOLDS, SEEDS, PATTERNS


@torch.inference_mode()
def predict(model, x, c, av, indices):
    rows = []
    for ix in np.array_split(indices, max(1, int(np.ceil(len(indices)/48)))):
        patterns = []
        for t, col, oct_present in PATTERNS:
            mask = av[ix].clone()
            if not t:
                mask[:, 0] = False
            if not col:
                mask[:, 1:25] = False
            if not oct_present:
                mask[:, 25:] = False
            patterns.append(model(x[ix], c[ix], mask)['logit'].sigmoid().cpu().numpy())
        rows.append(np.stack(patterns, 1))
    return np.concatenate(rows)


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda')
    config = tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'))
    for fold in FOLDS:
        data = tr.load_fold_data(config, fold)
        base = data['payload']
        x = base['cluster_features'].to(device)
        av = base['available'].bool().to(device)
        val = np.asarray(data['source_indices']['val'])
        for seed in SEEDS:
            out = OUT/fold/f'seed_{seed}'/'legacy_missingness.npz'
            if out.exists():
                continue
            src = ROOT/'results/cross_fitted_gain_v1'/fold/f'seed_{seed}'/'diagnostic_reference'
            meta = json.loads((src/'complete.json').read_text())
            cal = np.array(meta['cal_indices'])
            clinical, _ = clinical_matrix(base['ages'], base['hpvs'], base['tcts'], meta['fit_indices'])
            model = tr.create_model(config, data, device)
            model.load_state_dict(torch.load(src/'checkpoint.pt', map_location=device, weights_only=False)['model_state'])
            model.eval()
            c = torch.from_numpy(clinical).to(device)
            p, cp = predict(model, x, c, av, val), predict(model, x, c, av, cal)
            assert np.isfinite(p).all() and np.isfinite(cp).all()
            old = np.load(src.parent/'predictions.npz', allow_pickle=False)
            np.testing.assert_allclose(p[:, -1], old['full'], atol=2e-6, rtol=2e-5)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, indices=val, cal_indices=cal, legacy=p, cal_legacy=cp)
            print(json.dumps({'legacy_missingness_complete': fold, 'seed': seed}), flush=True)


if __name__ == '__main__':
    main()
