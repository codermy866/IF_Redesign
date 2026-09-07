"""Predeclared 2x2 unimodal protection x modality-dropout diagnostic screen."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score
import train_ices_backbone as tr
from run_cross_fitted_gain import calibration_split
from cervix_cogalign.cesl import clinical_matrix, hpv_group, tct_group, age_bin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/missing_modality_fusion_v1'
FOLDS = ('shiyan', 'enshi', 'wuhan', 'jingzhou', 'xiangyang')
SEEDS = (20260905, 20260906, 20260907)
ARMS = ('frozen_full', 'tune_full', 'frozen_missing', 'tune_missing')
PATTERNS = np.array([[bool(i & (1 << j)) for j in range(3)] for i in range(1, 8)])
NAMES = [''.join(m for m, present in zip('TCO', p) if present) for p in PATTERNS]


class Branch(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        self.encode = nn.Sequential(nn.Linear(14 if modality == 0 else 2048, 64), nn.LayerNorm(64), nn.GELU())
        if modality:
            self.position = nn.Parameter(torch.zeros(1, 12 if modality == 2 else 24, 64))
            self.score = nn.Linear(64, 1)
        self.head = nn.Linear(64, 1)

    def forward(self, x, clinical, available):
        if self.modality == 0:
            h = self.encode(clinical)
        else:
            sl = slice(1, 25) if self.modality == 1 else slice(25, 37)
            mask = available[:, sl]
            xx = torch.where(mask.unsqueeze(-1), x[:, sl], 0.)
            z = self.encode(xx) + self.position
            score = self.score(z).squeeze(-1).masked_fill(~mask, -1e9)
            weights = score.softmax(-1) * mask
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
            h = (z * weights.unsqueeze(-1)).sum(1)
        return h, self.head(h).squeeze(-1)


class Fusion(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(copy.deepcopy(branches))
        self.head = nn.Sequential(nn.Linear(195, 64), nn.GELU(), nn.Linear(64, 1))

    def representations(self, x, clinical, available):
        return torch.stack([b(x, clinical, available)[0] for b in self.branches], 1)

    def from_latents(self, h, pattern):
        if not pattern.any(1).all():
            raise ValueError('All modalities missing: refuse diagnostic prediction')
        h = torch.where(pattern.unsqueeze(-1), h, 0.)
        return self.head(torch.cat((h.flatten(1), pattern.float()), 1)).squeeze(-1)

    def forward(self, x, clinical, available, pattern):
        return self.from_latents(self.representations(x, clinical, available), pattern)


def bce(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


@torch.inference_mode()
def predict(model, x, c, av, indices, kind):
    rows = []
    for ix in np.array_split(indices, max(1, int(np.ceil(len(indices) / 64)))):
        if kind == 'branch':
            rows.append(model(x[ix], c[ix], av[ix])[1].sigmoid().cpu().numpy())
        else:
            h = model.representations(x[ix], c[ix], av[ix])
            rows.append(np.stack([model.from_latents(h, torch.as_tensor(p, device=x.device).expand(len(ix), -1)).sigmoid().cpu().numpy() for p in PATTERNS], 1))
    return np.concatenate(rows)


def cross_entropy(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def fit(model, kind, x, c, av, y, fit_ix, cal_ix, path, seed, arm=None):
    path.mkdir(parents=True, exist_ok=True)
    frozen = arm is not None and arm.startswith('frozen')
    if kind == 'fusion':
        for p in model.branches.parameters():
            p.requires_grad_(not frozen)
        groups = [{'params': model.head.parameters(), 'lr': .001}]
        if not frozen:
            groups.append({'params': model.branches.parameters(), 'lr': .0001})
    else:
        groups = [{'params': model.parameters(), 'lr': .001}]
    if (path / 'complete.json').exists():
        model.load_state_dict(torch.load(path/'checkpoint.pt', map_location=x.device, weights_only=True))
        return model.eval()
    optimizer = torch.optim.AdamW(groups, weight_decay=.0001)
    best = np.inf
    stale = 0
    history = []
    for epoch in range(80):
        model.train()
        order = np.random.default_rng(seed+epoch).permutation(fit_ix)
        mask_rng = np.random.default_rng(seed+100000+epoch)
        losses = []
        for start in range(0, len(order), 64):
            ix = order[start:start+64]
            if kind == 'branch':
                loss = bce(model(x[ix], c[ix], av[ix])[1], y[ix])
            else:
                h = model.representations(x[ix], c[ix], av[ix])
                full = torch.ones((len(ix), 3), dtype=torch.bool, device=x.device)
                loss = bce(model.from_latents(h, full), y[ix])
                if arm.endswith('missing'):
                    pattern = torch.as_tensor(PATTERNS[mask_rng.integers(7, size=len(ix))], device=x.device)
                    loss = loss + .5*bce(model.from_latents(h, pattern), y[ix])
                else:
                    loss = 1.5*loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        p = predict(model, x, c, av, cal_ix, kind)
        yy = y[cal_ix].cpu().numpy()
        score = float(cross_entropy(yy if kind == 'branch' else yy[:, None], p).mean())
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'cal_ce': score})
        if score < best - 1e-6:
            best = score
            stale = 0
            torch.save(model.state_dict(), path / 'checkpoint.pt')
        else:
            stale += 1
        if epoch % 10 == 0:
            print(json.dumps({'stage': path.name, 'epoch': epoch+1, 'internal_cal_ce': score}), flush=True)
        if stale >= 12:
            break
    model.load_state_dict(torch.load(path/'checkpoint.pt', map_location=x.device, weights_only=True))
    (path/'complete.json').write_text(json.dumps({'history': history, 'best_cal_ce': best, 'fit_indices': fit_ix.tolist(), 'cal_indices': cal_ix.tolist(), 'arm': arm}, indent=2))
    return model.eval()


def metrics(y, p, centers, threshold=None):
    result = {'macro_ap': tr.macro_auprc(y, p, centers), 'pooled_ap': float(average_precision_score(y, p)),
              'auroc': float(roc_auc_score(y, p)), 'ce': float(cross_entropy(y, p).mean()), 'brier': float(np.mean((y-p)**2))}
    if threshold is not None:
        result.update({'threshold': float(threshold), 'sensitivity': float(np.mean(p[y == 1] >= threshold)), 'specificity': float(np.mean(p[y == 0] < threshold))})
    return result


def threshold90(y, p):
    positives = np.sort(p[y == 1])
    if not len(positives):
        raise ValueError('No calibration positives')
    return positives[int(np.floor(.1*len(positives)))]


def availability_audit(base):
    av = base['available'].bool()
    profiles = torch.stack((av[:, 0], av[:, 1:25].any(1), av[:, 25:].any(1)), 1)
    assert profiles.all(), 'Protocol is paired synthetic dropout of the fully available cache; audit incomplete cohort separately'
    return {'patients': len(base['ids']), 'distinct_patients': len(set(base['patient_ids'])),
            'all_three_modalities': int(profiles.all(1).sum()), 'all_12_oct_positions': int(av[:, 25:].all(1).sum()),
            'age_unknown': sum(age_bin(a) == 'missing' for a in base['ages']),
            'hpv_unknown_or_unrecognized': sum(hpv_group(a) == 'unknown' for a in base['hpvs']),
            'tct_unknown_or_unrecognized': sum(tct_group(a) == 'unknown' for a in base['tcts'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=FOLDS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--restored-clinical', action='store_true')
    args = parser.parse_args()
    result_root = ROOT/'results/missing_modality_fusion_restored_v1' if args.restored_clinical else OUT
    path = result_root / args.fold / f'seed_{args.seed}'
    path.mkdir(parents=True, exist_ok=True)
    if (path/'complete.json').exists():
        return
    torch.set_num_threads(2)
    device = torch.device('cuda')
    config = tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'))
    data = tr.load_fold_data(config, args.fold)
    base = data['payload']
    if args.restored_clinical:
        sidecar = np.load(ROOT/'results/clinical_source_recovery_v1/clinical_sidecar.npz', allow_pickle=False)
        np.testing.assert_array_equal(sidecar['ids'], np.asarray(base['ids']))
        base = dict(base, hpvs=sidecar['hpvs'].tolist(), tcts=sidecar['tcts'].tolist())
    audit = availability_audit(base)
    assert audit['patients'] == audit['distinct_patients']
    source = np.asarray(data['source_indices']['train'])
    val = np.asarray(data['source_indices']['val'])
    outer = np.asarray(data['parts']['test']['index'])
    yy = base['labels'].numpy()
    centers = data['centres']
    fit_ix, cal_ix = calibration_split(source, centers, yy, args.seed+100)
    assert not set(source)&set(val) and not set(source)&set(outer) and not set(val)&set(outer)
    clinical, scaling = clinical_matrix(base['ages'], base['hpvs'], base['tcts'], fit_ix)
    x = base['cluster_features'].float().to(device)
    c = torch.from_numpy(clinical).to(device)
    av = base['available'].bool().to(device)
    y = base['labels'].float().to(device)
    branches = []
    for modality in range(3):
        tr.set_seed(args.seed+modality)
        branch = Branch(modality).to(device)
        branches.append(fit(branch, 'branch', x, c, av, y, fit_ix, cal_ix, path/f'branch_{modality}', args.seed))
    tr.set_seed(args.seed+200)
    template = Fusion(branches).to(device)
    arrays = {'indices': val, 'y': yy[val], 'centers': centers[val].astype(str), 'cal_indices': cal_ix, 'cal_y': yy[cal_ix]}
    uni_val = np.stack([predict(b, x, c, av, val, 'branch') for b in branches], 1)
    uni_cal = np.stack([predict(b, x, c, av, cal_ix, 'branch') for b in branches], 1)
    arrays.update(unimodal=uni_val, cal_unimodal=uni_cal)
    arrays['late'] = np.stack([uni_val[:, p].mean(1) for p in PATTERNS], 1)
    arrays['cal_late'] = np.stack([uni_cal[:, p].mean(1) for p in PATTERNS], 1)
    for arm in ARMS:
        model = fit(copy.deepcopy(template), 'fusion', x, c, av, y, fit_ix, cal_ix, path/arm, args.seed+200, arm)
        if arm.startswith('frozen'):
            for a, b in zip(model.branches.parameters(), template.branches.parameters()):
                assert torch.equal(a, b), 'Frozen encoder changed'
        arrays[arm] = predict(model, x, c, av, val, 'fusion')
        arrays['cal_'+arm] = predict(model, x, c, av, cal_ix, 'fusion')
        del model
    report = {}
    for mode in ('late',)+ARMS:
        report[mode] = {name: metrics(yy[val], arrays[mode][:, k], centers[val], threshold90(yy[cal_ix], arrays['cal_'+mode][:, k])) for k, name in enumerate(NAMES)}
    np.savez_compressed(path/'predictions.npz', **arrays)
    (path/'complete.json').write_text(json.dumps({'fold': args.fold, 'seed': args.seed, 'metrics': report, 'audit': audit,
        'scaling': scaling, 'source_n': len(source), 'fit_n': len(fit_ix), 'cal_n': len(cal_ix), 'cal_positive_n': int(yy[cal_ix].sum()),
        'val_n': len(val), 'test_labels_evaluated': False, 'restored_clinical': args.restored_clinical,
        'clinical_sidecar_sha256': hashlib.sha256((ROOT/'results/clinical_source_recovery_v1/clinical_sidecar.npz').read_bytes()).hexdigest() if args.restored_clinical else None,
        'protocol_sha256': hashlib.sha256((ROOT/'configs'/('CLINICAL_RECOVERY_REPLICATION.md' if args.restored_clinical else 'MISSING_MODALITY_FUSION_PROTOCOL.md')).read_bytes()).hexdigest()}, indent=2))
    print(json.dumps({'fold': args.fold, 'seed': args.seed, 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
