"""Observable acquisition and finite-perturbation sufficiency frontier screen."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
import train_ices_backbone as tr
from run_cvre_feasibility import kl, probability, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/csf_screen_v1'
MODES = ('total_kl', 'centered', 'coherence', 'stable', 'cross_center', 'random', 'attention')


def decomposition(pp, current):
    mean = pp.mean(-1)
    total = kl(pp, current).mean(-1)
    centered = kl(pp, mean[:, None]).mean(-1)
    gap = kl(mean, current)
    return total, centered, gap


@torch.inference_mode()
def rollout(model, x, c, available, donor, first, last, dc, centers,
            recipient_center, mode, seed, recipient_first, recipient_last):
    mask = available.clone(); mask[:, 25:] = False
    state = x * mask.unsqueeze(-1)
    predictions, perturbations, records = [], [], []
    rng = np.random.default_rng(seed)

    def record_prediction():
        p = float(probability(model, state, c, mask)[0])
        changed = state.repeat(2, 1, 1)
        selected = mask[0].clone(); selected[:25] = False
        changed[0, selected] = recipient_first[0, selected]
        changed[1, selected] = recipient_last[0, selected]
        q = probability(model, changed, c.repeat(2, 1), mask.repeat(2, 1))
        predictions.append(p); perturbations.append([p, *q.cpu().tolist()])

    record_prediction()
    for step in range(4):
        slots = torch.where(available[0] & ~mask[0])[0]
        if not len(slots):
            record_prediction(); continue
        distance = ((dc-c)**2).mean(1)
        visual = mask[0].clone(); visual[0] = False
        if visual.any():
            a = torch.nn.functional.normalize(donor[:, visual].float().flatten(1), dim=1)
            b = torch.nn.functional.normalize(state[:, visual].float().flatten(1), dim=1)
            distance += 1-(a*b).sum(1)
        eligible = np.arange(len(donor))
        if mode == 'cross_center':
            eligible = np.where(centers != recipient_center)[0]
            assert len(eligible), 'No cross-center source donors'
        eligible_t = torch.as_tensor(eligible, device=x.device)
        pool = eligible_t[torch.argsort(distance[eligible_t])[:32]].cpu().numpy()
        donors = rng.choice(pool, size=min(8, len(pool)), replace=False)
        width = len(donors); n = len(slots)*width
        xx = state.repeat(n, 1, 1); mm = mask.repeat(n, 1); cc = c.repeat(n, 1)
        ss = slots.repeat_interleave(width)
        dd = torch.as_tensor(np.tile(donors, len(slots)), device=x.device)
        ix = torch.arange(n, device=x.device)
        mm[ix, ss] = True; xx[ix, ss] = donor[dd, ss]
        result = model(xx, cc, mm)
        pp = result['logit'].sigmoid().reshape(-1, width)
        current = probability(model, state, c, mask)[0]
        total, centered, gap = decomposition(pp, current)
        assert torch.allclose(total, centered+gap, atol=2e-6, rtol=2e-5)
        xx[ix, ss] = first[dd, ss]; p1 = probability(model, xx, cc, mm)
        xx[ix, ss] = last[dd, ss]; p2 = probability(model, xx, cc, mm)
        sensitivity = (.5*(kl(p1, p2)+kl(p2, p1))).reshape(-1, width).mean(1)
        scores = {'total_kl': total, 'centered': centered, 'coherence': gap,
                  'stable': centered-sensitivity, 'cross_center': centered,
                  'random': torch.as_tensor(rng.random(len(slots)), device=x.device),
                  'attention': result['attention'][ix, ss].reshape(-1, width).mean(1)}
        choice = int(scores[mode].argmax()); slot = int(slots[choice])
        records.append({'step': step, 'slot': slot, 'total_kl': float(total[choice]),
                        'centered': float(centered[choice]), 'coherence': float(gap[choice]),
                        'sensitivity_proxy': float(sensitivity[choice]),
                        'score': float(scores[mode][choice]), 'donor_count': width})
        mask[0, slot] = True; state[0, slot] = x[0, slot]
        record_prediction()
    return predictions, perturbations, records


def frontier(y, p, cf, centers):
    risk = (p-y[:, None])**2
    regret = np.maximum(0, ((cf-y[:, None, None])**2).max(-1)-risk)
    worst = np.max([risk[centers == h].mean(0) for h in np.unique(centers)], axis=0)
    rows = np.column_stack((np.arange(p.shape[1]), regret.mean(0), worst))
    dominated = [bool(np.any(np.all(rows <= r, axis=1) & np.any(rows < r, axis=1))) for r in rows]
    return [{'budget': k, 'regret': float(r[1]), 'worst_center_brier': float(r[2]),
             'nondominated': not dominated[k]} for k, r in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fold', required=True)
    parser.add_argument('--seed', type=int, default=20260905)
    args = parser.parse_args(); output = OUT/args.fold/f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'complete.json').exists(): return
    torch.set_num_threads(2); device = torch.device('cuda')
    config = tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'))
    data = tr.load_fold_data(config, args.fold); base = data['payload']
    source = np.asarray(data['source_indices']['train']); val = np.asarray(data['source_indices']['val'])
    # Selection labels are used only to stratify source-training calibration.
    y = base['labels'].numpy(); centers = data['centres']
    rng = np.random.default_rng(20260906); cal = []
    for h in np.unique(centers[source]):
        for label in (0, 1):
            ix = source[(centers[source] == h) & (y[source] == label)]
            if len(ix) > 1: cal.extend(rng.permutation(ix)[:max(1, int(.2*len(ix)))])
    cal = np.sort(cal); fit = np.setdiff1d(source, cal)
    assert not set(fit)&set(cal) and not set(source)&set(val)
    model = tr.create_model(config, data, device).eval()
    cp = ROOT/'results/ices_v1/runs/cluster_m1_m2'/args.fold/f'seed_{args.seed}/checkpoint.pt'
    model.load_state_dict(torch.load(cp, map_location=device, weights_only=False)['model_state'])
    views = torch.load(ROOT/'results/ices_v1/supplement_m1_frame_sensitivity/features/m1_frame_sensitivity_resnet50.pt', map_location='cpu', weights_only=False)
    assert views['ids'] == base['ids']
    x = base['cluster_features']; clinical = data['clinical']
    donor = x[fit].to(device); first = views['frame1_features'][fit].to(device)
    last = views['frame10_features'][fit].to(device); dc = clinical[fit].to(device)
    arrays = {}; reports = {}; calibration_choices = {}
    for split, indices in (('cal', cal), ('val', val)):
        ps = {m: [] for m in MODES}; qs = {m: [] for m in MODES}; full = []; full_cf = []
        with (output/f'{split}_chains.jsonl').open('w') as log:
            for n, index in enumerate(indices):
                xx = x[index:index+1].to(device); cc = clinical[index:index+1].to(device)
                av = base['available'][index:index+1].bool().to(device)
                f1 = views['frame1_features'][index:index+1].to(device)
                f2 = views['frame10_features'][index:index+1].to(device)
                fp = float(probability(model, xx, cc, av)[0]); full.append(fp)
                full_cf.append([fp, float(probability(model, f1, cc, av)[0]), float(probability(model, f2, cc, av)[0])])
                for mode in MODES:
                    p, q, chain = rollout(model, xx, cc, av, donor, first, last, dc,
                                          centers[fit], centers[index], mode,
                                          args.seed+int(index)*100, f1, f2)
                    ps[mode].append(p); qs[mode].append(q)
                    log.write(json.dumps({'row': n, 'mode': mode, 'chain': chain})+'\n')
                if n%20 == 0: print(json.dumps({'fold': args.fold, 'seed': args.seed, 'split': split, 'done': n+1, 'total': len(indices)}), flush=True)
        yy = y[indices]; hh = centers[indices]; full = np.array(full); full_cf = np.array(full_cf)
        full_risk = max(np.mean((full[hh == h]-yy[hh == h])**2) for h in np.unique(hh))
        full_regret = float(np.mean(np.maximum(0, ((full_cf-yy[:, None])**2).max(1)-(full-yy)**2)))
        reports[split] = {'full': metrics(yy, full, hh), 'full_worst_brier': float(full_risk), 'full_regret': full_regret, 'policies': {}}
        arrays.update({split+'_y': yy, split+'_centers': hh.astype(str), split+'_indices': indices, split+'_full': full, split+'_full_cf': full_cf})
        for mode in MODES:
            p = np.array(ps[mode]); q = np.array(qs[mode]); f = frontier(yy, p, q, hh)
            if split == 'cal':
                feasible = [r['budget'] for r in f if r['worst_center_brier'] <= full_risk+.02 and r['regret'] <= .02]
                calibration_choices[mode] = {'budget': min(feasible) if feasible else 12,
                                             'feasible': bool(feasible) or full_regret <= .02,
                                             'fallback_full': not bool(feasible)}
            k = calibration_choices[mode]['budget']; selected = p[:, k] if k <= 4 else full
            reports[split]['policies'][mode] = {'frontier': f, 'budgets': [metrics(yy, p[:, k], hh) for k in range(5)],
                                              'selected': metrics(yy, selected, hh), 'selection': calibration_choices[mode]}
            arrays[split+'_'+mode] = p; arrays[split+'_'+mode+'_cf'] = q
            arrays[split+'_'+mode+'_selected'] = selected
        np.savez_compressed(output/'predictions.npz', **arrays)
    report = {'fold': args.fold, 'seed': args.seed, 'reports': reports, 'test_labels_evaluated': False,
              'donor_count': len(fit), 'n_cal': len(cal), 'n_val': len(val),
              'calibration_seen_by_frozen_predictor': True, 'clinical_counterfactual_validated': False,
              'protocol_sha256': hashlib.sha256((ROOT/'configs/CSF_SCREEN_PROTOCOL.md').read_bytes()).hexdigest()}
    (output/'complete.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'fold': args.fold, 'seed': args.seed, 'status': 'complete'}), flush=True)


if __name__ == '__main__': main()
