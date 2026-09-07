"""Supervised expected risk-reduction memory, with observed-state action inputs."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import train_ices_backbone as tr
from run_cvre_feasibility import probability, metrics
from run_csf_screen import frontier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/csf_memory_v1'
MODES = ('gain_mean', 'gain_transport')


def features(state, clinical, mask, p, slots, projection):
    # Caller passes already masked state; the second masking is deliberate.
    state = (state*mask.unsqueeze(-1)).float()
    colpo = state[:, 1:25].sum(1)/mask[:, 1:25].sum(1).clamp_min(1)[:, None]
    oct_mean = state[:, 25:].sum(1)/mask[:, 25:].sum(1).clamp_min(1)[:, None]
    basic = torch.cat((clinical.float(), colpo@projection, oct_mean@projection,
                       p.reshape(-1, 1), mask[:, 25:].float().mean(1, keepdim=True)), dim=1)
    onehot = torch.nn.functional.one_hot(slots-25, 12).float()
    basic = basic.expand(len(slots), -1)
    return torch.cat((basic, onehot, (basic[:, :, None]*onehot[:, None, :]).flatten(1)), dim=1).cpu().numpy()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fold', required=True)
    parser.add_argument('--seed', type=int, default=20260905); args = parser.parse_args()
    output = OUT/args.fold/f'seed_{args.seed}'; output.mkdir(parents=True, exist_ok=True)
    if (output/'complete.json').exists(): return
    torch.set_num_threads(2); device = torch.device('cuda')
    config = tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'))
    data = tr.load_fold_data(config, args.fold); base = data['payload']
    source = np.asarray(data['source_indices']['train']); val = np.asarray(data['source_indices']['val'])
    y = base['labels'].numpy(); centers = data['centres']
    rng = np.random.default_rng(20260906); cal = []
    for h in np.unique(centers[source]):
        for label in (0, 1):
            ix = source[(centers[source] == h)&(y[source] == label)]
            if len(ix) > 1: cal.extend(rng.permutation(ix)[:max(1, int(.2*len(ix)))])
    cal = np.sort(cal); fit = np.setdiff1d(source, cal)
    assert not set(fit)&set(cal) and not set(source)&set(val)
    model = tr.create_model(config, data, device).eval()
    cp = ROOT/'results/ices_v1/runs/cluster_m1_m2'/args.fold/f'seed_{args.seed}/checkpoint.pt'
    model.load_state_dict(torch.load(cp, map_location=device, weights_only=False)['model_state'])
    x = base['cluster_features'].to(device); clinical = data['clinical'].to(device)
    available = base['available'].bool().to(device)
    projection = torch.as_tensor(np.random.default_rng(20260906).normal(size=(2048, 16))/np.sqrt(2048), dtype=torch.float32, device=device)
    design, targets, groups = [], [], []
    for n, index in enumerate(fit):
        cc = clinical[index:index+1]; mask = available[index:index+1].clone(); mask[:, 25:] = False
        state = x[index:index+1]*mask.unsqueeze(-1)
        rng = np.random.default_rng(args.seed+int(index)*100)
        for step in range(4):
            slots = torch.where(available[index]&~mask[0])[0]
            p = probability(model, state, cc, mask)
            z = features(state, cc, mask, p, slots, projection)
            xx = state.repeat(len(slots), 1, 1); mm = mask.repeat(len(slots), 1)
            rows = torch.arange(len(slots), device=device); xx[rows, slots] = x[index, slots]; mm[rows, slots] = True
            after = probability(model, xx, cc.repeat(len(slots), 1), mm)
            target = (p-y[index])**2-(after-y[index])**2
            design.append(z); targets.append(target.cpu().numpy()); groups.extend([centers[index]]*len(slots))
            chosen = int(slots[rng.integers(len(slots))]); mask[0, chosen] = True; state[0, chosen] = x[index, chosen]
        if n%100 == 0: print(json.dumps({'fold': args.fold, 'seed': args.seed, 'training_cases': n+1, 'total': len(fit)}), flush=True)
    design = np.concatenate(design); target = np.concatenate(targets); groups = np.asarray(groups)

    def fit_value(indices):
        scaler = StandardScaler().fit(design[indices])
        hh = groups[indices]; weights = np.zeros(len(hh))
        for h in np.unique(hh): weights[hh == h] = len(hh)/(len(np.unique(hh))*sum(hh == h))
        estimator = Ridge(alpha=10).fit(scaler.transform(design[indices]), target[indices], sample_weight=weights)
        return scaler, estimator

    pooled = fit_value(np.arange(len(target)))
    experts = [fit_value(np.where(groups != h)[0]) for h in np.unique(groups)]
    joblib.dump({'pooled': pooled, 'experts': experts, 'projection': projection.cpu().numpy(), 'fit_indices': fit,
                 'excluded_cal_indices': cal, 'feature_description': 'clinical, observed projected means, current p, budget, action interactions'}, output/'value_memory.joblib')
    views = torch.load(ROOT/'results/ices_v1/supplement_m1_frame_sensitivity/features/m1_frame_sensitivity_resnet50.pt', map_location='cpu', weights_only=False)
    assert views['ids'] == base['ids']
    arrays = {}; reports = {}; choices = {}
    for split, indices in (('cal', cal), ('val', val)):
        ps = {m: [] for m in MODES}; qs = {m: [] for m in MODES}; full = []; full_cf = []
        with (output/f'{split}_chains.jsonl').open('w') as log:
            for n, index in enumerate(indices):
                cc = clinical[index:index+1]; av = available[index:index+1]
                f1 = views['frame1_features'][index:index+1].to(device); f2 = views['frame10_features'][index:index+1].to(device)
                fp = float(probability(model, x[index:index+1], cc, av)[0]); full.append(fp)
                full_cf.append([fp, float(probability(model, f1, cc, av)[0]), float(probability(model, f2, cc, av)[0])])
                for mode in MODES:
                    mask = av.clone(); mask[:, 25:] = False; state = x[index:index+1]*mask.unsqueeze(-1)
                    trajectory, perturb, chain = [], [], []
                    for step in range(5):
                        p = probability(model, state, cc, mask); trajectory.append(float(p[0]))
                        changed = state.repeat(2, 1, 1); selected = mask[0].clone(); selected[:25] = False
                        changed[0, selected] = f1[0, selected]; changed[1, selected] = f2[0, selected]
                        perturb.append([float(p[0]), *probability(model, changed, cc.repeat(2, 1), mask.repeat(2, 1)).cpu().tolist()])
                        if step == 4: break
                        slots = torch.where(av[0]&~mask[0])[0]
                        z = features(state, cc, mask, p, slots, projection)
                        models = [pooled] if mode == 'gain_mean' else experts
                        values = np.stack([r.predict(s.transform(z)) for s, r in models])
                        scores = values.min(0); choice = int(np.argmax(scores)); slot = int(slots[choice])
                        chain.append({'step': step, 'slot': slot, 'predicted_brier_gain': float(scores[choice]), 'p_before': float(p[0])})
                        mask[0, slot] = True; state[0, slot] = x[index, slot]
                    ps[mode].append(trajectory); qs[mode].append(perturb)
                    log.write(json.dumps({'row': n, 'mode': mode, 'chain': chain})+'\n')
        yy = y[indices]; hh = centers[indices]; full = np.array(full); full_cf = np.array(full_cf)
        full_risk = max(np.mean((full[hh == h]-yy[hh == h])**2) for h in np.unique(hh))
        full_regret = float(np.mean(np.maximum(0, ((full_cf-yy[:, None])**2).max(1)-(full-yy)**2)))
        reports[split] = {'full': metrics(yy, full, hh), 'full_worst_brier': float(full_risk), 'full_regret': full_regret, 'policies': {}}
        arrays.update({split+'_y': yy, split+'_centers': hh.astype(str), split+'_indices': indices, split+'_full': full, split+'_full_cf': full_cf})
        for mode in MODES:
            p = np.array(ps[mode]); q = np.array(qs[mode]); f = frontier(yy, p, q, hh)
            if split == 'cal':
                feasible = [r['budget'] for r in f if r['worst_center_brier'] <= full_risk+.02 and r['regret'] <= .02]
                choices[mode] = {'budget': min(feasible) if feasible else 12, 'feasible': bool(feasible) or full_regret <= .02, 'fallback_full': not bool(feasible)}
            k = choices[mode]['budget']; selected = p[:, k] if k <= 4 else full
            reports[split]['policies'][mode] = {'frontier': f, 'budgets': [metrics(yy, p[:, k], hh) for k in range(5)], 'selected': metrics(yy, selected, hh), 'selection': choices[mode]}
            arrays[split+'_'+mode] = p; arrays[split+'_'+mode+'_cf'] = q; arrays[split+'_'+mode+'_selected'] = selected
        np.savez_compressed(output/'predictions.npz', **arrays)
    (output/'complete.json').write_text(json.dumps({'fold': args.fold, 'seed': args.seed, 'reports': reports, 'n_fit': len(fit),
        'n_training_action_rows': len(target), 'n_cal': len(cal), 'n_val': len(val), 'test_labels_evaluated': False,
        'classifier_saw_fit_and_cal': True, 'independent_action_value_validation': True}, indent=2))
    print(json.dumps({'fold': args.fold, 'seed': args.seed, 'status': 'complete'}), flush=True)


if __name__ == '__main__': main()
