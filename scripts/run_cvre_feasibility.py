"""Source-only, observed-state acquisition screen; no hidden recipient lookahead.

Donors approximate future observations, not causal pathology interventions.
First/last within-position frame changes are repeat-view sensitivity proxies.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import train_ices_backbone as tr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/cvre_feasibility_v1'
POLICIES = ('information', 'robust', 'robust_no_cost', 'robust_unmatched', 'random', 'expected_attention')


def kl(p, q):
    p, q = p.clamp(1e-6, 1-1e-6), q.clamp(1e-6, 1-1e-6)
    return p * (p/q).log() + (1-p)*((1-p)/(1-q)).log()


def metrics(y, p, centers):
    return {'macro_auprc': tr.macro_auprc(y, p, centers),
            'auroc': float(roc_auc_score(y, p)),
            'brier': float(np.mean((p-y)**2))}


@torch.inference_mode()
def probability(model, x, c, mask):
    return model(x, c, mask)['logit'].sigmoid()


@torch.inference_mode()
def trajectory(model, x, c, available, donor_x, donor_first, donor_last,
               donor_clinical, mode, rng, beta=.005, steps=4):
    # x is a single case. Hidden positions are zeroed before every candidate
    # evaluation; only the selected action can reveal a recipient OCT token.
    mask = available.clone()
    mask[:,25:] = False
    state = x * mask.unsqueeze(-1)
    base = float(probability(model, state, c, mask)[0])
    records, probs, costs = [], [base], [0]
    for t in range(steps):
        slots = torch.where(available[0] & ~mask[0])[0]
        if not len(slots): break
        # Match on observed clinical and acquired visual state only. Labels,
        # held-out patients and hidden recipient OCT never enter donor choice.
        distance = ((donor_clinical-c)**2).mean(1)
        visual = mask[0].clone(); visual[0] = False
        if visual.any():
            a = torch.nn.functional.normalize(donor_x[:,visual].float().flatten(1),dim=1)
            b = torch.nn.functional.normalize(state[:,visual].float().flatten(1),dim=1)
            distance = distance + (1-(a*b).sum(1))
        pool = torch.argsort(distance)[:min(16,len(distance))].cpu().numpy()
        if mode == 'robust_unmatched': pool = np.arange(len(distance))
        donors = rng.choice(pool,size=min(3,len(pool)),replace=False)
        width = len(donors)
        n = len(slots)*width
        xx = state.repeat(n,1,1)
        mm = mask.repeat(n,1)
        cc = c.repeat(n,1)
        ss = slots.repeat_interleave(width)
        dd = torch.as_tensor(np.tile(donors,len(slots)),device=x.device)
        arange = torch.arange(n,device=x.device)
        mm[arange,ss] = True
        xx[arange,ss] = donor_x[dd,ss]
        predicted = model(xx,cc,mm)
        pp = predicted['logit'].sigmoid()
        current = probability(model,state,c,mask)[0]
        info = kl(pp,current).reshape(-1,width).mean(1)
        xx[arange,ss] = donor_first[dd,ss]
        pfirst = probability(model,xx,cc,mm)
        xx[arange,ss] = donor_last[dd,ss]
        plast = probability(model,xx,cc,mm)
        instability = (.5*(kl(pfirst,plast)+kl(plast,pfirst))).reshape(-1,width).mean(1)
        if mode == 'information': score = info
        elif mode == 'expected_attention': score = predicted['attention'][arange,ss].reshape(-1,width).mean(1)
        elif mode == 'random': score = torch.as_tensor(rng.random(len(slots)),device=x.device)
        else: score = info-instability
        choice = int(score.argmax())
        net = float(score[choice])-(0 if mode in ('robust_no_cost','random','expected_attention') else beta)
        if net <= 0:
            records.append({'step':t,'action':'stop','reason':'nonpositive_expected_net_value','max_net_value':net})
            break
        slot = int(slots[choice])
        mask[0,slot] = True
        state[0,slot] = x[0,slot]
        new_p = float(probability(model,state,c,mask)[0])
        records.append({'step':t,'action':f'oct_position_{slot-25:02d}',
                        'p_before':probs[-1],'p_after':new_p,'expected_kl':float(info[choice]),
                        'repeat_view_sensitivity':float(instability[choice]),'net_value':net,
                        'donor_count':width,'selected_visual_slot':slot})
        probs.append(new_p); costs.append(costs[-1]+1)
    if len(costs)-1 == steps:
        records.append({'step':steps,'action':'stop','reason':'pilot_budget_cap'})
    return probs[-1],costs[-1],records,probs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',required=True)
    parser.add_argument('--seed',type=int,default=20260905)
    args=parser.parse_args()
    config=tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'))
    data=tr.load_fold_data(config,args.fold)
    output=OUT/args.fold/f'seed_{args.seed}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists(): return
    torch.set_num_threads(2)
    device=torch.device('cuda')
    model=tr.create_model(config,data,device).eval()
    cp=ROOT/'results/ices_v1/runs/cluster_m1_m2'/args.fold/f'seed_{args.seed}/checkpoint.pt'
    model.load_state_dict(torch.load(cp,map_location=device,weights_only=False)['model_state'])
    source=data['source_indices']['train']; val=data['source_indices']['val']
    assert not set(source)&set(val)
    base=data['payload']; x=base['cluster_features']; clinical=data['clinical']
    other=torch.load(ROOT/'results/ices_v1/supplement_m1_frame_sensitivity/features/m1_frame_sensitivity_resnet50.pt',map_location='cpu',weights_only=False)
    assert other['ids']==base['ids']
    donor=x[source].to(device)
    first=other['frame1_features'][source].to(device)
    last=other['frame10_features'][source].to(device)
    dc=clinical[source].to(device)
    outputs={k:[] for k in POLICIES}
    costs={k:[] for k in POLICIES}
    prefixes={k:[] for k in POLICIES}
    full=[]
    with (output/'evidence_chains.jsonl').open('w') as log:
        for n,index in enumerate(val):
            xx=x[index:index+1].to(device); cc=clinical[index:index+1].to(device)
            available=base['available'][index:index+1].bool().to(device)
            full.append(float(probability(model,xx,cc,available)[0]))
            for policy in POLICIES:
                rng=np.random.default_rng(args.seed+index*100)
                p,c,chain,prefix=trajectory(model,xx,cc,available,donor,first,last,dc,policy,rng)
                outputs[policy].append(p);costs[policy].append(c);prefixes[policy].append(prefix)
                log.write(json.dumps({'validation_row':n,'policy':policy,'chain':chain})+'\n')
            if n%20==0: print(json.dumps({'fold':args.fold,'cases_complete':n+1,'total':len(val)}),flush=True)
    # Outcome labels first accessed for scoring after every trajectory is fixed.
    y=base['labels'][val].numpy(); centers=data['centres'][val]
    report={'full':metrics(y,np.array(full),centers),'policies':{},'test_labels_evaluated':False,
            'n_val':len(val),'seed':args.seed,'max_oct_budget':4,
            'interpretation':'development-only frozen-predictor acquisition screen; donor completion and repeat-view proxies, no identified causal intervention'}
    arrays={'y':y,'full':full,'centers':centers.astype(str),'indices':val}
    for policy in POLICIES:
        report['policies'][policy]={**metrics(y,np.array(outputs[policy]),centers),'mean_oct_units':float(np.mean(costs[policy])),'zero_oct_fraction':float(np.mean(np.array(costs[policy])==0))}
        arrays[policy]=outputs[policy];arrays[policy+'_cost']=costs[policy]
    for reference in ('random','expected_attention'):
        name=reference+'_robust_matched_cost'
        predictions=np.array([prefixes[reference][i][k] for i,k in enumerate(costs['robust'])])
        report['policies'][name]={**metrics(y,predictions,centers),'mean_oct_units':float(np.mean(costs['robust']))}
        arrays[name]=predictions;arrays[name+'_cost']=costs['robust']
    np.savez_compressed(output/'predictions.npz',**arrays)
    (output/'complete.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
