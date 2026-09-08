"""Source-only factorial pilot with reviewed site supervision and label-aware swaps."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
import train_ices_backbone as tr
from run_cvre_feasibility import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/oct_site_supervision_v1'


@torch.no_grad()
def evaluate(model, x, clinical, av, y, centers, sites, indices):
    model.eval(); scores = []; predictions = {k: [] for k in ('full', 'selected4', 'random4', 'zero')}
    rng = np.random.default_rng(20260906)
    for start in range(0, len(indices), 32):
        ix = indices[start:start+32]; xx=x[ix]; cc=clinical[ix]; aa=av[ix]
        result = model(xx, cc, aa); score = result['importance_logits'][:, 25:]
        scores.extend(score.cpu().numpy()); predictions['full'].extend(result['logit'].sigmoid().cpu().tolist())
        for kind in ('selected4', 'random4', 'zero'):
            mask=aa.clone(); mask[:,25:]=False
            if kind != 'zero':
                order = score.topk(4, dim=1).indices if kind=='selected4' else torch.as_tensor(np.stack([rng.choice(12,4,replace=False) for _ in ix]), device=x.device)
                mask.scatter_(1, order+25, True)
            predictions[kind].extend(model(xx*mask.unsqueeze(-1),cc,mask)['logit'].sigmoid().cpu().tolist())
    score=np.asarray(scores); target=sites[indices].cpu().numpy()
    case_aps=[]; recalls=[]
    for truth, ss in zip(target,score):
        if (truth < 0).any(): continue
        if 0 < truth.sum() < 12: case_aps.append(average_precision_score(truth,ss))
        if truth.sum() > 0: recalls.append(float(truth[np.argsort(-ss)[:4]].sum()/truth.sum()))
    yy=y[indices].cpu().numpy(); cc=centers[indices]
    report={k:metrics(yy,np.asarray(p),cc) for k,p in predictions.items()}
    report.update({'within_case_site_ap':float(np.mean(case_aps)) if case_aps else None,
                   'site_recall_at4':float(np.mean(recalls)) if recalls else None,
                   'random_expected_recall_at4':1/3, 'mixed_site_cases':len(case_aps), 'positive_site_cases':len(recalls)})
    return report, {'indices':indices,'y':yy,'centers':cc.astype(str),'site_labels':target,'site_scores':score,**predictions}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True);parser.add_argument('--arm',choices=['A00','A10','A01','A11'],required=True)
    parser.add_argument('--seed',type=int,default=20260905);args=parser.parse_args()
    output=OUT/'runs'/args.fold/f'seed_{args.seed}'/args.arm;output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);device=torch.device('cuda')
    config=tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'));data=tr.load_fold_data(config,args.fold);base=data['payload']
    annotations=np.load(OUT/'site_labels.npz',allow_pickle=False);assert list(annotations['ids'])==list(base['ids'])
    x=base['cluster_features'].to(device);clinical=data['clinical'].to(device);av=base['available'].bool().to(device)
    sites=torch.as_tensor(annotations['labels'],device=device).float();y=base['labels'].float().to(device);centers=data['centres']
    train=np.asarray(data['source_indices']['train']);val=np.asarray(data['source_indices']['val']);assert not set(train)&set(val)
    model=tr.create_model(config,data,device)
    checkpoint=ROOT/'results/ices_v1/runs/cluster_m1_m2'/args.fold/f'seed_{args.seed}/checkpoint.pt'
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model_state'])
    reference,_=evaluate(model,x,clinical,av,y,centers,sites,val)
    valid=sites[train]>=0; positives=sites[train][valid].sum();site_weight=(valid.sum()-positives)/positives.clamp_min(1)
    case_weight=(len(train)-y[train].sum())/y[train].sum().clamp_min(1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    site_cpu=sites.cpu().numpy();pools={}
    for center in np.unique(centers[train]):
        for position in range(12): pools[center,position]=train[(centers[train]!=center)&(site_cpu[train,position]==0)]
    m1=args.arm[1]=='1';m2=args.arm[2]=='1';history=[]
    for epoch in range(12):
        rng=np.random.default_rng(args.seed+epoch);order=rng.permutation(train);epoch_losses=[];npairs=0
        for start in range(0,len(order),32):
            ix=order[start:start+32];xx=x[ix];cc=clinical[ix];aa=av[ix];yy=y[ix]
            # Masks/permutation independent of intervention-specific RNG draws.
            mask_rng=np.random.default_rng(args.seed+epoch*10000+start)
            mask=aa.clone();mask[:,25:] &= torch.as_tensor(mask_rng.random((len(ix),12))<.5,device=device)
            torch.manual_seed(args.seed+epoch*10000+start)
            model.train();optimizer.zero_grad();pred=model(xx,cc,aa);partial=model(xx*mask.unsqueeze(-1),cc,mask)
            case_loss=F.binary_cross_entropy_with_logits(pred['logit'],yy,pos_weight=case_weight)+.5*F.binary_cross_entropy_with_logits(partial['logit'],yy,pos_weight=case_weight)
            loss=case_loss;site_loss=loss.new_zeros(());cf_loss=loss.new_zeros(())
            if m1:
                known=sites[ix]>=0
                if known.any():site_loss=F.binary_cross_entropy_with_logits(pred['importance_logits'][:,25:][known],sites[ix][known],pos_weight=site_weight)
                loss=loss+.25*site_loss
            if m2:
                rows=[];positions=[];donors=[]
                for row,index in enumerate(ix):
                    options=[j for j in range(12) if site_cpu[index,j]==1 and len(pools[centers[index],j])]
                    if options:
                        pos=int(mask_rng.choice(options));rows.append(row);positions.append(pos);donors.append(int(mask_rng.choice(pools[centers[index],pos])))
                if rows:
                    model.eval()  # same deterministic context, gradients retained
                    r=torch.as_tensor(rows,device=device);s=torch.as_tensor(positions,device=device)+25;dd=torch.as_tensor(donors,device=device)
                    original=model(xx[r],cc[r],aa[r])['importance_logits'][torch.arange(len(r),device=device),s]
                    changed=xx[r].clone();changed[torch.arange(len(r),device=device),s]=x[dd,s]
                    replaced=model(changed,cc[r],aa[r])['importance_logits'][torch.arange(len(r),device=device),s]
                    cf_loss=F.relu(.5-original+replaced).mean();loss=loss+.1*cf_loss;npairs+=len(rows)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            epoch_losses.append([float(case_loss.detach()),float(site_loss.detach()),float(cf_loss.detach())])
        record={'epoch':epoch+1,'mean_losses':np.mean(epoch_losses,axis=0).tolist(),'source_intervention_pairs':npairs};history.append(record)
        print(json.dumps({'fold':args.fold,'arm':args.arm,**record}),flush=True)
    report,arrays=evaluate(model,x,clinical,av,y,centers,sites,val)
    np.savez_compressed(output/'predictions.npz',**arrays)
    torch.save({'model_state':model.state_dict(),'arm':args.arm,'seed':args.seed,'epochs':12},output/'checkpoint.pt')
    (output/'complete.json').write_text(json.dumps({'fold':args.fold,'arm':args.arm,'seed':args.seed,'reference':reference,'validation':report,'history':history,'test_labels_evaluated':False,'site_labels_used_to_choose_validation_images':False,'regime':'offline full-feature retention, not prospective acquisition'},indent=2))


if __name__=='__main__':main()
