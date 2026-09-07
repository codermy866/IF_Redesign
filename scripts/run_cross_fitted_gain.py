"""Fresh nested diagnostic teachers -> matched OOF/seen gains -> frozen evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
import train_ices_backbone as tr
from cervix_cogalign.cesl import clinical_matrix, hpv_group
from cervix_cogalign.ices import random_observation_mask
from run_csf_memory import features as observed_features
from run_cvre_feasibility import probability, metrics, kl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cross_fitted_gain_v1'
MODES=('gain_oof','gain_seen','lesion_score','random','attention','entropy','kl','uncertainty')


def inner_folds(indices,centers,y,seed):
    rng=np.random.default_rng(seed);assignment={}
    for h in sorted(set(centers[indices])):
        for label in (0,1):
            group=indices[(centers[indices]==h)&(y[indices]==label)]
            offset=int(rng.integers(5))
            for j,index in enumerate(rng.permutation(group)):assignment[int(index)]=(j+offset)%5
    return [np.array(sorted(i for i,k in assignment.items()if k==fold),dtype=int)for fold in range(5)]


def calibration_split(indices,centers,y,seed):
    rng=np.random.default_rng(seed);cal=[]
    for h in sorted(set(centers[indices])):
        for label in (0,1):
            group=indices[(centers[indices]==h)&(y[indices]==label)]
            if len(group)>1:cal.extend(rng.permutation(group)[:max(1,int(.15*len(group)))])
    cal=np.array(sorted(cal),dtype=int)
    return np.setdiff1d(indices,cal),cal


def ce(y,p):
    p=np.clip(p,1e-6,1-1e-6)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def action_features(state,c,mask,p,slots,projection,candidates):
    observed=observed_features(state,c,mask,p,slots,projection)
    candidate=(candidates.float()@projection).cpu().numpy()
    return np.concatenate((observed,candidate),axis=1)


def train_teacher(config,data,x,av,fit,cal,gain,path,seed,device):
    path.mkdir(parents=True,exist_ok=True);base=data['payload'];centers=data['centres'];y=base['labels'].float().to(device)
    assert not set(fit)&set(cal) and not set(fit)&set(gain) and not set(cal)&set(gain)
    clinical,scaling=clinical_matrix(base['ages'],base['hpvs'],base['tcts'],fit)
    clinical=torch.from_numpy(clinical).to(device);tr.set_seed(seed)
    model=tr.create_model(config,data,device)
    if (path/'complete.json').exists():
        previous=json.loads((path/'complete.json').read_text())
        assert previous['fit_indices']==fit.tolist() and previous['cal_indices']==cal.tolist() and previous['gain_indices']==gain.tolist()
        model.load_state_dict(torch.load(path/'checkpoint.pt',map_location=device,weights_only=False)['model_state'])
        return model.eval(),clinical
    weight=(len(fit)-y[fit].sum())/y[fit].sum().clamp_min(1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    best=-np.inf;stale=0;history=[]
    for epoch in range(50):
        model.train();order=np.random.default_rng(seed+epoch).permutation(fit);losses=[]
        for start in range(0,len(order),48):
            ix=order[start:start+48];xx=x[ix];cc=clinical[ix];mask=av[ix];yy=y[ix]
            pred=model(xx,cc,mask)['logit'];aug=model(xx,cc,random_observation_mask(mask,.75))['logit']
            loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,yy,pos_weight=weight)+.5*torch.nn.functional.binary_cross_entropy_with_logits(aug,yy,pos_weight=weight)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            p=torch.cat([probability(model,x[ix],clinical[ix],av[ix])for ix in np.array_split(cal,max(1,int(np.ceil(len(cal)/48))))]).cpu().numpy()
        score=tr.macro_auprc(base['labels'][cal].numpy(),p,centers[cal]);history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'internal_cal_ap':score})
        if score>best+1e-6:
            best=score;stale=0;torch.save({'model_state':model.state_dict(),'clinical_scaling':scaling,'epoch':epoch+1},path/'checkpoint.pt')
        else:stale+=1
        if epoch%5==0:print(json.dumps({'teacher':path.name,'epoch':epoch+1,'cal_ap':score}),flush=True)
        if stale>=10:break
    metadata={'fit_indices':fit.tolist(),'cal_indices':cal.tolist(),'gain_indices':gain.tolist(),'clinical_scaling':scaling,
              'seed':seed,'initialization':'fresh_random_ICES; frozen_ImageNet_features','history':history,'best_internal_cal_ap':best}
    (path/'complete.json').write_text(json.dumps(metadata,indent=2))
    model.load_state_dict(torch.load(path/'checkpoint.pt',map_location=device,weights_only=False)['model_state'])
    return model.eval(),clinical


@torch.inference_mode()
def make_targets(teacher,teacher_c,x,av,selector_c,indices,y,site_labels,projection,seed):
    design=[];gains=[];lesions=[];rows=[]
    for index in indices:
        mask=av[index:index+1].clone();mask[:,25:]=False;state=x[index:index+1]*mask.unsqueeze(-1)
        rng=np.random.default_rng(seed+int(index)*100)
        for step in range(6):
            slots=torch.where(av[index]&~mask[0])[0];p0=probability(teacher,state,teacher_c[index:index+1],mask)
            z=action_features(state,selector_c[index:index+1],mask,p0,slots,projection,x[index,slots])
            n=len(slots);xx=state.repeat(n,1,1);mm=mask.repeat(n,1);rr=torch.arange(n,device=x.device)
            xx[rr,slots]=x[index,slots];mm[rr,slots]=True
            after=probability(teacher,xx,teacher_c[index:index+1].repeat(n,1),mm).cpu().numpy()
            gain=ce(y[index],float(p0[0]))-ce(y[index],after)
            design.append(z);gains.append(gain);lesions.append(site_labels[index,slots.cpu().numpy()-25]);rows.extend([(int(index),step,int(s))for s in slots.cpu().numpy()])
            chosen=int(slots[rng.integers(n)]);mask[0,chosen]=True;state[0,chosen]=x[index,chosen]
    return np.concatenate(design),np.concatenate(gains),np.concatenate(lesions),np.array(rows,dtype=int)


def fit_value(z,target,centers):
    scaler=StandardScaler().fit(z);weights=np.zeros(len(z))
    for h in np.unique(centers):weights[centers==h]=len(z)/(len(np.unique(centers))*sum(centers==h))
    model=Ridge(alpha=10).fit(scaler.transform(z),target,sample_weight=weights)
    return scaler,model


@torch.inference_mode()
def evaluate(model,model_c,selector_c,x,av,indices,projection,selectors,seed):
    predictions={m:[]for m in MODES};actions={m:[]for m in MODES};full=[];audit=[]
    for n,index in enumerate(indices):
        aa=av[index:index+1];cc=model_c[index:index+1];xx=x[index:index+1]
        initial=model(xx,cc,aa);attention=initial['attention'][0,25:];full.append(float(initial['logit'].sigmoid()[0]))
        for mode in MODES:
            mask=aa.clone();mask[:,25:]=False;state=xx*mask.unsqueeze(-1);path=[];chosen_slots=[]
            rng=np.random.default_rng(seed+int(index)*100)
            for step in range(7):
                p0=probability(model,state,cc,mask);path.append(float(p0[0]))
                if step==6:break
                slots=torch.where(aa[0]&~mask[0])[0];width=len(slots);rr=torch.arange(width,device=x.device)
                candidates=state.repeat(width,1,1);mm=mask.repeat(width,1);candidates[rr,slots]=x[index,slots];mm[rr,slots]=True
                pp=probability(model,candidates,cc.repeat(width,1),mm)
                if mode in selectors:
                    z=action_features(state,selector_c[index:index+1],mask,p0,slots,projection,x[index,slots])
                    scaler,regressor=selectors[mode];score=torch.as_tensor(regressor.predict(scaler.transform(z)),device=x.device)
                elif mode=='attention':score=attention[slots-25]
                elif mode=='random':score=torch.as_tensor(rng.random(width),device=x.device)
                elif mode=='entropy':
                    p=pp.clamp(1e-6,1-1e-6);score=-p*p.log()-(1-p)*(1-p).log()
                elif mode=='kl':score=kl(pp,p0[0])
                else:
                    model.train();torch.manual_seed(seed+int(index)*100+step)
                    draws=torch.stack([probability(model,candidates,cc.repeat(width,1),mm)for _ in range(5)])
                    model.eval();score=draws.var(0,unbiased=False)
                slot=int(slots[int(score.argmax())]);chosen_slots.append(slot-25);mask[0,slot]=True;state[0,slot]=x[index,slot]
                if mode=='attention' and step==0:
                    audit.append({'index':int(index),'p0':float(p0[0]),'candidate_p':pp.cpu().tolist(),'attention':attention.cpu().tolist()})
            predictions[mode].append(path);actions[mode].append(chosen_slots)
        if n%25==0:print(json.dumps({'validation_done':n+1,'total':len(indices)}),flush=True)
    return {m:np.array(v)for m,v in predictions.items()},{m:np.array(v)for m,v in actions.items()},np.array(full),audit


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--fold',required=True);parser.add_argument('--seed',type=int,default=20260905)
    parser.add_argument('--restored-clinical',action='store_true');args=parser.parse_args()
    if args.restored_clinical:OUT=ROOT/'results/cross_fitted_gain_restored_v1'
    output=OUT/args.fold/f'seed_{args.seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():return
    torch.set_num_threads(2);device=torch.device('cuda');config=tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'));data=tr.load_fold_data(config,args.fold);base=data['payload']
    if args.restored_clinical:
        sidecar=np.load(ROOT/'results/clinical_source_recovery_v1/clinical_sidecar.npz',allow_pickle=False)
        np.testing.assert_array_equal(sidecar['ids'],np.array(base['ids']))
        base=dict(base,hpvs=sidecar['hpvs'].tolist(),tcts=sidecar['tcts'].tolist());data['payload']=base
        clinical,scaling=clinical_matrix(base['ages'],base['hpvs'],base['tcts'],data['source_indices']['train'])
        data['clinical']=torch.from_numpy(clinical);data['clinical_scaling']=scaling
    assert len(set(base['patient_ids']))==len(base['patient_ids']), 'Implement grouped splitting for repeat patients'
    source=np.array(data['source_indices']['train']);val=np.array(data['source_indices']['val']);outer=np.array(data['parts']['test']['index'])
    assert not set(source)&set(val) and not set(source)&set(outer)
    y=base['labels'].numpy();centers=data['centres'];x=base['cluster_features'].to(device);av=base['available'].bool().to(device)
    selector_c=data['clinical'].to(device);folds=inner_folds(source,centers,y,args.seed)
    annotations=np.load(ROOT/'results/oct_site_supervision_v1/site_labels.npz',allow_pickle=False);assert list(annotations['ids'])==list(base['ids']);sites=annotations['labels']
    projection=torch.as_tensor(np.random.default_rng(args.seed).normal(size=(2048,16))/np.sqrt(2048),dtype=torch.float32,device=device)
    teachers=[];memberships=[]
    for inner,held in enumerate(folds):
        fit,cal=calibration_split(np.setdiff1d(source,held),centers,y,args.seed+inner)
        teacher,clinical=train_teacher(config,data,x,av,fit,cal,held,output/f'teacher_{inner}',args.seed+inner,device)
        teachers.append((teacher,clinical));memberships.append({'fit':fit,'cal':cal,'gain':held})
    fit,cal=calibration_split(source,centers,y,args.seed+100)
    diagnostic,diagnostic_c=train_teacher(config,data,x,av,fit,cal,np.concatenate((val,outer)),output/'diagnostic_reference',args.seed+100,device)
    owner={int(i):k for k,held in enumerate(folds)for i in held}
    seen={int(i):next((k for k,m in enumerate(memberships)if i in set(m['fit'])),None)for i in source}
    eligible=np.array([i for i in source if seen[int(i)] is not None]);target_parts={k:[]for k in ('oof','seen')};provenance=[]
    for kind,mapping in [('oof',owner),('seen',seen)]:
        for k,(teacher,clinical) in enumerate(teachers):
            indices=np.array([i for i in eligible if mapping[int(i)]==k])
            if not len(indices):continue
            assert (not set(indices)&set(memberships[k]['fit']) and not set(indices)&set(memberships[k]['cal'])) if kind=='oof' else set(indices)<=set(memberships[k]['fit'])
            z,g,l,rows=make_targets(teacher,clinical,x,av,selector_c,indices,y,sites,projection,args.seed)
            target_parts[kind].append((z,g,l,rows));provenance.extend([{'kind':kind,'patient_index':int(i),'teacher':k}for i in indices])
    targets={}
    for kind,parts in target_parts.items():
        arrays=[np.concatenate([p[j]for p in parts])for j in range(4)];order=np.lexsort((arrays[3][:,2],arrays[3][:,1],arrays[3][:,0]));targets[kind]=[a[order]for a in arrays]
    np.testing.assert_array_equal(targets['oof'][3],targets['seen'][3])
    selectors={kind:fit_value(targets[tag][0],targets[tag][1],centers[targets[tag][3][:,0]])for kind,tag in [('gain_oof','oof'),('gain_seen','seen')]}
    z,g,l,rows=targets['oof'];known=l>=0;selectors['lesion_score']=fit_value(z[known],l[known],centers[rows[known,0]])
    joblib.dump({'selectors':selectors,'projection':projection.cpu().numpy()},output/'value_models.joblib')
    np.savez_compressed(output/'gain_targets.npz',rows=rows,oof_gain=targets['oof'][1],seen_gain=targets['seen'][1],site_labels=l)
    (output/'target_provenance.json').write_text(json.dumps(provenance))
    (output/'phase.json').write_text(json.dumps({'phase':'validation','teachers_complete':6,'target_action_rows':len(rows)}))
    predicted,actions,full,audit=evaluate(diagnostic,diagnostic_c,selector_c,x,av,val,projection,selectors,args.seed)
    yy=y[val];hh=centers[val];report={};arrays={'indices':val,'y':yy,'centers':hh.astype(str),'full':full,'site_labels':sites[val]}
    for mode in MODES:
        report[mode]={'budgets':[metrics(yy,predicted[mode][:,k],hh)for k in range(7)]}
        recall=[]
        for i,selected in zip(val,actions[mode]):
            if (sites[i]>=0).all() and sites[i].sum()>0:recall.append(float(sites[i,selected[:4]].sum()/sites[i].sum()))
        report[mode]['mean_case_site_recall_at4']=float(np.mean(recall)) if recall else None
        arrays[mode]=predicted[mode];arrays[mode+'_actions']=actions[mode]
    attention_audit=[]
    for a in audit:
        i=a['index'];gain=ce(y[i],a['p0'])-ce(y[i],np.array(a['candidate_p']));rho=spearmanr(a['attention'],gain).statistic
        entry={'index':i,'center':str(centers[i]),'hpv_group':hpv_group(base['hpvs'][i]),'label':int(y[i]),
               'attention_gain_spearman':float(rho) if np.isfinite(rho) else None,'mean_gain':float(gain.mean())}
        if (sites[i]>=0).all() and 0<sites[i].sum()<12:entry['positive_minus_negative_site_gain']=float(gain[sites[i]==1].mean()-gain[sites[i]==0].mean())
        attention_audit.append(entry)
    np.savez_compressed(output/'predictions.npz',**arrays)
    (output/'attention_audit.json').write_text(json.dumps(attention_audit,ensure_ascii=False,indent=2))
    (output/'complete.json').write_text(json.dumps({'fold':args.fold,'seed':args.seed,'policies':report,'full':metrics(yy,full,hh),
        'source_n':len(source),'gain_patient_n':len(eligible),'val_n':len(val),'gain_action_rows':len(rows),
        'mean_oof_gain':float(targets['oof'][1].mean()),'mean_seen_gain':float(targets['seen'][1].mean()),
        'test_labels_evaluated':False,'fresh_teachers':True,'restored_clinical':args.restored_clinical,
        'clinical_sidecar_sha256':hashlib.sha256((ROOT/'results/clinical_source_recovery_v1/clinical_sidecar.npz').read_bytes()).hexdigest() if args.restored_clinical else None,
        'protocol_sha256':hashlib.sha256((ROOT/'configs'/('CROSS_FITTED_GAIN_RECOVERED_INPUT.md' if args.restored_clinical else 'CROSS_FITTED_GAIN_PROTOCOL.md')).read_bytes()).hexdigest()},indent=2))
    print(json.dumps({'fold':args.fold,'status':'complete'}),flush=True)


if __name__=='__main__':main()
