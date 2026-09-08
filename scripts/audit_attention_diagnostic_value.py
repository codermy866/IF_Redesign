"""Phase-1 audit of original frozen attention; no refitting or label-based selection."""
import json
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from run_cross_fitted_gain import ROOT,OUT,ce
import train_ices_backbone as tr
from cervix_cogalign.cesl import hpv_group
from run_cvre_feasibility import probability


@torch.inference_mode()
def main():
    torch.set_num_threads(2);device=torch.device('cpu');OUT.mkdir(parents=True,exist_ok=True)
    config=tr.read_config(str(ROOT/'configs/ices_v1_exploratory.json'));rows=[]
    annotations=np.load(ROOT/'results/oct_site_supervision_v1/site_labels.npz',allow_pickle=False)
    for fold in ('shiyan','enshi','wuhan','jingzhou','xiangyang'):
        data=tr.load_fold_data(config,fold);base=data['payload'];assert list(annotations['ids'])==list(base['ids'])
        model=tr.create_model(config,data,device).eval();cp=ROOT/'results/ices_v1/runs/cluster_m1_m2'/fold/'seed_20260905/checkpoint.pt'
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model_state'])
        for index in data['source_indices']['val']:
            x=base['cluster_features'][index:index+1];c=data['clinical'][index:index+1];av=base['available'][index:index+1].bool()
            attention=model(x,c,av)['attention'][0,25:].numpy();mask=av.clone();mask[:,25:]=False
            p0=float(probability(model,x*mask.unsqueeze(-1),c,mask)[0]);xx=(x*mask.unsqueeze(-1)).repeat(12,1,1);mm=mask.repeat(12,1)
            rr=torch.arange(12);slots=rr+25;xx[rr,slots]=x[0,slots];mm[rr,slots]=True
            p=probability(model,xx,c.repeat(12,1),mm).numpy();y=int(base['labels'][index]);gain=ce(y,p0)-ce(y,p)
            selected=np.argsort(-attention)[:4];labels=annotations['labels'][index];rho=spearmanr(attention,gain).statistic
            row={'fold':fold,'index':int(index),'label':y,'center':str(data['centres'][index]),'hpv_group':hpv_group(base['hpvs'][index]),
                 'attention_gain_rho':float(rho) if np.isfinite(rho) else None,'selected_minus_all_gain':float(gain[selected].mean()-gain.mean()),
                 'mean_candidate_gain':float(gain.mean())}
            if (labels>=0).all():
                row['selected_positive_site_fraction']=float(labels[selected].mean());row['all_positive_site_fraction']=float(labels.mean())
                if 0<labels.sum()<12:
                    row['within_case_site_ap']=float(average_precision_score(labels,attention))
                    row['positive_minus_negative_gain']=float(gain[labels==1].mean()-gain[labels==0].mean())
            rows.append(row)
        print(json.dumps({'attention_audit_fold':fold,'status':'complete'}),flush=True)
    fields=('attention_gain_rho','selected_minus_all_gain','selected_positive_site_fraction','all_positive_site_fraction','positive_minus_negative_gain','within_case_site_ap')
    groups={}
    for key in ('fold','label','hpv_group','center'):
        groups[key]={}
        for value in sorted({str(r[key])for r in rows}):
            part=[r for r in rows if str(r[key])==value];stats={'appearances':len(part),'unique_patients':len({r['index']for r in part})}
            for field in fields:
                # Average repeated appearances within patient before overall mean.
                by_patient={i:[r[field]for r in part if r['index']==i and r.get(field) is not None]for i in {r['index']for r in part}}
                values=[np.mean(v)for v in by_patient.values()if v];stats[field]=float(np.mean(values)) if values else None
            groups[key][value]=stats
    (OUT/'phase1_attention_rows.json').write_text(json.dumps(rows,ensure_ascii=False))
    (OUT/'phase1_attention_summary.json').write_text(json.dumps({'groups':groups,'test_labels_evaluated':False,
         'clinical_claim':'OCT reread sites are not biopsy-confirmed pathology regions; model gain is not clinical causal effect; HPV grouping combines16/18'},ensure_ascii=False,indent=2))
    print(json.dumps(groups['label'],indent=2),flush=True)


if __name__=='__main__':main()
