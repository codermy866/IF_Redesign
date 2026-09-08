"""Report genuine completed predictions only, paired patient bootstrap and CIs."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return dict(AUROC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                AUPRC=float(average_precision_score(y,p)) if y.sum() else None,
                F1=float(f1_score(y,p>=.5,zero_division=0)))


def parsed_chain(text):
    try:
        start=text.index("{"); value=json.JSONDecoder().raw_decode(text[start:])[0]
        if not isinstance(value.get("evidence"),list):return None
        return value
    except (ValueError,TypeError,AttributeError):return None


def grounded_fraction(text, observed):
    chain=parsed_chain(text)
    if chain is None:return None
    sites=[]
    for item in chain["evidence"]:
        try:sites.append(int(item["site"]))
        except (KeyError,ValueError,TypeError):return None
    return float(np.mean([s in observed for s in sites])) if sites else None


def interval(values):
    valid=[x for x in values if x is not None and np.isfinite(x)]
    return np.quantile(valid,[.025,.975]).tolist() if valid else None


def bootstrap(y,p,centers,draws,seed=20260907,other=None):
    rng=np.random.default_rng(seed);groups=[np.where(centers==h)[0] for h in np.unique(centers)]
    values=defaultdict(list)
    for _ in range(draws):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        score=metrics(y[ix],p[ix]);control=metrics(y[ix],other[ix]) if other is not None else None
        for k,v in score.items():
            values[k].append(v if control is None else (v-control[k] if v is not None and control[k] is not None else None))
    return {k:dict(CI95=interval(v),valid_draws=sum(x is not None for x in v)) for k,v in values.items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/dea_v1.json");args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());out=ROOT/cfg["output"]
    records=[];pooled=defaultdict(lambda:defaultdict(list))
    for path in sorted((out/"runs").glob("*/*/seed_*/*/predictions_*.jsonl")):
        if "smoke" in path.name:continue
        split=path.stem.removeprefix("predictions_");arm=path.parent.name;seed=int(path.parents[1].name.split("_")[1]);fold=path.parents[2].name;backbone=path.parents[3].name
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if not (path.parent/f"evaluation_{split}.json").exists():continue
        y=np.array([r["label"]for r in rows]);p=np.array([r["probability"]for r in rows]);centers=np.array([r["center"]for r in rows])
        report=dict(backbone=backbone,fold=fold,seed=seed,arm=arm,split=split,n=len(rows),metrics=metrics(y,p),
                    per_center={h:metrics(y[centers==h],p[centers==h])for h in np.unique(centers)},
                    uncertainty=bootstrap(y,p,centers,cfg["bootstrap_replicates"]))
        grounding=[grounded_fraction(r["response"],r["sites"])for r in rows]
        report["format_valid_fraction"]=float(np.mean([parsed_chain(r["response"])is not None for r in rows]))
        report["citation_valid_fraction"]=float(np.mean([x for x in grounding if x is not None])) if any(x is not None for x in grounding) else None
        report["citation_valid_n"]=sum(x is not None for x in grounding)
        teacher=out/"teacher_audits"/fold/f"seed_{seed}.npz"
        if teacher.exists():
            cache=np.load(teacher);lookup={int(i):g for i,g in zip(cache["indices"],cache["true_gain"])}
            correlations=[]
            for r in rows:
                if r["index"] in lookup:
                    g=lookup[r["index"]][np.array(r["sites"])-1];scores=r["importance_scores"]
                    if np.ptp(g)>0 and np.ptp(scores)>0:correlations.append(float(spearmanr(g,scores).statistic))
            report["evidence_ranking_spearman"]=float(np.mean(correlations)) if correlations else None
            report["ranking_valid_n"]=len(correlations)
        effects={}
        for intervention in ("remove_high","remove_low","replace_high_normal"):
            subset=[r for r in rows if intervention in r["counterfactuals"]]
            if not subset:continue
            original=np.array([r["probability"]for r in subset]);cf=np.array([r["counterfactuals"][intervention]["probability"]for r in subset])
            valid=[]
            for r in subset:
                c=r["counterfactuals"][intervention];observed=set(r["sites"])-{c["removed_site"]}
                if "donor_site" in c:observed.add(c["donor_site"])
                valid.append(grounded_fraction(c["response"],observed))
            effects[intervention]=dict(n=len(subset),prediction_consistency=float(np.mean((original>=.5)==(cf>=.5))),
                absolute_probability_change=float(np.mean(abs(original-cf))),
                citation_valid_fraction=float(np.mean([v for v in valid if v is not None])) if any(v is not None for v in valid) else None,
                inherited_patient_label_metrics=metrics([r["label"]for r in subset],cf),
                counterfactual_pathology_known=False)
        report["counterfactuals"]=effects;records.append(report)
        # Test rows appear in exactly one outer fold; ensemble before resampling.
        if split=="test":
            for r in rows:pooled[backbone,arm][r["id"]].append((seed,r))
    ensembles={};vectors={}
    for (backbone,arm),patients in pooled.items():
        complete={k:v for k,v in patients.items() if {s for s,r in v}==set(cfg["seeds"]) and len(v)==len(cfg["seeds"])}
        if not complete:continue
        ids=sorted(complete);y=np.array([complete[k][0][1]["label"]for k in ids]);p=np.array([np.mean([r["probability"]for s,r in complete[k]])for k in ids]);h=np.array([complete[k][0][1]["center"]for k in ids])
        key=f"{backbone}/{arm}";vectors[key]=(ids,y,p,h)
        ensembles[key]=dict(n=len(ids),metrics=metrics(y,p),uncertainty=bootstrap(y,p,h,cfg["bootstrap_replicates"]))
    comparisons={}
    for backbone in cfg["backbones"]:
        for a,b in (("alignment","sft"),("alignment_dpo","alignment"),("alignment","alignment_no_rank")):
            ka,kb=f"{backbone}/{a}",f"{backbone}/{b}"
            if ka not in vectors or kb not in vectors:continue
            ia,ya,pa,ha=vectors[ka];ib,yb,pb,hb=vectors[kb]
            if ia!=ib:continue
            np.testing.assert_array_equal(ya,yb)
            delta={m:(ensembles[ka]["metrics"][m]-ensembles[kb]["metrics"][m]) if ensembles[ka]["metrics"][m] is not None and ensembles[kb]["metrics"][m] is not None else None for m in ("AUROC","AUPRC","F1")}
            comparisons[f"{ka} minus {b}"]=dict(delta=delta,paired_patient_bootstrap=bootstrap(ya,pa,ha,cfg["bootstrap_replicates"],other=pb))
    out.mkdir(parents=True,exist_ok=True)
    result=dict(status="no_completed_evaluations" if not records else "results_available",completed_evaluations=len(records),runs=records,
                three_seed_test_ensembles=ensembles,paired_comparisons=comparisons,
                limitations=["Citation validity is not morphological correctness or causal faithfulness",
                             "Counterfactual images have no new pathology labels; inherited-label metrics measure robustness only",
                             "Source folds overlap; do not treat fold means as independent patients"])
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(status=result["status"],completed_evaluations=len(records))))


if __name__=="__main__":main()
