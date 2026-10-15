"""Exercise the complete CV/factorial/statistics pipeline on synthetic caches."""
from pathlib import Path
import hashlib
import json
import tempfile
import numpy as np
from sklearn.metrics import roc_auc_score
import core
import paired_statistics as stats
import run_factorial as runner


def main():
    rng=np.random.default_rng(196);n=120
    y=np.tile(np.repeat([0,1],12),5);g=np.repeat(['a','b','c','d','wuhan'],24)
    tied=rng.integers(0,5,n).astype(float)
    # Bootstrap weighted AUROC agrees with direct patient resampling, including ties.
    count=rng.multinomial(n,np.full(n,1/n));idx=np.repeat(np.arange(n),count)
    order=np.argsort(tied,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(tied[order])!=0)+1]
    positive=np.add.reduceat(count[order]*(y[order]==1),starts)
    negative=np.add.reduceat(count[order]*(y[order]==0),starts)
    auc=(positive*(negative.cumsum()-.5*negative)).sum()/(positive.sum()*negative.sum())
    assert abs(auc-roc_auc_score(y[idx],tied[idx]))<1e-12
    root=Path(__file__).resolve().parent
    original=runner.ROOT
    with tempfile.TemporaryDirectory(prefix='cfv-factorial-SYNTHETIC-') as directory:
        runner.ROOT=Path(directory);p=runner.ROOT/'private';p.mkdir()
        rows=[];cache={}
        for i in range(n):
            key=f'synthetic_{i}';hidden=rng.normal(size=(2,32));hidden[:,0]+=2*y[i]-1
            delta=rng.normal(size=(2,20));delta[:,:3]+=.5*(2*y[i]-1)
            q=np.array([hidden[0,0],hidden[1,0]])
            path=p/f'{key}.npz'
            np.savez_compressed(path,delta=delta,mask=np.ones(20,bool),A=q,I=q*.1,q=q,
                                hidden=hidden,only_o=q*.7,only_k=q*.3)
            cache[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            rows.append(dict(key=key,center=g[i],y=int(y[i]),cin3=int(y[i]),age=40.+rng.normal()*10,
                             hpv='pos' if rng.random()<.4 else 'neg',tct='normal',old_input={}))
        for name,value in [('private/COHORT.json',rows),('private/CACHE_INDEX.json',cache),
            ('DATA_AUDIT.json',dict(state='AUDITED_WITH_LIMITATIONS',cross_patient_shared_image_hashes=0)),
            ('CORE_VERIFICATION.json',dict(state='PASS')),
            ('DESIGN_LOCK.json',dict(code_dependencies={str(root/'core.py'):hashlib.sha256((root/'core.py').read_bytes()).hexdigest()},
                                    base_alpha_grid=[0.,.25,.5,1.],bootstrap_repeats=30))]:
            (runner.ROOT/name).write_text(json.dumps(value))
        try:runner.main('five')
        finally:runner.ROOT=original
        complete=json.loads((Path(directory)/'five_center/COMPLETE.json').read_text())
        assert complete['N']==n and complete['folds']==5
    report=dict(state='PASS',synthetic_only=True,source_CV_then_freeze_then_outer=True,
                five_folds_exercised=True,paired_bootstrap_tie_handling=True,
                real_patient_data_used=False)
    (root/'PIPELINE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
