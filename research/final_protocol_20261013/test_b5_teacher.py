"""Synthetic exclusion/cache/masking test; not evidence of value learnability."""
import runtime
from runtime import HERE,save
import tempfile
from pathlib import Path
import numpy as np
import b5_values
import run_b5
import joblib

def main():
    rng=np.random.default_rng(204);n=80
    rows=[dict(key=f'synthetic_{i}',center=['one','two','three','four','five'][i//16],y=i%2,age=20+i%40,hpv='positive' if i%3 else 'negative',tct='normal' if i%4 else 'ascus') for i in range(n)]
    y=np.tile([0,1],n//2);groups=np.repeat(['one','two','three','four','five'],16)
    full=rng.normal(0,.05,(n,5120));hidden=full[:,None,:]+rng.normal(0,.005,(n,13,5120));hidden[:,0]=full
    frames=rng.normal(0,.03,(n,12,10,64)).astype('float32');valid=np.ones((n,12,10),bool)
    a=dict(rows=rows,y=y.copy(),groups=groups,hidden=hidden,native=rng.normal(0,.1,(n,13)),frame_hidden=frames,valid=valid)
    source=np.arange(16,64);held=np.r_[np.arange(16),np.arange(64,80)]
    current=b5_values.HERE
    with tempfile.TemporaryDirectory(prefix='rime_b5_teacher_test_') as temp:
        destination=Path(temp);b5_values.HERE=destination
        (destination/'CONFIG.json').write_bytes((current/'CONFIG.json').read_bytes())
        (destination/'INPUT_LOCK.json').write_bytes((current/'INPUT_LOCK.json').read_bytes())
        teacher=b5_values.teacher(a,source);assert np.array_equal(teacher['train'],source)
        baseline=teacher['prediction'].copy();a['y'][held]=1-a['y'][held]
        reused=b5_values.teacher(a,source);assert np.array_equal(baseline,reused['prediction'])
        a['y'][held]=1-a['y'][held]
        u=b5_values.targets(a,source,destination/'audit.json')
        assert np.max(abs(u[source]))<=1 and np.count_nonzero(u[held])==0
        import json
        audit=json.loads((destination/'audit.json').read_text())
        for fold in audit['audit']:
            assert not set(fold['teacher_train'])&set(fold['recipients'])
            assert set(fold['teacher_train']).issubset(source) and set(fold['recipients']).issubset(source)
            assert len(fold['teacher_train'])>=27
        # Entire auxiliary/source-selection/refit workflow and resumed target
        # diagnostics, with all three seeds and no real patient tables.
        cfg=runtime.config();cfg.update(max_epochs=2,patience=1,B5_aux_weights=[.1],B5_teacher_epochs=13)
        (destination/'CONFIG.json').write_text(json.dumps(cfg,indent=2)+'\n')
        runtime.HERE=destination;run_b5.HERE=destination;run_b5.verify=lambda:None
        outer='one';tr=np.flatnonzero(groups!=outer)
        folder=destination/'private/head_runs'/outer;folder.mkdir(parents=True)
        save(folder/'D_RIME_LAPLACE_SOURCE_SELECTION_LOCK.json',dict(selected_tau=.3,selected_lambda=.003))
        features=destination/'private/fold_features'/outer;features.mkdir(parents=True)
        h=rng.uniform(-.4,.4,(n,12,10,8));b=rng.normal(size=(n,34));anchor=rng.normal(0,.1,n)
        for name,fit in {'outer':tr,**{c:tr[groups[tr]!=c] for c in sorted(set(groups[tr]))}}.items():
            joblib.dump(dict(h=h,valid=valid,sigma=.25,train=fit,anchor=dict(b=b,anchor=anchor,semantic_x=b[:,:26],initial_semantic_weight=np.zeros(26))),features/(name+'.joblib'))
        run_b5.fit_fold(a,outer,source_only=True)
        assert not list(folder.glob('*.npz'))
        # Explicit numerical refit test only. Real controller requires all
        # five source gates before this phase can be admitted.
        run_b5.fit_fold(a,outer,source_only=False)
        prediction_files=list(folder.glob('*.npz'));assert len(prediction_files)==4
        hashes={p.name:runtime.sha(p) for p in prediction_files};run_b5.fit_fold(a,outer,source_only=False)
        assert hashes=={p.name:runtime.sha(p) for p in prediction_files}
        for path in prediction_files:
            with np.load(path) as result:
                assert result['seed_logits'].shape==(3,16) and np.isfinite(result['seed_logits']).all()
                assert np.array_equal(result['source_indices'],tr)
    b5_values.HERE=current;runtime.HERE=current
    save(HERE/'B5_TEACHER_TEST_GATE.json',dict(state='PASS',scope='synthetic implementation/exclusion only',
        crossfit_patient_exclusion=True,held_label_perturbation_leaves_teacher_prediction_identical=True,
        all13_condition_inputs_used=True,source_only_preprocess_and_teacher_cache=True,
        auxiliary_source_selection_refit_and_resume=True,all_three_seeds_retained=True,
        real_full_native_target_extraction_complete=False,conditional_value_learnability_supported=False))

if __name__=='__main__':main()
