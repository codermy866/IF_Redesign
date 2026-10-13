"""Small end-to-end source CV/resume checks, kept out of clinical tables."""
import runtime
from runtime import HERE,BASE,save,config
import ast,json,tempfile
from pathlib import Path
import numpy as np,torch,joblib
import core_heads,run_heads
from b5_values import shuffle,advantage

def main():
    torch.set_num_threads(1);rng=np.random.default_rng(91)
    rows=[dict(key=f'synthetic_{i}',center=f'centre{i//8}',y=i%2) for i in range(40)]
    y=np.array([r['y'] for r in rows]);g=np.array([r['center'] for r in rows])
    outer='centre0';tr=np.flatnonzero(g!=outer);valid=np.ones((40,12,10),bool)
    h=rng.uniform(-.4,.4,size=(40,12,10,8));b=rng.normal(size=(40,34));anchor=rng.normal(0,.1,size=40)
    original_anchor=ast.parse((BASE/'198_rime_jev_cv1_20261013/cv1_anchor.py').read_text())
    current_anchor=ast.parse((HERE/'anchor_adapter.py').read_text())
    old=next(n for n in original_anchor.body if isinstance(n,ast.FunctionDef) and n.name=='build_anchor')
    new=next(n for n in current_anchor.body if isinstance(n,ast.FunctionDef) and n.name=='build_anchor')
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    with tempfile.TemporaryDirectory(prefix='rime_final_execution_') as temp:
        dest=Path(temp);folder=dest/'private/fold_features'/outer;folder.mkdir(parents=True)
        cfg=config();cfg.update(attention_diversity_weight=.1,taus=[.3],lambdas=[.003],seeds=[17,29,43],max_epochs=2,patience=1,warmup_grid=[0,1],B2_variants=['geometry','local','conditional','local_capacity_matched','context_shuffled'])
        save(dest/'CONFIG.json',cfg)
        for name,fit in {'outer':tr,**{c:tr[g[tr]!=c] for c in sorted(set(g[tr]))}}.items():
            part=dict(h=h,valid=valid,sigma=.25,train=fit,anchor=dict(b=b,anchor=anchor,semantic_x=b[:,:26],initial_semantic_weight=np.zeros(26)))
            joblib.dump(part,folder/(name+'.joblib'))
        core_heads.DEST=dest;core_heads.load_patients=lambda:rows;core_heads.prepare_fold=lambda _:None
        core_heads.fit_fold(outer,'cpu')
        run_heads.HERE=dest;run_heads.verify=lambda:None;run_heads.config=lambda:cfg;run_heads.load_patients=lambda:rows
        run_heads.run_control(outer,'B1','cpu');run_heads.run_control(outer,'B2','cpu')
        from factorial_inputs import tensor_inputs,shuffled_scoring_context
        from rime_dataset import atoms
        from robustness import changed
        from if_redesign.rime.readout import RIMEHead
        for name,(pool,readout) in core_heads.METHODS.items():
            ti=tensor_inputs(part,np.arange(4),name,.3)
            reference=tensor_inputs(part,np.arange(4),'C_MEASURE_MOMENT' if readout=='moment' else 'D_RIME_LAPLACE',.3)
            assert torch.equal(ti['h'],reference['h']) and torch.equal(ti['valid'],reference['valid']) and torch.equal(ti['coordinate_scale'],reference['coordinate_scale'])
            model=RIMEHead(34,.3,pool,readout).double()
            with torch.no_grad():model.beta.fill_(.05)
            original={k:v for k,v in ti.items() if k!='indices'}
            copied=np.pad(h[:4],((0,0),(0,0),(0,2),(0,0)));copied[:,0,-2:]=h[:4,0,:1]
            mask=np.pad(valid[:4],((0,0),(0,0),(0,2)));mask[:,0,-2:]=True
            altered=changed(original,part,copied,mask,name)
            with torch.no_grad():assert torch.allclose(model(**original),model(**altered),atol=1e-10,rtol=0)
        ti=tensor_inputs(part,tr,'D_RIME_LAPLACE',.3);vi=tensor_inputs(part,np.arange(8),'D_RIME_LAPLACE',.3)
        st,sv,audit=shuffled_scoring_context(ti,vi,17)
        assert torch.equal(st['anchor'],ti['anchor']) and torch.equal(sv['anchor'],vi['anchor'])
        assert set(audit['validation_donors']).issubset(set(tr)) and not np.any(audit['train_donors']==tr)
        core_heads.fit_fold(outer,'cpu',methods={'ATTENTION_DIVERSITY_LAPLACE':('attention','laplace'),'MEASURE_LSE56':('measure','lse')})
        files=sorted((dest/'private/head_runs'/outer).glob('*.npz'));assert len(files)==11
        for path in files:
            with np.load(path) as values:
                assert values['seed_logits'].shape==(3,8) and np.isfinite(values['seed_logits']).all()
                assert np.array_equal(values['source_indices'],tr) and len(values['source_y'])==len(tr)
        before={p.name:runtime.sha(p) for p in files}
        core_heads.fit_fold(outer,'cpu');run_heads.run_control(outer,'B1','cpu');run_heads.run_control(outer,'B2','cpu')
        assert before=={p.name:runtime.sha(p) for p in files}
    values=np.linspace(-.8,.8,40*12).reshape(40,12);shuffled,records=shuffle(values,tr,g,17)
    assert np.array_equal(shuffled[g==outer],values[g==outer])
    for record in records:
        assert not np.intersect1d(record['recipient'],np.flatnonzero(g==outer)).size
        assert not np.any(record['recipient']==record['donor'])
        assert np.array_equal(shuffled[record['recipient'],record['site']],values[record['donor'],record['site']])
    p=np.tile(np.linspace(.1,.9,13),(40,1));a=advantage(p,y)
    assert np.all(a<=1) and np.all(a>=-1) and np.any(a<0) and np.any(a>0)
    save(HERE/'EXECUTION_TEST_GATE.json',dict(state='PASS',scope='synthetic execution only; never clinical outcomes',
        source_cv_nested_indices=True,all_three_seeds=True,all_four_core_cells=True,B1_B2_runs=True,
        resume_preserves_complete_predictions=True,shuffle_excludes_outer_and_has_no_self_donors=True,
        common_dedup_inputs_and_copy_invariance_in_all_four_cells=True,context_shuffle_keeps_anchor_and_uses_source_donors=True,equal56_LSE_and_attention_diversity_controls=True,independent_anchor_adapter_AST_matches_audited_CV1_index_fix=True))

if __name__=='__main__':main()
