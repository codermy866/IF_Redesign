"""Full native montage/context recomputation on the pre-existing200-patient subset."""
import runtime
from runtime import HERE,BASE,config,save,sha
import argparse,json,time,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
OLD=BASE/'196_controlled_cfv_jev_20261012'
def plan():
    import rime_dataset
    rows=rime_dataset.load_patients();lookup={r['key']:r for r in rows}
    subset=json.loads((BASE/'197_rime_jev_20261012/private/MECHANISM_PATIENTS.json').read_text())
    jobs=[]
    for case in subset:
        row=lookup[case['key']]
        for repeat in range(3):
            seed=int(hashlib.sha256(('mask201:'+str(repeat)+':'+row['key']).encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed);frames=np.stack([rng.permutation(10) for _ in range(12)]);sites=rng.permutation(12)
            for kind,budgets in [('frames',config()['frame_budgets']),('sites',config()['site_budgets'])]:
                for budget in budgets:
                    keep=np.zeros((12,10),bool)
                    if kind=='frames':
                        for s in range(12):keep[s,frames[s,:budget]]=True
                    else:keep[sites[:budget]]=True
                    changed={str(s):[t for t in [0,3,6,9] if not keep[s,t]] for s in range(12)}
                    changed={s:tt for s,tt in changed.items() if tt}
                    identity=dict(key=row['key'],changed_montage_frames=changed,remove_colposcopy=False)
                    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
                    jobs.append(dict(key=row['key'],center=row['center'],budget_type=kind,budget=budget,repeat=repeat,keep=keep.tolist(),inclusion_probability=case['inclusion_probability'],identity=identity,token=token))
    save(HERE/'private/NATIVE_ROBUSTNESS_QUEUE.json',jobs)
    save(HERE/'NATIVE_ROBUSTNESS_PLAN.json',dict(frozen_unix=time.time(),patients=len(subset),conditions=len(jobs),unique_native_inputs=len(set(j['token'] for j in jobs)),sampling='Existing Stage197 predeclared centre x CIN2 stratified40-per-centre subset; inverse inclusion weights retained',same_masks_as_fixed_anchor=True,three_repeats=True,independent_confirmation=False,source_subset_sha256=sha(BASE/'197_rime_jev_20261012/private/MECHANISM_PATIENTS.json')))

def images(item,changed):
    paths=[item['oct_images'].get(str(i)) for i in range(1,13)]+item['colpo_paths'][:8]
    paths +=[None]*(20-len(paths));out=[]
    for path in paths:
        if path:
            with Image.open(path) as im:out.append(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR))
        else:out.append(Image.new('RGB',(224,224),(128,128,128)))
    for s,tiles in changed.items():
        for t in tiles:
            at=[0,3,6,9].index(t)
            out[int(s)].paste((128,128,128),((at%2)*112,(at//2)*112,(at%2+1)*112,(at//2+1)*112))
    return out

def worker(split):
    sys.path.insert(0,str(OLD))
    from qwen27b.interventions import InterventionJEV
    start=time.monotonic();model=InterventionJEV(BASE/'190_autotrust_jev27b_20261011/model')
    manifest=json.loads((OLD/'private/LABEL_FREE_INPUTS.json').read_text());cache=json.loads((OLD/'private/CACHE_INDEX.json').read_text())
    jobs=json.loads((HERE/'private/NATIVE_ROBUSTNESS_QUEUE.json').read_text());unique={}
    for job in jobs:unique.setdefault(job['token'],job)
    selected=list(unique.values())[split::2];targetdir=HERE/'private/native_robustness';targetdir.mkdir(exist_ok=True)
    checked=set()
    for index,job in enumerate(selected):
        if time.time()>config()['deadline_unix']:raise TimeoutError('Original deadline')
        target=targetdir/(job['token']+'.npz')
        if target.exists():continue
        item=manifest[job['key']]
        if job['key'] not in checked:
            inp=model.inputs(images(item,{}));features=model.encode(inp)
            if not model.preflight_done:model.parity(inp,features)
            full=model.forward(inp,features if model.vision_cache else None,hidden=True)
            with np.load(cache[job['key']]['path']) as prior:
                q_error=float(abs(full['q'][0]-prior['q'][0]))
                hidden_error=float(abs(full['hidden'][0]-model.null['hidden'][0]-prior['hidden'][0]).max())
            if q_error>.001 or hidden_error>.001:
                save(HERE/f'NATIVE_GATE_FAILURE_{split}.json',dict(state='BLOCKED_NATIVE_REFERENCE_PARITY',q_error=q_error,hidden_error=hidden_error,expected_tolerance=.001,no_native_performance_claim=True))
                raise RuntimeError('Native source-input parity gate failed')
            checked.add(job['key'])
        inp=model.inputs(images(item,job['identity']['changed_montage_frames']))
        feature=model.encode(inp);out=model.forward(inp,feature if model.vision_cache else None,hidden=True)
        temp=target.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,hidden=out['hidden'][0]-model.null['hidden'][0],A=out['q'][0]-model.null['q'][0],identity=np.array(json.dumps(job['identity'],sort_keys=True)))
        temp.replace(target)
        save(HERE/f'NATIVE_ROBUSTNESS_WORKER_{split}.json',dict(state='RUNNING',completed=index+1,total=len(selected),pid=__import__('os').getpid(),seconds=time.monotonic()-start,labels_passed_to_model=False))
        print('NATIVE_COMPLETE',split,index+1,len(selected),flush=True)
    save(HERE/f'NATIVE_ROBUSTNESS_WORKER_{split}.json',dict(state='COMPLETE',completed=len(selected),total=len(selected),seconds=time.monotonic()-start,parity=model.parity_report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--split',type=int,choices=[0,1]);a=p.parse_args()
    if a.plan:plan()
    else:worker(a.split)
