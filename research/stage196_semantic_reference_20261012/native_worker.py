"""Missing-patient inference only; one pinned frozen model per assigned GPU."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.umask(0o077)
from pathlib import Path
import argparse
import json
import hashlib
import time
import numpy as np
from PIL import Image
from qwen27b.interventions import InterventionJEV

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def views(item):
    paths=[item['oct_images'].get(str(i)) for i in range(1,13)]+item['colpo_paths'][:8]
    paths.extend([None]*(20-len(paths)));result=[]
    for path in paths:
        if path:
            with Image.open(path) as im:result.append(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR))
        else:result.append(Image.new('RGB',(224,224),(128,128,128)))
    return result


def main(split,mechanism=False):
    manifest=json.loads((ROOT/'private/LABEL_FREE_INPUTS.json').read_text())
    assert all(set(row)=={'oct_images','colpo_paths','mask'} for row in manifest.values())
    queue=json.loads((ROOT/('private/EXACT_DELETION_QUEUE.json' if mechanism else 'private/MISSING_FROZEN_FEATURES.json')).read_text())[split::2]
    model=InterventionJEV(BASE/'190_autotrust_jev27b_20261011/model',device=0)
    out=ROOT/('private/exact_deletions' if mechanism else 'private/frozen_cache');out.mkdir(exist_ok=True)
    started=time.monotonic()
    for i,job in enumerate(queue):
        key=job['key'] if mechanism else job
        cache_identity=json.dumps(dict(key=key,slots=job['slots']),sort_keys=True) if mechanism else key
        target=out/(hashlib.sha256(cache_identity.encode()).hexdigest()[:24]+'.npz')
        if not target.exists():
            item=manifest[key];patient_views=views(item);mask=np.asarray(item['mask'],bool)
            if mechanism:
                mask=mask.copy()
                for slot in job['slots']:
                    patient_views[slot]=Image.new('RGB',(224,224),(128,128,128));mask[slot]=False
            row=model.extract(patient_views,mask)
            row.update(model_revision=np.array('fee8c1d19d60216f8e9bcbb348141c0aa87ee2cc'),
                       input_manifest_sha256=np.array(hashlib.sha256((ROOT/'private/LABEL_FREE_INPUTS.json').read_bytes()).hexdigest()))
            tmp=target.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**row)
            tmp.replace(target)
        save(ROOT/f'NATIVE_WORKER_{split}.json',dict(state='EXACT_DELETION' if mechanism else 'INFER_FROZEN_MISSING',pid=os.getpid(),completed=i+1,total=len(queue),seconds=time.monotonic()-started))
        print('FROZEN_FEATURE',split,i+1,len(queue),flush=True)
    save(ROOT/f'NATIVE_WORKER_{split}.json',dict(state='COMPLETE',pid=os.getpid(),completed=len(queue),total=len(queue),parity=model.parity_report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',type=int,choices=[0,1],required=True)
    p.add_argument('--mechanism',action='store_true');args=p.parse_args()
    main(args.split,args.mechanism)
