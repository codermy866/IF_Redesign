"""Complete native site removals, reusable frozen vision and rerun language context."""
import runtime
from runtime import HERE,BASE,RECOVERY,save,config,verify
import sys,time,json,hashlib,argparse,os
import numpy as np
from PIL import Image
from rime_dataset import load_patients
sys.path.insert(0,str(BASE/'196_controlled_cfv_jev_20261012'))
from qwen27b.interventions import InterventionJEV

def token(key,site):
    identity=dict(key=key,changed_montage_frames={site:[0,3,6,9]},remove_colposcopy=False)
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),identity

def worker(split):
    verify();rows=load_patients()[split::2];out=HERE/'private/b5_native';out.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((BASE/'196_controlled_cfv_jev_20261012/private/LABEL_FREE_INPUTS.json').read_text())
    model=None;start=time.monotonic();new=0;reused=0
    for i,row in enumerate(rows):
        verify();missing=[]
        for site in range(12):
            name,identity=token(row['key'],site);path=out/(name+'.npz')
            legacy=RECOVERY/'private/deletion_native'/path.name
            if not path.exists() and legacy.exists():
                with np.load(legacy,allow_pickle=False) as f:
                    assert json.loads(str(f['identity']))==json.loads(json.dumps(identity,sort_keys=True))
                path.symlink_to(legacy);reused+=1
            if not path.exists():missing.append((site,path,identity))
        if missing:
            if model is None:model=InterventionJEV(BASE/'190_autotrust_jev27b_20261011/model')
            item=inputs[row['key']];paths=[item['oct_images'].get(str(s)) for s in range(1,13)]+item['colpo_paths'][:8];paths +=[None]*(20-len(paths))
            images=[]
            for path in paths:
                if path:
                    with Image.open(path) as im:images.append(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR))
                else:images.append(Image.new('RGB',(224,224),(128,128,128)))
            inp=model.inputs(images);vision=model.encode(inp)
            if not model.preflight_done:
                model.parity(inp,vision);save(HERE/f'B5_NATIVE_PARITY_{split}.json',model.parity_report)
            for site,path,identity in missing:
                verify();value=model.replaced(inp,vision,[site],hidden=True,cached=model.vision_cache)
                temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,hidden=value['hidden'][0]-model.null['hidden'][0],A=value['q'][0]-model.null['q'][0],identity=np.array(json.dumps(identity,sort_keys=True)))
                try:os.link(temp,path)
                except FileExistsError:pass
                finally:temp.unlink(missing_ok=True)
                new+=1
        save(HERE/f'B5_NATIVE_WORKER_{split}.json',dict(state='RUNNING',pid=os.getpid(),patients=i+1,total=len(rows),new_native_calls=new,reused_calls=reused,seconds=time.monotonic()-start,full_context_recomputed=True))
    save(HERE/f'B5_NATIVE_WORKER_{split}.json',dict(state='COMPLETE',pid=os.getpid(),patients=len(rows),total=len(rows),new_native_calls=new,reused_calls=reused,seconds=time.monotonic()-start,full_context_recomputed=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',type=int,required=True);worker(p.parse_args().split)
