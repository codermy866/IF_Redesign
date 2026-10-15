"""Independent single-frame JEV hidden contrasts, with a fixed gray template.

Labels are deliberately absent from this worker's inputs. The existing frozen
JEV loader checks released adapter tensor equality before every process runs.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2';os.umask(0o077)
import sys, argparse, json, hashlib, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'03_new_experiment_design/196_controlled_cfv_jev_20261012'
DEST=ROOT/'03_new_experiment_design/197_rime_jev_20261012'
sys.path.insert(0,str(OLD))
from qwen27b.native import NativeJEV
from qwen27b.adapter import raw_oct
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
class FrameJEV(NativeJEV):
    def __init__(self,bundle):
        super().__init__(bundle)
        from qwen27b.adapter import MedicalQwen
        self.template=MedicalQwen.inputs(self,[Image.new('RGB',(224,224),(128,128,128)) for _ in range(20)],'reread')
        with self.torch.inference_mode():
            self.gray_features=tuple(self.model.get_base_model().get_image_features(self.template['pixel_values'],self.template['image_grid_thw']).pooler_output)
        self.null=self.forward(self.gray_features)
    def forward(self,features,direct=None):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        kwargs={k:v for k,v in self.template.items() if k!='pixel_values'}
        if direct is None:kwargs['mm_encoder_outputs']={'image':BaseModelOutputWithPooling(pooler_output=features)}
        else:kwargs['pixel_values']=direct
        with self.torch.inference_mode():
            out=self.model(**kwargs,use_cache=False,output_hidden_states=True,logits_to_keep=1)
            h=out.hidden_states[-1][0,-1].float().cpu().numpy()
            logits=out.logits[0,-1,self.ids].float().cpu().numpy()
        if not np.isfinite(h).all():raise RuntimeError('Nonfinite hidden state')
        return h,logits
    def extract(self,path,parity=False):
        image=raw_oct(path)
        pixel=self.processor.image_processor(images=[image],return_tensors='pt')
        grid=pixel['image_grid_thw'].to(self.device)
        assert grid.tolist()==[[1,14,14]]
        values=pixel['pixel_values'].to(self.device)
        # Keep the canonical 20-image encoder batch. Single-image encoder batches
        # can round differently in BF16, even though the vision encoder has no
        # cross-image attention. Do not silently loosen the parity requirement.
        direct=self.template['pixel_values'].clone();direct[:196]=values
        with self.torch.inference_mode():
            feat=tuple(self.model.get_base_model().get_image_features(direct,self.template['image_grid_thw']).pooler_output)
        h,q=self.forward(feat);difference=h-self.null[0]
        report=None
        if parity:
            dh,dq=self.forward(None,direct)
            report=dict(hidden_max_error=float(np.max(abs(dh-h))),logit_max_error=float(np.max(abs(dq-q))))
            if report['hidden_max_error']>1e-5 or report['logit_max_error']>1e-5:raise RuntimeError('Single-frame cached-vision parity failed')
        return difference,q-self.null[1],report
def main(args):
    cfg=json.loads((DEST/'INPUT_LOCK.json').read_text())
    if args.preflight:jobs=json.loads((DEST/'private/PREFLIGHT_QUEUE.json').read_text())
    else:
        jobs=[]
        for key,r in json.loads((DEST/'private/FRAME_INPUTS.json').read_text()).items():
            for s,paths in r['oct_sites'].items():
                for t,path in enumerate(paths):jobs.append(dict(key=key,site=int(s),frame=t,path=path))
    jobs=jobs[args.split::args.splits]
    out=DEST/'private/frame_cache';out.mkdir(exist_ok=True)
    expected_hashes=json.loads((DEST/'private/FRAME_IMAGE_HASHES.json').read_text())
    model=FrameJEV(cfg['model_bundle']);start=time.monotonic();parity=None;calls=0
    for j,job in enumerate(jobs):
        image_sha=sha(job['path'])
        if expected_hashes[str(Path(job['path']).resolve())]!=image_sha:raise RuntimeError('Image changed since global data audit')
        identity=dict(image_sha256=image_sha,revision=cfg['bundle_revision'],template='independent_slot0_reread_v1',preprocess='ResizeShortest256CenterCrop224')
        token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        target=out/(token+'.npz')
        if not target.exists():
            t=time.monotonic();h,q,check=model.extract(job['path'],parity=parity is None);calls+=1
            if check is not None:parity=check
            tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,hidden=h,logits=q,identity=np.array(json.dumps(identity,sort_keys=True)),seconds=np.array(time.monotonic()-t))
            try:os.link(tmp,target)
            except FileExistsError:
                with np.load(target,allow_pickle=False) as existing:
                    if not np.array_equal(existing['hidden'],h):raise RuntimeError('Duplicate image inference differed across workers')
            finally:tmp.unlink(missing_ok=True)
        save(DEST/f'FRAME_WORKER_{args.split}.json',dict(state='PREFLIGHT' if args.preflight else 'EXTRACTING',pid=os.getpid(),completed=j+1,total=len(jobs),new_calls=calls,seconds=time.monotonic()-start))
        if j%10==0:print('FRAME',j+1,len(jobs),'seconds',time.monotonic()-start,flush=True)
    save(DEST/f'FRAME_WORKER_{args.split}.json',dict(state='COMPLETE',pid=os.getpid(),completed=len(jobs),total=len(jobs),new_calls=calls,seconds=time.monotonic()-start,parity=parity,peak_memory_GiB=model.torch.cuda.max_memory_allocated()/1024**3))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--split',type=int,default=0);p.add_argument('--splits',type=int,default=1);main(p.parse_args())
