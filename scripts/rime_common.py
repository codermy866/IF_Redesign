import os,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'03_new_experiment_design'
OLD=BASE/'196_controlled_cfv_jev_20261012'
DEST=BASE/'197_rime_jev_20261012'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [serial(v) for v in x]
        return x
    t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(serial(v),indent=2,allow_nan=False)+'\n');t.replace(p)
def frame_identity(path):
    cfg=json.loads((DEST/'INPUT_LOCK.json').read_text())
    actual=sha(path)
    if not hasattr(frame_identity,'expected'):
        frame_identity.expected=json.loads((DEST/'private/FRAME_IMAGE_HASHES.json').read_text())
    if frame_identity.expected[str(Path(path).resolve())]!=actual:raise RuntimeError('Image identity changed since audit')
    identity=dict(image_sha256=actual,revision=cfg['bundle_revision'],template='independent_slot0_reread_v1',preprocess='ResizeShortest256CenterCrop224')
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return DEST/'private/frame_cache'/(token+'.npz'),identity
def markdown(frame):
    """Dependency-free table writer; do not require tabulate on experiment hosts."""
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6g}'
        return str(v).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)
