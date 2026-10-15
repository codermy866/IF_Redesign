import os,sys,json,time,hashlib
from pathlib import Path
os.umask(0o077)
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
HERE=Path(__file__).resolve().parent
BASE=HERE.parent;ROOT=BASE.parent;PREVIOUS=BASE/'200_ARDR_JEV_FINAL_20261014'
CPU=BASE/'.venv/bin/python'
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(HERE/'code'))
def config():return json.loads((HERE/'CONFIG.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,obj):
    import numpy as np
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,dict):return {str(k):convert(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [convert(v) for v in x]
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(convert(obj),indent=2,allow_nan=False)+'\n');tmp.replace(path)
def verify():
    manifest=json.loads((HERE/'EXPERIMENT_MANIFEST.json').read_text())
    for path,want in manifest['execution_files'].items():
        if sha(path)!=want:raise RuntimeError('Frozen execution code changed: '+Path(path).name)
    if time.time()>config()['deadline_unix']:raise TimeoutError('Original campaign deadline reached')
