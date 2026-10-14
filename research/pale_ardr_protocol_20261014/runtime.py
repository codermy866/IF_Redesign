import os,sys,json,hashlib,time
from pathlib import Path
os.umask(0o077)
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parent
OLD=BASE/'199_final_core_confirmation_20261013'
RECOVERY=BASE/'197_rime_jev_20261012_indexfix_recovery_20261013'
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
        if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [convert(v) for v in x]
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(convert(obj),indent=2,allow_nan=False)+'\n');temp.replace(path)
def verify():
    manifest=json.loads((HERE/'EXPERIMENT_MANIFEST.json').read_text())
    for path,digest in manifest['execution_files'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen ARDR dependency changed: '+Path(path).name)
    if time.time()>config()['deadline_unix']:raise TimeoutError('Unchanged original deadline exhausted')
