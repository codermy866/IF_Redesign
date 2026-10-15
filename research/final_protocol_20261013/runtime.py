"""Independent namespace; never modify a Stage197 dependency."""
import os,sys,json,time,hashlib
from pathlib import Path
os.umask(0o077)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'03_new_experiment_design'
ORIGINAL=BASE/'197_rime_jev_20261012'
RECOVERY=BASE/'197_rime_jev_20261012_indexfix_recovery_20261013'
CPU=BASE/'.venv/bin/python'
GPU=BASE/'190_autotrust_jev27b_20261011/.venv/bin/python'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(HERE/'code'))
import rime_common
rime_common.DEST=HERE
from rime_common import save,sha
import rime_dataset
from anchor_adapter import build_anchor
rime_dataset.build_anchor=build_anchor
def config():return json.loads((HERE/'CONFIG.json').read_text())
def verify():
    lock=json.loads((HERE/'CODE_AND_DATA_MANIFEST.json').read_text())
    for path,digest in lock['files'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen final dependency changed: '+path)
    original=json.loads((ORIGINAL/'CODE_LOCK.json').read_text())
    for path,digest in original['files'].items():
        if sha(path)!=digest:raise RuntimeError('Protected original dependency changed: '+path)
    inputs=json.loads((HERE/'INPUT_LOCK.json').read_text())
    if sha(inputs['cohort_path'])!=inputs['cohort_sha256']:raise RuntimeError('Cohort changed')
    if time.time()>=config()['deadline_unix']:raise TimeoutError('Original 168h deadline reached')
