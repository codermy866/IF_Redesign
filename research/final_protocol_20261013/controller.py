"""Core-first finite queue, with original recovery and no competing 27B copy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import runtime
from runtime import HERE,ROOT,BASE,ORIGINAL,RECOVERY,CPU,save,sha,config,verify
import sys,json,time,subprocess,signal,atexit,traceback,fcntl,shutil
from pathlib import Path
OWNED=[]

def state(phase,**fields):
    save(HERE/'STATE.json',dict(state=phase,pid=os.getpid(),time=time.time(),completed=False,**fields))
    (HERE/'RUN_STATE.md').write_text('# Current execution\n\n'+phase+'\n\n'+json.dumps(fields,indent=2)+'\n\nAll clinical tables are pending until produced from all five held-centre predictions. Original 168h deadline is unchanged.\n')

def alive(pid,needle):
    path=Path('/proc')/str(pid)/'cmdline'
    return path.exists() and needle.encode() in path.read_bytes()

def spawn(script,args=(),device=None):
    verify();env=os.environ.copy()
    if device is not None:env['CUDA_VISIBLE_DEVICES']=str(device)
    log=(HERE/(script.stem+'_'+('_'.join(str(a) for a in args).replace('/','_'))+'.log')).open('a')
    child=subprocess.Popen([str(CPU),'-u',str(script),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    OWNED.append(child);log.close();return child

def stop_owned():
    for child in OWNED:
        if child.poll() is None:
            try:
                # The recovery wrapper owns detached native grandchildren;
                # Python SIGINT runs its atexit child cleanup before exiting.
                receipt=HERE/'ORIGINAL_RECOVERY_LAUNCH.json'
                recovery_pid=json.loads(receipt.read_text())['pid'] if receipt.exists() else None
                os.killpg(child.pid,signal.SIGINT if child.pid==recovery_pid else signal.SIGTERM)
            except ProcessLookupError:pass
atexit.register(stop_owned)
def on_signal(signum,frame):stop_owned();raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)

def run(script,args=()):
    child=spawn(script,args)
    while child.poll() is None:verify();time.sleep(10)
    if child.returncode:raise RuntimeError(script.name+' failed; preserve log and diagnose')

def wait_original():
    deadline=config()['deadline_unix']
    while time.time()<deadline:
        verify();s=json.loads((ORIGINAL/'STATE.json').read_text())
        if (ORIGINAL/'FEATURE_MANIFEST.json').exists() and s['state']=='FAILED_REQUIRES_DIAGNOSIS':
            text='\n'.join(p.read_text(errors='replace') for p in ORIGINAL.glob('heads_*.log'))
            known='IndexError' in text and 'xx[train]' in text and 'build_anchor' in text
            launch=json.loads((ORIGINAL/'CONTROLLER_LAUNCH.json').read_text())
            running=alive(launch['pid'],'run_rime_jev.py')
            # Wait for any original sibling head child to exit as well.
            sibling=False
            for proc in Path('/proc').glob('[0-9]*/cmdline'):
                try:cmd=proc.read_bytes()
                except (PermissionError,FileNotFoundError,ProcessLookupError):continue
                if str(ROOT/'scripts/run_rime_heads.py').encode() in cmd:sibling=True
            if known and not running and not sibling:
                save(HERE/'ORIGINAL_FAILURE_RECEIPT.json',dict(state=s,state_sha256=sha(ORIGINAL/'STATE.json'),known_indexing_error_only=True))
                return
            if not known:raise RuntimeError('Unknown original failure; no automatic intervention')
        if s.get('completed'):
            save(HERE/'ORIGINAL_FAILURE_RECEIPT.json',dict(state='ORIGINAL_COMPLETED',index_correction_not_needed=False));return
        state('WAIT_SHARED_NATIVE_FEATURES_AND_ORIGINAL_SAFE_HANDOVER',original_state=s['state'],deadline_unix=deadline)
        time.sleep(20)
    raise TimeoutError('Shared cache unfinished at original deadline')

def attach_cache():
    for name in ['private/frame_hidden.npy','private/frame_mask.npy','private/frame_cache','private/FRAME_IMAGE_HASHES.json','private/PACKED_FRAME_INDEX.json']:
        source=ORIGINAL/name;target=HERE/name
        if not source.exists():raise RuntimeError('Missing shared frozen cache: '+name)
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source,target_is_directory=source.is_dir())
    shutil.copyfile(ORIGINAL/'FEATURE_MANIFEST.json',HERE/'FEATURE_MANIFEST.json')
    save(HERE/'SHARED_CACHE_RECEIPT.json',dict(original_feature_manifest_sha256=sha(ORIGINAL/'FEATURE_MANIFEST.json'),no_backbone_reinference=True))

def heads(phase,gpu=True):
    todo=[c for c in ['enshi','jingzhou','shiyan','wuhan','xiangyang'] if not (HERE/'private/head_runs'/c/(('COMPLETE' if phase=='core' else phase+'_COMPLETE')+'.json')).exists()]
    active={}
    while todo or active:
        verify()
        for slot,(child,outer) in list(active.items()):
            code=child.poll()
            if code is None:continue
            del active[slot]
            if code:raise RuntimeError(f'{phase} fold {outer} failed; no new search or silent seed exclusion')
        slots=[0,1] if gpu else [None]
        for slot in slots:
            if slot in active or not todo:continue
            if gpu:
                free=int(subprocess.check_output(['nvidia-smi',f'--id={slot}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<4000:continue
            outer=todo.pop(0);child=spawn(HERE/'run_heads.py',['--phase',phase,'--fold',outer,'--device','cuda:0' if gpu else 'cpu'],slot)
            active[slot]=(child,outer)
        state(phase.upper()+'_SOURCE_CV_AND_FROZEN_OUTER_PREDICTIONS',workers=[dict(pid=p.pid,outer=c,device=s) for s,(p,c) in active.items()],queued=todo)
        time.sleep(10)
    run(HERE/'summarize.py',['--phase',phase])

def resume_original_protocol():
    """Reuse parity-verified joint-training cells, never copy new controls as old."""
    for outer in ['enshi','jingzhou','shiyan','wuhan','xiangyang']:
        source=HERE/'private/fold_features'/outer;target=RECOVERY/'private/fold_features'/outer
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source,target_is_directory=True)
        source=HERE/'private/head_runs'/outer;target=RECOVERY/'private/head_runs'/outer;target.mkdir(parents=True,exist_ok=True)
        for method in config()['core_methods']:
            for path in source.glob(method+'*'):
                if path.is_file() and not (target/path.name).exists():os.link(path,target/path.name)
    save(HERE/'ORIGINAL_RECOVERY_REUSE_RECEIPT.json',dict(core_joint_exact_parity=True,features_shared_read_only=True,new_warmup_and_scores_not_reused_as_original=True))
    wrapper=BASE/'198_rime_jev_cv1_20261013/run_original_integrity_recovery.py'
    child=spawn(wrapper)
    save(HERE/'ORIGINAL_RECOVERY_LAUNCH.json',dict(pid=child.pid,original_tasks_preserved=True))
    return child

def main():
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    if json.loads((HERE/'PREFLIGHT_GATE.json').read_text())['state']!='PASS':raise RuntimeError('Numerical preflight not passed')
    wait_original();attach_cache();heads('core')
    state('CORE_CONTRIBUTIONS_REVIEWED_BEFORE_OPTIMIZATION',review='CORE_CONTRIBUTION_REVIEW.json')
    recovery=resume_original_protocol()
    # Preserve the original campaign's GPU priority; small controls use CPU.
    heads('B1',gpu=False);heads('B2',gpu=False)
    state('WAIT_ORIGINAL_GPU_TASKS_BEFORE_B5_FULL_NATIVE_RECOMPUTATION',original_recovery_pid=recovery.pid)
    while recovery.poll() is None:verify();time.sleep(20)
    if recovery.returncode:
        save(HERE/'ORIGINAL_RECOVERY_FAILURE.json',dict(returncode=recovery.returncode,original_outputs_preserved=True))
    run(HERE/'run_b5.py')
    run(HERE/'finalize.py')
    save(HERE/'STATE.json',dict(state='FINAL_PROTOCOL_FINISHED_WITH_EXPLICIT_GATES',pid=os.getpid(),completed=True,independent_confirmation=False))

if __name__=='__main__':
    try:main()
    except TimeoutError as exc:
        stop_owned()
        with (HERE/'partial_finalize.log').open('a') as log:
            subprocess.run([str(CPU),str(HERE/'finalize.py')],stdout=log,stderr=subprocess.STDOUT,timeout=120)
        state('ORIGINAL_BUDGET_EXHAUSTED_PARTIAL_DELIVERY',error=str(exc))
    except Exception as exc:
        stop_owned();state('FAILED_REQUIRES_DIAGNOSIS',error=str(exc));traceback.print_exc();raise
