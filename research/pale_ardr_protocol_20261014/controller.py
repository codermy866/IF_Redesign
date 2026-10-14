"""Detached, fixed-budget source-pilot -> six-arm five-centre confirmation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
import runtime
from runtime import HERE,CPU,config,verify,save
import json,time,signal,atexit,subprocess,fcntl,traceback
OWNED=[]
def state(phase,**fields):
    save(HERE/'STATE.json',dict(state=phase,pid=os.getpid(),time=time.time(),completed=False,**fields))
    (HERE/'RUN_STATE.md').write_text('# ARDR execution\n\n'+phase+'\n\n'+json.dumps(fields,indent=2)+'\n\nCommon FR1 repaired inputs for A0-A5; frozen old results preserved. No outer label based tuning. Original deadline unchanged.\n')
def stop():
    for child in OWNED:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
atexit.register(stop)
def on_signal(number,frame):stop();raise SystemExit(128+number)
signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
def spawn(script,args=(),gpu=None,name=None):
    verify();env=os.environ.copy()
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (HERE/'logs'/(name or (HERE/script).stem+'.log')).open('a') as log:
        child=subprocess.Popen([str(CPU),'-u',str(HERE/script),*map(str,args)],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    OWNED.append(child);return child
def await_child(child):
    while child.poll() is None:verify();time.sleep(10)
    if child.returncode:raise RuntimeError('Owned worker failed: '+str(child.pid))
def main():
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();cfg=config()
    for name in ['PREFLIGHT_GATE.json','LEGACY_REPRODUCTION.json','REPAIR_INITIAL_SOURCE_GATE.json']:
        assert json.loads((HERE/name).read_text())['state']=='PASS',name
    state('SOURCE_ONLY_FIXED_PILOT',outer=cfg['pilot_outer'],held_source=cfg['pilot_held_source'])
    if not (HERE/'PILOT_GATE.json').exists():
        child=spawn('worker.py',['--outer',cfg['pilot_outer'],'--device','cuda:0','--pilot'],gpu=1,name='pilot_gpu1.log')
        save(HERE/'PILOT_LAUNCH.json',dict(pid=child.pid,gpu=1,outer_labels_used=False));await_child(child)
    gate=json.loads((HERE/'PILOT_GATE.json').read_text())
    if not gate['full_matrix_allowed']:
        child=spawn('summarize.py');await_child(child)
        save(HERE/'STATE.json',dict(state='STOPPED_AFTER_NEGATIVE_SOURCE_PILOT',completed=True,gate=gate,decision='ARDR不纳入主方法'));return
    todo=[c for c in cfg['centers'] if not (HERE/'private/runs'/c/'COMPLETE.json').exists()];active={}
    while todo or active:
        verify()
        for gpu,(child,outer) in list(active.items()):
            if child.poll() is None:continue
            if child.returncode:raise RuntimeError('Full ARDR worker failed in '+outer)
            del active[gpu]
        for gpu in [0,1]:
            if gpu in active or not todo:continue
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<2048:continue
            outer=todo.pop(0);child=spawn('worker.py',['--outer',outer,'--device','cuda:0'],gpu=gpu,name='outer_'+outer+'.log');active[gpu]=(child,outer)
        state('FULL_FIXED_SIX_ARM_SOURCE_CV',workers=[dict(pid=c.pid,outer=o,gpu=gg) for gg,(c,o) in active.items()],queued=todo)
        time.sleep(10)
    state('PAIRED_PATIENT_STATISTICS_AND_FINAL_ADMISSION')
    child=spawn('summarize.py');await_child(child)
    save(HERE/'STATE.json',dict(state='COMPLETE_WITH_FINAL_VERDICT',pid=os.getpid(),completed=True,decision=json.loads((HERE/'FINAL_VERDICT.json').read_text())['decision'],independent_confirmation=False))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        stop();state('FAILED_REQUIRES_DIAGNOSIS',error=str(exc));traceback.print_exc();raise
