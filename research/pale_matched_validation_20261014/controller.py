"""Resumable private queue; only owns processes launched by this controller."""
import runtime
from runtime import HERE,CPU,config,verify,save
import subprocess,time,os,fcntl,atexit,signal,json
children=[]
def stop():
    for p in children:
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
atexit.register(stop)
def state(phase,**kw):
    save(HERE/'STATE.json',dict(state=phase,pid=os.getpid(),time=time.time(),completed=False,**kw))
    (HERE/'RUN_STATE.md').write_text('# PALE fixed validation\n\n'+phase+'\n\n'+json.dumps(kw,indent=2)+'\n')
def runphase(phase):
    todo=[c for c in config()['centers'] if not (HERE/'private/runs'/c/(phase.upper()+'_COMPLETE.json')).exists()];active={}
    while todo or active:
        verify()
        for gpu,(p,outer) in list(active.items()):
            if p.poll() is None:continue
            if p.returncode:raise RuntimeError('Owned worker failed in '+outer+' '+phase)
            del active[gpu]
        for gpu in [0,1]:
            if gpu in active or not todo:continue
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<4096:continue
            outer=todo.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            with (HERE/'logs'/f'{phase}_{outer}.log').open('a') as log:
                p=subprocess.Popen([str(CPU),'-u',str(HERE/'worker.py'),'--outer',outer,'--phase',phase,'--device','cuda:0'],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(p);active[gpu]=(p,outer)
        state(phase.upper()+'_SOURCE_CV',active=[dict(gpu=g,pid=p.pid,outer=c) for g,(p,c) in active.items()],queued=todo)
        time.sleep(10)
def main():
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
    assert json.loads((HERE/'PREFLIGHT_GATE.json').read_text())['state']=='PASS'
    assert json.loads((HERE/'AUDIT_GATE.json').read_text())['state']=='PASS'
    runphase('core')
    state('CORE_REPORT_PENDING')
    if (HERE/'report.py').exists():
        with (HERE/'logs/core_report.log').open('a') as log:
            result=subprocess.run([str(CPU),str(HERE/'report.py'),'--phase','core'],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Core report failed')
    state('RECENT_BASELINE_ADAPTER_GATE_PENDING',core_completed=True)
    # Explicit gate receipt supplied by baseline implementation preflight.
    while not (HERE/'RECENT_ADAPTER_GATE.json').exists():verify();time.sleep(20)
    gate=json.loads((HERE/'RECENT_ADAPTER_GATE.json').read_text())
    if gate.get('controller_script'):
        with (HERE/'logs/recent_controller.log').open('a') as log:
            p=subprocess.Popen([str(CPU),str(HERE/gate['controller_script'])],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
        while p.poll() is None:verify();state('RECENT_BASELINE_EXECUTION',child_pid=p.pid);time.sleep(20)
        if p.returncode:save(HERE/'RECENT_EXECUTION_FAILURE.json',dict(returncode=p.returncode,negative_status=True))
    runphase('extension')
    state('FINAL_REPORT_PENDING',core_completed=True,extension_completed=True)
    with (HERE/'logs/final_report.log').open('a') as log:
        result=subprocess.run([str(CPU),str(HERE/'report.py'),'--phase','all'],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Final report failed')
    save(HERE/'STATE.json',dict(state='COMPLETE_WITH_DECLARED_LIMITATIONS',completed=True,pid=os.getpid(),independent_confirmation=False))
if __name__=='__main__':
    try:main()
    except Exception as exc:stop();state('FAILED_REQUIRES_DIAGNOSIS',error=str(exc));raise
