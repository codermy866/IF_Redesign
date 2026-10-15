import runtime
from runtime import HERE,CPU,save,verify
import time,subprocess,json,os,fcntl
from recent_training import cfg
def main():
    lock=(HERE/'recent_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=cfg();todo=[];active={};failures=[]
    kinds=c['kinds']+c['simple']
    for track in c['tracks']:
        for kind in kinds:
            if kind in c['simple'] and track=='B':continue
            for outer in c['centers']:
                if not (HERE/'private/recent_runs'/outer/f'TRACK_{track}_{kind}.npz').exists():todo.append((track,kind,outer))
    while todo or active:
        verify()
        if time.time()>c['deadline_unix']:break
        for gpu,(p,job) in list(active.items()):
            if p.poll() is None:continue
            if p.returncode:failures.append(dict(job=job,pid=p.pid,returncode=p.returncode))
            del active[gpu]
            if (HERE/'recent_report.py').exists():
                with (HERE/'logs/recent_partial_report.log').open('a') as log:subprocess.run([str(CPU),str(HERE/'recent_report.py')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        for gpu in [0,1]:
            if gpu in active or not todo:continue
            free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<4096:continue
            track,kind,outer=todo.pop(0);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            with (HERE/'logs'/f'recent_{track}_{kind}_{outer}.log').open('a') as log:
                p=subprocess.Popen([str(CPU),'-u',str(HERE/'recent_worker.py'),'--outer',outer,'--kind',kind,'--track',track],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(p,(track,kind,outer))
        save(HERE/'RECENT_STATE.json',dict(state='RUNNING',pid=os.getpid(),time=time.time(),active=[dict(gpu=g,pid=p.pid,job=job) for g,(p,job) in active.items()],queued=todo,failures=failures))
        time.sleep(15)
    save(HERE/'RECENT_STATE.json',dict(state='COMPLETE_WITH_FAILURES' if failures else 'COMPLETE',failures=failures,completed=not todo and not active,remaining=todo,time=time.time()))
    if (HERE/'recent_report.py').exists():subprocess.run([str(CPU),str(HERE/'recent_report.py')],cwd=HERE,check=True)
if __name__=='__main__':main()
