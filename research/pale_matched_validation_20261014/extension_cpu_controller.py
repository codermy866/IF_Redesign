import runtime;import subprocess,json,time,fcntl;from runtime import HERE,CPU,config,save,verify;lock=(HERE/"extension_cpu.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);active={};todo=[c for c in config()["centers"] if not (HERE/"private/runs"/c/"EXTENSION_COMPLETE.json").exists()]
while todo or active:
 verify()
 for slot,(child,outer) in list(active.items()):
  if child.poll() is not None:
   if child.returncode: raise RuntimeError("CPU extension worker failed: "+outer)
   del active[slot]
 for slot in [0,1]:
  if slot in active or not todo:continue
  outer=todo.pop(0);log=(HERE/"logs"/("extension_CPU_"+outer+".log")).open("a");child=subprocess.Popen([str(CPU),"-u",str(HERE/"worker.py"),"--outer",outer,"--phase","extension","--device","cpu"],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[slot]=(child,outer)
 save(HERE/"EXTENSION_CPU_STATE.json",dict(state="RUNNING",active=[dict(pid=c.pid,outer=o) for c,o in active.values()],queued=todo,time=time.time()));time.sleep(15)
save(HERE/"EXTENSION_CPU_STATE.json",dict(state="COMPLETE",completed=True,time=time.time()))