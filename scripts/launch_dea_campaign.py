"""Durable, finite two-A6000 queue; never terminate existing GPU jobs."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv/bin/python")
LOCK = threading.Lock()


def run_command(args, log, gpu=None):
    env=dict(os.environ,PYTHONPATH=str(ROOT/"src")+":"+str(ROOT/"scripts"),OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",PYTHONUNBUFFERED="1")
    if gpu is not None:env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as handle:
        return subprocess.run([PYTHON,*args],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode


def idle(gpu,limit):
    query=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    usage={int(line.split(",")[0]):int(line.split(",")[1])for line in query.stdout.splitlines()}
    return usage.get(gpu,10**9)<limit


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/dea_v1.json");p.add_argument("--prepare",action="store_true");p.add_argument("--wait-data-pid",type=int);a=p.parse_args()
    cfg=json.loads((ROOT/a.config).read_text());out=ROOT/cfg["output"];out.mkdir(parents=True,exist_ok=True)
    lockfile=(out/("prepare.lock" if a.prepare else "campaign.lock")).open("w")
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.prepare:
        if a.wait_data_pid:
            while Path(f"/proc/{a.wait_data_pid}").exists():
                (out/"prepare_status.json").write_text(json.dumps(dict(status="waiting_for_initial_builder",pid=a.wait_data_pid)))
                time.sleep(15)
        failed=set()
        while True:
            pending=[]
            for fold in cfg["folds"]:
                for seed in cfg["seeds"]:
                    audit=out/"datasets"/fold/f"seed_{seed}"/"audit.json"
                    if audit.exists() and json.loads(audit.read_text()).get("schema_version")==2:continue
                    key=f"{fold}/{seed}"
                    if key in failed:continue
                    source=ROOT/cfg["gain_root"]/fold/f"seed_{seed}"/"gain_targets.npz"
                    if not source.exists():pending.append(key);continue
                    code=run_command(["scripts/build_dea_dataset.py","--config",a.config,"--fold",fold,"--seed",str(seed)],out/"logs"/f"build_{fold}_{seed}.log")
                    if code:failed.add(key)
                    (out/"prepare_status.json").write_text(json.dumps(dict(status="building",last=key,last_exit=code,failed=sorted(failed),waiting_for_oof=pending)))
            (out/"prepare_status.json").write_text(json.dumps(dict(status="waiting_for_oof" if pending else ("failed" if failed else "complete"),failed=sorted(failed),waiting_for_oof=pending)))
            if not pending:break
            time.sleep(30)
        return
    jobs=[]
    for fold in cfg["folds"]:
        for seed in cfg["seeds"]:
            source=ROOT/cfg["gain_root"]/fold/f"seed_{seed}"/"gain_targets.npz"
            if not source.exists():jobs.append(dict(kind="recover_oof",fold=fold,seed=seed,status="pending"))
    for seed in cfg["seeds"]:
        for fold in cfg["folds"]:
            for backbone in cfg["backbones"]:
                jobs.append(dict(kind="vlm",backbone=backbone,fold=fold,seed=seed,status="pending"))
    status_path=out/"campaign_status.json"

    def save():
        temp=out/"campaign_status.tmp"
        temp.write_text(json.dumps(dict(pid=os.getpid(),updated=time.time(),
            status=("complete" if all(j["status"]=="complete"for j in jobs) else
                    "failed" if all(j["status"] in ("complete","failed","blocked_dependency")for j in jobs) else "active"),
            primary_cells=180,additional_rank_ablation_cells=15,jobs=jobs),indent=2))
        temp.replace(status_path)

    def ready(job):
        if job["kind"]=="recover_oof":return True
        audit=out/"datasets"/job["fold"]/f"seed_{job['seed']}"/"audit.json"
        if not audit.exists() or json.loads(audit.read_text()).get("schema_version")!=2:
            job["waiting_for"]="audited_dataset_v2";return False
        model_status=out/"model_status.json"
        state=json.loads(model_status.read_text()) if model_status.exists() else {}
        if state.get(job["backbone"],{}).get("status")!="downloaded":
            job["waiting_for"]="model_download";return False
        job.pop("waiting_for",None);return True

    conversion_lock=threading.Lock()

    def execute(job,gpu):
        fold,seed=job["fold"],job["seed"]
        if job["kind"]=="recover_oof":
            return run_command(["scripts/run_cross_fitted_gain.py","--fold",fold,"--seed",str(seed),"--restored-clinical"],out/"logs"/f"recover_{fold}_{seed}.log",gpu)
        backbone=job["backbone"]
        if backbone=="llava_med":
            with conversion_lock:
                code=run_command(["scripts/convert_dea_llava.py"],out/"logs"/"llava_conversion.log")
                if code:return code
        arms=["sft","alignment","alignment_dpo","zero_shot"]
        if backbone=="qwen3b":arms.insert(2,"alignment_no_rank")
        common=["--config",a.config,"--fold",fold,"--seed",str(seed),"--backbone",backbone]
        # Each new backbone must pass an actual backward smoke before full training.
        smoke=out/"smoke_runs"/backbone/fold/f"seed_{seed}"/"alignment"/"smoke_complete.json"
        if not smoke.exists():
            code=run_command(["scripts/run_dea.py",*common,"--arm","alignment","--max-steps","1"],out/"logs"/f"{backbone}_{fold}_{seed}_smoke.log",gpu)
            if code:return code
        for arm in arms:
            log=out/"logs"/f"{backbone}_{fold}_{seed}_{arm}.log"
            if arm!="zero_shot":
                code=run_command(["scripts/run_dea.py",*common,"--arm",arm],log,gpu)
                if code:return code
            for split in ("val","test"):
                marker=out/"runs"/backbone/fold/f"seed_{seed}"/arm/f"evaluation_{split}.json"
                if marker.exists():continue
                code=run_command(["scripts/run_dea.py",*common,"--arm",arm,"--mode","evaluate","--split",split],log,gpu)
                if code:return code
        code=run_command(["scripts/audit_dea_teacher.py","--config",a.config,"--fold",fold,"--seed",str(seed)],out/"logs"/f"teacher_{fold}_{seed}.log")
        if code:return code
        return 0

    def worker(gpu):
        while True:
            with LOCK:
                pending=[j for j in jobs if j["status"]=="pending"]
                if not pending:return
                available=idle(gpu,cfg["gpu_max_existing_memory_mb"])
                selected=next((j for j in pending if ready(j)),None) if available else None
                if selected:
                    selected.update(status="running",gpu=gpu,started=time.time())
                else:
                    for j in pending:
                        if not available:j["waiting_for"]="existing_GPU_jobs"
                save()
            if selected is None:
                time.sleep(20);continue
            try:code=execute(selected,gpu)
            except Exception as e:code=1;selected["error"]=repr(e)
            with LOCK:
                selected.update(status="complete" if code==0 else "failed",exit_code=code,finished=time.time());save()
            # Metrics aggregation is serialized to avoid concurrent output writes.
            if code==0:
                with LOCK:run_command(["scripts/summarize_dea.py","--config",a.config],out/"logs"/"summary.log")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu)for gpu in (0,1)]
        for future in futures:future.result()
    with LOCK:save()


if __name__=="__main__":main()
