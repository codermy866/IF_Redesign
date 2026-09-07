"""Finite 15-run recovered-clinical gain replication, persistent per-job logs."""
import concurrent.futures
import json
import os
import subprocess
from run_cross_fitted_gain import ROOT

OUT = ROOT/'results/cross_fitted_gain_restored_v1'
FOLDS = ('shiyan', 'enshi', 'wuhan', 'jingzhou', 'xiangyang')
SEEDS = (20260905, 20260906, 20260907)


def worker(gpu, jobs):
    states = []
    for fold, seed in jobs:
        path = OUT/fold/f'seed_{seed}'
        path.mkdir(parents=True, exist_ok=True)
        if (path/'complete.json').exists():
            continue
        with (path/'run.log').open('a') as log:
            process = subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_cross_fitted_gain.py'), '--fold', fold, '--seed', str(seed), '--restored-clinical'],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT)
        states.append({'fold': fold, 'seed': seed, 'exit_code': process.returncode})
        (OUT/f'worker_{gpu}.json').write_text(json.dumps(states, indent=2))
        if process.returncode:
            break
    return states


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(fold, seed) for seed in SEEDS for fold in FOLDS]
    state = {'status': 'running', 'planned_runs': 15, 'planned_diagnostic_models': 90, 'protocol': 'CROSS_FITTED_GAIN_RECOVERED_INPUT.md'}
    (OUT/'progress.json').write_text(json.dumps(state, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(worker, gpu, jobs[gpu::2]) for gpu in (0, 1)]
        statuses = [r for t in tasks for r in t.result()]
    n = len(list(OUT.glob('*/seed_*/complete.json')))
    state.update(status='summarizing' if n == 15 else 'failed', completed_runs=n, jobs=statuses)
    (OUT/'progress.json').write_text(json.dumps(state, indent=2))
    if n != 15:
        raise RuntimeError('Incomplete matrix; inspect worker logs')
    subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/summarize_cross_fitted_gain_replication.py'), '--output', str(OUT)], cwd=ROOT, check=True)
    state['status'] = 'complete'
    (OUT/'progress.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
