"""Finite, resumable two-GPU experiment queue. Does not terminate other jobs."""
import concurrent.futures
import argparse
import json
import os
import subprocess
from run_missing_modality_fusion import ROOT, OUT, FOLDS, SEEDS


def worker(gpu, jobs, restored=False):
    rows = []
    for fold, seed in jobs:
        path = OUT/fold/f'seed_{seed}'
        path.mkdir(parents=True, exist_ok=True)
        if (path/'complete.json').exists():
            continue
        with (path/'run.log').open('a') as log:
            p = subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/run_missing_modality_fusion.py'), '--fold', fold, '--seed', str(seed)]+(['--restored-clinical'] if restored else []),
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT)
        rows.append({'fold': fold, 'seed': seed, 'exit_code': p.returncode})
        (OUT/f'worker_{gpu}.json').write_text(json.dumps(rows, indent=2))
        if p.returncode:
            break
    return rows


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored-clinical', action='store_true')
    args = parser.parse_args()
    if args.restored_clinical:
        OUT = ROOT/'results/missing_modality_fusion_restored_v1'
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(f, s) for s in SEEDS for f in FOLDS]
    (OUT/'progress.json').write_text(json.dumps({'status': 'running', 'planned_runs': 15, 'planned_unimodal': 45, 'planned_fusion': 60}))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(worker, g, jobs[g::2], args.restored_clinical) for g in (0, 1)]
        rows = [r for task in tasks for r in task.result()]
    n = len(list(OUT.glob('*/seed_*/complete.json')))
    state = {'status': 'summarizing' if n == 15 else 'failed', 'completed_runs': n, 'planned_runs': 15, 'jobs': rows}
    (OUT/'progress.json').write_text(json.dumps(state, indent=2))
    if n == 15:
        if not args.restored_clinical:
            subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/audit_ices_modality_missingness.py')], cwd=ROOT, check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/summarize_missing_modality_fusion.py'), '--output', str(OUT)], cwd=ROOT, check=True)
        state['status'] = 'complete'
        (OUT/'progress.json').write_text(json.dumps(state, indent=2))
    else:
        raise RuntimeError('Incomplete queue; inspect logs, do not label complete')


if __name__ == '__main__':
    main()
