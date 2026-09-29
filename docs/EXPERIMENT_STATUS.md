# Experimental Status

This document records the current aggregate experimental status without exposing patient-level data.

## Completed confirmatory evaluation

- Four source centres with strict patient-level leave-one-centre-out evaluation.
- Qwen3-VL-2B-Instruct main MLLM experiments.
- Phi-3.5-Vision-Instruct full-LoRA cross-family experiments.
- Phi-3.5-Vision frozen-representation transfer experiments.
- 144 total MLLM runs across confirmatory and transfer settings.

## Factorial stage-structure audit

A frozen-feature factorial comparison separates four structures:

- flat fusion;
- prior supplied as an ordinary feature;
- direct prior anchoring;
- prior anchoring with baseline-referenced contrast.

The current result supports explicit prior anchoring as the transferable structural principle. Baseline subtraction is not required.

## SOPI validation

Completed analyses include:

- Brier and log-score innovation;
- within-centre patient permutation;
- posterior-movement versus outcome-aligned innovation;
- controlled nuisance perturbation;
- finite-sample bias/coverage analysis;
- stagewise additivity checks.

A key falsification result is that within-centre permutation can increase posterior movement while making Brier/log innovation strongly negative. Therefore posterior movement is not a valid substitute for task-relevant information gain.

## Boundary experiments

The following hypotheses were tested and are not retained as the primary method:

- local semantic recognition as a proxy for patient-level innovation;
- hard clinical-subspace orthogonality;
- worst-centre robustness as a proxy for information value;
- exact committor transition value;
- latent committor/innovation steering.

These negative results are retained as falsification boundaries rather than additional method modules.

## Public-repository policy

The repository intentionally excludes patient identifiers, raw clinical data, OCT images, model weights, private OOF predictions, and manuscript files. Public code is restricted to the reusable method, validation utilities, and mathematical tests.
