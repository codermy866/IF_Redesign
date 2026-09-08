# Seed-only replication amendment

After the complete seed20260905 screen, retain the exact fixed protocol and run
seed20260906 and seed20260907 for all five source folds and every comparator.
No hyperparameter, architecture, budget, loss, cohort, selector input or endpoint
changes. Archive the first pilot summary separately; report all15 fold-seed runs.

Rationale: OOF gain exceeded seen-patient gain in4/5 pilot folds, but did not
consistently beat entropy/random; primary paired CIs overlapped zero. Replication
tests robustness, not a license to search a winning seed. First-seed evidence is
developmental, not an independent confirmatory test after this amendment.

Primary analysis averages seed-specific paired metric differences within each
source fold. Bootstrap identical patient draws across seeds. Do not count seeds
or overlapping folds as independent patients, and do not call averaged metrics
a probability ensemble. No outer-hospital outcomes are unlocked by this amendment.
