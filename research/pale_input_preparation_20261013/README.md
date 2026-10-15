# Private input preparation adapters

`anchor_adapter.py` preserves the corrected Stage199 source-only patient anchor
builder. `fr1_fold_data.py` preserves Stage200 shared FR1 source centering,
gray reanchoring and source-fitted Laplace coordinate scaling. The latter is
originally named `data.py`; it is renamed here only to disambiguate its role.
These snippets require their original private campaign runtime. The portable
projection is `if_redesign.pale_ardr.preprocessing.repair_project`. The original
frame/anchor preparation helpers are retained in root `scripts/rime_dataset.py`.
