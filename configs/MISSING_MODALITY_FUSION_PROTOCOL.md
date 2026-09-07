# Missing-Modality Fusion Protocol

This public protocol file documents the reusable code path only. Provide a
de-identified feature cache and fold assignment table through
`configs/ices_v1_exploratory.json`.

The implementation compares unimodal branches, simple late averaging, frozen
fusion, trainable fusion, and modality-dropout fusion under paired synthetic
availability patterns. It refuses an all-missing diagnostic prediction.

No data, centre identifiers, trained weights, logs, or result tables are
included in this repository.
