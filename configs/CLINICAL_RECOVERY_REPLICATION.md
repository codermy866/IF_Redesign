# Clinical-Input Recovery Protocol

This public stub exists so the reusable scripts can hash a protocol document
when a user supplies their own de-identified clinical sidecar.

The repository does not include private source tables or a sidecar. Users must
construct any replacement clinical fields outside version control and verify
that joins are outcome-blind, exact, and patient-level disjoint across splits.
