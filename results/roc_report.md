# Detector ROC/AUPRC (continuous scores, observable-only)

Pooled over 4 agents' honest-mode traces. Unit: one retrieved memory in an attack episode.
Positive class: the memory's labels include `poison` (evaluation only; the label is not a
gate input). n=18942, positives=5170 (27.3%). Intervals are cluster-bootstrap
95% percentile intervals over (agent, seed), n_clusters=20.

| Detector | ROC-AUC [cluster 95%] | AUPRC [cluster 95%] |
|---|--:|--:|
| TGM (graded term-matching) | 0.649 [0.598, 0.691] | 0.561 [0.505, 0.634] |
| Provenance (graded authority + text) | 0.603 [0.530, 0.663] | 0.515 [0.427, 0.610] |

Baseline: a no-skill ranker has ROC-AUC 0.5 and AUPRC 0.273 (the positive prevalence).
Graded TGM point estimate 0.649; graded provenance 0.603.
These scores are per-memory text/authority functions. They are not the binary action gates.
The binary gates are documented in `results/generated/eleven_condition_table.md`.

Cluster intervals that cover 0.5 do not establish ranking skill beyond chance on this sample.
