# Sampling budget ablation (k=25→50, r=2 fixed) — 2026-08-16, FIXED generator

Re-run of the 08-13 ablation after the §1.0c generator-reasoning fix (the earlier version
compared two arms that both used the accidental default-reasoning generator; this one uses
the audited `gpt-5.5-off` config on both arms and the regenerated k=25 canonical files).
Sycophancy_pushback dev (34 subjects); side files `budget_k50_syc_<model>.json`.

| model | k=25 dev r | k=50 dev r | k25↔k50 agreement r |
|---|---|---|---|
| llama-3.3-70b | +0.764 | +0.743 | +0.935 |
| gpt-5.4-nano-low | +0.198 | +0.149 | +0.846 |

Reading: unchanged from the earlier run — doubling the scenario budget moves dev r by ≤0.05
in either direction (llama −0.02, nano −0.05, both within noise; agreement r 0.85–0.94).
k=25×r=2 is not the binding constraint where the method works, and more budget does not
rescue it where it is weak. Pre-empts "watching fails because it was under-resourced";
the paper states k×r as a design constant.
