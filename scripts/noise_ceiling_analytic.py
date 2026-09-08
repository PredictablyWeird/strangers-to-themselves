"""Cross-check: analytic reliability (observed variance minus unbiased binomial noise variance),
with a bootstrap CI over conditions; plus target distributions for the anomalous cells."""
import json
from statistics import mean, pstdev
import random
from behavior_prediction import common, splits
from behavior_prediction.evals import get_spec

POOL = ["llama-3.3-70b", "llama-4-maverick", "deepseek-v4-flash-low", "qwen3.7-plus-low"]
rng = random.Random(7)

def analytic_rel(rows):
    """rows: list of (rate, n). Returns (rel, ceiling) from var decomposition."""
    if len(rows) < 5:
        return None, None
    rates = [r for r, _ in rows]
    var_obs = pstdev(rates) ** 2
    if var_obs < 1e-12:
        return 0.0, 0.0
    noise = mean(r * (1 - r) / (n - 1) if n > 1 else 0.25 for r, n in rows)
    rel = max(0.0, (var_obs - noise)) / var_obs
    return rel, rel ** 0.5

def boot_ci(rows, reps=1000):
    vals = []
    for _ in range(reps):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        rel, _ = analytic_rel(sample)
        if rel is not None:
            vals.append(rel ** 0.5)
    vals.sort()
    return vals[int(0.05 * len(vals))], vals[int(0.95 * len(vals))]

for ev in ["sycophancy_pushback", "capability_mmlu", "propensitybench"]:
    spec = get_spec(ev)
    manifest = splits.load_manifest(ev)
    for slug in POOL:
        targets = common.load_json(f'results/{ev}/{slug}/targets.json')["targets"]
        dev = splits.split_keys(spec, targets, manifest, "dev")
        rows = [(t["rate"], t["n"]) for k, t in targets.items()
                if k in dev and t.get("rate") is not None and t.get("n")]
        rel, ceil = analytic_rel(rows)
        lo, hi = boot_ci(rows)
        sd = pstdev([r for r, _ in rows])
        print(f"{ev:<20} {slug:<24} rel={rel:+.2f} ceiling={ceil:+.2f} [{lo:+.2f},{hi:+.2f}]  "
              f"target sd={sd:.2f} mean n={mean(n for _, n in rows):.0f}")
    print()

# anomalous cell: deepseek sycophancy target distribution
t = common.load_json('results/sycophancy_pushback/deepseek-v4-flash-low/targets.json')["targets"]
manifest = splits.load_manifest("sycophancy_pushback")
spec = get_spec("sycophancy_pushback")
dev = splits.split_keys(spec, t, manifest, "dev")
rows = sorted((v["rate"], v["count"], v["n"], k) for k, v in t.items() if k in dev)
print("deepseek sycophancy dev targets (rate, flips, n):")
print("  zeros:", sum(1 for r, *_ in rows if r == 0), "of", len(rows))
print("  nonzero:", [(f"{r:.2f}", c, n, k[:22]) for r, c, n, k in rows if r > 0])
