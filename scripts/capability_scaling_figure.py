#!/usr/bin/env python3
"""Capability-vs-self-knowledge scatter (fig_scaling): per-model measured MMLU accuracy
(the capability_mmlu targets, so each entry's reasoning setting is priced in) against the
model's macro test-split prediction correlation, for self-report and item-informed
self-prediction. Replaces coarse tier grouping with the actual capability axis.

Reads results/reports/evaluation_test.json + results/capability_mmlu/*/targets.json.
Writes paper/generated/fig_scaling.{png,pdf} plus the macros quoted in app:abl-settings
(paper/generated/scaling.tex), and prints the correlations.
"""
import json
import random
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from behavior_prediction import metrics

POOL = ['deepseek-v4-flash-low', 'gemini-3.1-flash-lite-low', 'gpt-5.4-nano-low',
        'llama-3.3-70b', 'llama-4-maverick', 'qwen3.7-plus-low',
        'claude-sonnet-5-off', 'claude-sonnet-5-low', 'gpt-5.5-off', 'gpt-5.5-low',
        'deepseek-v4-pro-off', 'deepseek-v4-pro-low']
FRONTIER = {'claude-sonnet-5-off', 'claude-sonnet-5-low', 'gpt-5.5-off', 'gpt-5.5-low',
            'deepseek-v4-pro-off', 'deepseek-v4-pro-low'}
SHORT = {'deepseek-v4-flash-low': 'DS-V4-Flash', 'gemini-3.1-flash-lite-low': 'Gemini-3.1-FL',
         'gpt-5.4-nano-low': 'GPT-5.4-nano', 'llama-3.3-70b': 'Llama-3.3-70B',
         'llama-4-maverick': 'Llama-4-Mav', 'qwen3.7-plus-low': 'Qwen3.7-Plus',
         'claude-sonnet-5-off': 'Sonnet-5 (off)', 'claude-sonnet-5-low': 'Sonnet-5 (low)',
         'gpt-5.5-off': 'GPT-5.5 (off)', 'gpt-5.5-low': 'GPT-5.5 (low)',
         'deepseek-v4-pro-off': 'DS-V4-Pro (off)', 'deepseek-v4-pro-low': 'DS-V4-Pro (low)'}


def spearman(pairs):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    xs, ys = zip(*pairs)
    return metrics.pearson(list(zip(rank(xs), rank(ys))))


PERMS, BOOT, SEED = 10_000, 10_000, 1234


def perm_p(pairs):
    """Two-sided permutation p for r: how often a shuffled pairing matches |r_obs|."""
    r_obs = metrics.pearson(pairs)
    if r_obs is None:          # constant on one axis: no association to test
        return 1.0
    obs = abs(r_obs)
    xs, ys = [x for x, _ in pairs], [y for _, y in pairs]
    rng = random.Random(SEED)
    hits = 0
    for _ in range(PERMS):
        rng.shuffle(ys)
        r = metrics.pearson(list(zip(xs, ys)))
        hits += r is not None and abs(r) >= obs
    return (hits + 1) / (PERMS + 1)


def boot_ci(pairs):
    """Percentile bootstrap 95% CI for r, resampling (model) points."""
    rng = random.Random(SEED + 1)
    rs = []
    for _ in range(BOOT):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        r = metrics.pearson(draw)
        if r is not None:
            rs.append(r)
    rs.sort()
    return rs[int(0.025 * len(rs))], rs[int(0.975 * len(rs)) - 1]


def main():
    doc = json.load(open('results/reports/evaluation_test.json'))
    pmm = {(r['method'], r['model']): r['mean_r'] for r in doc['per_method_model']}
    acc = {}
    for m in POOL:
        t = json.load(open(f'results/capability_mmlu/{m}/targets.json'))['targets']
        acc[m] = mean(v['rate'] for v in t.values() if v.get('rate') is not None)

    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), sharex=True)
    for ax, method, title in [(axes[0], 'self_report', 'self-report'),
                              (axes[1], 'informed_oracle', 'item-informed self-prediction')]:
        pairs = [(acc[m], pmm[(method, m)]) for m in POOL if pmm.get((method, m)) is not None]
        r = metrics.pearson(pairs)
        rho = spearman(pairs)
        # least-squares fit line over the observed x-range
        xs, ys = zip(*pairs)
        mx, my = mean(xs), mean(ys)
        beta = sum((x - mx) * (y - my) for x, y in pairs) / sum((x - mx) ** 2 for x in xs)
        x0, x1 = min(xs) - 0.005, max(xs) + 0.005
        ax.plot([x0, x1], [my + beta * (x0 - mx), my + beta * (x1 - mx)],
                color='#777777', lw=1.2, ls='-', zorder=2)
        for m in POOL:
            y = pmm.get((method, m))
            if y is None:
                continue
            fr = m in FRONTIER
            ax.scatter(acc[m], y, s=34, color='#4c72b0' if fr else '#dd8452',
                       marker='s' if fr else 'o', zorder=3)
            ax.annotate(SHORT[m], (acc[m], y), textcoords='offset points',
                        xytext=(4, 3), fontsize=5.5, color='#444444')
        p = perm_p(pairs)
        lo, hi = boot_ci(pairs)
        stats[method] = (r, rho, p, lo, hi, len(pairs))
        ax.set_title(f'{title}  ($r={r:+.2f}$, $\\rho={rho:+.2f}$)', fontsize=9)
        ax.set_xlabel('measured MMLU accuracy (this paper\'s subset, reasoning as run)',
                      fontsize=8)
        ax.axhline(0, ls='--', lw=0.8, color='gray')
        ax.tick_params(labelsize=8)
        print(f'{method}: pearson {r:+.3f}  spearman {rho:+.3f}  '
              f'p={p:.3f}  95% CI [{lo:+.2f}, {hi:+.2f}]  n={len(pairs)}')
    axes[0].set_ylabel('macro test correlation', fontsize=8)
    fig.tight_layout()
    out = Path('paper/generated')
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'fig_scaling.{ext}', dpi=180, bbox_inches='tight')

    # app:abl-settings quotes r, its p and whether the CI covers zero; export them so the
    # prose cannot drift from the figure (the p and CI were previously not computed at all).
    tex = ['% Auto-generated by scripts/capability_scaling_figure.py -- DO NOT EDIT BY HAND.',
           '% Association between measured MMLU accuracy and macro test correlation, over the',
           '% twelve pool entries. p = two-sided permutation over pairings; CI = percentile',
           '% bootstrap over entries.']
    for method, tag in (('self_report', 'Sr'), ('informed_oracle', 'Io')):
        r, rho, p, lo, hi, n = stats[method]
        tex += [f'\\newcommand{{\\scale{tag}R}}{{{r:+.2f}}}',
                f'\\newcommand{{\\scale{tag}Rho}}{{{rho:+.2f}}}',
                f'\\newcommand{{\\scale{tag}P}}{{{p:.2f}}}',
                f'\\newcommand{{\\scale{tag}CILo}}{{{lo:+.2f}}}',
                f'\\newcommand{{\\scale{tag}CIHi}}{{{hi:+.2f}}}',
                f'\\newcommand{{\\scale{tag}N}}{{{n}}}']
    (out / 'scaling.tex').write_text('\n'.join(tex) + '\n')
    print('Wrote paper/generated/fig_scaling.{png,pdf} and scaling.tex')


if __name__ == '__main__':
    main()
