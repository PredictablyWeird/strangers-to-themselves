#!/usr/bin/env python3
"""Repair frozen-test report_mean / oracle_report_mean predictions (2026-08-19).

Bug: the 12 frozen-test lanes computed the donor-derived methods (report_mean,
oracle_report_mean — zero API calls, they average the donor pool's stored
self_report / informed_oracle predictions) whenever each lane reached them,
racing the donor lanes' own progress. Test entries therefore averaged PARTIAL
donor sets (contributor n as low as 1, vs n=8 on dev). Dev entries are
untouched by this script.

Fix, now that all 12 lanes are complete: strip the test-split keys from every
tuned report_mean/oracle_report_mean prediction file (backing the files up
first), then re-run the canonical test top-up (bp-benchmark --include-test
--tuned-only) so the entries are recomputed from the full donor files.
Deterministic given the donor files; no model calls.

Usage: python scripts/fix_reportmean_test_donors.py --backup DIR   (strip step)
Then:  bp-benchmark --phase generate --include-test --tuned-only \
           --methods report_mean,oracle_report_mean --evals <8> --models <12>
"""
import argparse
import json
import tarfile
from pathlib import Path

from behavior_prediction import common, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.tuning import ACTIVE_EVALS

POOL = ['deepseek-v4-flash-low', 'gemini-3.1-flash-lite-low', 'gpt-5.4-nano-low',
        'llama-3.3-70b', 'llama-4-maverick', 'qwen3.7-plus-low',
        'claude-sonnet-5-off', 'claude-sonnet-5-low', 'gpt-5.5-off', 'gpt-5.5-low',
        'deepseek-v4-pro-off', 'deepseek-v4-pro-low']
METHODS = ['report_mean', 'oracle_report_mean']


def test_pred_keys(spec, targets, manifest):
    """Test-split keys in the *prediction* file's key space (contrast keys for bias evals)."""
    keys = splits.split_keys(spec, targets, manifest, 'test')
    if getattr(spec, 'scoring_semantics', 'absolute_rate') != 'bias_contrast':
        return set(keys)
    out = set()
    for k in keys:
        c = spec.condition_contrast(targets[k]['condition'])
        if c is not None:
            out.add(c['key'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backup', required=True, help='directory for the pre-strip tar backup')
    args = ap.parse_args()

    root = Path(common.results_root())
    bak = Path(args.backup) / 'reportmean_test_prestrip.tar'
    bak.parent.mkdir(parents=True, exist_ok=True)
    stripped = files = 0
    with tarfile.open(bak, 'w') as tf:
        for ev in ACTIVE_EVALS:
            spec = get_spec(ev)
            manifest = splits.load_manifest(ev)
            tuned = json.load(open(root / ev / 'tuning.json'))['methods']
            for bm in METHODS:
                if not tuned.get(bm, {}).get('best_setting'):
                    continue
                fname = tuned[bm]['best_setting'] + '.json'
                for m in POOL:
                    f = root / ev / m / 'predictions' / fname
                    if not f.exists():
                        continue
                    targets = json.load(open(root / ev / m / 'targets.json'))['targets']
                    tk = test_pred_keys(spec, targets, manifest)
                    doc = json.load(open(f))
                    keep = {k: v for k, v in doc['predictions'].items() if k not in tk}
                    removed = len(doc['predictions']) - len(keep)
                    if removed:
                        tf.add(f, arcname=str(f.relative_to(root)))
                        doc['predictions'] = keep
                        common.save_json(doc, str(f))
                        stripped += removed
                        files += 1
    print(f'backup: {bak}')
    print(f'stripped {stripped} test entries across {files} files')


if __name__ == '__main__':
    main()
