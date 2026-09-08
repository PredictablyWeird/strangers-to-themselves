#!/usr/bin/env python3
"""Build the public release tree from the private working repository.

    python scripts/make_release.py SRC DST [--min-size-mb 1.0] [--exclude PREFIX ...]

Copies every git-tracked file of SRC into DST (minus the excluded prefixes), plus the
gitignored-but-required ``behavioral_sampling`` prediction files (``--include-ignored``), and
slims the large prediction files: for any ``results/**/*.json`` above the size threshold, the
per-condition ``raw`` (verbatim model transcripts) and ``scenarios`` (generated scenario
texts) fields are dropped from each entry under ``predictions``. Everything the scoring
pipeline reads (``predicted_rate``, ``n``, ``samples``, ``parse_failures``, ...) is kept
untouched, so ``bp-evaluate`` reproduces the paper's numbers from the slimmed tree. A
``release_note`` block is added to each slimmed file naming the dropped fields and pointing
at the full-record archive.

The un-slimmed files, the raw Inspect logs and the transcript/reasoning side-files are
published separately (see README, "Full artifacts").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

STRIP_FIELDS = ("raw", "scenarios")
ARCHIVE_NOTE = ("Full record (with these fields) in the companion Zenodo archive; "
                "see README 'Full artifacts'.")


def tracked_files(src: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(src), "ls-files", "-z"],
                         check=True, capture_output=True).stdout
    return [p.decode() for p in out.split(b"\0") if p]


def slim(doc: dict) -> tuple[dict, int]:
    """Drop STRIP_FIELDS from every entry under ``predictions``; return (doc, n_fields_dropped)."""
    preds = doc.get("predictions")
    dropped = 0
    if isinstance(preds, dict):
        for entry in preds.values():
            if isinstance(entry, dict):
                for f in STRIP_FIELDS:
                    if f in entry:
                        entry.pop(f)
                        dropped += 1
    if dropped:
        doc["release_note"] = {"stripped_fields": list(STRIP_FIELDS), "archive": ARCHIVE_NOTE}
    return doc, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--min-size-mb", type=float, default=1.0,
                    help="only JSON files at least this large are slimmed (default 1.0)")
    ap.add_argument("--exclude", nargs="*", default=[".claude/"],
                    help="tracked path prefixes to leave out of the release")
    ap.add_argument("--include-ignored", nargs="*",
                    default=["results/*/*/predictions/behavioral_sampling*.json"],
                    help="globs (relative to SRC) of untracked files that the scoring pipeline "
                         "needs and which the release therefore ships anyway")
    ap.add_argument("--drop-gitignore-lines", nargs="*",
                    default=["results/**/predictions/behavioral_sampling*.json",
                             ".claude/settings.local.json", ".claude/scheduled_tasks.lock"],
                    help="pattern lines to remove from the release .gitignore (their now-orphaned "
                         "comment lines go with them)")
    args = ap.parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"refusing to write into non-empty {dst}")
    threshold = int(args.min_size_mb * 1e6)

    files = [f for f in tracked_files(src) if not any(f.startswith(e) for e in args.exclude)]
    tracked = set(files)
    for g in args.include_ignored:
        files += sorted(str(p.relative_to(src)) for p in src.glob(g)
                        if p.is_file() and str(p.relative_to(src)) not in tracked)
    before = after = 0
    slimmed: list[tuple[str, int, int]] = []
    for rel in files:
        s, d = src / rel, dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        size = s.stat().st_size
        before += size
        if rel.startswith("results/") and rel.endswith(".json") and size >= threshold:
            with s.open() as fh:
                doc = json.load(fh)
            doc, dropped = slim(doc) if isinstance(doc, dict) else (doc, 0)
            if dropped:
                with d.open("w") as fh:
                    json.dump(doc, fh, indent=2)
                    fh.write("\n")
                slimmed.append((rel, size, d.stat().st_size))
                after += d.stat().st_size
                continue
        shutil.copy2(s, d)
        after += size

    # The release .gitignore must not ignore files the release ships, and carries no local
    # editor/agent state: drop those pattern lines, then any comment run left without a pattern.
    gi = dst / ".gitignore"
    if gi.exists() and args.drop_gitignore_lines:
        lines = [l for l in gi.read_text().splitlines() if l.strip() not in args.drop_gitignore_lines]
        out: list[str] = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("#"):
                j = i
                while j < len(lines) and lines[j].startswith("#"):
                    j += 1
                if j < len(lines) and lines[j].strip():   # comment run followed by a pattern
                    out += lines[i:j]
                i = j
                continue
            if lines[i].strip() or (out and out[-1].strip()):   # collapse repeated blank lines
                out.append(lines[i])
            i += 1
        gi.write_text("\n".join(out).rstrip("\n") + "\n")

    print(f"{len(files)} files copied; {len(slimmed)} slimmed")
    print(f"total {before/1e6:.0f} MB -> {after/1e6:.0f} MB")
    for rel, b, a in sorted(slimmed, key=lambda t: t[1] - t[2], reverse=True)[:15]:
        print(f"  {b/1e6:6.1f} -> {a/1e6:5.1f} MB  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
