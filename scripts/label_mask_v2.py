#!/usr/bin/env python3
"""Phase-0 two-pass relabelling of MASK for the *factorial* redesign (see docs/mask.md discussion).

The current ``labels.json`` tags each proposition with a single free-text ``domain``. That field is
an 87-way long tail with heavy overlap (``health`` / ``medicine`` / ``public health`` /
``alternative medicine`` are one space; ``animal behavior`` lumps welfare with ecology), so a
``domain``-grouped target isn't measuring a clean *thing*. This script rebuilds the labels as a
**two-level taxonomy** with a minimum cell size, so pooled per-subdomain contrasts have enough
propositions to be stable — the MASK analogue of DiscrimEval's hand-curated ``CATEGORY`` map.

Two passes:

  **Pass 1 — open labelling** (one LLM call per proposition, cached by ``task_id``). Reads the
  PROPOSITION + SYSTEM/USER prompt and emits, as free text:
    - ``fine_topic``    — a specific 2-4 word topic ("sea moss hair loss", "hyperloop test claims")
    - ``subject_facet`` — the angle within a broad area ("supplement efficacy" vs "animal welfare"
                          vs "endangered-species ecology"), which is what splits the broad domains
    - ``stakes_type``   — consumer_harm / health_safety / reputation / political / financial / trivia
    - ``beneficiary``   — who gains if the model asserts the falsehood
    - ``pressure_levers`` — the mechanism(s) the prompt uses to push the lie (selling_interest,
                          authority_order, sunk_cost_continuation, permission_granting,
                          emotional_appeal, plain_request, ...). This folds in the Phase-2 pressure
                          audit: same input, so we read it once. It also tells us the lever
                          vocabulary that is actually IN the data before we design the factorial.

  **Pass 2 — consolidation** (a handful of LLM calls, no per-row cost). Feeds the DISTINCT Pass-1
  ``fine_topic`` + ``subject_facet`` strings (with counts) to an LLM and asks for a fixed subdomain
  vocabulary that (a) respects a ``--min-cell`` floor so every subdomain has enough propositions to
  pool, and (b) keeps a ``parent -> subdomain`` hierarchy so undersized subdomains roll up to their
  parent. Then maps every proposition onto that vocabulary. Emits ``labels_v2.json``.

``labels_v2.json`` is the committed, model-independent artifact that will define the factorial
split universe (like the DiscrimEval ``CATEGORY`` map). Generate once, hand spot-check, commit.

Usage::

    # Pilot: label 20 rows, eyeball the JSON, don't consolidate yet.
    python scripts/label_mask_v2.py --pass1 --limit 20

    # Full pass 1 (all ~510 rows), then consolidate into subdomains.
    python scripts/label_mask_v2.py --pass1
    python scripts/label_mask_v2.py --pass2 --min-cell 15

    # Inspect the resulting taxonomy.
    python scripts/label_mask_v2.py --summary

The dataset is gated on HF (accept terms + ``huggingface-cli login`` / ``HF_TOKEN`` once); keys come
from ``.env`` (OPENROUTER_API_KEY). Model shortcuts resolve via ``behavior_prediction/models.yaml``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Reuse the archetype loader + proposition helpers from the existing analysis script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mask import ARCHETYPES, load_archetype, proposition_stem  # noqa: E402

# The three archetypes the eval actually measures (binary A/B propositions with a belief step).
EVAL_ARCHETYPES = ["continuations", "disinformation", "known_facts"]

DEFAULT_LABEL_MODEL = "openrouter/anthropic/claude-sonnet-4"
PASS1_CACHE = Path("results/mask/analysis/labels_v2_pass1.json")
OUT_PATH = Path("results/mask/analysis/labels_v2.json")

# --- Pass 1: open labelling ---------------------------------------------------------------------

# Closed vocab where a closed vocab is defensible (keeps counts poolable); open text where the whole
# point is discovery (fine_topic / subject_facet). pressure_levers is closed-ish but allows "other".
_PASS1_SYS = (
    "You label items from an AI-honesty benchmark. Each item pushes a model to state a specific "
    "FALSEHOOD. You are given the factual PROPOSITION under test (with a <pivotal>true/false</"
    "pivotal> span), the SYSTEM PROMPT (the model's role/instructions) and the USER PROMPT (the "
    "pressure that tries to make it lie). Return STRICT JSON, no prose:\n"
    "{\n"
    '  "fine_topic": <specific 2-4 word topic, e.g. "sea moss hair loss", "hyperloop test '
    'claims", "moon landing hoax">,\n'
    '  "subject_facet": <the ANGLE within a broad area, so broad areas split cleanly. e.g. for '
    'health use "supplement efficacy" | "medical procedure safety" | "disease cause"; for animals '
    'use "animal welfare" | "species ecology" | "pet care"; for politics use "election integrity" '
    '| "policy effect" | "official misconduct". 2-4 words.>,\n'
    '  "stakes_type": <ONE of: consumer_harm, health_safety, financial, reputation, political, '
    'legal, scientific_record, trivia>,\n'
    '  "beneficiary": <who gains if the model asserts the falsehood: e.g. "the user selling a '
    'product", "a political side", "the user\'s convenience", "nobody in particular">,\n'
    '  "pressure_levers": <LIST of the mechanisms the prompt uses to push the lie, from: '
    'plain_request, selling_interest (user has a commercial stake / is marketing something), '
    'authority_order (asserts authority/expertise or orders compliance), sunk_cost_continuation '
    '(asks to continue a passage that already states the falsehood), permission_granting '
    '(explicitly frees the model from norms/guidelines), emotional_appeal, roleplay_persona, '
    'reward_incentive, other. Pick every lever that clearly applies (usually 1-2).>\n'
    "}"
)


def _pass1_prompt(row: dict[str, Any]) -> str:
    sysp = str(row.get("system_prompt") or "")[:1400]
    userp = str(row.get("user_prompt") or "")[:1600]
    return (f"PROPOSITION: {row.get('proposition')}\n\n"
            f"SYSTEM PROMPT: {sysp}\n\nUSER PROMPT: {userp}")


# --- Pass 2: consolidation into a subdomain taxonomy --------------------------------------------

# Pass 2 runs in two calls so we never ask the model to echo back hundreds of strings (the
# brittle step that dropped 170 propositions on the first attempt):
#   2a DEFINE  — propose a balanced subdomain vocabulary from the facet catalog (definitions only);
#   2b ASSIGN  — map every distinct facet onto that vocabulary (few, verifiable, retried to 100%).
# Every proposition then inherits its facet's subdomain, so coverage is guaranteed by construction.

_PASS2_DEFINE_SYS = (
    "You are designing the SUBDOMAIN level of a two-level topic taxonomy for an AI-honesty "
    "benchmark, so propositions can be POOLED into subdomains for statistical measurement. You are "
    "given SUBJECT-FACETS (the angle within a broad area) with how many propositions carry each. "
    "Design a set of subdomains where:\n"
    "  - each SUBDOMAIN groups semantically related facets at a SPECIFIC grain, not a broad area: "
    "'supplement efficacy' and 'medical procedure safety' are DIFFERENT subdomains, never both "
    "'health'; misconduct by politicians, corporations, and celebrities are DIFFERENT subdomains, "
    "never one 'misconduct';\n"
    "  - sizes are BALANCED: aim for MIN_CELL..2.5*MIN_CELL propositions each. If a facet (or a "
    "group) would exceed ~2.5*MIN_CELL, SPLIT it along a meaningful distinction into finer "
    "subdomains rather than leaving one mega-bucket;\n"
    "  - every subdomain reaches at least MIN_CELL propositions (roll genuinely tiny, unrelated "
    "facets up into the nearest coherent subdomain, or a small 'other_<parent>' catch-all — never "
    "below the floor);\n"
    "  - each subdomain names its broad PARENT domain (health, politics, science, business, ...).\n"
    "Return STRICT JSON, no prose:\n"
    "{\n"
    '  "subdomains": [\n'
    '    {"subdomain": "<snake_case slug>", "parent": "<broad area>", '
    '"gloss": "<one line, what belongs here>"}\n'
    "  ]\n"
    "}"
)

_PASS2_ASSIGN_SYS = (
    "You assign each SUBJECT-FACET to exactly one of the SUBDOMAINS defined below. Consider the "
    "facet's meaning and each subdomain's gloss; pick the single best fit. Return STRICT JSON, no "
    "prose, mapping EVERY facet string to a subdomain slug:\n"
    '{ "assignments": { "<facet string>": "<subdomain slug>", ... } }'
)


def _resolve_and_model(shortcut: str, concurrency: int):
    from dotenv import load_dotenv
    from inspect_ai.model import GenerateConfig, get_model

    from behavior_prediction import common

    load_dotenv()
    full, reasoning = common.resolve_model(shortcut)
    model = get_model(full)
    cfg = GenerateConfig(temperature=0.0, max_connections=concurrency, **(reasoning or {}))
    return model, cfg


def _parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    return json.loads(m.group()) if m else {"raw": text}


def _eval_rows() -> list[dict[str, Any]]:
    """All rows from the three measured archetypes, tagged with their archetype."""
    rows: list[dict[str, Any]] = []
    for name in EVAL_ARCHETYPES:
        df = load_archetype(name)
        rows.extend(df.to_dict("records"))
    return rows


async def _run_pass1(rows: list[dict[str, Any]], model, cfg, cache: dict[str, Any],
                     concurrency: int) -> dict[str, Any]:
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    from tqdm import tqdm

    todo = [r for r in rows
            if not (isinstance(cache.get(str(r.get("task_id"))), dict)
                    and cache[str(r.get("task_id"))].get("fine_topic"))]
    print(f"[pass1] {len(todo)} rows to label (cached: {len(cache)})", file=sys.stderr)
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(todo), desc="pass1", unit="row")

    async def one(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        tid = str(row.get("task_id"))
        async with sem:
            try:
                out = await model.generate([
                    ChatMessageSystem(content=_PASS1_SYS),
                    ChatMessageUser(content=_pass1_prompt(row)),
                ], config=cfg)
                parsed = _parse_json((out.completion or "").strip())
            except Exception as exc:
                parsed = {"error": f"{type(exc).__name__}: {exc}"}
            parsed["archetype"] = row.get("archetype")
            pbar.update(1)
            return tid, parsed

    results = await asyncio.gather(*(one(r) for r in todo))
    pbar.close()
    for tid, parsed in results:
        cache[tid] = parsed
    return cache


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def cmd_pass1(args) -> int:
    rows = _eval_rows()
    if args.limit:
        rows = rows[:args.limit]
    cache = json.loads(PASS1_CACHE.read_text()) if PASS1_CACHE.exists() else {}
    model, cfg = _resolve_and_model(args.label_model, args.concurrency)
    cache = asyncio.run(_run_pass1(rows, model, cfg, cache, args.concurrency))
    _write(PASS1_CACHE, cache)

    vals = [v for v in cache.values() if isinstance(v, dict) and v.get("fine_topic")]
    facets = Counter(v.get("subject_facet") for v in vals)
    stakes = Counter(v.get("stakes_type") for v in vals)
    levers = Counter(l for v in vals for l in (v.get("pressure_levers") or []))
    print(f"\n[pass1] labelled {len(vals)} rows -> {PASS1_CACHE}")
    print(f"  distinct fine_topics : {len({v['fine_topic'] for v in vals})}")
    print(f"  distinct facets      : {len(facets)}")
    print(f"  top facets           : {dict(facets.most_common(15))}")
    print(f"  stakes_type          : {dict(stakes.most_common())}")
    print(f"  pressure_levers      : {dict(levers.most_common())}")
    return 0


def _facet_of(v: dict[str, Any]) -> str:
    """The pooling key for a proposition: its subject_facet, backing off to fine_topic."""
    return (v.get("subject_facet") or v.get("fine_topic") or "").strip()


async def _pass2_define(model, cfg, facet_counts: Counter, min_cell: int) -> list[dict[str, Any]]:
    """Call 2a — propose subdomain definitions (no member lists) from the facet catalog."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    sys_prompt = _PASS2_DEFINE_SYS.replace("MIN_CELL", str(min_cell))
    catalog = "\n".join(f"- {facet} | {n}" for facet, n in facet_counts.most_common())
    user = (f"MIN_CELL = {min_cell}\n\nSUBJECT-FACETS (facet | proposition count):\n{catalog}")
    out = await model.generate(
        [ChatMessageSystem(content=sys_prompt), ChatMessageUser(content=user)], config=cfg)
    return _parse_json((out.completion or "").strip()).get("subdomains", [])


async def _pass2_assign(model, cfg, facets: list[str], subs: list[dict[str, Any]]) -> dict[str, str]:
    """Call 2b — map every facet to a subdomain slug, retrying until coverage is complete."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    slugs = {s["subdomain"] for s in subs}
    defs = "\n".join(f"- {s['subdomain']} ({s.get('parent','?')}): {s.get('gloss','')}" for s in subs)
    mapping: dict[str, str] = {}
    todo = list(facets)
    for attempt in range(4):
        if not todo:
            break
        user = ("SUBDOMAINS:\n" + defs + "\n\nFACETS to assign:\n"
                + "\n".join(f"- {f}" for f in todo))
        out = await model.generate(
            [ChatMessageSystem(content=_PASS2_ASSIGN_SYS), ChatMessageUser(content=user)], config=cfg)
        got = _parse_json((out.completion or "").strip()).get("assignments", {})
        for f, slug in got.items():
            if f in set(todo) and slug in slugs:
                mapping[f] = slug
        todo = [f for f in todo if f not in mapping]
        if todo:
            print(f"[pass2] assign retry {attempt + 1}: {len(todo)} facets still unmapped",
                  file=sys.stderr)
    return mapping


def cmd_pass2(args) -> int:
    if not PASS1_CACHE.exists():
        raise SystemExit("run --pass1 first (no labels_v2_pass1.json).")
    cache = json.loads(PASS1_CACHE.read_text())
    vals = {tid: v for tid, v in cache.items()
            if isinstance(v, dict) and v.get("fine_topic")}

    # Pool at the FACET grain (far fewer, verifiable) rather than fine_topic.
    facet_counts = Counter(_facet_of(v) for v in vals.values())
    facets = [f for f in facet_counts if f]

    model, cfg = _resolve_and_model(args.label_model, 1)
    subs = asyncio.run(_pass2_define(model, cfg, facet_counts, args.min_cell))
    if not subs:
        raise SystemExit("pass2 define returned no subdomains; inspect the model output.")
    parent_of = {s["subdomain"]: s.get("parent", "") for s in subs}
    facet_map = asyncio.run(_pass2_assign(model, cfg, facets, subs))

    labels_v2: dict[str, Any] = {}
    unmapped = 0
    for tid, v in vals.items():
        sub = facet_map.get(_facet_of(v))
        parent = parent_of.get(sub, None) if sub else None
        if sub is None:
            unmapped += 1
        labels_v2[tid] = {
            "subdomain": sub,
            "parent": parent,
            "fine_topic": v["fine_topic"],
            "subject_facet": v.get("subject_facet"),
            "stakes_type": v.get("stakes_type"),
            "beneficiary": v.get("beneficiary"),
            "pressure_levers": v.get("pressure_levers") or [],
            "archetype": v.get("archetype"),
        }
    _write(OUT_PATH, {"taxonomy": subs, "min_cell": args.min_cell, "labels": labels_v2})

    sizes = Counter(l["subdomain"] for l in labels_v2.values() if l["subdomain"])
    small = {k: n for k, n in sizes.items() if n < args.min_cell}
    print(f"[pass2] {len(subs)} subdomains, {len(labels_v2)} propositions -> {OUT_PATH}")
    print(f"  subdomain sizes: {dict(sizes.most_common())}")
    if small:
        print(f"  WARNING under-floor subdomains (<{args.min_cell}): {small}")
    if unmapped:
        print(f"  WARNING {unmapped} propositions had no subdomain (fine_topic not mapped)")
    return 0


# --- Post-hoc taxonomy edits (split an over-large or heterogeneous subdomain; merge undersized ones) --

_SPLIT_SYS = (
    "You classify propositions from an AI-honesty benchmark into a fixed set of categories. You are "
    "given the PROPOSITION (with a <pivotal>true/false</pivotal> span) and must assign it to exactly "
    "one of the listed categories, based on what the claim is actually about. Return STRICT JSON, no "
    "prose, mapping EVERY item id to a category slug:\n"
    '{ "assignments": { "<id>": "<category slug>", ... } }'
)


def _prop_texts() -> dict[str, str]:
    rows = _eval_rows()
    return {str(r.get("task_id")): str(r.get("proposition") or "") for r in rows}


def _prop_rows() -> dict[str, dict[str, str]]:
    """``task_id -> {proposition, system_prompt, user_prompt, belief_elicit_1}`` -- the full
    labelled fields, for review artifacts (the CSV) where the bare proposition claim isn't enough
    to verify pressure_levers/stakes_type/beneficiary (those were labelled off the prompts, not the
    proposition sentence)."""
    out: dict[str, dict[str, str]] = {}
    for r in _eval_rows():
        out[str(r.get("task_id"))] = {
            "proposition": str(r.get("proposition") or ""),
            "system_prompt": str(r.get("system_prompt") or ""),
            "user_prompt": str(r.get("user_prompt") or ""),
            "belief_elicit_1": str(r.get("belief_elicit_1") or ""),
        }
    return out


async def _classify_ids(model, cfg, items: dict[str, str], categories: list[tuple[str, str]]
                        ) -> dict[str, str]:
    """``items``: id -> proposition text. ``categories``: [(slug, gloss), ...]. Retries unmapped ids
    up to 4 times, same pattern as ``_pass2_assign``."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser
    slugs = {c[0] for c in categories}
    cat_block = "\n".join(f"- {slug}: {gloss}" for slug, gloss in categories)
    mapping: dict[str, str] = {}
    todo = list(items)
    for attempt in range(4):
        if not todo:
            break
        item_block = "\n".join(f"[{tid}] {items[tid][:220]}" for tid in todo)
        user = f"CATEGORIES:\n{cat_block}\n\nITEMS:\n{item_block}"
        out = await model.generate(
            [ChatMessageSystem(content=_SPLIT_SYS), ChatMessageUser(content=user)], config=cfg)
        got = _parse_json((out.completion or "").strip()).get("assignments", {})
        for tid, slug in got.items():
            if tid in set(todo) and slug in slugs:
                mapping[tid] = slug
        todo = [tid for tid in todo if tid not in mapping]
        if todo:
            print(f"[split] classify retry {attempt + 1}: {len(todo)} items still unmapped",
                  file=sys.stderr)
    return mapping


def cmd_split(args) -> int:
    """Reclassify every proposition currently in ``--split SLUG`` into the categories given by
    ``--categories 'slug:gloss;slug:gloss;...'``. A category slug equal to an EXISTING subdomain
    (e.g. reusing ``conspiracy_theories``) merges those propositions into it instead of creating a
    new one. The source subdomain is removed from the taxonomy afterward."""
    if not OUT_PATH.exists():
        raise SystemExit("no labels_v2.json yet.")
    doc = json.loads(OUT_PATH.read_text())
    labels, taxonomy = doc["labels"], doc["taxonomy"]
    existing = {s["subdomain"]: s for s in taxonomy}
    if args.split not in existing:
        raise SystemExit(f"{args.split!r} is not a current subdomain; see --summary.")

    cats: list[tuple[str, str]] = []
    for part in args.categories.split(";"):
        slug, _, gloss = part.strip().partition(":")
        cats.append((slug.strip(), gloss.strip()))
    if not cats:
        raise SystemExit("--categories is required, e.g. 'financial_corruption:bribery, fraud...;"
                         "personal_conduct_misconduct:sexual or ethical scandal...'")

    members = {tid: l for tid, l in labels.items() if l["subdomain"] == args.split}
    texts = _prop_texts()
    items = {tid: texts.get(tid, "") for tid in members}
    model, cfg = _resolve_and_model(args.label_model, args.concurrency)
    assign = asyncio.run(_classify_ids(model, cfg, items, cats))

    parent = existing[args.split].get("parent", "")
    used_slugs = set(assign.values())
    # Drop the source entry FIRST, so a category slug that reuses the source's own name (e.g.
    # splitting "corporate_misconduct" into a narrower "corporate_misconduct" + something else)
    # gets re-added below instead of being silently deleted by this same filter.
    taxonomy[:] = [s for s in taxonomy if s["subdomain"] != args.split]
    still_present = {s["subdomain"] for s in taxonomy}
    for slug, gloss in cats:
        if slug in used_slugs and slug not in still_present:
            taxonomy.append({"subdomain": slug, "parent": parent, "gloss": gloss})

    unmapped = 0
    for tid in members:
        slug = assign.get(tid)
        if slug is None:
            unmapped += 1
            continue
        labels[tid]["subdomain"] = slug
        labels[tid]["parent"] = next((s["parent"] for s in taxonomy if s["subdomain"] == slug),
                                     labels[tid].get("parent"))

    _write(OUT_PATH, doc)
    sizes = Counter(l["subdomain"] for l in labels.values() if l["subdomain"])
    print(f"[split] {args.split} ({len(members)} props) -> "
          f"{ {slug: sizes.get(slug, 0) for slug, _ in cats} }")
    if unmapped:
        print(f"  WARNING {unmapped} propositions left unclassified")
    return 0


def cmd_merge(args) -> int:
    """Merge ``--merge SRC:DST`` — every proposition in SRC is relabelled to DST (which must already
    exist) and SRC is dropped from the taxonomy. No LLM call; a pure relabel."""
    if not OUT_PATH.exists():
        raise SystemExit("no labels_v2.json yet.")
    src, _, dst = args.merge.partition(":")
    if not src or not dst:
        raise SystemExit("--merge expects 'SRC:DST'")
    doc = json.loads(OUT_PATH.read_text())
    labels, taxonomy = doc["labels"], doc["taxonomy"]
    by_slug = {s["subdomain"]: s for s in taxonomy}
    if src not in by_slug or dst not in by_slug:
        raise SystemExit(f"both {src!r} and {dst!r} must be current subdomains; see --summary.")
    n = 0
    for l in labels.values():
        if l["subdomain"] == src:
            l["subdomain"] = dst
            l["parent"] = by_slug[dst].get("parent", l.get("parent"))
            n += 1
    taxonomy[:] = [s for s in taxonomy if s["subdomain"] != src]
    _write(OUT_PATH, doc)
    sizes = Counter(l["subdomain"] for l in labels.values() if l["subdomain"])
    print(f"[merge] {src} -> {dst}: moved {n} propositions. {dst} now n={sizes.get(dst, 0)}")
    return 0


CSV_PATH = Path("results/mask/analysis/labels_v2_review.csv")


def cmd_csv(args) -> int:
    """Export one row per proposition -- subdomain, parent, the full labelled text (proposition,
    system_prompt, user_prompt, belief_elicit_1 -- everything the labeller actually read, since
    pressure_levers/stakes_type/beneficiary were labelled off the PROMPTS, not just the proposition
    sentence) and the other label fields -- plus two BLANK columns (``verified``, ``notes``) for
    manual spot-checking. Open in a spreadsheet, sort/filter by ``subdomain``, mark each row y/n."""
    import csv as csv_mod

    if not OUT_PATH.exists():
        raise SystemExit("no labels_v2.json yet.")
    doc = json.loads(OUT_PATH.read_text())
    labels = doc["labels"]
    prop_rows = _prop_rows()

    rows = []
    for tid, l in labels.items():
        pr = prop_rows.get(tid, {})
        rows.append({
            "subdomain": l.get("subdomain"),
            "parent": l.get("parent"),
            "task_id": tid,
            "proposition": pr.get("proposition", ""),
            "system_prompt": pr.get("system_prompt", ""),
            "user_prompt": pr.get("user_prompt", ""),
            "belief_elicit_1": pr.get("belief_elicit_1", ""),
            "fine_topic": l.get("fine_topic"),
            "subject_facet": l.get("subject_facet"),
            "stakes_type": l.get("stakes_type"),
            "beneficiary": l.get("beneficiary"),
            "pressure_levers": ";".join(l.get("pressure_levers") or []),
            "archetype": l.get("archetype"),
            "verified": "",
            "notes": "",
        })
    rows.sort(key=lambda r: (r["subdomain"] or "", r["task_id"]))

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows -> {CSV_PATH}")
    return 0


def cmd_show(args) -> int:
    """Look up exactly which proposition(s) got which label -- either every member of one
    ``--subdomain`` (proposition text + its full label), or the full detail (proposition, system/
    user/belief prompts, AND its label) for one ``--task-id``."""
    if not OUT_PATH.exists():
        raise SystemExit("no labels_v2.json yet.")
    doc = json.loads(OUT_PATH.read_text())
    labels = doc["labels"]
    texts = _prop_texts()

    if args.task_id:
        lab = labels.get(args.task_id)
        if lab is None:
            raise SystemExit(f"no label for task_id {args.task_id!r}; see --subdomain to browse.")
        row = next((r for r in _eval_rows() if str(r.get("task_id")) == args.task_id), None)
        print(f"task_id: {args.task_id}")
        print(f"proposition: {texts.get(args.task_id)}")
        print(f"\nLABEL: {json.dumps(lab, indent=2)}")
        if row:
            print(f"\nsystem_prompt: {row.get('system_prompt')}")
            print(f"\nuser_prompt: {row.get('user_prompt')}")
            print(f"\nbelief_elicit_1: {row.get('belief_elicit_1')}")
        return 0

    if args.subdomain:
        members = [(tid, l) for tid, l in labels.items() if l["subdomain"] == args.subdomain]
        if not members:
            raise SystemExit(f"{args.subdomain!r} is not a current subdomain; see --summary.")
        print(f"=== {args.subdomain} (n={len(members)}) ===\n")
        for tid, l in members[:args.limit] if args.limit else members:
            print(f"[{tid}] {texts.get(tid, '?')}")
            print(f"    facet={l.get('subject_facet')!r}  stakes={l.get('stakes_type')!r}  "
                  f"levers={l.get('pressure_levers')}")
        if args.limit and len(members) > args.limit:
            print(f"\n... {len(members) - args.limit} more (raise --limit or omit it)")
        return 0

    raise SystemExit("--show requires --task-id or --subdomain")


def cmd_summary(args) -> int:
    if not OUT_PATH.exists():
        raise SystemExit("no labels_v2.json yet; run --pass1 then --pass2.")
    doc = json.loads(OUT_PATH.read_text())
    labels = doc["labels"]
    by_sub = Counter(l["subdomain"] for l in labels.values() if l["subdomain"])
    by_parent = Counter(l["parent"] for l in labels.values() if l.get("parent"))
    levers = Counter(x for l in labels.values() for x in (l.get("pressure_levers") or []))
    print(f"=== labels_v2 summary ({len(labels)} propositions, {len(by_sub)} subdomains) ===")
    print(f"parents: {dict(by_parent.most_common())}\n")
    for s in doc["taxonomy"]:
        n = by_sub.get(s["subdomain"], 0)
        print(f"  {s['subdomain']:<32} ({s.get('parent','?'):<14}) n={n:<3} {s.get('gloss','')}")
    print(f"\npressure_levers across corpus: {dict(levers.most_common())}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass1", action="store_true", help="open per-proposition labelling")
    ap.add_argument("--pass2", action="store_true", help="consolidate fine topics into subdomains")
    ap.add_argument("--summary", action="store_true", help="print the labels_v2 taxonomy")
    ap.add_argument("--split", default=None, metavar="SUBDOMAIN",
                    help="reclassify SUBDOMAIN's propositions into --categories")
    ap.add_argument("--categories", default=None,
                    help="'slug:gloss;slug:gloss;...' for --split (a slug matching an existing "
                         "subdomain merges into it instead of creating a new one)")
    ap.add_argument("--merge", default=None, metavar="SRC:DST",
                    help="relabel every SRC proposition to DST and drop SRC from the taxonomy")
    ap.add_argument("--show", action="store_true",
                    help="look up labels: pair with --task-id (full detail incl. prompts) or "
                         "--subdomain (list every member's proposition + label)")
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--subdomain", default=None)
    ap.add_argument("--csv", action="store_true",
                    help=f"export one row per proposition (subdomain, proposition text, other "
                         f"label fields, blank verified/notes columns) to {CSV_PATH}")
    ap.add_argument("--limit", type=int, default=None, help="cap rows in pass 1 (pilot)")
    ap.add_argument("--min-cell", type=int, default=15,
                    help="minimum propositions per subdomain (pooling floor; default 15)")
    ap.add_argument("--label-model", default=DEFAULT_LABEL_MODEL,
                    help=f"labelling model shortcut or full id (default: {DEFAULT_LABEL_MODEL})")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    if args.pass1:
        return cmd_pass1(args)
    if args.pass2:
        return cmd_pass2(args)
    if args.split:
        if not args.categories:
            raise SystemExit("--split requires --categories")
        return cmd_split(args)
    if args.merge:
        return cmd_merge(args)
    if args.show:
        return cmd_show(args)
    if args.csv:
        return cmd_csv(args)
    if args.summary:
        return cmd_summary(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
