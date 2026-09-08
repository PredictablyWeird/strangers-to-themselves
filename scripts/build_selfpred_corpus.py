#!/usr/bin/env python3
"""Build the self-prediction corpus: topic clusters, LLM labels, and a preview payload.

Spec: docs/selfpred-corpus-v1.md. A *condition* is (category x topic cluster x polarity). The
prediction target is the model's own affirmative-answer rate on that family, and — the primary
target — the polarity **gap**, which the v0 headroom check showed is the model-specific part
(self-report r +0.273/+0.615 vs a cross-model prior's -0.066).

Clustering is TF-IDF + spherical k-means in numpy (no sklearn/torch in this venv), seeded, so the
assignment is reproducible. Labels are one short noun phrase per cluster, written by an LLM from six
example items plus the cluster's distinctive terms.

Stages (state under --out):
  cluster  per category: TF-IDF over item text, k-means -> cluster assignment + top terms
  label    one LLM call per cluster -> a short human-readable name
  preview  emit corpus_preview.json for the HTML viewer (scripts/render_corpus_viewer.py)

Usage:
  .venv/bin/python scripts/build_selfpred_corpus.py --evals-repo <clone> --stage cluster --k 10
  .venv/bin/python scripts/build_selfpred_corpus.py --evals-repo <clone> --stage label
  .venv/bin/python scripts/build_selfpred_corpus.py --evals-repo <clone> --stage preview
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_introspection as intro  # noqa: E402  (load_jsonl / save_jsonl)
import screen_condition_families as S  # noqa: E402
import selfpred_headroom as H  # noqa: E402

STOP = set("""the a an and or but if then than that this these those is are was were be been being do
does did doing have has had having i you he she it we they them his her its their my your our will
would shall should can could may might must not no nor so as at by for with about into over after
under of on in to from up down out off again once here there all any both each few more most other
some such only own same too very just now what which who whom whose when where why how something
statement following say would like want think feel believe answer question yes""".split())


def item_text(it: dict) -> str:
    return (it.get("statement") or it["question"]).strip()


def tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP]


def tfidf(texts: list[str], max_vocab: int = 2000) -> tuple[np.ndarray, list[str]]:
    docs = [tokens(t) for t in texts]
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    vocab = [w for w, _ in sorted(df.items(), key=lambda kv: -kv[1])[:max_vocab]]
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for r, d in enumerate(docs):
        for w in d:
            if w in idx:
                X[r, idx[w]] += 1.0
    idf = np.log((1 + len(docs)) / (1 + np.array([df[w] for w in vocab], dtype=np.float32))) + 1.0
    X *= idf
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-9), vocab


def _assign_capped(sim: np.ndarray, cap: int) -> np.ndarray:
    """Greedy assignment of rows to columns, no column exceeding ``cap``. Highest similarity first."""
    n, k = sim.shape
    order = np.argsort(-sim, axis=None)
    labels = np.full(n, -1, dtype=int)
    counts = np.zeros(k, dtype=int)
    for flat in order:
        i, j = divmod(int(flat), k)
        if labels[i] == -1 and counts[j] < cap:
            labels[i] = j
            counts[j] += 1
    for i in np.where(labels == -1)[0]:  # leftovers -> emptiest cluster
        j = int(np.argmin(counts))
        labels[i] = j
        counts[j] += 1
    return labels


def kmeans(X: np.ndarray, k: int, seed: int, pol: np.ndarray | None = None,
           iters: int = 15) -> np.ndarray:
    """Spherical k-means with k-means++ init.

    When ``pol`` (a boolean POS mask) is given, each polarity half is assigned under a size cap of
    ceil(n_half / k), so every (cluster x polarity) condition ends up with ~n/(2k) items. Balance is
    what makes a condition's rate estimable and its polarity *gap* well-defined; unconstrained
    k-means produced clusters of size 1 and empty halves. The cost is slightly less topical purity.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d = np.maximum(1.0 - np.max(X @ np.array(centers).T, axis=1), 0) ** 2
        centers.append(X[rng.choice(n, p=d / d.sum() if d.sum() > 0 else None)])
    C = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        sim = X @ C.T
        if pol is None:
            labels = np.argmax(sim, axis=1)
        else:
            for half in (pol, ~pol):
                idx = np.where(half)[0]
                if idx.size:
                    labels[idx] = _assign_capped(sim[idx], int(np.ceil(idx.size / k)))
        for j in range(k):
            m = labels == j
            if m.any():
                v = X[m].sum(0)
                C[j] = v / max(np.linalg.norm(v), 1e-9)
            else:
                C[j] = X[np.argmin(np.max(X @ C.T, axis=1))]
    return labels


def top_terms(X: np.ndarray, vocab: list[str], mask: np.ndarray, n: int = 6) -> list[str]:
    if not mask.any():
        return []
    mean_in, mean_out = X[mask].mean(0), X[~mask].mean(0) if (~mask).any() else 0
    score = mean_in - mean_out
    return [vocab[i] for i in np.argsort(-score)[:n]]


def stage_cluster(repo: Path, out: Path, k: int, seed: int, max_items: int) -> None:
    cats = S.load_categories(repo)
    doc = {}
    for ci, (cat, items) in enumerate(sorted(cats.items()), 1):
        items = items[:max_items]
        texts = [item_text(it) for it in items]
        X, vocab = tfidf(texts)
        kk = min(k, max(2, len(items) // 40))  # >= ~20 items per (cluster x polarity) cell
        pol = np.array([H.polarity(it) == "POS" for it in items])
        lab = kmeans(X, kk, seed, pol=pol)
        clusters = []
        for j in range(kk):
            m = lab == j
            idxs = [i for i in range(len(items)) if m[i]]
            pols = [H.polarity(items[i]) for i in idxs]
            clusters.append({
                "cluster": j,
                "n": len(idxs),
                "n_pos": sum(1 for p in pols if p == "POS"),
                "n_neg": sum(1 for p in pols if p == "NEG"),
                "terms": top_terms(X, vocab, m),
                "examples": [texts[i] for i in idxs[:6]],
                "item_indices": idxs,
            })
        doc[cat] = {"k": kk, "n_items": len(items), "clusters": clusters}
        if ci % 25 == 0:
            print(f"  clustered {ci}/{len(cats)}")
    (out / "clusters.json").write_text(json.dumps(doc, indent=1))
    sizes = [c["n"] for v in doc.values() for c in v["clusters"]]
    cells = [min(c["n_pos"], c["n_neg"]) for v in doc.values() for c in v["clusters"]]
    print(f"\n{len(doc)} categories, {sum(len(v['clusters']) for v in doc.values())} clusters")
    print(f"cluster size: min {min(sizes)} median {int(np.median(sizes))} max {max(sizes)}")
    print(f"smaller polarity half per cluster: min {min(cells)} median {int(np.median(cells))}")
    print(f"conditions (cluster x polarity): {2*sum(len(v['clusters']) for v in doc.values())}")


LABEL_PROMPT = """A set of questions probing "{cat}" has been split into {k} groups by topic.

{groups}

Name each group with a short noun phrase (2-6 words) saying what its questions are about. Each name
will be shown to a model as "questions about <name>", so the {k} names MUST be clearly different from
one another — they are what distinguishes these groups. Do not simply restate "{cat}".

Reply with exactly {k} lines, in order, formatted `1. <name>`. Nothing else."""


def _label_prompt(cat: str, clusters: list[dict]) -> str:
    groups = "\n\n".join(
        f"Group {c['cluster'] + 1} — distinctive words: {', '.join(c['terms'])}\n"
        + "\n".join(f"  - {e}" for e in c["examples"][:3])
        for c in clusters)
    return LABEL_PROMPT.format(cat=cat.split("/")[-1].replace("-", " "),
                               k=len(clusters), groups=groups)


def stage_label(out: Path, workers: int) -> None:
    """One call per *category*, naming all its clusters together.

    Labelling clusters independently produced near-duplicate names (28.7% of sibling clusters shared
    a label): the model sees one group, is anchored on the parent category, and paraphrases it. A
    condition's label is the only thing distinguishing it from its siblings, so colliding labels make
    the condition-rate target unlearnable. Naming them jointly forces contrast."""
    doc = json.loads((out / "clusters.json").read_text())
    cats = sorted(doc)
    prompts = [_label_prompt(cat, doc[cat]["clusters"]) for cat in cats]
    print(f"labelling {sum(len(doc[c]['clusters']) for c in cats)} clusters "
          f"in {len(cats)} calls (one per category)")
    replies = S.elicit_together(prompts, workers, max_tokens=320)

    for cat, reply in zip(cats, replies):
        clusters = doc[cat]["clusters"]
        found = dict(re.findall(r"^\s*(\d+)[.)]\s*(.+?)\s*$", reply or "", re.M))
        for c in clusters:
            name = found.get(str(c["cluster"] + 1), "").strip(" \"'.")
            c["label"] = name[:60] or f"group {c['cluster'] + 1}"
    (out / "clusters.json").write_text(json.dumps(doc, indent=1))

    labs = [c["label"] for v in doc.values() for c in v["clusters"]]
    dup = sum(len(v["clusters"]) - len({c["label"].lower() for c in v["clusters"]})
              for v in doc.values())
    print(f"wrote {len(labs)} labels; within-category duplicates {dup}/{len(labs)} "
          f"({dup/len(labs):.1%}); fallbacks {sum(1 for l in labs if l.startswith('group '))}")


# --- elicitation -----------------------------------------------------------------------------

def _elicit_model(prompts: list[str], model: str, workers: int,
                  extra_body: dict | None = None, max_tokens: int = 8) -> list[str]:
    """Like ``S.elicit_together`` but against an arbitrary model string (e.g. an endpoint name).

    ``extra_body`` carries provider passthrough — for Together's reasoning models the only way to
    get a direct answer is ``{"chat_template_kwargs": {"enable_thinking": False}}``; without it they
    spend the whole token budget thinking and return an empty string.
    """
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    # Use the OpenAI-compatible gateway at api.together.XYZ, NOT the native ``together`` SDK. The SDK
    # defaults to api.together.AI, where a dedicated-endpoint LoRA 404s "Model not found" (verified
    # 2026-07-23); the .xyz OpenAI-compatible route — the same one inspect_ai's together provider and
    # our tau2 fix use — routes to the dedicated endpoint by the endpoint-name model string.
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"],
                    base_url="https://api.together.xyz/v1")
    kw = {"extra_body": extra_body} if extra_body else {}

    def one(p: str) -> str:
        """An empty completion is a dropped request, not an answer — retry it like an exception.

        Together drops a few percent of requests under sustained load and returns 200 with empty
        content. Those look like refusals downstream and, once written, the resumable cache treats
        them as done. Retrying recovers essentially all of them (measured: a category that produced
        75/2000 empties returned 0/240 on retest).
        """
        for attempt in range(6):
            try:
                r = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": p}],
                    temperature=0.0, max_tokens=max_tokens, **kw)
                text = r.choices[0].message.content or ""
                if text.strip():
                    return text
            except Exception:  # noqa: BLE001 — retry transient API errors
                pass
            if attempt < 5:
                time.sleep(2 ** attempt)
        return ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, prompts))
    # Only a catastrophic rate means the endpoint is actually down (e.g. stopped mid-run). Anything
    # below that has already survived six retries and is a genuine non-answer.
    empty = sum(1 for a in out if not a.strip())
    if empty > 0.10 * len(out):
        raise RuntimeError(f"{empty}/{len(out)} calls still empty after retries — the endpoint is "
                           f"down or dropping most requests. Nothing written; re-run to resume.")
    if empty:
        print(f"  ({empty}/{len(out)} empty after retries — recorded as unparseable)")
    return out


def stage_elicit(repo: Path, out: Path, max_items: int, workers: int,
                 limit_categories: int | None, model_name: str | None = None,
                 behavior_file: str = "behavior_llama.jsonl",
                 model_shortcut: str | None = None,
                 extra_body_from: str | None = None) -> None:
    """One greedy call per item: the model's own answer. Resumable by (category, item_idx).

    Items are individually near-deterministic (Probe 2), so no resampling — a condition's rate comes
    from across-item variance, not from resampling one item.

    ``model_name`` defaults to the serverless base model. For a later round of the self-prediction
    fixed-point iteration, pass a finetuned model's dedicated ENDPOINT NAME so we measure *that*
    model's behavior (which is what the next round's labels must be).

    ``model_shortcut`` instead resolves a models.yaml entry, picking up its ``extra_body`` — needed
    for Together models whose reasoning must be disabled (e.g. ``gemma-4-31b``).

    ``extra_body_from`` supplies that ``extra_body`` while still calling ``model_name``. A finetuned
    endpoint has no models.yaml entry, but a LoRA inherits its base's chat template — and measuring
    tuned behavior under a different template than the base corpus would confound the finetune's
    effect with the template change.
    """
    cats = S.load_categories(repo)
    names = sorted(cats)[:limit_categories] if limit_categories else sorted(cats)
    done_path = out / behavior_file
    done = {(r["category"], r["idx"]) for r in map(json.loads, done_path.open())} \
        if done_path.exists() else set()
    todo = [(cat, i, it) for cat in names for i, it in enumerate(cats[cat][:max_items])
            if (cat, i) not in done]
    print(f"{len(todo)} items to elicit ({len(done)} cached) over {len(names)} categories")
    if not todo:
        return

    CHUNK = 2000  # flush periodically so a crash never loses more than one chunk
    extra_body = None
    if model_shortcut:
        from behavior_prediction import common
        full, cfg = common.resolve_model(model_shortcut)
        model = full.split("/", 1)[1] if full.startswith("together/") else full
        extra_body = cfg.get("extra_body")
    else:
        model = model_name or S.TOGETHER_MODEL
    if extra_body_from:
        from behavior_prediction import common
        extra_body = common.resolve_model(extra_body_from)[1].get("extra_body")
    print(f"model: {model}" + (f"  extra_body={extra_body}" if extra_body else ""))
    with done_path.open("a") as fh:
        for s in range(0, len(todo), CHUNK):
            chunk = todo[s:s + CHUNK]
            answers = _elicit_model([it["question"] + S.INSTRUCTION for _, _, it in chunk],
                                    model, workers, extra_body)
            for (cat, i, it), a in zip(chunk, answers):
                fh.write(json.dumps({"category": cat, "idx": i, "answer": a,
                                     "aff": H.said_affirmative(a, H.affirmative(it)),
                                     "pol": H.polarity(it)}) + "\n")
            fh.flush()
            print(f"  {min(s + CHUNK, len(todo))}/{len(todo)}")

    recs = [json.loads(l) for l in done_path.open()]
    bad = sum(1 for r in recs if r["aff"] is None)
    print(f"elicited {len(recs)} items ({bad} unparseable, {bad/len(recs):.1%} — refusals)")



# --- Binder property pairs, from THIS model's own responses ----------------------------------

def stage_binder(out: Path, workers: int, n_rows: int, seed: int,
                 model_name: str | None, model_shortcut: str | None,
                 rows_src: Path) -> None:
    """Collect the finetuning model's OWN object-level responses to Binder hypotheticals.

    Self-prediction means predicting *your own* behavior. The Binder half of the training mix must
    therefore be regenerated for every model, and for every round of the fixed-point iteration —
    round 2 trains on v1's behavior, so its Binder pairs must be v1's responses, not the base's.

    `train_rows.jsonl` (the sampled hypotheticals) is model-independent and is reused; only the
    responses are re-collected. mmlu_non_cot rows are skipped: 20.4% of our capability_mmlu eval
    items live in that subset.
    """
    rows = [json.loads(l) for l in (rows_src / "train_rows.jsonl").open()
            if json.loads(l)["original_dataset"] != "mmlu_non_cot"]
    random.Random(seed).shuffle(rows)
    rows = rows[:n_rows]
    if not (out / "train_rows.jsonl").exists():
        intro.save_jsonl(rows, out / "train_rows.jsonl")

    done_path = out / "binder_object_level.jsonl"
    done = {(r["property"], r["row_idx"]) for r in map(json.loads, done_path.open())} \
        if done_path.exists() else set()
    todo = [r for r in rows if (r["property"], r["row_idx"]) not in done]
    print(f"{len(todo)} Binder object-level prompts to collect ({len(done)} cached)")
    if not todo:
        return

    extra_body = None
    if model_shortcut:
        from behavior_prediction import common
        full, cfg = common.resolve_model(model_shortcut)
        model = full.split("/", 1)[1] if full.startswith("together/") else full
        extra_body = cfg.get("extra_body")
    else:
        model = model_name or S.TOGETHER_MODEL
    print(f"model: {model}" + (f"  extra_body={extra_body}" if extra_body else ""))

    CHUNK = 1000
    with done_path.open("a") as fh:
        for s0 in range(0, len(todo), CHUNK):
            chunk = todo[s0:s0 + CHUNK]
            answers = _elicit_model([r["object_level_prompt"] for r in chunk], model, workers,
                                    extra_body, max_tokens=200)
            for r, a in zip(chunk, answers):
                fh.write(json.dumps({"property": r["property"], "row_idx": r["row_idx"],
                                     "response": a,
                                     "target": intro.extract_property(r["property"], a, r)},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {min(s0 + CHUNK, len(todo))}/{len(todo)}")

    recs = [json.loads(l) for l in done_path.open()]
    bad = sum(1 for r in recs if r["target"] is None)
    print(f"collected {len(recs)} responses ({bad} without extractable target, dropped)")


# --- labels ----------------------------------------------------------------------------------

def _yes_rate(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def stage_labels(out: Path, behavior_file: str = "behavior_llama.jsonl") -> None:
    """Pooled category x polarity yes-rates, plus empirical-Bayes shrunk per-cluster targets.

    Probe 5 found the topic-cluster axis carries no learnable signal, so a per-cluster estimate from
    ~20 items is mostly measurement noise. Shrink each cluster's estimate toward its category value
    by w = s^2/(s^2 + noise^2), where s^2 is the *true* between-cluster variance (observed variance
    minus the sampling variance) — pooled over categories so w is stable.
    """
    doc = json.loads((out / "clusters.json").read_text())
    recs = [json.loads(l) for l in (out / behavior_file).open()]
    by_item = {(r["category"], r["idx"]): r for r in recs if r["aff"] is not None}

    cl_of = {(cat, i): c["cluster"]
             for cat, v in doc.items() for c in v["clusters"] for i in c["item_indices"]}

    cell: dict[tuple[str, int, str], list[float]] = {}   # (cat, cluster, pol) -> [aff]
    cat_half: dict[tuple[str, str], list[float]] = {}    # (cat, pol) -> [aff]
    for (cat, i), r in by_item.items():
        c = cl_of.get((cat, i))
        if c is None:
            continue
        cell.setdefault((cat, c, r["pol"]), []).append(float(r["aff"]))
        cat_half.setdefault((cat, r["pol"]), []).append(float(r["aff"]))

    # variance components, pooled across categories (per polarity-half, which is what we label)
    obs_var, samp_var = [], []
    for cat, v in doc.items():
        for pol in ("POS", "NEG"):
            rs = [(_yes_rate(cell[(cat, c["cluster"], pol)]), len(cell[(cat, c["cluster"], pol)]))
                  for c in v["clusters"] if cell.get((cat, c["cluster"], pol))]
            rs = [(r, n) for r, n in rs if r is not None and n >= 5]
            if len(rs) < 3:
                continue
            mean_r = sum(r for r, _ in rs) / len(rs)
            obs_var.append(sum((r - mean_r) ** 2 for r, _ in rs) / len(rs))
            samp_var.append(sum(r * (1 - r) / n for r, n in rs) / len(rs))
    ov, sv = sum(obs_var) / len(obs_var), sum(samp_var) / len(samp_var)
    s2 = max(ov - sv, 0.0)
    w = s2 / (s2 + sv) if (s2 + sv) > 0 else 0.0
    print(f"between-cluster var {s2:.5f} (sd {s2**0.5:.3f}), sampling var {sv:.5f} "
          f"(sd {sv**0.5:.3f}) -> shrinkage w = {w:.3f}")

    labels = {"shrinkage_w": w, "between_cluster_sd": s2 ** 0.5, "sampling_sd": sv ** 0.5,
              "category": {}, "cluster": {}}
    for cat, v in doc.items():
        for pol in ("POS", "NEG"):
            y_cat = _yes_rate(cat_half.get((cat, pol), []))
            if y_cat is None:
                continue
            labels["category"][f"{cat}|{pol}"] = {"yes": y_cat, "n": len(cat_half[(cat, pol)])}
            for c in v["clusters"]:
                vals = cell.get((cat, c["cluster"], pol))
                if not vals:
                    continue
                y_cl = _yes_rate(vals)
                labels["cluster"][f"{cat}|{c['cluster']}|{pol}"] = {
                    "yes_raw": y_cl, "yes": y_cat + w * (y_cl - y_cat), "n": len(vals)}
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"wrote labels.json: {len(labels['category'])} category halves, "
          f"{len(labels['cluster'])} cluster halves")


# --- splits ----------------------------------------------------------------------------------

def stage_splits(out: Path, n_heldout: int, seed: int) -> None:
    """Seeded category-level split. Ring B (new category) is the claim worth making; the
    'new cluster, seen category' ring is meaningless (Probe 5) and does not exist."""
    doc = json.loads((out / "clusters.json").read_text())
    cats = sorted(doc)
    rng = random.Random(seed)
    held = sorted(rng.sample(cats, min(n_heldout, len(cats))))
    train = [c for c in cats if c not in set(held)]
    assert not (set(train) & set(held))
    (out / "splits_categories.json").write_text(json.dumps(
        {"seed": seed, "train": train, "heldout": held}, indent=1))
    print(f"train {len(train)} categories, held-out {len(held)}")
    print("held-out sample:", ", ".join(c.split('/')[-1] for c in held[:5]))


# --- SFT pairs -------------------------------------------------------------------------------

def _binder_pairs(n: int, seed: int, binder_dir: Path) -> list[dict]:
    """Binder property pairs from ``binder_dir``, EXCLUDING mmlu_non_cot (20.4% of our
    capability_mmlu eval items live in that subset — docs/rate-self-prediction-finetuning.md).

    ``binder_dir`` MUST hold the *finetuning model's own* object-level responses. Self-prediction
    means predicting your own behavior: mixing in another model's responses (or, across a
    fixed-point round, the previous round's) teaches the model to predict something it is not.
    Regenerate with ``--stage binder`` for every model and every round.
    """
    if n <= 0:
        return []
    rows_path = binder_dir / "train_rows.jsonl"
    obj_path = binder_dir / "binder_object_level.jsonl"
    if not obj_path.exists():
        sys.exit(f"{obj_path} missing — run `--stage binder --model-shortcut <model>` first, or "
                 f"pass --n-binder 0 to train on rate pairs only")
    rows = {(r["property"], r["row_idx"]): r for r in map(json.loads, rows_path.open())}
    pairs = []
    for o in map(json.loads, obj_path.open()):
        r = rows.get((o["property"], o["row_idx"]))
        if not r or o["target"] is None or r["original_dataset"] == "mmlu_non_cot":
            continue
        pairs.append({"prompt": r["hypothetical_prompt"], "completion": o["target"],
                      "source": "binder"})
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


def stage_pairs(repo: Path, out: Path, variants: int, n_binder: int, seed: int,
                holdout_items: int, binder_dir: Path | None = None,
                donor_labels: list[Path] | None = None) -> None:
    """Emit sft_train.jsonl in the {"prompt","completion"} schema finetune_together already consumes.

    One pair per (train category, cluster, polarity, variant). The prompt shows K example items and
    NEVER names the category, so the model must infer the family from item content — the test-time
    skill. The last ``holdout_items`` items of each cluster half are reserved for Ring A and never
    appear in a training prompt.

    ``donor_labels`` (labels.json paths from OTHER models' corpus passes) additionally emits one
    residual-target pair per cell: the completion is the signed deviation of this model's rate from
    the donor mean — the model-specific part of the rate, which is what our evals reward. Example
    sets are drawn from an independent rng stream so the two objectives don't share prompts.
    """
    cats = S.load_categories(repo)
    doc = json.loads((out / "clusters.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    split = json.loads((out / "splits_categories.json").read_text())
    donors = [json.loads(p.read_text()) for p in donor_labels or []]
    rng = random.Random(seed)
    rng_dev = random.Random(seed + 100_000)

    pairs, dev_pairs, used_items = [], [], {}
    for cat in split["train"]:
        items = cats[cat]
        aff = H.affirmative(items[0])
        for c in doc[cat]["clusters"]:
            for pol in ("POS", "NEG"):
                key = f"{cat}|{c['cluster']}|{pol}"
                idxs = [i for i in c["item_indices"] if H.polarity(items[i]) == pol]
                pool = idxs[:-holdout_items] if len(idxs) > holdout_items + H.K_EXAMPLES else idxs
                used_items[key] = pool
                lab = labels["cluster"].get(key)
                if lab is None or len(pool) < H.K_EXAMPLES:
                    continue
                for _ in range(variants):
                    ex = [items[i] for i in rng.sample(pool, H.K_EXAMPLES)]
                    pairs.append({"prompt": H.selfreport_prompt(ex, aff, rng),
                                  "completion": str(round(lab["yes"] * 100)),
                                  "source": "selfpred", "category": cat})
                if donors and all(key in d["cluster"] for d in donors):
                    dmean = sum(d["cluster"][key]["yes"] for d in donors) / len(donors)
                    for _ in range(variants):
                        ex = [items[i] for i in rng_dev.sample(pool, H.K_EXAMPLES)]
                        dev_pairs.append({"prompt": H.deviation_prompt(ex, aff, rng_dev),
                                          "completion": str(round((lab["yes"] - dmean) * 100)),
                                          "source": "selfpred_dev", "category": cat})
    print(f"{len(pairs)} rate pairs from {len(split['train'])} train categories")
    if donors:
        print(f"{len(dev_pairs)} deviation pairs (donor mean over {len(donors)} models)")

    held = set(split["heldout"])
    assert not any(p["category"] in held for p in pairs + dev_pairs), \
        "held-out category leaked into training"

    binder = _binder_pairs(n_binder, seed, binder_dir or out)
    print(f"{len(binder)} binder pairs (this model's OWN responses; mmlu_non_cot excluded)")
    allp = pairs + dev_pairs + binder
    rng.shuffle(allp)
    intro.save_jsonl([{"prompt": p["prompt"], "completion": p["completion"]} for p in allp],
                     out / "sft_train.jsonl")
    (out / "pairs_meta.json").write_text(json.dumps(
        {"n_selfpred": len(pairs), "n_deviation": len(dev_pairs), "n_binder": len(binder),
         "total": len(allp), "variants": variants, "holdout_items": holdout_items,
         "train_categories": len(split["train"]),
         "donor_labels": [str(p) for p in donor_labels or []]}, indent=1))
    print(f"wrote {out/'sft_train.jsonl'} ({len(allp)} pairs)")


def _measured_by_category(repo: Path) -> dict[str, dict]:
    """Category-level measured rate + polarity gap for both models, from the v0 screen answers.
    Empty if the screen has not been run. Cluster-level rates require the full elicitation pass."""
    out: dict[str, dict] = {}
    for model in ("llama", "qwen"):
        if not (Path("logs/condition_family_screen") / f"screen_{model}.jsonl").exists():
            continue
        m = H.measured(model, repo, 40, 0)
        for (cat, pol), v in m.items():
            out.setdefault(cat, {}).setdefault(model, {})[pol] = v
    for cat, per in out.items():
        for model, halves in per.items():
            if "POS" in halves and "NEG" in halves:
                halves["gap"] = halves["POS"] + halves["NEG"] - 1
                halves["rate"] = (halves["POS"] + 1 - halves["NEG"]) / 2
    return out


def _trim(q: str, n: int = 260) -> str:
    q = " ".join(q.split())
    return q if len(q) <= n else q[: n - 1] + "…"


def stage_preview(repo: Path, out: Path) -> None:
    """Assemble everything the HTML viewer needs: per-category clusters, real example items, and the
    measured rates. The prompts themselves are reconstructed in the viewer from the same templates,
    so the page cannot drift from the scripts."""
    cats = S.load_categories(repo)
    doc = json.loads((out / "clusters.json").read_text())
    meas = _measured_by_category(repo)
    payload = {"n_categories": len(doc), "categories": [],
               "n_clusters": sum(len(v["clusters"]) for v in doc.values()),
               "n_conditions": 2 * sum(len(v["clusters"]) for v in doc.values()),
               "n_items": sum(v["n_items"] for v in doc.values()),
               "measure_instruction": S.INSTRUCTION.strip(),
               "k_examples": H.K_EXAMPLES}
    for cat, v in sorted(doc.items()):
        items = cats[cat]
        aff = H.affirmative(items[0])
        cl = []
        for c in v["clusters"]:
            idxs = c["item_indices"]
            pos = [items[i] for i in idxs if H.polarity(items[i]) == "POS"][:2]
            neg = [items[i] for i in idxs if H.polarity(items[i]) == "NEG"][:2]
            cl.append({
                "cluster": c["cluster"], "label": c.get("label", ""), "terms": c["terms"][:4],
                "n_pos": c["n_pos"], "n_neg": c["n_neg"],
                "pos": [{"q": _trim(it["question"]),
                         "a": it["answer_matching_behavior"].strip()} for it in pos],
                "neg": [{"q": _trim(it["question"]),
                         "a": it["answer_matching_behavior"].strip()} for it in neg],
            })
        payload["categories"].append({
            "name": cat, "suite": cat.split("/")[0], "affirmative": aff,
            "n_items": v["n_items"], "measured": meas.get(cat, {}), "clusters": cl})
    p = out / "corpus_preview.json"
    p.write_text(json.dumps(payload))
    print(f"wrote {p} ({p.stat().st_size/1e6:.2f} MB): {payload['n_categories']} categories, "
          f"{payload['n_clusters']} clusters, {payload['n_conditions']} conditions, "
          f"{sum(1 for c in payload['categories'] if c['measured'])} with measured rates")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("logs/selfpred_corpus"))
    ap.add_argument("--stage", required=True,
                    choices=["cluster", "label", "elicit", "binder", "labels", "splits", "pairs",
                             "preview"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-categories", type=int, default=None, help="smoke runs")
    ap.add_argument("--heldout-categories", type=int, default=40)
    ap.add_argument("--variants", type=int, default=2, help="example-sets per (cluster, polarity)")
    ap.add_argument("--n-binder", type=int, default=5000)
    ap.add_argument("--binder-rows", type=Path,
                    default=Path("logs/introspection_finetune_llama30k"),
                    help="dir holding train_rows.jsonl (model-independent hypotheticals)")
    ap.add_argument("--binder-dir", type=Path, default=None,
                    help="dir holding THIS model's binder_object_level.jsonl (default: --out)")
    ap.add_argument("--n-binder-rows", type=int, default=6500,
                    help="object-level prompts to collect (yields ~n-binder pairs after drops)")
    ap.add_argument("--holdout-items", type=int, default=5,
                    help="items per cluster-half reserved for Ring A (never in a training prompt)")
    ap.add_argument("--donor-labels", type=Path, nargs="+", default=None,
                    help="labels.json paths from OTHER models' passes; adds residual-target "
                         "(deviation) pairs with completion = own rate - donor mean, in points")
    ap.add_argument("--model-name", default=None,
                    help="Together model/ENDPOINT NAME to elicit behavior from (default: base)")
    ap.add_argument("--model-shortcut", default=None,
                    help="models.yaml shortcut; picks up its extra_body (e.g. gemma-4-31b)")
    ap.add_argument("--extra-body-from", default=None,
                    help="take extra_body from this shortcut while calling --model-name (for a "
                         "finetuned endpoint, which has no models.yaml entry)")
    ap.add_argument("--behavior-file", default="behavior_llama.jsonl",
                    help="behavior file to write (elicit) / read (labels)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "cluster":
        stage_cluster(args.evals_repo, args.out, args.k, args.seed, args.max_items)
    elif args.stage == "label":
        stage_label(args.out, args.workers)
    elif args.stage == "elicit":
        stage_elicit(args.evals_repo, args.out, args.max_items, args.workers, args.limit_categories,
                     args.model_name, args.behavior_file, args.model_shortcut, args.extra_body_from)
    elif args.stage == "binder":
        stage_binder(args.out, args.workers, args.n_binder_rows, args.seed,
                     args.model_name, args.model_shortcut, args.binder_rows)
    elif args.stage == "labels":
        stage_labels(args.out, args.behavior_file)
    elif args.stage == "splits":
        stage_splits(args.out, args.heldout_categories, args.seed)
    elif args.stage == "pairs":
        stage_pairs(args.evals_repo, args.out, args.variants, args.n_binder, args.seed,
                    args.holdout_items, args.binder_dir, args.donor_labels)
    else:
        stage_preview(args.evals_repo, args.out)


if __name__ == "__main__":
    main()
