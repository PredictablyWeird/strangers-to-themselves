#!/usr/bin/env python3
"""Before/after report: how well does Llama predict its own behavior on the self-prediction corpus?

Asks the base and the finetuned model the SAME self-report prompt the corpus trains on, on two
generalization rings, and scores both against the model's measured behavior.

  Ring A  new items, seen category  — prompts built from items held out of every training prompt
  Ring B  new category             — 40 categories the finetune never saw. The claim worth making.

(The "new cluster, seen category" ring does not exist: Probe 5 showed clusters share a target, so it
would measure memorisation. See docs/selfpred-corpus-v1.md.)

**All metrics are computed at CATEGORY level.** A category's ~10 cluster prompts are averaged to one
predicted yes-rate per polarity, then `rate` and `gap` are derived. Ten clusters sharing a target are
ten replicates, not ten observations; scoring per-cluster would inflate n tenfold and understate
standard errors ~3x.

Targets:  rate = (yes_pos + 1 - yes_neg) / 2     (shared across models — a pool prior predicts it)
          gap  = yes_pos + yes_neg - 1           (this model's acquiescence bias — the headline)

Stages:
  predict  ask a model the self-report prompts for both rings
  drift    re-elicit the tuned model's ACTUAL behavior on held-out categories (self-fulfilment check)
  report   score everything, write results/reports/selfpred_finetune_llama.md + .json

Usage:
  .venv/bin/python scripts/selfpred_report.py --evals-repo <clone> --stage predict --model base
  .venv/bin/python scripts/selfpred_report.py --evals-repo <clone> --stage predict --model tuned
  .venv/bin/python scripts/selfpred_report.py --evals-repo <clone> --stage drift
  .venv/bin/python scripts/selfpred_report.py --evals-repo <clone> --stage report
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_condition_families as S  # noqa: E402
import selfpred_headroom as H  # noqa: E402

CORPUS = Path("logs/selfpred_corpus")
RUN = Path("logs/selfpred_llama")


# --- model plumbing --------------------------------------------------------------------------

def _tuned_model(run: Path) -> str:
    """The dedicated endpoint's NAME — passing the finetuned base model id 400s (non-serverless)."""
    rec = json.loads((run / "together_endpoint.json").read_text())
    if rec.get("deleted"):
        sys.exit("the endpoint has been torn down; redeploy before predicting against `tuned`")
    return rec["endpoint_name"]


def _elicit(prompts: list[str], model: str, workers: int, max_tokens: int,
            extra_body: dict | None = None) -> list[str]:
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    # OpenAI-compatible api.together.XYZ gateway, not the native together SDK (api.together.AI),
    # which 404s on dedicated-endpoint LoRAs (verified 2026-07-23). Same fix as build_selfpred_corpus.
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"],
                    base_url="https://api.together.xyz/v1")

    def one(p: str) -> str:
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": p}],
                    temperature=0.0, max_tokens=max_tokens,
                    **({"extra_body": extra_body} if extra_body else {}))
                return r.choices[0].message.content or ""
            except Exception:  # noqa: BLE001 — retry transient API errors
                if attempt == 3:
                    return ""
                time.sleep(2 ** (attempt + 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, prompts))


# --- prompt construction ---------------------------------------------------------------------

def _ring_prompts(repo: Path, holdout_items: int, seed: int,
                  deviation: bool = False) -> list[dict]:
    """One self-report prompt per (category, cluster, polarity), for both rings.

    Ring A uses the trailing ``holdout_items`` of each cluster half — the exact items
    ``build_selfpred_corpus.stage_pairs`` excludes from training prompts. Ring B uses any items,
    since the whole category is unseen.

    ``deviation`` additionally emits one residual-objective prompt per cell (obj="dev": how much
    more/fewer than a typical model — the residual-target corpus's second objective), from an
    independent rng stream, mirroring ``stage_pairs``. Rate rows carry obj="rate".
    """
    cats = S.load_categories(repo)
    doc = json.loads((CORPUS / "clusters.json").read_text())
    split = json.loads((CORPUS / "splits_categories.json").read_text())
    held = set(split["heldout"])
    rng = random.Random(seed)
    rng_dev = random.Random(seed + 100_000)

    out = []
    for cat, v in doc.items():
        items = cats[cat]
        aff = H.affirmative(items[0])
        ring = "B" if cat in held else "A"
        for c in v["clusters"]:
            for pol in ("POS", "NEG"):
                idxs = [i for i in c["item_indices"] if H.polarity(items[i]) == pol]
                if ring == "A":
                    pool = idxs[-holdout_items:] if len(idxs) > holdout_items + H.K_EXAMPLES else []
                else:
                    pool = idxs
                if len(pool) < min(H.K_EXAMPLES, 3):
                    continue
                base = {"ring": ring, "category": cat, "cluster": c["cluster"], "pol": pol,
                        "aff": aff}
                ex = [items[i] for i in rng.sample(pool, min(H.K_EXAMPLES, len(pool)))]
                out.append({**base, "obj": "rate", "prompt": H.selfreport_prompt(ex, aff, rng)})
                if deviation:
                    ex = [items[i] for i in rng_dev.sample(pool, min(H.K_EXAMPLES, len(pool)))]
                    out.append({**base, "obj": "dev",
                                "prompt": H.deviation_prompt(ex, aff, rng_dev)})
    return out


# --- stages ----------------------------------------------------------------------------------

def stage_predict(repo: Path, run: Path, which: str, workers: int, holdout_items: int,
                  seed: int, base_shortcut: str | None = None, deviation: bool = False) -> None:
    """``base_shortcut`` names the models.yaml entry for the BASE model. Its ``extra_body`` applies to
    the tuned model too — a LoRA inherits its base's chat template, so gemma's tuned endpoint also
    needs ``enable_thinking=false`` or it returns an empty completion."""
    from behavior_prediction import common
    extra_body = None
    if base_shortcut:
        full, cfg = common.resolve_model(base_shortcut)
        extra_body = cfg.get("extra_body")
        base_model = full.split("/", 1)[1] if full.startswith("together/") else full
    else:
        base_model = S.TOGETHER_MODEL
    model = base_model if which == "base" else _tuned_model(run)
    rows = _ring_prompts(repo, holdout_items, seed, deviation)
    print(f"{len(rows)} prompts ({sum(1 for r in rows if r['ring']=='B')} Ring B) -> {which}")
    answers = _elicit([r["prompt"] for r in rows], model, workers, 8, extra_body)
    for r, a in zip(rows, answers):
        r["raw"] = a
        r["pred"] = (H.parse_signed(a) if r.get("obj") == "dev"
                     else common.parse_percentage(a))
        r.pop("prompt")
    path = run / f"pred_{which}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    bad = sum(1 for r in rows if r["pred"] is None)
    print(f"wrote {path} ({bad} unparseable)")


def stage_drift(repo: Path, run: Path, n_items: int, workers: int,
                base_shortcut: str | None = None) -> None:
    """Self-fulfilment check: did the finetune move the model's ACTUAL behavior?

    If predictions match tuned behavior only because behavior moved toward the predictions, that is
    not introspection. Re-elicit behavior on the held-out categories and compare with the base
    model's measured behavior from the corpus pass.
    """
    cats = S.load_categories(repo)
    split = json.loads((CORPUS / "splits_categories.json").read_text())
    model = _tuned_model(run)
    extra_body = None
    if base_shortcut:
        from behavior_prediction import common
        extra_body = common.resolve_model(base_shortcut)[1].get("extra_body")
    rng = random.Random(0)
    jobs = []
    for cat in split["heldout"]:
        items = cats[cat][:400]
        for i in rng.sample(range(len(items)), min(n_items, len(items))):
            jobs.append((cat, i, items[i]))
    print(f"re-eliciting {len(jobs)} items on {len(split['heldout'])} held-out categories -> tuned")
    answers = _elicit([it["question"] + S.INSTRUCTION for _, _, it in jobs], model, workers, 8,
                      extra_body)
    recs = [{"category": c, "idx": i, "answer": a,
             "aff": H.said_affirmative(a, H.affirmative(it)), "pol": H.polarity(it)}
            for (c, i, it), a in zip(jobs, answers)]
    (run / "behavior_tuned.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"wrote behavior_tuned.jsonl ({sum(1 for r in recs if r['aff'] is None)} unparseable)")


# --- scoring ---------------------------------------------------------------------------------

def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else 0.0


def _halves(recs: list[dict], key) -> dict[tuple[str, str], float]:
    """(category, polarity) -> mean of ``key`` over that category's cluster prompts / items."""
    acc: dict[tuple[str, str], list[float]] = {}
    for r in recs:
        v = key(r)
        if v is not None:
            acc.setdefault((r["category"], r["pol"]), []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _derive(halves: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    out = {}
    for cat in {c for c, _ in halves}:
        p, n = halves.get((cat, "POS")), halves.get((cat, "NEG"))
        if p is not None and n is not None:
            out[cat] = {"yes_pos": p, "yes_neg": n, "rate": (p + 1 - n) / 2, "gap": p + n - 1}
    return out


def _measured(path: Path) -> dict[str, dict[str, float]]:
    recs = [json.loads(l) for l in path.open()]
    return _derive(_halves(recs, lambda r: r["aff"]))


def _score(pred: dict, meas: dict, cats: list[str]) -> dict:
    cats = [c for c in cats if c in pred and c in meas]
    out = {"n_categories": len(cats)}
    for q in ("rate", "gap"):
        p = [pred[c][q] for c in cats]
        m = [meas[c][q] for c in cats]
        out[q] = {"r": _corr(p, m), "mae": st.mean(abs(a - b) for a, b in zip(p, m)),
                  "bias": st.mean(p) - st.mean(m),
                  "sd_pred": st.pstdev(p) if len(p) > 1 else 0.0,
                  "sd_measured": st.pstdev(m) if len(m) > 1 else 0.0}
    return out


def _derive_dev(halves: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    """Per-category residual quantities from per-half predicted DEVIATIONS. Constants difference
    out, so rate has no +1 and gap no -1."""
    out = {}
    for cat in {c for c, _ in halves}:
        p, n = halves.get((cat, "POS")), halves.get((cat, "NEG"))
        if p is not None and n is not None:
            out[cat] = {"rate": (p - n) / 2, "gap": p + n}
    return out


def _residual(meas: dict, donor_meas: list[dict]) -> dict[str, dict[str, float]]:
    """Measured residual per category: this model's rate/gap minus the donor-pool mean."""
    out = {}
    for c in meas:
        if all(c in d for d in donor_meas):
            out[c] = {q: meas[c][q] - st.mean(d[c][q] for d in donor_meas)
                      for q in ("rate", "gap")}
    return out


def stage_report(repo: Path, run: Path, out_md: Path, behavior: Path | None = None,
                 donor_behavior: list[Path] | None = None) -> None:
    split = json.loads((CORPUS / "splits_categories.json").read_text())
    # The "base" role is whichever model produced the TRAINING LABELS: base-Llama in round 1, the
    # previous round's tuned model thereafter.
    meas_base = _measured(behavior or (CORPUS / "behavior_llama.jsonl"))
    preds, dev_preds = {}, {}
    for which in ("base", "tuned"):
        p = run / f"pred_{which}.jsonl"
        if p.exists():
            recs = [json.loads(l) for l in p.open()]
            preds[which] = {ring: _derive(_halves([r for r in recs if r["ring"] == ring
                                                   and r.get("obj", "rate") == "rate"],
                                                  lambda r: r["pred"]))
                            for ring in ("A", "B")}
            dev = [r for r in recs if r.get("obj") == "dev"]
            if dev:
                dev_preds[which] = {ring: _derive_dev(_halves([r for r in dev
                                                               if r["ring"] == ring],
                                                              lambda r: r["pred"]))
                                    for ring in ("A", "B")}
    if "base" not in preds:
        sys.exit("run `predict --model base` first")

    doc = {"heldout_categories": split["heldout"], "train_categories": len(split["train"]),
           "rings": {}, "baselines": {}, "drift": None}
    for ring, cats in (("A", split["train"]), ("B", split["heldout"])):
        doc["rings"][ring] = {w: _score(preds[w][ring], meas_base, cats)
                              for w in preds if ring in preds[w]}

    # Baseline: the cross-model prior — Qwen's measured gap, free from the earlier screen.
    qpath = Path("logs/condition_family_screen/screen_qwen.jsonl")
    if qpath.exists():
        qrecs = []
        cats_all = S.load_categories(repo)
        rows = S.sample_items(cats_all, 40, 0)  # same seed/n the screen used
        pol = {(c, i): H.polarity(it) for c, i, it in rows}
        aff = {c: H.affirmative(cats_all[c][0]) for c in cats_all}
        for l in qpath.open():
            r = json.loads(l)
            v = H.said_affirmative(r["answer"], aff[r["category"]])
            if v is not None and (r["category"], r["idx"]) in pol:
                qrecs.append({"category": r["category"], "pol": pol[(r["category"], r["idx"])],
                              "aff": v})
        qmeas = _derive(_halves(qrecs, lambda r: r["aff"]))
        held = [c for c in split["heldout"] if c in qmeas and c in meas_base]
        doc["baselines"]["cross_model_prior"] = {
            "n_categories": len(held),
            "gap_r": _corr([qmeas[c]["gap"] for c in held], [meas_base[c]["gap"] for c in held]),
            "rate_r": _corr([qmeas[c]["rate"] for c in held], [meas_base[c]["rate"] for c in held])}

    # Noise ceiling: labels come from 200 items/half, so gap SE = sqrt(2 * .25 / 200).
    n_half = 200
    gap_se = math.sqrt(2 * 0.25 / n_half)
    gap_sd = st.pstdev([meas_base[c]["gap"] for c in split["heldout"] if c in meas_base])
    rel = max(gap_sd ** 2 - gap_se ** 2, 0) / gap_sd ** 2 if gap_sd else 0
    doc["baselines"]["noise_ceiling"] = {"gap_se": gap_se, "gap_sd": gap_sd,
                                         "reliability": rel, "max_r": math.sqrt(rel)}

    # Drift / self-fulfilment, and the comparison that actually answers "does it know itself":
    # each model scored against ITS OWN behavior. The training labels are the BASE model's behavior,
    # so `tuned vs base behavior` measures how well the finetune learned the label function, while
    # `tuned vs tuned behavior` measures self-knowledge of the model that now exists.
    dpath = run / "behavior_tuned.jsonl"
    if dpath.exists() and "tuned" in preds:
        meas_tuned = _measured(dpath)
        shared = [c for c in meas_tuned if c in meas_base]
        signed = st.mean(meas_tuned[c]["gap"] - meas_base[c]["gap"] for c in shared)
        toward = st.mean(abs(preds["tuned"]["B"][c]["gap"] - meas_tuned[c]["gap"])
                         - abs(preds["tuned"]["B"][c]["gap"] - meas_base[c]["gap"])
                         for c in shared if c in preds["tuned"]["B"])
        # Tuned behavior may be measured on fewer items than the corpus pass, so its labels are
        # noisier and attenuate the correlation. Read the actual per-half count rather than assume.
        counts: dict[tuple[str, str], int] = {}
        for line in dpath.open():
            r = json.loads(line)
            if r["aff"] is not None:
                counts[(r["category"], r["pol"])] = counts.get((r["category"], r["pol"]), 0) + 1
        n_half_tuned = st.median(counts.values()) if counts else 50
        se_t = math.sqrt(2 * 0.25 / n_half_tuned)
        sd_t = st.pstdev([meas_tuned[c]["gap"] for c in shared])
        rel_t = max(sd_t ** 2 - se_t ** 2, 0) / sd_t ** 2 if sd_t else 0
        r_self_tuned = _score(preds["tuned"]["B"], meas_tuned, shared)["gap"]["r"]
        doc["drift"] = {
            "n_categories": len(shared),
            "mean_abs_delta_rate": st.mean(abs(meas_tuned[c]["rate"] - meas_base[c]["rate"])
                                           for c in shared),
            "mean_abs_delta_gap": st.mean(abs(meas_tuned[c]["gap"] - meas_base[c]["gap"])
                                          for c in shared),
            "signed_delta_gap": signed,
            "r_base_vs_tuned_behavior": _corr([meas_base[c]["gap"] for c in shared],
                                              [meas_tuned[c]["gap"] for c in shared]),
            "moved_toward_predictions": toward,
            "tuned_pred_vs_tuned_behavior_gap_r": r_self_tuned,
            "tuned_pred_vs_base_behavior_gap_r": _score(preds["tuned"]["B"], meas_base, shared)["gap"]["r"],
            "tuned_self_r_disattenuated": r_self_tuned / math.sqrt(rel_t) if rel_t else None,
            "tuned_behavior_reliability": rel_t,
            "n_half_tuned": n_half_tuned,
        }
        doc["self_knowledge"] = {
            "base_vs_own_behavior": doc["rings"]["B"]["base"]["gap"]["r"],
            "tuned_vs_own_behavior": r_self_tuned,
            # the scored n is the Ring-B categories that also have tuned behavior, not every
            # category the drift pass happened to measure
            "n_categories": _score(preds["tuned"]["B"], meas_tuned, shared)["n_categories"],
        }

    out_md.parent.mkdir(parents=True, exist_ok=True)
    # Residual objective (the residual-target corpus): deviation predictions scored against
    # measured residuals — this model's behavior minus the donor-pool mean. The residual is the
    # model-specific part of the rate by construction; a constant predictor and the donor pool
    # itself both score r = 0 on it.
    if dev_preds and donor_behavior:
        donor_meas = [_measured(p) for p in donor_behavior]
        resid_label = _residual(meas_base, donor_meas)
        doc["residual"] = {"donor_files": [str(p) for p in donor_behavior], "rings": {}}
        for ring, cats_r in (("A", split["train"]), ("B", split["heldout"])):
            doc["residual"]["rings"][ring] = {w: _score(dev_preds[w][ring], resid_label, cats_r)
                                              for w in dev_preds if ring in dev_preds[w]}
        # parametric noise ceiling on the rate residual (200 items/half own + 2 donors)
        se_half2 = 0.25 / 200
        se_resid = math.sqrt(se_half2 / 2 * (1 + 1 / max(len(donor_meas), 1)))
        sd_resid = st.pstdev([resid_label[c]["rate"] for c in split["heldout"]
                              if c in resid_label])
        rel_resid = max(sd_resid ** 2 - se_resid ** 2, 0) / sd_resid ** 2 if sd_resid else 0
        doc["residual"]["noise_ceiling"] = {"rate_se": se_resid, "rate_sd": sd_resid,
                                            "reliability": rel_resid,
                                            "max_r": math.sqrt(rel_resid)}
        # honest scoring: the tuned model's deviation predictions against its OWN re-elicited
        # residual (tuned behavior minus the same donor mean), not the training labels
        if dpath.exists() and "tuned" in dev_preds:
            resid_tuned = _residual(_measured(dpath), donor_meas)
            shared_r = [c for c in resid_tuned if c in resid_label and c in split["heldout"]]
            doc["residual"]["self_knowledge"] = {
                "base_vs_own": _score(dev_preds["base"]["B"], resid_label, split["heldout"]),
                "tuned_vs_own": _score(dev_preds["tuned"]["B"], resid_tuned, shared_r),
                "tuned_vs_training_labels": _score(dev_preds["tuned"]["B"], resid_label,
                                                   shared_r),
            }

    (out_md.with_suffix(".json")).write_text(json.dumps(doc, indent=1))
    out_md.write_text(_render(doc))
    print(_render(doc))
    print(f"\nwrote {out_md} and {out_md.with_suffix('.json')}")


def _f(v, spec="+.3f"):
    return "—" if v is None else format(v, spec)


def _render(d: dict) -> str:
    L = ["# Self-prediction finetune — Llama-3.3-70B, before vs after", ""]
    L.append(f"Trained on {d['train_categories']} categories of Anthropic's model-written evals; "
             f"**{len(d['heldout_categories'])} categories held out entirely** (Ring B). Metrics are "
             "computed per category (clusters are prompt variations, not observations).")
    L.append("")
    L.append("`rate = (yes_pos + 1 - yes_neg)/2` is shared across models; "
             "`gap = yes_pos + yes_neg - 1` is this model's own acquiescence bias — the headline.")
    L.append("")
    for ring, title in (("B", "Ring B — new category (the claim)"),
                        ("A", "Ring A — new items, seen category")):
        r = d["rings"].get(ring, {})
        if not r:
            continue
        L += [f"## {title}", "",
              "| quantity | model | r | MAE | bias | sd(pred) | sd(measured) |",
              "|---|---|---|---|---|---|---|"]
        for q in ("gap", "rate"):
            for w in ("base", "tuned"):
                if w not in r:
                    continue
                s = r[w][q]
                L.append(f"| {q} | {w} | **{_f(s['r'])}** | {_f(s['mae'],'.3f')} | "
                         f"{_f(s['bias'])} | {_f(s['sd_pred'],'.3f')} | {_f(s['sd_measured'],'.3f')} |")
        L += ["", f"n = {r[list(r)[0]]['n_categories']} categories", ""]

    b = d["baselines"]
    L += ["## Baselines", ""]
    if "cross_model_prior" in b:
        c = b["cross_model_prior"]
        L.append(f"- **Cross-model prior** (predict this model's behavior from *another* model's "
                 f"measured behavior — here Qwen's, n={c['n_categories']}): gap r = "
                 f"{_f(c['gap_r'])}, rate r = {_f(c['rate_r'])}. A method that only beats this on "
                 "`rate` has learned no self-knowledge.")
    nc = b["noise_ceiling"]
    L.append(f"- **Noise ceiling** on the gap: labels use 200 items/half so SE = {nc['gap_se']:.3f} "
             f"against a measured sd of {nc['gap_sd']:.3f}; reliability {nc['reliability']:.2f}, so "
             f"the best attainable r is **{nc['max_r']:.2f}**, not 1.0.")
    L.append("- **Constant predictor** scores r = 0 by construction.")
    L.append("")

    if d.get("self_knowledge"):
        sk = d["self_knowledge"]
        L += ["## Does the model know *itself*? (each model scored against its own behavior)", "",
              "The training labels are the **base** model's behavior. So `tuned vs base behavior` "
              "measures how well the finetune learned that label function on unseen categories, "
              "while `tuned vs its own behavior` measures self-knowledge of the model that now "
              "exists. Only the second is introspection.", "",
              "| model | predicts | gap r |", "|---|---|---|",
              f"| base | its own behavior | {_f(sk['base_vs_own_behavior'])} |",
              f"| tuned | its own behavior | **{_f(sk['tuned_vs_own_behavior'])}** |",
              f"| tuned | base's behavior (the training target) | {_f(d['drift']['tuned_pred_vs_base_behavior_gap_r'])} |",
              f"| cross-model prior | Llama's behavior | {_f(d['baselines'].get('cross_model_prior',{}).get('gap_r'))} |",
              "", f"n = {sk['n_categories']} held-out categories.", ""]

    if d.get("drift"):
        dr = d["drift"]
        moved = "AWAY from" if dr["moved_toward_predictions"] > 0 else "TOWARD"
        gap_sd = d["baselines"]["noise_ceiling"]["gap_sd"]
        frac = dr["mean_abs_delta_gap"] / gap_sd if gap_sd else 0
        L += ["## Behavior drift / self-fulfilment check", "",
              f"Finetuning shifted actual behavior on held-out categories: mean |Δrate| = "
              f"{dr['mean_abs_delta_rate']:.3f} but mean |Δgap| = **{dr['mean_abs_delta_gap']:.3f}** "
              f"(signed {dr['signed_delta_gap']:+.3f}), i.e. {frac:.0%} of the between-category gap "
              f"sd ({gap_sd:.3f}). The finetune left *what* the model does largely intact and moved "
              f"*how acquiescent* it is. r(prev gap, new gap) = {_f(dr['r_base_vs_tuned_behavior'])}.",
              "",
              f"- tuned predictions vs **base** behavior:  gap r = {_f(dr['tuned_pred_vs_base_behavior_gap_r'])}",
              f"- tuned predictions vs **tuned** behavior: gap r = {_f(dr['tuned_pred_vs_tuned_behavior_gap_r'])}"
              + (f" (disattenuated {_f(dr['tuned_self_r_disattenuated'])}; tuned behavior measured on "
                 f"{int(dr['n_half_tuned'])} items/half, reliability "
                 f"{dr['tuned_behavior_reliability']:.2f})"
                 if dr.get("tuned_self_r_disattenuated") else ""),
              "",
              f"Behavior moved **{moved}** the model's own predictions "
              f"(|pred−new| − |pred−prev| = {dr['moved_toward_predictions']:+.3f}).", ""]
        if dr["moved_toward_predictions"] > 0:
            L += ["Moving *away* rules out self-fulfilment — the predictions did not come true "
                  "because behavior chased them.", ""]
        else:
            L += ["Moving *toward* the predictions is the **self-fulfilment signature**: some of the "
                  "agreement may be behavior chasing the prediction rather than the model knowing "
                  f"itself. Weigh it against the size of the drift (mean |Δgap| = "
                  f"{dr['mean_abs_delta_gap']:.3f}); a small drift can only manufacture a small part "
                  "of the correlation.", ""]
        r_self = dr["tuned_pred_vs_tuned_behavior_gap_r"] or 0
        r_targ = dr["tuned_pred_vs_base_behavior_gap_r"] or 0
        if r_targ - r_self > 0.10:
            L += ["Instead the predictor and the predicted **drifted apart**: the finetune installed "
                  "accurate knowledge of the *pre-finetune* model while changing the model. The "
                  "tuned model describes the model it used to be. Iterate: re-elicit this model's "
                  "behavior and train the next round on it.", ""]
        else:
            L += [f"And the model now predicts *itself* almost as well as it predicts its training "
                  f"target ({_f(r_self)} vs {_f(r_targ)}). The predictor and the predicted have "
                  "**converged** — this round is close to a fixed point of "
                  "`G(M) = finetune(base, behavior(M))`.", ""]

    if d.get("residual"):
        res = d["residual"]
        L += ["## Residual objective — deviation from the donor-pool mean", "",
              "Predictions of the *signed deviation* from a typical model, scored against measured "
              "residuals (this model's behavior minus the donor mean). The donor pool itself scores "
              "r = 0 on this quantity by construction — any positive r is model-specific signal.", ""]
        for ring, title in (("B", "Ring B — new category"), ("A", "Ring A — new items")):
            r = res["rings"].get(ring, {})
            if not r:
                continue
            L += [f"### {title}", "",
                  "| quantity | model | r | MAE | bias | sd(pred) | sd(measured) |",
                  "|---|---|---|---|---|---|---|"]
            for q in ("rate", "gap"):
                for w in ("base", "tuned"):
                    if w not in r:
                        continue
                    s = r[w][q]
                    L.append(f"| resid {q} | {w} | **{_f(s['r'])}** | {_f(s['mae'],'.3f')} | "
                             f"{_f(s['bias'])} | {_f(s['sd_pred'],'.3f')} | "
                             f"{_f(s['sd_measured'],'.3f')} |")
            L += ["", f"n = {r[list(r)[0]]['n_categories']} categories", ""]
        nc = res.get("noise_ceiling")
        if nc:
            L.append(f"Noise ceiling on the rate residual: SE {nc['rate_se']:.3f} vs sd "
                     f"{nc['rate_sd']:.3f} → reliability {nc['reliability']:.2f}, max attainable "
                     f"r **{nc['max_r']:.2f}**.")
            L.append("")
        if res.get("self_knowledge"):
            sk = res["self_knowledge"]
            L += ["### Honest scoring (Ring B, each model vs its OWN residual)", "",
                  "| model | predicts | resid rate r | resid gap r |", "|---|---|---|---|",
                  f"| base | its own residual | {_f(sk['base_vs_own']['rate']['r'])} | "
                  f"{_f(sk['base_vs_own']['gap']['r'])} |",
                  f"| tuned | its own re-elicited residual | "
                  f"**{_f(sk['tuned_vs_own']['rate']['r'])}** | "
                  f"**{_f(sk['tuned_vs_own']['gap']['r'])}** |",
                  f"| tuned | the training-label residual | "
                  f"{_f(sk['tuned_vs_training_labels']['rate']['r'])} | "
                  f"{_f(sk['tuned_vs_training_labels']['gap']['r'])} |",
                  "", f"n = {sk['tuned_vs_own']['n_categories']} held-out categories.", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals-repo", type=Path, required=True)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--stage", choices=["predict", "drift", "report"], required=True)
    ap.add_argument("--model", choices=["base", "tuned"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--holdout-items", type=int, default=5)
    ap.add_argument("--drift-items", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base-shortcut", default=None,
                    help="models.yaml shortcut for the BASE model (picks up extra_body)")
    ap.add_argument("--out", type=Path,
                    default=Path("results/reports/selfpred_finetune_llama.md"))
    ap.add_argument("--behavior", type=Path, default=None,
                    help="behavior file of the model whose behavior was the training label "
                         "(default: the corpus's base-model behavior)")
    ap.add_argument("--splits-dir", type=Path, default=None, help="override corpus dir for splits")
    ap.add_argument("--deviation", action="store_true",
                    help="predict: also ask the residual-objective (deviation) prompts")
    ap.add_argument("--donor-behavior", type=Path, nargs="+", default=None,
                    help="report: donor models' behavior files; scores deviation predictions "
                         "against measured residuals (own minus donor mean)")
    args = ap.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)

    if args.stage == "predict":
        if not args.model:
            sys.exit("--model {base,tuned} required")
        stage_predict(args.evals_repo, args.run, args.model, args.workers, args.holdout_items,
                      args.seed, args.base_shortcut, args.deviation)
    elif args.stage == "drift":
        stage_drift(args.evals_repo, args.run, args.drift_items, args.workers, args.base_shortcut)
    else:
        stage_report(args.evals_repo, args.run, args.out, args.behavior, args.donor_behavior)


if __name__ == "__main__":
    main()
