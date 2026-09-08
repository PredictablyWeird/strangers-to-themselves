"""Statistical rigor pass: significance + equivalence macros for the paper.

Reads the canonical ``results/reports/evaluation.json`` (default pool = the merged 12
models since 2026-08-07; the paper's numbers). Writes:

  paper/generated/stats.tex        -- one ``\\newcommand`` per test statistic
  paper/generated/table_cell_n.tex -- appendix table: dev-split n (conditions) per (eval, model)

Tests (all on scored cells, i.e. after the noise-ceiling filter, matching every number in the
paper). Each paired comparison reports the mean difference over shared (eval, model) cells, a
10k percentile-bootstrap 95% CI (resampling cells), and a two-sided sign-flip permutation
p-value (exact when the cell count allows full enumeration, else 100k Monte-Carlo):

  SrXmm       self_report vs cross_model_mean            (all shared cells)
  IoXmm       informed_oracle vs cross_model_mean        (all shared cells)
  IoXmmDiscrim  the DiscrimEval exception                (DiscrimEval cells only)
  PwSrPB      pairwise vs self_report                    (PropensityBench cells only)

Frontier scale claim (revised 2026-08-15): DIRECTIONAL primary — "scale does not
help": the group difference (frontier - small) in per-model self_report macro means over the
four text evals every model shares, with its one-sided 95% upper bound (the 95th bootstrap
percentile) capping any frontier advantage. TOST equivalence is SECONDARY, at a margin that
must be small relative to the ~0.1-0.3 effect scale of the methods themselves (0.05, not the
old 0.10 — a 0.10 margin is larger than the entire self-report macro).

Deterministic (seed 1234). Invoked automatically by ``bp-evaluate`` after a default-pool run
(alongside scripts/spearman_robustness.py); safe to run standalone.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from behavior_prediction import method_names

SEED = 1234
BOOT = 10_000
SPLIT = "dev"   # overwritten in main() from the evaluation doc; captions read it lazily
PERM_MC = 100_000
EXACT_MAX = 20            # full sign-flip enumeration up to 2^20 combos
#: Equivalence margin, DERIVED non-inferiority style (2026-08-15): one fifth of the
#: item-informed self-prediction (informed_oracle) dev macro — the strongest asking-channel
#: method (anchoring on self_report was rejected: a method that doesn't work is no reference).
#: 0.20 x (+0.25) = 0.05. FROZEN from the final dev artifacts at pre-registration pin time;
#: NEVER recomputed on the test split (re-derive here manually at the stage-7 freeze if the
#: informed_oracle dev macro moves). The primary claim is estimation (the 95% CI's upper end
#: bounds the advantage); this margin only feeds the secondary TOST / the E-N-F ladder cuts.
MARGIN = 0.05
TEXT_EVALS = ["sycophancy_pushback", "discrimeval", "capability_mmlu", "reward_hacking"]
FRONTIER = ["claude-sonnet-5-low", "claude-sonnet-5-off", "gpt-5.5-low", "gpt-5.5-off",
            "deepseek-v4-pro-low", "deepseek-v4-pro-off"]
SMALL = ["deepseek-v4-flash-low", "gemini-3.1-flash-lite-low", "gpt-5.4-nano-low",
         "llama-3.3-70b", "llama-4-maverick", "qwen3.7-plus-low"]

OUT_STATS = Path("paper/generated/stats.tex")
OUT_NTAB = Path("paper/generated/table_cell_n.tex")
OUT_FRTAB = Path("paper/generated/table_frontier.tex")

# tab:frontier layout (row order matches the paper's prose; per_method_model is canonical).
FR_ROWS = [("claude-sonnet-5-off", "Claude Sonnet 5 (off)"),
           ("claude-sonnet-5-low", "Claude Sonnet 5 (low)"),
           ("gpt-5.5-off", "GPT-5.5 (off)"),
           ("gpt-5.5-low", "GPT-5.5 (low)"),
           ("deepseek-v4-pro-off", "DeepSeek V4 Pro (off)"),
           ("deepseek-v4-pro-low", "DeepSeek V4 Pro (low)")]
FR_METHODS = ["self_report", "pairwise", "informed_oracle", "generic_oracle",
              "cross_model_mean"]
FR_MACRO = {"claude-sonnet-5-off": "SonnetOff", "claude-sonnet-5-low": "SonnetLow",
            "gpt-5.5-off": "GptOff", "gpt-5.5-low": "GptLow",
            "deepseek-v4-pro-off": "DsOff", "deepseek-v4-pro-low": "DsLow"}


def _cells(doc) -> dict[tuple[str, str, str], float]:
    return {(c["eval"], c["model"], c["method"]): c["r"] for c in doc["cells"]}


def _fmt(x: float) -> str:
    return f"{x:+.2f}"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def paired(cells, m_a: str, m_b: str, evals: list[str] | None = None) -> dict:
    """Mean r(m_a) - r(m_b) over shared (eval, model) cells + bootstrap CI + sign-flip p."""
    keys = sorted({(e, m) for (e, m, meth) in cells if meth == m_a and (e, m, m_b) in cells
                   and (evals is None or e in evals)})
    d = np.array([cells[(e, m, m_a)] - cells[(e, m, m_b)] for e, m in keys])
    n = len(d)
    if n == 0:   # method pair not (yet) in the evaluation doc — emit NaNs rather than crash
        nan = float("nan")
        return {"n": 0, "diff": nan, "lo": nan, "hi": nan, "p": nan, "n_pos": 0, "n_neg": 0}
    rng = np.random.default_rng(SEED)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    obs = abs(d.mean())
    if n <= EXACT_MAX:                     # exact two-sided sign-flip permutation
        hits, total = 0, 2 ** n
        for signs in itertools.product((1.0, -1.0), repeat=n):
            if abs((d * signs).mean()) >= obs - 1e-12:
                hits += 1
        p = hits / total
    else:
        signs = rng.choice([1.0, -1.0], size=(PERM_MC, n))
        p = ((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum() + 1) / (PERM_MC + 1)
    return {"n": n, "diff": d.mean(), "lo": lo95, "hi": hi95, "p": p,
            "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())}


def claim_tier(lo95: float, hi95: float, m: float) -> str:
    """Pre-registered claim ladder (docs/test-prespecification.md §3): which sentence the
    95% CI of an advantage difference licenses, strongest first. ALL intervals are 95%
    (never 90%) — TOST equivalence runs through the 95% CI (alpha=0.025 per side).

      E  equivalence: CI within (-m, +m)          -> "statistically equivalent"
      D  bounded no-advantage: hi < +m, lo <= -m  -> "no more accurate; advantage <= hi"
      N  null, unbounded: CI straddles 0, hi >= m -> "no significant difference" (cannot
                                                     bound the advantage below m)
      S  small advantage: lo > 0, hi < m          -> "small but significant advantage <= hi"
      F  substantial advantage: lo > 0, hi >= m   -> claim fails; report as measured
    (CI entirely below 0 falls under D — the comparison subject is strictly better.)"""
    if lo95 > 0:
        return "S" if hi95 < m else "F"
    if -m < lo95 and hi95 < m:
        return "E"
    if hi95 < m:
        return "D"
    return "N"


def frontier_tost(fcells) -> dict:
    """Group difference in per-model self_report macro means (frontier - small). 95% CI
    ONLY: the equivalence check and the advantage bound both use the 95% interval."""
    def model_mean(m):
        rs = [fcells[(e, m, "self_report")] for e in TEXT_EVALS if (e, m, "self_report") in fcells]
        return float(np.mean(rs))
    f = np.array([model_mean(m) for m in FRONTIER])
    s = np.array([model_mean(m) for m in SMALL])
    rng = np.random.default_rng(SEED)
    boots = np.array([f[rng.integers(0, len(f), len(f))].mean()
                      - s[rng.integers(0, len(s), len(s))].mean() for _ in range(BOOT)])
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    return {"diff": f.mean() - s.mean(), "lo95": lo95, "hi95": hi95,
            "equiv": bool(lo95 > -MARGIN and hi95 < MARGIN),
            "tier": claim_tier(lo95, hi95, MARGIN),
            "f_mean": f.mean(), "s_mean": s.mean()}


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def emit(name: str, r: dict) -> list[str]:
    return [macro(f"stat{name}Diff", _fmt(r["diff"])),
            macro(f"stat{name}CILo", _fmt(r["lo"])),
            macro(f"stat{name}CIHi", _fmt(r["hi"])),
            macro(f"stat{name}P", _fmt_p(r["p"])),
            macro(f"stat{name}N", str(r["n"])),
            macro(f"stat{name}NPos", str(r["n_pos"]))]


def frontier_exports(fdoc) -> tuple[str, list[str]]:
    """tab:frontier (fully generated) + the prose macros of sec:results-frontier."""
    mm = {(x["method"], x["model"]): x["mean_r"] for x in fdoc["per_method_model"]}
    small_mean = {meth: float(np.mean([mm[(meth, m)] for m in SMALL if (meth, m) in mm]))
                  for meth in FR_METHODS}
    lines = [
        "% Auto-generated by scripts/significance_tests.py -- DO NOT EDIT BY HAND.",
        "\\begin{table}[h]", "  \\centering",
        "  \\caption{The frontier settings of the pool (" + SPLIT + " split; mean Pearson $r$ over each",
        "           model's scoreable evaluations, at the settings frozen on the small-model",
        "           selection pool). The bottom row is the small-tier mean from the same",
        "           evaluation.",
        "           Ask-channel methods stay far below the shared-structure predictor",
        "           (the cross-model behavior mean)",
        "           at every scale and reasoning setting.}",
        "  \\label{tab:frontier}", "  \\footnotesize", "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\begin{tabular}{l" + "c" * len(FR_METHODS) + "}", "    \\toprule",
        "    Model (reasoning) & " + " & ".join(method_names.short(m) for m in FR_METHODS)
        + " \\\\", "    \\midrule",
    ]
    for slug, disp in FR_ROWS:
        cells = " & ".join(f"${_fmt(mm[(meth, slug)])}$" if (meth, slug) in mm else "--"
                           for meth in FR_METHODS)
        lines.append(f"    {disp:<23} & {cells} \\\\")
    lines.append("    \\midrule")
    lines.append("    Small-tier mean & "
                 + " & ".join(f"${_fmt(small_mean[meth])}$" for meth in FR_METHODS) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]

    io = {slug: mm[("informed_oracle", slug)] for slug, _ in FR_ROWS}
    sr = {slug: mm[("self_report", slug)] for slug, _ in FR_ROWS}
    fr_io_mean = float(np.mean(list(io.values())))
    sr_reason_max = max(abs(sr[f"{b}-low"] - sr[f"{b}-off"])
                        for b in ("claude-sonnet-5", "gpt-5.5", "deepseek-v4-pro"))
    macros = ["", "% Frontier-tier prose numbers (per_method_model means of the frontier report).",
              macro("statFrIoFrontierMean", _fmt(fr_io_mean)),
              macro("statFrIoSmallMean", _fmt(small_mean["informed_oracle"])),
              macro("statFrIoDiff", _fmt(fr_io_mean - small_mean["informed_oracle"])),
              macro("statFrSrLo", _fmt(min(sr.values()))),
              macro("statFrSrHi", _fmt(max(sr.values()))),
              macro("statFrSrReasonMax", _fmt(sr_reason_max))]
    macros += [macro(f"statFrIo{FR_MACRO[slug]}", _fmt(io[slug])) for slug, _ in FR_ROWS]
    return "\n".join(lines), macros


def n_table(doc) -> str:
    ceil = {(c["eval"], c["model"]): c for c in doc["ceilings"]}
    weak = {(w["eval"], w["model"]) for w in doc["dropped_weak"]}
    evals, models = doc["evals"], doc["models"]
    short = {m: m.replace("-low", "").replace("-3.3-70b", "-3.3").replace("3.1-", "") for m in models}
    lines = [
        "% Auto-generated by scripts/significance_tests.py -- DO NOT EDIT BY HAND.",
        "\\begin{table}[h]", "  \\centering",
        "  \\caption{" + SPLIT.capitalize() + "-split cell sizes: number of scored conditions per (evaluation, model)."
        " $\\dagger$ = cell dropped by the noise-ceiling filter (behavior too unreliable or"
        " degenerate to score); its predictions enter no aggregate.}",
        "  \\label{tab:cell-n}", "  \\scriptsize", "  \\setlength{\\tabcolsep}{2pt}",
        "  \\begin{tabular}{l" + "c" * len(models) + "}", "    \\toprule",
        "    Evaluation & " + " & ".join(f"\\rotatebox{{90}}{{\\texttt{{{short[m]}}}}}"
                                          for m in models) + " \\\\",
        "    \\midrule",
    ]
    for e in evals:
        row = []
        for m in models:
            c = ceil.get((e, m))
            n = c.get("n_conditions") if c else None   # bare {eval, model} = ceiling not computable
            cell = "--" if not n else (f"{n}$^\\dagger$" if (e, m) in weak else str(n))
            row.append(cell)
        eshort = {"sycophancy_pushback": "sycophancy", "propensitybench": "PropB",
                  "capability_mmlu": "capability", "reward_hacking": "reward hack",
                  "tau2_policy": "$\\tau^2$", "discrimeval": "DiscrimEval",
                  "tau2_transfer": "$\\tau^2$ transfer",
                  "mask_subdomain_pressure": "MASK pressure"}.get(e, e.replace("_", "\\_"))
        lines.append(f"    {eshort} & " + " & ".join(row) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> int:
    doc = json.load(open("results/reports/evaluation.json"))
    global SPLIT
    SPLIT = doc.get("split", "dev")
    cells = _cells(doc)
    fcells = cells   # merged pool: tier stats come from the same doc

    lines = ["% Auto-generated by scripts/significance_tests.py -- DO NOT EDIT BY HAND.",
             f"% Paired mean differences over shared scored (eval, model) cells; {BOOT} bootstrap",
             "% resamples (95% CI); two-sided sign-flip permutation p (exact when cells <= "
             f"{EXACT_MAX}).", ""]
    for name, a, b, evs in [
        ("SrXmm", "cross_model_mean", "self_report", None),
        ("IoXmm", "cross_model_mean", "informed_oracle", None),
        ("IoXmmDiscrim", "informed_oracle", "cross_model_mean", ["discrimeval"]),
        ("PwSrPB", "pairwise", "self_report", ["propensitybench"]),
        # Generic-self-image pillar: the indexical null (does "you" beat "a generic
        # assistant"?) and the shared-prior test (does the self beat the mean of the other
        # models' self-reports?). Both expected ~0; the CI is the claim.
        ("SrGeneric", "self_report", "generic_report", None),
        ("SrReportmean", "self_report", "report_mean", None),
        # Oracle tier: does holding INFORMATION fixed and changing the SUBJECT cost anything?
        ("IoGeneric", "informed_oracle", "generic_oracle", None),
        ("IoOrm", "informed_oracle", "oracle_report_mean", None),
        # History tier: same examples, same conversation format, only WHO answers changes
        # (few_shot = the model itself, few_shot_other = the fixed analyst); and the format
        # effect at fixed answerer-is-other (conversation vs one text block).
        ("FsFso", "few_shot", "few_shot_other", None),
        ("FsoLlm", "few_shot_other", "llm_prediction", None),
    ]:
        r = paired(cells, a, b, evs)
        lines += emit(name, r)
        lines.append(f"% {name}: {a} - {b}"
                     f"{' on ' + ','.join(evs) if evs else ''}: {r['n_pos']}/{r['n']} cells positive")
        # Ladder claims (docs/test-prespecification.md §3): stamp the tier the 95% CI attains.
        # IoGeneric is C3(a) — the self-vs-generic advantage; same margin as the frontier claim.
        if name == "IoGeneric":
            tier = claim_tier(r["lo"], r["hi"], MARGIN)
            lines += [macro("statIoGenericTier", tier),
                      macro("statIoGenericUB", _fmt(r["hi"])),
                      f"% claim tier (C3a, self - generic): {tier}  "
                      f"(E equiv / D bounded / N null / S small adv / F fails); "
                      f"self-advantage capped at {r['hi']:+.3f} (95% CI upper end)"]
    t = frontier_tost(fcells)
    lines += ["",
              "% Frontier scale claim: per-model self_report macro mean over the four shared",
              "% text evals. 95% CI ONLY (never 90%). statFrontierSrUB = the CI's upper end,",
              "% capping any frontier advantage; TOST equivalence through the SAME 95% CI",
              f"% (margin +/-{MARGIN}, alpha=0.025 per side). Claim ladder: see claim_tier().",
              macro("statFrontierSrDiff", _fmt(t["diff"])),
              macro("statFrontierSrUB", _fmt(t["hi95"])),
              macro("statFrontierSrCILo", _fmt(t["lo95"])),
              macro("statFrontierSrCIHi", _fmt(t["hi95"])),
              macro("statFrontierSrMean", _fmt(t["f_mean"])),
              macro("statSmallSrMean", _fmt(t["s_mean"])),
              macro("statEquivMargin", f"{MARGIN:g}"),
              f"% claim tier attained: {t['tier']}  (E equiv / D bounded / N null / "
              f"S small adv / F fails)",
              f"% frontier advantage capped at {t['hi95']:+.3f} (95% CI upper end); "
              f"equivalence at {MARGIN}: {'PASS' if t['equiv'] else 'FAIL'}"]
    frtab, frmacros = frontier_exports(doc)
    lines += frmacros
    OUT_STATS.write_text("\n".join(lines) + "\n")
    OUT_NTAB.write_text(n_table(doc) + "\n")
    OUT_FRTAB.write_text(frtab + "\n")
    print(f"Wrote {OUT_STATS}, {OUT_NTAB} and {OUT_FRTAB}")
    for ln in lines:
        if ln.startswith("%") and (":" in ln or "PASS" in ln or "FAIL" in ln):
            print(" ", ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
