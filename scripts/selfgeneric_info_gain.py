"""Paired information gain, self framing vs generic framing (analysis 3 of the analyst-gain set).

Within each (model, eval) cell: gain_self = r(informed_oracle) - r(self_report);
gain_generic = r(generic_oracle) - r(generic_report). If the self question extracts privileged
information once items are shown, gain_self should exceed gain_generic beyond what the self
arm's lower (denial-suppressed) starting point explains — so the *levels* reached are reported
alongside. Cells require all four methods scored (post ceiling filter) in the evaluation JSON.
Bootstrap 95% CI over cells (paired).

    python scripts/selfgeneric_info_gain.py            # both splits
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
QUAD = ("self_report", "informed_oracle", "generic_report", "generic_oracle")


def cells(split: str) -> dict[tuple[str, str], dict[str, float]]:
    doc = json.load(open(ROOT / f"results/reports/evaluation{'_test' if split == 'test' else ''}.json"))
    out: dict[tuple[str, str], dict[str, float]] = {}
    for c in doc["cells"]:
        if c["method"] in QUAD and c.get("r") is not None:
            out.setdefault((c["eval"], c["model"]), {})[c["method"]] = c["r"]
    return {k: v for k, v in out.items() if all(m in v for m in QUAD)}


def ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    bs = sorted(mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def fmt(v: float) -> str:
    return f"{v:+.2f}"


def report(split: str) -> list[str]:
    C = cells(split)
    lines = [f"===== {split} split: {len(C)} complete cells (all four of {', '.join(QUAD)}) ====="]
    by_eval: dict[str, list[tuple[str, str]]] = {}
    for (ev, m) in C:
        by_eval.setdefault(ev, []).append((ev, m))
    hdr = (f"{'eval':22s} {'n':>3s}  {'self_rep':>8s} {'inf_orc':>8s} {'gain_s':>7s}   "
           f"{'gen_rep':>8s} {'gen_orc':>8s} {'gain_g':>7s}   {'gs-gg':>6s}  {'95% CI':>14s}")
    lines.append(hdr)

    def row(name: str, keys: list[tuple[str, str]]) -> str:
        v = [C[k] for k in keys]
        gs = [x["informed_oracle"] - x["self_report"] for x in v]
        gg = [x["generic_oracle"] - x["generic_report"] for x in v]
        d = [a - b for a, b in zip(gs, gg)]
        lo, hi = ci(d) if len(d) > 1 else (d[0], d[0])
        return (f"{name:22s} {len(v):>3d}  {fmt(mean(x['self_report'] for x in v)):>8s} "
                f"{fmt(mean(x['informed_oracle'] for x in v)):>8s} {fmt(mean(gs)):>7s}   "
                f"{fmt(mean(x['generic_report'] for x in v)):>8s} "
                f"{fmt(mean(x['generic_oracle'] for x in v)):>8s} {fmt(mean(gg)):>7s}   "
                f"{fmt(mean(d)):>6s}  [{fmt(lo)}, {fmt(hi)}]")

    for ev in sorted(by_eval):
        lines.append(row(ev, by_eval[ev]))
    lines.append(row("ALL (paired cells)", list(C)))
    # Level reached: does the self arm end ABOVE the generic arm after information?
    end = [C[k]["informed_oracle"] - C[k]["generic_oracle"] for k in C]
    lo, hi = ci(end)
    lines.append(f"  informed_oracle - generic_oracle (level reached): {fmt(mean(end))} "
                 f"[{fmt(lo)}, {fmt(hi)}]")
    start = [C[k]["self_report"] - C[k]["generic_report"] for k in C]
    lo, hi = ci(start)
    lines.append(f"  self_report - generic_report (starting level):   {fmt(mean(start))} "
                 f"[{fmt(lo)}, {fmt(hi)}]")
    lines.append("")
    return lines


def main() -> None:
    out = ["# Paired information gain: self framing vs generic framing",
           "# gain_s = r(informed_oracle) - r(self_report); gain_g = r(generic_oracle) - r(generic_report)",
           "# gs-gg > 0 with informed_oracle > generic_oracle would indicate information unlocks",
           "# self-specific access; gs-gg > 0 with equal end levels = recovery from a suppressed base.",
           ""]
    for split in ("dev", "test"):
        out += report(split)
    text = "\n".join(out)
    print(text)
    (ROOT / "results/reports/selfgeneric_info_gain.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
