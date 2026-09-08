# paper/

Only the **generated** artifacts are versioned here. `generated/` holds the result macros,
tables and figures written by the scripts named in `scripts/README.md` — regenerate those
rather than editing values by hand.

## Sources are kept in Overleaf, not in git

The LaTeX sources are maintained separately and are deliberately not tracked:

| file | needed for |
| --- | --- |
| `main.tex` | the paper itself |
| `references.bib` | bibliography |
| `neurips_2026.sty` | vendored style file (loads `natbib` — do not add `\usepackage{natbib}`) |

To build, copy all three into this directory, then:

```bash
cd paper && latexmk -pdf main.tex
```

Without them the document cannot be compiled, and these two run lanes fail at their
LaTeX step (everything before it still succeeds):

- `scripts/runs/run_results_qwen.sh` — "recompile paper PDF"
- `scripts/runs/run_stage7_dev_eval.sh` — "LaTeX build check"

Regenerating `generated/` does not need the sources and is unaffected.
