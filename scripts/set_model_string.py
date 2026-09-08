#!/usr/bin/env python3
"""Point a models.yaml shortcut at a freshly-deployed Together endpoint.

Together mints a new name suffix every time a dedicated endpoint is created, so the model string
committed in models.yaml goes stale the moment its endpoint is torn down. After `finetune_together.py
deploy`, this rewrites the shortcut's `model:` line in place (comments and layout preserved — the file
is documentation as much as config).

    python scripts/set_model_string.py llama-3.3-70b-intro30k-tg logs/introspection_finetune_llama30k
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

YAML = Path(__file__).resolve().parent.parent / "behavior_prediction" / "models.yaml"


def main() -> int:
    if len(sys.argv) != 3:
        return print(__doc__) or 2
    shortcut, out = sys.argv[1], Path(sys.argv[2])
    rec = json.loads((out / "together_endpoint.json").read_text())
    if rec.get("deleted"):
        sys.exit(f"{out}/together_endpoint.json is marked deleted — deploy before patching")
    new = f"together/{rec['endpoint_name']}"

    text = YAML.read_text()
    # The shortcut's block runs from its `  <name>:` key to the next key at the same indent.
    m = re.search(rf"^  {re.escape(shortcut)}:\n(?:.*\n)*?(?=  \S|\Z)", text, re.M)
    if not m:
        sys.exit(f"shortcut {shortcut!r} not found in {YAML}")
    block = m.group(0)
    patched, n = re.subn(r"^(\s+model:\s*).*$", lambda g: g.group(1) + new, block, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"no `model:` line inside the {shortcut!r} block")
    if patched == block:
        print(f"{shortcut}: already points at {new}")
        return 0
    YAML.write_text(text[: m.start()] + patched + text[m.end():])
    print(f"{shortcut} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
