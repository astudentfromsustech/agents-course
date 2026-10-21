#!/usr/bin/env bash
# Re-run the notebook and commit it WITH its outputs.
#
# The notebook is shipped executed, so every result is readable without a
# kernel. That means any edit to vit_mnist.py or to a cell leaves the stored
# outputs stale — run this afterwards.
#
#   bash practice/p2_vit_mnist/rebuild.sh
set -euo pipefail
cd "$(dirname "$0")"
BIN="$HOME/.conda/envs/courses_self-attention/bin"

"$BIN/jupyter" nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=python3 \
  vit_mnist.ipynb

"$BIN/python" - <<'PY'
import json
nb = json.load(open("vit_mnist.ipynb"))
import re
html = "".join("".join(o.get("data", {}).get("text/html", []))
               for c in nb["cells"] for o in c.get("outputs", []))
errs = sum(o.get("output_type") == "error"
           for c in nb["cells"] for o in c.get("outputs", []))
imgs = (sum("image/png" in o.get("data", {})
            for c in nb["cells"] for o in c.get("outputs", []))
        + len(re.findall(r"data:image/png;base64", html)))
dets = len(re.findall(r"<details", html))
blank = [i for i, c in enumerate(nb["cells"])
         if c["cell_type"] == "code" and not c.get("outputs")]
print(f"  {imgs} images · {dets} click-to-expand blocks · {errs} errors · "
      f"cells with no output: {blank or 'none'}")
assert errs == 0, "notebook has error outputs — fix before committing"
assert not blank, "some code cell produced nothing"
assert dets >= 10, "the fold blocks are missing"
PY
"$BIN/python" - <<'PY'
import ast
tree = ast.parse(open("vit_mnist.py").read())
bad = []
for n in tree.body:
    if not isinstance(n, ast.FunctionDef):
        continue
    src = ast.unparse(n)
    takes_path = "path" in [a.arg for a in n.args.args + n.args.kwonlyargs]
    saves_fig = "savefig" in src                      # a FIGURE function, not load()
    if takes_path and saves_fig:
        top = [ast.unparse(r.value) for r in n.body if isinstance(r, ast.Return) and r.value]
        if top and top[-1] != "path":
            bad.append(f"{n.name} returns {top[-1]}")
print(f"  figure functions returning their path: {'all' if not bad else 'PROBLEM'}")
assert not bad, "fold() cannot embed a figure unless the function returns its path: " + "; ".join(bad)
PY
echo "  ok — commit vit_mnist.ipynb with its outputs"
