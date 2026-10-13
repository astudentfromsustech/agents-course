#!/usr/bin/env python3
"""Validate every notebook under practice/.

Catches the failure that shipped once: nbformat concatenates a cell's `source`
list with NO separator, so each line must carry its own trailing newline. A list
built with str.split("\n") loses them, the notebook renders as one run-together
line in Jupyter and Colab, and the code does not parse — while any check that
re-joins with "\n" says it is fine. Verify the way the consumer reads it.

    python3 resources/_check_notebooks.py          # report
    python3 resources/_check_notebooks.py --fix    # repair newlines in place
"""
import glob, json, sys

FIX = "--fix" in sys.argv
bad = fixed = 0

for path in sorted(glob.glob("practice/**/*.ipynb", recursive=True)):
    nb = json.load(open(path))
    problems, dirty = [], False
    for i, c in enumerate(nb.get("cells", [])):
        src = c.get("source", [])
        if not isinstance(src, list) or not src:
            continue
        missing = [j for j, l in enumerate(src[:-1]) if not l.endswith("\n")]
        if missing:
            problems.append(f"cell {i} ({c['cell_type']}): {len(missing)} line(s) without a newline")
            if FIX:
                c["source"] = [l if l.endswith("\n") else l + "\n" for l in src[:-1]] + [src[-1].rstrip("\n")]
                dirty = True
        if c["cell_type"] == "code":
            # IPython magics (!cmd, %magic, %%cellmagic) are legal in a notebook but not
            # in Python — IPython rewrites them before the kernel compiles anything.
            # Blank them out, preserving indentation, so the rest still gets checked.
            lines = []
            for l in "".join(c["source"]).split("\n"):
                stripped = l.lstrip()
                if stripped.startswith(("!", "%")):
                    lines.append(" " * (len(l) - len(stripped)) + "pass")
                else:
                    lines.append(l)
            try:
                compile("\n".join(lines), f"{path}:cell{i}", "exec")
            except SyntaxError as e:
                problems.append(f"cell {i}: does NOT parse as the kernel reads it — {e.msg}")

    if dirty:
        json.dump(nb, open(path, "w"), indent=1)
        fixed += 1
    if problems:
        bad += 1
        print(f"  {path}")
        for p in problems:
            print(f"      {p}")
    else:
        n = sum(c["cell_type"] == "code" for c in nb["cells"])
        print(f"  ok   {path}  ({n} code cells, all parse)")

if FIX and fixed:
    print(f"\nrepaired {fixed} notebook(s) — re-run without --fix to confirm")
    sys.exit(0)
print(f"\n{bad} notebook(s) with problems")
sys.exit(1 if bad else 0)
