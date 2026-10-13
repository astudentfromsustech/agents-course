#!/usr/bin/env python3
"""Generate practice/p1_induction_heads/induction.ipynb from the canonical cells."""
import json, pathlib

REPO = "astudentfromsustech/agents-course"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/practice/p1_induction_heads/induction.ipynb"

def md(s):
    L = s.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in L[:-1]] + [L[-1]]}

def code(s):
    L = s.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in L[:-1]] + [L[-1]]}

cells = [
md(f"""
# Watch multi-head attention learn an algorithm

[![Open In Colab]({'https://colab.research.google.com/assets/colab-badge.svg'})]({COLAB})

**Runs on your Mac's GPU or on Colab's — the next cell tells you which, and nothing else changes.**

`p0` built the attention forward pass by hand. This trains a 41k-parameter transformer until it
*discovers* a circuit, then measures which head learned what. About 20 s either way.
"""),

code("""
import sys, platform

IN_COLAB = "google.colab" in sys.modules
try:
    import torch
except ImportError:                      # Colab has torch; a bare local kernel might not
    !pip install -q torch matplotlib
    import torch

if torch.cuda.is_available():
    DEV, WHERE = "cuda", torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    DEV, WHERE = "mps", f"Apple {platform.processor() or 'Silicon'} GPU (Metal)"
else:
    DEV, WHERE = "cpu", "CPU only — this still works, just slower"

print(f"frontend : {'Colab' if IN_COLAB else 'local Jupyter'}")
print(f"compute  : {DEV}  ({WHERE})")
print(f"torch    : {torch.__version__}")
"""),

md("""
## The task

A random block, repeated to fill the sequence — **with the block length drawn fresh for every
example**:

```
[ 7 3 9 1 | 7 3 9 1 | 7 3 ... ]   period 4
[ 2 8 5 | 2 8 5 | 2 8 5 | ... ]   period 3
```

The first pass through a block is unpredictable. Everything after it is determined — *if* you can
do one thing: **I am at token 7; where did 7 appear before; what came after it there.**

The random period is the methodological point. With a *fixed* period, a head scores perfectly by
always looking 31 positions back, matching no tokens at all. Randomising it removes that shortcut.
"""),

code("""
import math, time, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
V, T, D, H, L = 32, 64, 64, 2, 2        # vocab, seq len, d_model, heads/layer, layers
HALF = T // 2

def batch(n):
    \"\"\"n sequences, each a random block repeated to length T, with its OWN period.\"\"\"
    P = torch.randint(4, HALF + 1, (n,))
    rows = []
    for p in P.tolist():
        base = torch.randint(0, V, (p,))
        rows.append(base.repeat(math.ceil(T / p))[:T])
    return torch.stack(rows).to(DEV), P.to(DEV)

x, P = batch(2)
print("periods:", P.tolist())
print("sequence 0:", x[0][:20].tolist(), "...")
"""),

md("""
## The model

Attention-only — no MLPs — so the circuit you find is made of attention alone.
"""),

code("""
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)

    def forward(self, x, keep=False):
        B, t, _ = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)
        att = att.masked_fill(torch.tril(torch.ones(t, t, device=x.device)) == 0, float("-inf"))
        att = att.softmax(-1)
        if keep:
            self.last = att.detach()
        return self.proj((att @ v).transpose(1, 2).reshape(B, t, D))

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Attn() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)

    def forward(self, idx, keep=False):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = x + b(self.ln(x), keep)
        return self.head(self.ln(x))

model = Tiny().to(DEV)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters on {DEV}")
"""),

md("""
## Scoring: induction, and the control that makes it mean something

Two numbers per head. The second is the one that matters:

- **induction** — attention from *i* to *i−P+1*, the token that followed the previous occurrence
- **best fixed** — the best score any single *fixed* offset could get on the same batch

A positional reflex scores high on the second and low on the first. **A head only earns the name
if it beats its own shortcut.**
"""),

code("""
def loss_on(x, P):
    logits, tgt = model(x)[:, :-1], x[:, 1:]
    t = torch.arange(T - 1, device=DEV)[None, :]
    learnable = t >= P[:, None]              # only positions whose answer is determined
    return F.cross_entropy(logits[learnable], tgt[learnable])

def scores(n=128):
    x, P = batch(n)
    model(x, keep=True)
    rows = torch.arange(n, device=DEV)[:, None]
    i = torch.arange(1, T, device=DEV)[None, :]
    valid = i >= P[:, None]
    tgt = (i - P[:, None] + 1).clamp(min=0)
    out = {}
    for li, blk in enumerate(model.blocks):
        for h in range(H):
            A = blk.last[:, h]
            ind = A[rows, i, tgt][valid].mean().item()
            per_off = [A[rows, i, (i - o).clamp(min=0)][valid].mean().item()
                       for o in range(1, HALF + 1)]
            best = max(range(len(per_off)), key=per_off.__getitem__)
            out[f"L{li}H{h}"] = (ind, per_off[best], best + 1)
    chance = float((1.0 / (i + 1).float().expand(n, -1)[valid]).mean())
    return out, chance
"""),

md("""
## Train

Watch the `best induction head` column. It stays near chance for a few hundred steps, then climbs.
"""),

code("""
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
hist, t0 = [], time.time()
for step in range(3001):
    xb, Pb = batch(64)
    loss = loss_on(xb, Pb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0:
        hist.append((step, loss.item()))
    if step % 500 == 0:
        sc, _ = scores(64)
        best = max(sc, key=lambda k: sc[k][0])
        print(f"step {step:5}  loss {loss.item():.4f}   best induction head {best} {sc[best][0]:.3f}")
print(f"\\ntrained in {time.time()-t0:.1f}s on {DEV}")
"""),

md("""
## What each head became
"""),

code("""
sc, chance = scores()
print(f"chance = {chance:.3f}\\n")
print("            induction  best fixed   at      what this head is")
for k, (ind, fx, off) in sorted(sc.items(), key=lambda kv: -kv[1][0]):
    if ind > 8 * chance and ind > fx:
        what = "INDUCTION HEAD — matches on content"
    elif fx > 8 * chance and fx > ind:
        what = f"positional — always looks {off} back" + ("  <- previous-token head" if off == 1 else "")
    else:
        what = "neither"
    print(f"  {k}       {ind:.3f}      {fx:.3f}      {off:3}    {what}")
"""),

md("""
**The circuit has two parts, in two layers, and neither works alone:**

| layer 0 | a **previous-token head** writes "what came before me" into each position |
|---|---|
| layer 1 | an **induction head** reads that, matches the current token against it, and copies what followed |

That composition is why this task needs two layers — and why multi-head matters: different jobs,
different heads, running side by side without mixing.
"""),

code("""
steps, losses = zip(*hist)
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=140)
ax.plot(steps, losses, lw=1.4, color="#b4541f")
ax.set_yscale("log")
ax.set_xlabel("training step"); ax.set_ylabel("loss on predictable positions (log)")
ax.set_title("Loss does not fall smoothly — it stalls, then drops", fontsize=10)
ax.axhline(math.log(V), ls=":", lw=1, color="#777")
ax.spines[["top", "right"]].set_visible(False)
plt.show()
"""),

code("""
xv, Pv = batch(1); model(xv, keep=True)
fig, axes = plt.subplots(L, H, figsize=(6.4, 6.0), dpi=140)
for li, blk in enumerate(model.blocks):
    for h in range(H):
        a = axes[li, h]
        a.imshow(blk.last[0, h].cpu(), cmap="magma", vmin=0, vmax=1)
        i_, f_, o_ = sc[f"L{li}H{h}"]
        a.set_title(f"L{li}H{h}  ind {i_:.2f} · fixed {f_:.2f}@{o_}", fontsize=8)
        a.set_xlabel("key position", fontsize=7); a.set_ylabel("query position", fontsize=7)
        a.tick_params(labelsize=6)
        p = int(Pv[0]); a.plot([1, T - p], [p, T], lw=.6, ls="--", color="#4ad", alpha=.8)
fig.suptitle(f"Attention per head (period {int(Pv[0])}). Dashed = the induction stripe.", fontsize=9)
plt.tight_layout(); plt.show()
"""),

md("""
## Things worth trying

Each is one edit and a re-run.

1. **Make the period fixed** — in `batch`, replace the random `P` with a constant. Re-train. Watch
   the control column expose the shortcut: heads score high on *fixed* and low on *induction*.
2. **Drop to one layer** — set `L = 1`. It should fail; one layer cannot compose the two steps.
3. **More heads** — set `H = 4`. Do you get more induction heads, or redundant ones?
4. **Longer sequences** — raise `T`. Does the circuit still form, and how much slower?
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path("practice/p1_induction_heads/induction.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
