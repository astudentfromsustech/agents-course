#!/usr/bin/env python3
"""
Watch multi-head attention learn an algorithm — and measure which head learned it.

p0 built the attention forward pass by hand. This is the other half: a tiny
transformer TRAINS on a task with a known answer, and you find the circuit.

The task: a random block, repeated to fill the sequence — with the block length
chosen AT RANDOM for every example.
    [ 7 3 9 1 | 7 3 9 1 | 7 3 ... ]   period 4
    [ 2 8 5 | 2 8 5 | 2 8 5 | ... ]   period 3
The random period is the point. If the period were fixed, a head could score perfectly
by always attending "31 positions back" — a positional reflex that matches no tokens at
all. With the period varying per example, the only thing that works is the real move:
"I am at token 7; where did 7 appear before; what came after it there." That is an
INDUCTION HEAD, and a 2-layer attention-only transformer discovers it on its own.

Run:  python induction.py          # train, score the heads, write figures
"""
import math, time, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
V, T, D, H, L = 32, 64, 64, 2, 2        # vocab, seq len, d_model, heads/layer, layers
HALF = T // 2


def batch(n):
    """n sequences, each a random block repeated to length T, with its OWN period.
    Returns (tokens, period). The per-row period is what kills the positional shortcut."""
    P = torch.randint(4, HALF + 1, (n,))
    rows = []
    for p in P.tolist():
        base = torch.randint(0, V, (p,))
        rows.append(base.repeat(math.ceil(T / p))[:T])
    return torch.stack(rows).to(DEV), P.to(DEV)


class Attn(nn.Module):
    """Multi-head self-attention, causal. Written out so every tensor is visible."""
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)

    def forward(self, x, keep=False):
        B, t, _ = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        # (B, t, D) -> (B, H, t, d_head): this split is the whole of "multi-head"
        q, k, v = (z.view(B, t, H, D // H).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // H)
        att = att.masked_fill(torch.tril(torch.ones(t, t, device=x.device)) == 0, float("-inf"))
        att = att.softmax(-1)
        if keep:
            self.last = att.detach()                       # (B, H, t, t) for inspection
        y = (att @ v).transpose(1, 2).reshape(B, t, D)     # heads concatenated
        return self.proj(y)


class Tiny(nn.Module):
    """Attention-only: no MLPs, so the circuit you find is made of attention alone."""
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.ModuleList([Attn() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(D), nn.Linear(D, V, bias=False)

    def forward(self, idx, keep=False):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = x + b(self.ln(x), keep)                    # residual stream
        return self.head(self.ln(x))


def loss_on(model, x, P):
    """Score only positions whose answer is actually determined: predicting x[t+1]
    is possible once t >= P, i.e. the block has been seen once already."""
    logits, tgt = model(x)[:, :-1], x[:, 1:]
    t = torch.arange(T - 1, device=DEV)[None, :]
    learnable = t >= P[:, None]
    return F.cross_entropy(logits[learnable], tgt[learnable])


def induction_scores(model, n=128):
    """Two numbers per head, and the gap between them is the finding.

    induction : attention from i to i-P+1, where P is THIS row's period — the token
                that followed the previous occurrence of the current token.
    best_fixed: the best score any single fixed offset could get on the same batch.
                A positional reflex scores high here and low on induction.
    """
    x, P = batch(n)
    model(x, keep=True)
    rows = torch.arange(n, device=DEV)[:, None]
    i = torch.arange(1, T, device=DEV)[None, :]
    valid = i >= P[:, None]                                # induction defined here
    tgt = (i - P[:, None] + 1).clamp(min=0)
    out = {}
    for li, blk in enumerate(model.blocks):
        for h in range(H):
            A = blk.last[:, h]                             # (n, T, T)
            ind = A[rows, i, tgt][valid].mean().item()
            per_off = [A[rows, i, (i - off).clamp(min=0)][valid].mean().item()
                       for off in range(1, HALF + 1)]
            best_off = int(max(range(len(per_off)), key=per_off.__getitem__)) + 1
            out[f"L{li}H{h}"] = (ind, max(per_off), best_off)
    # chance: attention spread uniformly over everything a query is allowed to see
    chance = float((1.0 / (i + 1).float().expand(n, -1)[valid]).mean())
    return out, chance


def main():
    model = Tiny().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    hist, t0 = [], time.time()
    print(f"device {DEV} · {sum(p.numel() for p in model.parameters()):,} params\n")
    for step in range(3001):
        xb, Pb = batch(64)
        loss = loss_on(model, xb, Pb)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            hist.append((step, loss.item()))
        if step % 500 == 0:
            sc, _ = induction_scores(model, 64)
            best = max(sc, key=lambda k: sc[k][0])
            print(f"step {step:5}  loss {loss.item():.4f}   best induction head {best} {sc[best][0]:.3f}")
    print(f"\ntrained in {time.time()-t0:.1f}s")

    sc, chance = induction_scores(model)
    print(f"\nchance ≈ {chance:.3f} (attention spread evenly over everything a query can see)\n")
    print("            induction  best fixed   at      what this head is")
    print("            (content)  (position)  offset")
    for k, (ind, fx, off) in sorted(sc.items(), key=lambda kv: -kv[1][0]):
        if ind > 8 * chance and ind > fx:
            what = "INDUCTION HEAD — matches on content"
        elif fx > 8 * chance and fx > ind:
            what = f"positional — always looks {off} back" + (
                   "  ← previous-token head" if off == 1 else "")
        else:
            what = "neither"
        print(f"  {k}       {ind:.3f}      {fx:.3f}      {off:3}    {what}")
    print("""
The middle column is the control, and it is the point: the best score any FIXED
offset could get on the same batch. Without it, a head that blindly looks k steps
back can masquerade as induction — which it did, until the block period was
randomised. A head only earns the name if it beats its own shortcut.

What you should find is a two-part circuit, not one clever head:
  layer 0  a PREVIOUS-TOKEN head (offset 1) writes 'what came before me' into each position
  layer 1  an INDUCTION head reads that, matches the current token against it, and copies
That composition is why this needs two layers. One layer cannot do it.""")

    # ── figures ────────────────────────────────────────────────────────────────
    steps, losses = zip(*hist)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    ax.plot(steps, losses, lw=1.4, color="#b4541f")
    ax.set_yscale("log")        # linear hides the plateau; the phase change is the point
    ax.set_xlabel("training step"); ax.set_ylabel("loss on predictable positions (nats, log)")
    ax.set_title("Loss does not fall smoothly — it stalls, then drops", fontsize=10)
    ax.axhline(math.log(V), ls=":", lw=1, color="#777")
    ax.annotate(f"chance = ln({V}) = {math.log(V):.2f}", (steps[-1], math.log(V)),
                ha="right", va="bottom", fontsize=7, color="#777")
    ax.annotate("plateau — the circuit\nhas not formed yet", xy=(330, 0.45), xytext=(330, 1.6),
                fontsize=7, color="#444", ha="center",
                arrowprops=dict(arrowstyle="-", lw=.6, color="#999"))
    ax.annotate("induction forms", xy=(690, 0.05), xytext=(900, 0.22),
                fontsize=7, color="#b4541f", ha="left",
                arrowprops=dict(arrowstyle="->", lw=.7, color="#b4541f"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("loss.png"); plt.close(fig)

    x, Pv = batch(1); model(x, keep=True)
    fig, axes = plt.subplots(L, H, figsize=(6.4, 6.0), dpi=200)
    for li, blk in enumerate(model.blocks):
        for h in range(H):
            a = axes[li, h]
            a.imshow(blk.last[0, h].cpu(), cmap="magma", vmin=0, vmax=1)
            _i, _f, _o = sc[f'L{li}H{h}']
            a.set_title(f"L{li}H{h}  ind {_i:.2f} · fixed {_f:.2f}@{_o}", fontsize=7.5)
            a.set_xlabel("key position", fontsize=7); a.set_ylabel("query position", fontsize=7)
            a.tick_params(labelsize=6)
            p = int(Pv[0]); a.plot([1, T - p], [p, T], lw=0.6, ls="--", color="#4ad", alpha=.8)
    fig.suptitle(f"Attention per head on one sequence (period {int(Pv[0])}). The dashed line is the\n"
                 "induction stripe: query i attending to i−P+1, after the previous occurrence.", fontsize=9)
    fig.tight_layout(); fig.savefig("heads.png"); plt.close(fig)
    print("\nwrote loss.png and heads.png")


if __name__ == "__main__":
    main()
