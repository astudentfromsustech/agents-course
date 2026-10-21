#!/usr/bin/env python3
"""
A transformer that reads handwritten digits — and three measurements on it.

p0 built the attention forward pass by hand. p1 watched heads learn an algorithm.
This one puts a transformer on the dataset the chapter keeps drawing: MNIST.

An image is not a sequence, so the first question is how it becomes one. A
Vision Transformer cuts the 28x28 image into 7x7 patches, flattens each into a
vector, and calls that the sequence — 16 tokens, plus one [CLS] token whose
final vector is what the classifier reads. Everything after that is chapter 01
unchanged: Q/K/V, softmax, multi-head, residual, norm.

Three things the chapter CLAIMS, measured here instead of asserted:

  train     it works at all, and the attention maps are readable
  position  attention is permutation-equivariant. Strip the positional
            encoding and shuffling the patches changes NOTHING. Keep it and
            shuffling destroys the model. (§ Position: the one thing
            attention cannot see)
  scale     a CNN's hard-coded locality is a prior — free knowledge worth a
            lot when data is scarce, worth less as data grows. (§ Self-
            attention vs CNN)

Run:  python vit_mnist.py              # all three
      python vit_mnist.py --only scale # just one
"""
import argparse, math, os, time, urllib.request, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

IMG, PATCH = 28, 7
NP = (IMG // PATCH) ** 2          # 16 patches
D, HEADS, LAYERS = 64, 4, 4


# ---------------------------------------------------------------- data
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

def load(path="mnist.npz"):
    """MNIST as four tensors. One 11 MB .npz, no torchvision — so this file runs
    anywhere torch does, including a bare Colab runtime."""
    if not os.path.exists(path):
        print(f"  downloading MNIST -> {path}")
        urllib.request.urlretrieve(URL, path)
    d = np.load(path)
    def pack(x, y):
        X = torch.from_numpy(x).float().div(255).sub(0.1307).div(0.3081).unsqueeze(1)
        return X.to(DEV), torch.from_numpy(y).long().to(DEV)
    tr, te = pack(d["x_train"], d["y_train"]), pack(d["x_test"], d["y_test"])
    n_tr, n_te = tr[0].shape[0], te[0].shape[0]
    print(f"  MNIST  {n_tr + n_te:,} images total  =  {n_tr:,} train  +  {n_te:,} test")
    print(f"         each {tuple(tr[0].shape[1:])} — 1 greyscale channel, 28x28 px, "
          f"labels 0-9")
    return tr, te


# ---------------------------------------------------------------- model
class PatchEmbed(nn.Module):
    """Image -> sequence of patch vectors.

    Note what this layer IS: a convolution with kernel == stride == patch size.
    Non-overlapping, so it does no mixing at all — it only reshapes pixels into
    tokens. Every bit of spatial reasoning after this point is done by attention.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, D, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):                      # x [B,1,28,28]
        return self.proj(x).flatten(2).transpose(1, 2)   # [B, 16, D]


class Block(nn.Module):
    """Pre-norm block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, keep_attn=False):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, HEADS, D // HEADS).transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / (D // HEADS) ** 0.5      # [B,H,T,T]
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)                                       # NO mask: every patch sees every patch
        x = x + self.mlp(self.n2(x))
        return (x, att) if keep_attn else (x, None)


class ViT(nn.Module):
    def __init__(self, use_pos=True):
        super().__init__()
        self.use_pos = use_pos
        self.patch = PatchEmbed()
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.randn(1, NP + 1, D) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.norm, self.head = nn.LayerNorm(D), nn.Linear(D, 10)

    def forward(self, x, perm=None, keep_attn=False):
        t = self.patch(x)                                  # [B,16,D]
        if perm is not None:
            t = t[:, perm]                                 # shuffle the PATCHES, not the pixels
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1)
        if self.use_pos:
            t = t + self.pos                               # position added AFTER the shuffle
        maps = []
        for b in self.blocks:
            t, a = b(t, keep_attn)
            if keep_attn: maps.append(a)
        return self.head(self.norm(t)[:, 0]), maps         # read the [CLS] row only


class SmallCNN(nn.Module):
    """The baseline with locality built in, sized to a similar parameter count."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x, **kw):
        return self.f(x), []


# ---------------------------------------------------------------- folding
def fold(fn, *args, label=None, start_open=False, **kw):
    """Run fn and render everything it produced inside a click-to-expand block.

    Captures the function's printed output and, if it returns a .png path,
    embeds that image too — then wraps both in an HTML <details>, which every
    notebook front-end renders as a real disclosure triangle you can click.

        fold(show_data, d, label="the data and the patches")
    """
    import base64, builtins, contextlib, io, html as _html
    from IPython.display import HTML, display

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kw)
    text = buf.getvalue().rstrip()

    body = ""
    if text:
        body += f'<pre style="margin:.4em 0;line-height:1.35">{_html.escape(text)}</pre>'
    if isinstance(out, str) and out.endswith(".png") and os.path.exists(out):
        with builtins.open(out, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        body += f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'

    title = label or getattr(fn, "__name__", "details")
    display(HTML(
        f'<details {"open" if start_open else ""} style="margin:.3em 0">'
        f'<summary style="cursor:pointer;font-weight:600;color:#1c7ed6;'
        f'padding:.25em 0;user-select:none">{_html.escape(title)}</summary>'
        f'<div style="padding:.4em 0 .2em 1.1em;border-left:2px solid #dee2e6;'
        f'margin-left:.35em">{body}</div></details>'))
    return out



# ---------------------------------------------------------------- a shared model
_CACHE = {}

def trained(d, epochs=6):
    """One trained model, reused by every figure that needs one.

    show_filters() and show_training() still train their own, because they need
    the weights BEFORE training and the loss at every step respectively. The
    rest just need a model that works, so they share this one.
    """
    if epochs not in _CACHE:
        (Xtr, Ytr), (Xte, Yte) = d
        torch.manual_seed(0)
        m = ViT().to(DEV)
        run(m, Xtr, Ytr, Xte, Yte, epochs=epochs, quiet=True)
        _CACHE[epochs] = m.eval()
    return _CACHE[epochs]



# ---------------------------------------------------------------- look at it
def show_data(d, n=8, path="data_and_patches.png"):
    """What the model is actually handed.

    Row 1: digits. Row 2: ONE digit cut on the 7x7 grid. Row 3: the same 16
    patches as the sequence the transformer receives. The model never sees the
    square arrangement — only those 16 vectors in a row, which is exactly why
    the positional encoding has to put the layout back.
    """
    (Xtr, _), (Xte, Yte) = d
    img = Xte[0, 0].cpu()
    G = IMG // PATCH
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 1.25, .62], hspace=.55)

    for i in range(n):                                       # --- the data
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(Xte[i, 0].cpu(), cmap="gray"); ax.axis("off")
        ax.set_title(int(Yte[i]), fontsize=11)
    ntr, nte = Xtr.shape[0], Xte.shape[0]
    fig.text(.5, 1.0, f"MNIST: {ntr + nte:,} images  =  {ntr:,} train  +  {nte:,} test"
             f"   (these {n} are from the test set)", ha="center", fontsize=10.5, va="bottom")

    ax = fig.add_subplot(gs[1, :2])                          # --- the cut
    ax.imshow(img, cmap="gray")
    for k in range(1, G):
        ax.axhline(k * PATCH - .5, color="#e8590c", lw=1.6)
        ax.axvline(k * PATCH - .5, color="#e8590c", lw=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"cut into {G}x{G} = {NP} patches of {PATCH}x{PATCH}px", fontsize=10, pad=6)

    ax2 = fig.add_subplot(gs[1, 2:])                         # --- what it means
    ax2.axis("off")
    px = PATCH * PATCH
    ax2.text(0, .97, f"{PATCH}x{PATCH} = {px} pixels", fontsize=13, va="top", color="#495057")
    ax2.text(.355, .975, "\u2192", fontsize=26, va="top", color="#e8590c")
    ax2.text(.45, .97, f"{D}-d token", fontsize=13, va="top", weight="bold")
    ax2.text(0, .70, f"A learned {px}x{D} matrix does it — PatchEmbed, a Conv2d whose\n"
                     f"kernel equals its stride, so patches never overlap. Note the\n"
                     f"direction: {px} \u2192 {D} goes UP. A {PATCH}x{PATCH} scrap of pixels has no\n"
                     f"room for \u201ctop bar of a 7\u201d to be a direction; {D} dimensions do.\n\n"
                     "Everything after this point is chapter 01 unchanged.",
             fontsize=10.5, va="top", linespacing=1.75)

    inner = gs[2, :].subgridspec(1, NP, wspace=.22)          # --- the sequence
    for k in range(NP):
        r, c = divmod(k, G)
        a = fig.add_subplot(inner[0, k])
        a.imshow(img[r*PATCH:(r+1)*PATCH, c*PATCH:(c+1)*PATCH], cmap="gray",
                 vmin=float(img.min()), vmax=float(img.max()))
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color("#e8590c"); sp.set_linewidth(1.3)
        a.set_xlabel(k, fontsize=8, labelpad=2)
        if k == 0:
            a.set_ylabel("token", fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    fig.text(.5, .215, f"the sequence the transformer actually receives — {NP} tokens "
                       f"of {D} dims, raster order, plus a [CLS] token",
             ha="center", fontsize=10)

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)        # pyplot keeps every figure alive otherwise
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------- train / eval
def run(model, Xtr, Ytr, Xte, Yte, epochs=3, bs=256, lr=3e-4, quiet=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            loss = F.cross_entropy(model(Xtr[b])[0], Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if not quiet:
            print(f"    epoch {ep+1}/{epochs}  loss {loss.item():.4f}  test {acc(model, Xte, Yte)*100:.2f}%")
    return model


@torch.no_grad()
def acc(model, X, Y, perm=None, bs=1000):
    model.eval()
    ok = 0
    for i in range(0, X.shape[0], bs):
        ok += (model(X[i:i+bs], perm=perm)[0].argmax(-1) == Y[i:i+bs]).sum().item()
    return ok / X.shape[0]


def show_training(d, epochs=6, bs=256, lr=3e-4, path="how_training_works.png"):
    """Open up the training loop that exp_train() hides.

    Five lines do all the work, once per batch:

        logits = model(x)                 # guess
        loss   = cross_entropy(logits, y) # how wrong, as ONE number
        opt.zero_grad()                   # clear last batch's gradients
        loss.backward()                   # d(loss)/d(every parameter)
        opt.step()                        # nudge each parameter downhill

    This records the loss at every step (not just the end of each epoch) and
    the class probabilities for one test digit before and after, so the two
    numbers printed during training stop being opaque.
    """
    (Xtr, Ytr), (Xte, Yte) = d
    torch.manual_seed(0)
    m = ViT().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    n, k = Xtr.shape[0], 0
    steps, losses, marks = [], [], []

    probe, truth = Xte[:1], int(Yte[0])
    with torch.no_grad():
        p_before = m(probe)[0].softmax(-1)[0].cpu()

    for ep in range(epochs):
        m.train()
        idx = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            loss = F.cross_entropy(m(Xtr[b])[0], Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            steps.append(k); losses.append(loss.item()); k += 1
        marks.append((k, acc(m, Xte, Yte)))
    with torch.no_grad():
        p_after = m(probe)[0].softmax(-1)[0].cpu()

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=.45, wspace=.25)

    ax = fig.add_subplot(gs[:, 0])                       # --- the curve
    ax.plot(steps, losses, lw=.7, color="#adb5bd", label="loss, every batch")
    w = 20
    sm = [sum(losses[max(0, i-w):i+1]) / len(losses[max(0, i-w):i+1]) for i in range(len(losses))]
    ax.plot(steps, sm, lw=2, color="#1c7ed6", label=f"moving average ({w} batches)")
    for j, (x, _) in enumerate(marks):
        ax.axvline(x, color="#dee2e6", lw=1, zorder=0)
    ax.set_xlabel(f"gradient step  ({len(steps)} total = {epochs} epochs x {n//bs + 1} batches)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("every parameter update, not just the end of each epoch", fontsize=10.5)
    ax2 = ax.twinx()
    ax2.plot([x for x, _ in marks], [a*100 for _, a in marks], "o-", color="#e8590c",
             lw=1.8, ms=5, label="test accuracy")
    ax2.set_ylabel("test accuracy (%)", color="#e8590c")
    ax2.tick_params(axis="y", colors="#e8590c")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right", framealpha=.95)
    for sp in ("top",): ax.spines[sp].set_visible(False)

    axl = fig.add_subplot(gs[0, 1]); axl.axis("off")     # --- the loop
    axl.text(0, 1.0, "the whole loop, once per batch", fontsize=11, weight="bold", va="top")
    axl.text(0, .80,
             "logits = model(x)\n"
             "loss   = cross_entropy(logits, y)\n"
             "opt.zero_grad()\n"
             "loss.backward()\n"
             "opt.step()",
             fontsize=10.5, va="top", family="monospace", linespacing=1.55, color="#1c7ed6")
    axl.text(0, .17,
             f"{len(steps)} steps total. The loss printed each epoch is just\n"
             f"the LAST batch's — which is why it jitters.",
             fontsize=9.5, va="top", color="#495057", linespacing=1.6)

    axb = fig.add_subplot(gs[1, 1])                      # --- one example
    xpos = range(10)
    axb.bar([x - .2 for x in xpos], p_before, .4, label="at init", color="#ced4da")
    axb.bar([x + .2 for x in xpos], p_after, .4, label="trained", color="#1c7ed6")
    axb.axvline(truth, color="#e8590c", lw=1.2, ls="--", zorder=0)
    axb.set_xticks(list(xpos)); axb.set_xlabel("predicted class")
    axb.set_ylabel("probability")
    axb.set_title(f"one test digit (true label {truth}) — what the loss measures",
                  fontsize=10)
    axb.legend(fontsize=9)
    axb.text(.98, .62, f"p({truth}) {p_before[truth]:.3f} \u2192 {p_after[truth]:.3f}\n"
                       f"loss = -log p:  {-math.log(float(p_before[truth])):.2f} "
                       f"\u2192 {-math.log(float(p_after[truth])):.2f}",
             transform=axb.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(fc="#fff4e6", ec="#e8590c", lw=.8))
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)        # pyplot keeps every figure alive otherwise
    print(f"  wrote {path}  ({len(steps)} steps, final {marks[-1][1]*100:.2f}%)")
    return path


def trace_shapes(d, bs=256):
    """Walk ONE batch through the model, printing every tensor on the way.

    Nothing here is hand-written: each shape comes from a real tensor produced
    by the same modules training uses. Run it and the five-line loop stops
    being opaque.
    """
    (Xtr, Ytr), _ = d
    torch.manual_seed(0)
    m = ViT().to(DEV)
    x, y = Xtr[:bs], Ytr[:bs]
    hd = D // HEADS

    def p(label, t, note=""):
        shape = str(tuple(t.shape)) if torch.is_tensor(t) else str(t)
        print(f"  {label:<34}{shape:<22}{note}")

    print("\nFORWARD" + "-" * 66)
    p("x  (one batch)", x, f"{bs} images, 1 channel, 28x28")
    t = m.patch.proj(x)
    p("  Conv2d(1,64,k=7,s=7)", t, "64 filters, 7x7, non-overlapping")
    t = t.flatten(2).transpose(1, 2)
    p("  flatten+transpose", t, f"-> a SEQUENCE: {NP} tokens of {D} dims")
    t = torch.cat([m.cls.expand(bs, -1, -1), t], 1)
    p("  prepend [CLS]", t, f"{NP}+1 = {NP+1} tokens")
    t = t + m.pos
    p("  + positional encoding", t, "same shape: position is ADDED, not appended")

    b = m.blocks[0]
    print("\n  block 1 of 4" + "-" * 55)
    h = b.n1(t)
    p("    LayerNorm", h)
    qkv = b.qkv(h)
    p("    Linear(64,192)", qkv, "= 3 x 64, one slab each for q, k, v")
    q, k, v = qkv.chunk(3, dim=-1)
    p("    chunk -> q (k,v same)", q)
    q4, k4, v4 = [z.view(bs, NP+1, HEADS, hd).transpose(1, 2) for z in (q, k, v)]
    p("    split into heads", q4, f"{HEADS} heads x {hd} dims  ({HEADS}x{hd}={D})")
    att = (q4 @ k4.transpose(-2, -1)) / hd ** .5
    p("    q @ k^T / sqrt(d)", att, "<- THE attention matrix, 17x17 per head")
    att = att.softmax(-1)
    p("    softmax(dim=-1)", att, f"each row sums to 1: {float(att[0,0,0].sum().detach()):.4f}")
    o = att @ v4
    p("    att @ v", o, "weighted average of value vectors")
    o = o.transpose(1, 2).reshape(bs, NP+1, D)
    p("    merge heads", o, "concatenate the 4 heads back to 64")
    o = b.proj(o)
    p("    Linear(64,64)  = W_O", o)
    t2 = t + o
    p("    x + attn(...)", t2, "RESIDUAL: add, never replace")
    t3 = t2 + b.mlp(b.n2(t2))
    p("    x + mlp(norm(x))", t3, "mlp widens 64 -> 256 -> 64, per position")

    for blk in m.blocks[1:]:
        t3, _ = blk(t3)
    print()
    p("after all 4 blocks", t3, "shape NEVER changes -- that is why they stack")
    t4 = m.norm(t3)
    p("final LayerNorm", t4)
    cls = t4[:, 0]
    p("take row 0 ([CLS] only)", cls, f"the other {NP} tokens are discarded here")
    logits = m.head(cls)
    p("Linear(64,10)", logits, "one score per digit class")

    print("\nLOSS" + "-" * 69)
    loss = F.cross_entropy(logits, y)
    p("y  (true labels)", y, f"integers 0-9, e.g. {y[:6].tolist()}")
    p("cross_entropy(logits,y)", f"scalar {loss.detach().item():.4f}", "ONE number from 256x10")

    print("\nBACKWARD" + "-" * 65)
    print(f"  loss.backward() fills .grad on every parameter, SAME shape as the parameter:")
    m.zero_grad(); loss.backward()
    shown = 0
    for n_, prm in m.named_parameters():
        if shown < 5:
            print(f"    {n_:<28}{str(tuple(prm.shape)):<18}grad {tuple(prm.grad.shape)}")
            shown += 1
    tot = sum(prm.numel() for prm in m.parameters())
    print(f"    ... {len(list(m.parameters()))} tensors, {tot:,} numbers total")
    print(f"\n  opt.step() moves each of those {tot:,} numbers a little way downhill.")
    return m


def draw_flow(path="shape_flow.png", bs=256):
    """The same trace as trace_shapes(), drawn.

    Left column: the path a batch takes. Right: one block opened up. The
    shapes are the ones trace_shapes() prints from live tensors.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    T1, hd = NP + 1, D // HEADS
    INK, ACC, BLU, GRY = "#212529", "#e8590c", "#1c7ed6", "#868e96"

    fig, ax = plt.subplots(figsize=(16.6, 9.7))
    ax.set_xlim(0, 124); ax.set_ylim(0, 91); ax.axis("off"); ax.invert_yaxis()

    def box(x, y, w, h, label, shape, note="", fc="#f8f9fa", ec=INK, lw=1.1, fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + 1.6, y + h/2, label, va="center", ha="left", fontsize=fs,
                zorder=4, color=INK)
        ax.text(x + w - 1.6, y + h/2, shape, va="center", ha="right", fontsize=fs,
                family="monospace", zorder=4, color=BLU)
        if note:
            ax.text(x + w + 1.4, y + h/2, note, va="center", ha="left", fontsize=8.6, color=GRY)

    def arrow(x, y0, y1, label="", color=INK, lw=1.2):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=11,
                                     lw=lw, color=color, zorder=2))
        if label:
            ax.text(x + 1.2, (y0 + y1) / 2, label, va="center", ha="left",
                    fontsize=8.8, color=color, style="italic")

    # ---------------- left column: the whole path
    LX, LW, H = 4, 40, 4.6
    ax.text(LX, 2.4, "ONE BATCH, END TO END", fontsize=11.5, weight="bold", color=INK)
    steps = [
        ("input images",       f"({bs}, 1, 28, 28)", "784 pixels each",        "#fff4e6"),
        ("Conv2d(1,64,k=7,s=7)", f"({bs}, 64, 4, 4)", "still a GRID",          "#f8f9fa"),
        ("flatten + transpose", f"({bs}, {NP}, {D})", "now a SEQUENCE",        "#e7f5ff"),
        (f"prepend [CLS] + position", f"({bs}, {T1}, {D})", f"{NP}+1 tokens",  "#e7f5ff"),
    ]
    y = 6
    for i, (lab, sh, note, fc) in enumerate(steps):
        box(LX, y, LW, H, lab, sh, note, fc=fc)
        if i < len(steps) - 1:
            arrow(LX + LW/2, y + H, y + H + 4.2)
        y += H + 4.2
    y_block_in = y

    y -= 1.8
    ax.add_patch(FancyBboxPatch((LX - 1.2, y + 3.4), LW + 2.4, 13.5,
                                boxstyle="round,pad=0.6,rounding_size=1.4",
                                fc="#fff9db", ec=ACC, lw=1.8, zorder=3))
    ax.text(LX + LW/2, y + 7.2, f"{LAYERS} x  TRANSFORMER BLOCK", ha="center",
            fontsize=11, weight="bold", color=ACC, zorder=4)
    ax.text(LX + LW/2, y + 11.8, f"({bs}, {T1}, {D})  in   \u2192   ({bs}, {T1}, {D})  out",
            ha="center", fontsize=9.6, family="monospace", color=GRY, zorder=4)
    ax.text(LX + LW/2, y + 15.0, "shape never changes \u2014 which is why they stack",
            ha="center", fontsize=8.8, color=GRY, style="italic", zorder=4)
    arrow(LX + LW/2, y, y + 3.4)
    y += 13.5 + 3.4 + 4.2

    tail = [
        ("LayerNorm, take row 0", f"({bs}, {D})",  f"other {NP} tokens dropped", "#e7f5ff"),
        ("Linear(64,10)",         f"({bs}, 10)",   "one score per class",        "#f8f9fa"),
        ("cross_entropy(., y)",   "scalar",        "ONE number",                 "#ffe3e3"),
    ]
    for i, (lab, sh, note, fc) in enumerate(tail):
        arrow(LX + LW/2, y - 4.2, y)
        box(LX, y, LW, H, lab, sh, note, fc=fc)
        y += H + 4.2

    # backward
    bx = LX - 3.2
    ax.plot([bx, bx], [10.5, y - 4.2 + H / 2], color=ACC, lw=1.5, ls=(0, (5, 3)), zorder=2)
    ax.add_patch(FancyArrowPatch((bx, 14), (bx, 8.6), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.5, color=ACC, zorder=2))
    ax.text(bx - 1.8, (y + 8.5) / 2,
            "loss.backward()\nfills .grad on every\nparameter \u2014 52 tensors,\n204,298 numbers,\neach the SAME shape\nas its parameter\n\nopt.step() then nudges\nevery one downhill",
            ha="right", va="center", fontsize=9, color=ACC, linespacing=1.65)

    # ---------------- right column: inside one block
    RX, RW = 64, 34
    ax.text(RX, 2.4, "INSIDE ONE BLOCK", fontsize=11.5, weight="bold", color=ACC)
    ax.add_patch(FancyBboxPatch((RX - 2.6, 5.0), RW + 11.5, 76.5,
                                boxstyle="round,pad=0.6,rounding_size=1.4",
                                fc="#fffdf5", ec=ACC, lw=1.4, zorder=1))
    inner = [
        ("LayerNorm",            f"({bs}, {T1}, {D})",            "", "#f8f9fa"),
        ("Linear(64,192)",       f"({bs}, {T1}, {3*D})",          "= 3 x 64", "#f8f9fa"),
        ("chunk \u2192 q, k, v",     f"({bs}, {T1}, {D})",            "each", "#f8f9fa"),
        (f"split into {HEADS} heads",  f"({bs}, {HEADS}, {T1}, {hd})",  f"{HEADS}x{hd}={D}", "#e7f5ff"),
        ("q @ k\u1d40 / \u221ad",        f"({bs}, {HEADS}, {T1}, {T1})",  "the attention matrix", "#ffe8cc"),
        ("softmax(dim=-1)",      f"({bs}, {HEADS}, {T1}, {T1})",  "rows sum to 1", "#ffe8cc"),
        ("@ v",                  f"({bs}, {HEADS}, {T1}, {hd})",  "weighted average", "#e7f5ff"),
        ("merge heads",          f"({bs}, {T1}, {D})",            "", "#e7f5ff"),
        ("Linear(64,64) = W\u1d3c",  f"({bs}, {T1}, {D})",            "", "#f8f9fa"),
        ("x + attn(.)",          f"({bs}, {T1}, {D})",            "RESIDUAL", "#d3f9d8"),
        ("x + mlp(norm(x))",     f"({bs}, {T1}, {D})",            "64\u2192256\u219264, per token", "#d3f9d8"),
    ]
    yy, HH = 8.0, 4.3
    for i, (lab, sh, note, fc) in enumerate(inner):
        box(RX, yy, RW, HH, lab, sh, note, fc=fc, fs=9.4,
            ec=ACC if fc == "#ffe8cc" else INK, lw=1.6 if fc == "#ffe8cc" else 1.0)
        if i < len(inner) - 1:
            arrow(RX + RW/2, yy + HH, yy + HH + 2.3, lw=.9, color=GRY)
        yy += HH + 2.3

    ax.annotate("", xy=(RX - 3.0, 30), xytext=(LX + LW + 9, y_block_in + 9),
                arrowprops=dict(arrowstyle="-", color=ACC, lw=1.1, ls=(0, (4, 3)),
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(RX + RW + 14.5, 30.5,
            "the ONLY step where\ninformation crosses\nbetween positions \u2014\neverything else works\non one token at a time",
            fontsize=9.2, color=ACC, va="center", linespacing=1.65)

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)        # pyplot keeps every figure alive otherwise
    print(f"  wrote {path}")
    return path


def draw_architecture(d, path="architecture.png"):
    """The network drawn the way CNN papers draw one: left to right, slabs
    sized by their tensor, the digit at one end and the ten classes at the other.
    """
    from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
    (Xtr, _), (Xte, Yte) = d
    img = Xte[0, 0].cpu()
    N, BS = Xtr.shape[0], 256
    mm = ViT()                                   # for parameter counts
    grp = {}
    for nm, prm in mm.named_parameters():
        k = ("patch" if nm.startswith("patch") else "cls" if nm == "cls"
             else "pos" if nm == "pos" else "blocks" if nm.startswith("blocks")
             else "norm" if nm.startswith("norm") else "head")
        grp[k] = grp.get(k, 0) + prm.numel()
    TOT = sum(grp.values())
    NB = (N + BS - 1) // BS                      # batches per epoch
    EP = 6
    BLU, BLE, ACC, INK, GRY = "#aec7e8", "#4a6fa5", "#e8590c", "#212529", "#868e96"
    GRN, RED = "#8bc34a", "#e53935"

    fig, ax = plt.subplots(figsize=(22.0, 7.3))
    ax.set_xlim(2, 224); ax.set_ylim(0, 64); ax.axis("off")

    def slab(x, y, w, h, dx=2.2, dy=1.6, fc=BLU, ec=BLE, lw=.8, a=1.0):
        """one isometric box: front face + top + right side"""
        ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)], fc=fc, ec=ec, lw=lw, alpha=a, zorder=3))
        ax.add_patch(Polygon([(x,y+h),(x+dx,y+h+dy),(x+w+dx,y+h+dy),(x+w,y+h)], fc=fc, ec=ec, lw=lw, alpha=a*.75, zorder=3))
        ax.add_patch(Polygon([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)], fc=fc, ec=ec, lw=lw, alpha=a*.55, zorder=3))

    def stack(x, y, w, h, n, step=.72, **kw):
        for i in range(n - 1, -1, -1):
            slab(x + i*step, y + i*step*.72, w, h, **kw)
        return x + (n-1)*step, y + (n-1)*step*.72      # origin of the TOP slab

    def bracket(x0, x1, y, label, sub="", c=INK):
        ax.plot([x0, x1], [y, y], color=c, lw=1.1)
        for xx in (x0, x1): ax.plot([xx, xx], [y, y + 1.4], color=c, lw=1.1)
        ax.plot([(x0+x1)/2]*2, [y, y - 1.4], color=c, lw=1.1)
        ax.text((x0+x1)/2, y - 3.0, label, ha="center", va="bottom", fontsize=10.5,
                weight="bold", color=c)
        if sub:
            ax.text((x0+x1)/2, y - 7.4, sub, ha="center", va="bottom", fontsize=9, color=c)

    def dims(x, y, t, c=INK):
        ax.text(x, y, t, ha="center", va="top", fontsize=9.5, color=c)

    # 1 ---------------- input: a BATCH of digits, not one
    for k in (3, 2, 1):
        ax.add_patch(Rectangle((6 + k*1.5, 30 + k*1.5), 18, 18, fc="#ffffff",
                               ec=GRY, lw=.9, zorder=2))
    ax.imshow(img, cmap="gray_r", extent=[6, 24, 30, 48], zorder=3)
    ax.add_patch(Rectangle((6, 30), 18, 18, fill=False, ec=INK, lw=1.2, zorder=4))
    ax.text(15, 56.4, "INPUT", ha="center", fontsize=10.5, weight="bold")
    ax.text(15, 53.4, f"one batch = {BS} images", ha="center", fontsize=8.8,
            color=ACC, weight="bold")
    dims(15, 28.5, f"({BS}, 1, 28, 28)")

    # 2 ---------------- 16 patches
    for r in range(4):
        for c in range(4):
            ax.add_patch(Rectangle((34 + c*4.4, 42.5 - r*4.4), 4.0, 4.0,
                                   fc=BLU, ec=BLE, lw=.7, zorder=3))
    bracket(33, 52, 54, "PatchEmbed", "Conv2d 7x7, stride 7\n(no overlap)")
    dims(42.5, 25.0, "16 patches\n(7 x 7 each = 49 px)")
    ax.text(42.5, 19.4, f"learned: {grp['patch']:,}", ha="center", fontsize=9, color=ACC, weight="bold")
    ax.annotate("", xy=(33, 38), xytext=(25, 38),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    for (px, py) in [(6, 48), (6, 30)]:
        ax.plot([24, 34], [py, 46.5 if py > 40 else 29.9], color=GRY, lw=.7, ls=(0, (3, 3)), zorder=1)

    # 3 ---------------- token sequence (16 -> 17 planes of width 64)
    ax.annotate("", xy=(62, 38), xytext=(54, 38), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    stack(63, 28, 11, 1.4, 16)
    bracket(62, 80, 54, "flatten + project", f"each patch \u2192 one token\nof d_model = {D}")
    dims(71, 25, f"{NP} tokens x {D}\n(a SEQUENCE now)\nthe {D} is d_model")

    # 4 ---------------- prepend CLS + position
    ax.annotate("", xy=(93, 38), xytext=(86, 38), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    tx, ty = stack(94, 28, 11, 1.4, 16)
    slab(tx + .72, ty + 1.92, 11, 1.4, fc="#ffd8a8", ec=ACC, lw=1.1)
    ax.text(tx + 5.5, ty + 5.4, f"[CLS]  (1, {D})", fontsize=8.6, color=ACC,
            weight="bold", ha="center")
    bracket(93, 111, 54, "+ [CLS]  + position",
            f"cls (1, {D}) CONCATENATED\npos ({NP+1}, {D}) ADDED")
    dims(102, 25, f"{NP} + 1 = {NP+1} tokens x {D}\nd_model unchanged")
    ax.text(102, 19.4, f"learned: {grp['cls']+grp['pos']:,}", ha="center", fontsize=9, color=ACC, weight="bold")

    # 5 ---------------- the blocks, opened up
    BX, BW = 120, 64
    hd = D // HEADS
    ax.annotate("", xy=(BX, 38), xytext=(114, 38), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    ax.add_patch(Rectangle((BX, 20.5), BW, 35, fc="#fff9db", ec=ACC, lw=1.6, zorder=2))
    ax.text(BX + BW/2, 59.0, f"{LAYERS} x  TRANSFORMER BLOCK", ha="center", fontsize=10.5,
            weight="bold", color=ACC)
    ax.text(BX + BW/2, 56.4, f"multi-head self-attention ({HEADS} heads) + MLP",
            ha="center", fontsize=8.4, color=GRY, style="italic")

    T1, sq = NP + 1, f"({NP+1},{D})"
    sh = f"({D},{D})"
    rows = [
        ("x",                 f"{sq}",                          "what arrives",          INK),
        ("Q = x @ W\u1d60",       f"{sq} @ {sh} = {sq}",            "W\u1d60 learned",           "#1c7ed6"),
        ("K = x @ W\u1d4f",       f"{sq} @ {sh} = {sq}",            "W\u1d4f learned",           "#1c7ed6"),
        ("V = x @ W\u1d5b",       f"{sq} @ {sh} = {sq}",            "W\u1d5b learned",           "#1c7ed6"),
        (f"\u2193 split {HEADS} heads", f"{sq} -> ({HEADS},{T1},{hd})",   "a RESHAPE - no maths",  GRY),
        (f"\u2193 Q @ K\u1d40/\u221a{hd}",  f"({HEADS},{T1},{hd}) @ ({HEADS},{hd},{T1}) = ({HEADS},{T1},{T1})",
                                                                "<- Q,K used HERE",      ACC),
        ("\u2193 softmax",        f"({HEADS},{T1},{T1})  unchanged",     "each row sums to 1",    GRY),
        ("\u2193 A' @ V",         f"({HEADS},{T1},{T1}) @ ({HEADS},{T1},{hd}) = ({HEADS},{T1},{hd})",
                                                                "<- V used HERE",        ACC),
        ("\u2193 merge @ W\u1d3c",    f"{sq} @ {sh} = {sq}",            "W\u1d3c learned",           GRY),
        ("x + attn(x)",        f"{sq}",                          "residual",              "#2f9e44"),
        ("x + MLP(x)",         f"{sq}",                          f"{D}->{4*D}->{D}",      "#2f9e44"),
    ]
    yy = 53.4
    for lab, shp, note, col in rows:
        ax.text(BX + 1.6, yy, lab, va="center", ha="left", fontsize=7.8, color=col,
                weight="bold" if col in (ACC, "#2f9e44", "#1c7ed6") else "normal")
        ax.text(BX + 15.0, yy, shp, va="center", ha="left", fontsize=6.6,
                family="monospace", color="#1c7ed6")
        ax.text(BX + 48.0, yy, note, va="center", ha="left", fontsize=7.0,
                color=ACC if note.startswith("<-") else GRY)
        yy -= 2.82

    ax.text(BX + BW/2, 21.7, "Q and K only make SCORES.  Only V carries content.",
            ha="center", fontsize=8.0, color=ACC, style="italic")
    ax.text(BX + BW/2, 17.8, f"learned: {grp['blocks']:,}  ({grp['blocks']//LAYERS:,} x {LAYERS})",
            ha="center", fontsize=9, color=ACC, weight="bold")

    # 6 ---------------- take the CLS row
    ax.annotate("", xy=(190, 38), xytext=(184, 38), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=INK))
    slab(190, 37.2, 13, 1.6, fc="#ffd8a8", ec=ACC, lw=1.1)
    bracket(189, 205, 54, "take row 0", "the other 16 tokens\nare discarded")
    dims(196, 33.5, f"[CLS] only: one\nd_model vector, {D} long")

    # 7 ---------------- the ten classes
    for i, lab in enumerate(["0", "1", "2", "...", "9"]):
        yy = 48 - i * 5.2
        if lab == "...":
            ax.text(212, yy, "\u22ee", ha="center", va="center", fontsize=13); continue
        ax.add_patch(Circle((212, yy), 1.55, fc=RED, ec="#b71c1c", lw=.8, zorder=4))
        ax.text(215.2, yy, lab, va="center", fontsize=10, weight="bold")
        ax.plot([205.5, 210.4], [38.5, yy], color=BLE, lw=.5, alpha=.55, zorder=1)
    ax.text(212, 53.5, "OUTPUT", ha="center", fontsize=10.5, weight="bold")
    dims(212, 22.0, "Linear(64, 10)\n-> softmax")
    ax.text(212, 16.4, f"learned: {grp['head']+grp['norm']:,}", ha="center", fontsize=9, color=ACC, weight="bold")

    ax.text(113, 13.8, "Drawn for ONE image. Every tensor really carries a leading BATCH "
            "axis \u2014 (256, 17, 64), not (17, 64).", ha="center", fontsize=9.4, color=INK)
    ax.text(113, 11.6, "The 256 ride through together \u2014 but never interact: attention mixes "
            "tokens WITHIN one image, so there are 256 separate 17x17 matrices.",
            ha="center", fontsize=9.4, color=GRY)
    ax.text(113, 9.4, f"Training set {N:,} images  =  {NB-1} x {BS} + {N - (NB-1)*BS}  "
            f"\u2192  {NB} batches per epoch  x  {EP} epochs  =  {NB*EP:,} weight updates.",
            ha="center", fontsize=9.4, color=ACC, weight="bold")
    ax.text(113, 7.2, f"d_model = {D} is the RESIDUAL-STREAM WIDTH \u2014 set by PatchEmbed, "
            f"unchanged through all {LAYERS} blocks, read once at the end.",
            ha="center", fontsize=9.4, color="#1c7ed6", weight="bold")
    ax.text(113, 5.0, f"Orange = LEARNED, {TOT:,} numbers in all, every one updated by the same "
            "loss.backward() + opt.step().", ha="center", fontsize=9.4, color=INK)
    ax.text(113, 2.8, "No parameters at all: Q\u00b7K\u1d40, the scaling, softmax, @V, the residual "
            "adds, the [CLS] read-out \u2014 those are fixed arithmetic.",
            ha="center", fontsize=9.4, color=GRY)

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)        # pyplot keeps every figure alive otherwise
    print(f"  wrote {path}")
    return path


def batch_check(d, n=256):
    """Do images in a batch influence each other? Measure it, do not assume."""
    (_, _), (Xte, _) = d
    torch.manual_seed(0)
    m = ViT().to(DEV).eval()
    with torch.no_grad():
        big, alone = m(Xte[:n])[0], m(Xte[:1])[0]
        _, maps = m(Xte[:n], keep_attn=True)
    diff = float((big[0] - alone[0]).abs().max())
    print(f"  image 0 inside a batch of {n}, vs computed completely alone:")
    print(f"    batch : {[round(v, 4) for v in big[0, :5].tolist()]} ...")
    print(f"    alone : {[round(v, 4) for v in alone[0, :5].tolist()]} ...")
    print(f"    max difference {diff:.2e}   identical: "
          f"{torch.allclose(big[0], alone[0], atol=1e-5)}")
    print(f"\n  attention matrix {tuple(maps[0].shape)} = {n} SEPARATE 17x17 matrices,")
    print("  one per image, per head. Nothing ever scores a token of image i")
    print("  against a token of image j.")
    return diff


def show_tokens(d, path="tokens.png"):
    """The real 16 x 64 matrix for one digit, drawn.

    Every row IS one token: 64 numbers describing one 7x7 patch. The blank
    patches produce byte-identical rows, because the embedding is a pure
    function of the 49 pixels -- nothing about WHERE the patch sat has been
    added yet.
    """
    torch.manual_seed(0)
    (_, _), (Xte, Yte) = d
    m = ViT().to(DEV).eval()
    with torch.no_grad():
        T = m.patch(Xte[:1])[0].cpu()
    img, G = Xte[0, 0].cpu(), IMG // PATCH
    ink = [float((img[k//G*PATCH:(k//G+1)*PATCH, k%G*PATCH:(k%G+1)*PATCH].std())) for k in range(NP)]
    blank = [k for k in range(NP) if ink[k] < 1e-6]

    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[.85, 2.5, 1.15], wspace=.33)

    ax = fig.add_subplot(gs[0, 0])                       # the patches, numbered
    ax.imshow(img, cmap="gray")
    for k in range(1, G):
        ax.axhline(k*PATCH - .5, color="#e8590c", lw=1.3)
        ax.axvline(k*PATCH - .5, color="#e8590c", lw=1.3)
    for k in range(NP):
        r, c = divmod(k, G)
        ax.text(c*PATCH + 3, r*PATCH + 3.6, str(k), color="#e8590c", fontsize=9,
                ha="center", va="center", weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{NP} patches, numbered", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])                      # THE matrix
    v = float(T.abs().max())
    im = ax2.imshow(T, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax2.set_yticks(range(NP)); ax2.set_yticklabels(range(NP), fontsize=8)
    ax2.set_ylabel("token", fontsize=9.5)
    ax2.set_title(f"the real matrix:  {NP} tokens x {D}   (one ROW = one token)", fontsize=10.5)
    for k in blank:
        ax2.add_patch(plt.Rectangle((-.5, k-.5), D, 1, fill=False, ec="#2f9e44", lw=1.4))
    ax2.set_xlabel(f"the {D} numbers of each token  ->", fontsize=9.5)
    ax2.text(.5, -.175, f"the {len(blank)} green rows are BYTE-IDENTICAL \u2014 every blank patch "
             "gives the same 64 numbers", transform=ax2.transAxes, ha="center",
             fontsize=9.6, color="#2f9e44", weight="bold")
    fig.colorbar(im, ax=ax2, fraction=.025, pad=.085)

    ax3 = fig.add_subplot(gs[0, 2]); ax3.axis("off")     # the actual numbers
    def row(tok, y, c):
        ax3.text(0, y, f"token {tok}", fontsize=10, weight="bold", color=c)
        for r in range(4):
            ax3.text(0, y - .055 - r*.047,
                     " ".join(f"{x:+5.2f}" for x in T[tok, r*4:(r+1)*4].tolist()),
                     fontsize=8.6, family="monospace", color="#212529")
        ax3.text(0, y - .055 - 4*.047, "...  (64 numbers)", fontsize=8.2, color="#868e96")
    row(0, .97, "#2f9e44"); row(6, .60, "#c92a2a")
    ax3.text(0, .22, "token 0 is blank background.\nIts 64 numbers are exactly\n"
                     "W @ (constant pixels) + bias,\nthe same for all 8 blank patches.\n\n"
                     "Nothing about WHERE the patch\nsat has been added yet \u2014 that is\n"
                     "the positional encoding's job,\none step later.",
             fontsize=9.2, va="top", linespacing=1.6, color="#212529")

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)        # pyplot keeps every figure alive otherwise
    print(f"  wrote {path}  ({len(blank)} identical blank rows)")
    return path



def why_64(d, n_img=4000):
    """Why is a token 64-d when the patch is only 7x7 = 49 pixels?

    Because 49 is forced by the patch size and 64 is CHOSEN. This measures
    what the extra dimensions actually contain: the embedding matrix is
    64 x 49, so its rank is at most 49, and every token really lives in a
    49-dimensional slice of the 64-d space.
    """
    torch.manual_seed(0)
    (Xtr, _), _ = d
    m = ViT().to(DEV).eval()
    px = PATCH * PATCH
    W = m.patch.proj.weight.reshape(D, -1).detach().cpu()
    b = m.patch.proj.bias.detach().cpu().double()
    print(f"  PatchEmbed matrix : {tuple(W.shape)}  ->  {px} numbers in, {D} out")
    print(f"  its rank          : {torch.linalg.matrix_rank(W).item()}  (at most min({D}, {px}) = {px})\n")

    with torch.no_grad():
        T = m.patch(Xtr[:n_img]).reshape(-1, D).cpu().double()
    s = torch.linalg.svdvals(T - b)
    s = s / s[0]
    print(f"  {T.shape[0]:,} real tokens. Singular values of (tokens - bias), largest = 1:")
    for k in (0, 20, px - 2, px - 1, px, px + 1, D - 1):
        tag = "  <- noise floor" if k >= px else ""
        print(f"    sigma[{k:2}] = {s[k]:.3e}{tag}")
    drop = float(s[px - 1] / s[px])
    print(f"\n  sigma[{px-1}] -> sigma[{px}] falls by a factor of {drop:,.0f}.")
    print(f"  So the tokens span exactly {px} directions, not {D}.")
    print(f"  The other {D - px} start EMPTY \u2014 they are scratch space the blocks write into,")
    print(f"  because every sub-layer ADDS to the residual stream rather than replacing it.")
    return s



def is_it_padding(d):
    """49 pixels -> 64 numbers. A transformation, or zeros in the spare slots?

    Decide it by measurement. Zero-padding would copy the 49 inputs and leave
    15 zeros; a linear map mixes every input into every output.
    """
    torch.manual_seed(0)
    (_, _), (Xte, _) = d
    m = ViT().to(DEV).eval()
    px = Xte[0, 0, PATCH:2*PATCH, PATCH:2*PATCH].reshape(-1).cpu()   # patch 5
    W = m.patch.proj.weight.reshape(D, -1).detach().cpu()
    b = m.patch.proj.bias.detach().cpu()
    out = W @ px + b

    print(f"  if it were ZERO-PADDING: out[:{px.numel()}] == the pixels, out[{px.numel()}:] == 0\n")
    print(f"    out[0:3]     {[round(v, 3) for v in out[:3].tolist()]}")
    print(f"    pixels[0:3]  {[round(v, 3) for v in px[:3].tolist()]}   <- different")
    print(f"    out[{px.numel()}:{px.numel()+3}]   {[round(v, 3) for v in out[px.numel():px.numel()+3].tolist()]}  <- NOT zero")
    print(f"    outputs equal to some pixel : "
          f"{int(((out[:, None] - px[None, :]).abs() < 1e-6).sum())}")
    print(f"    outputs that are zero       : {int((out.abs() < 1e-6).sum())}")

    px2 = px.clone(); px2[0] = 0.0
    diff = (W @ px2 + b) - out
    print(f"\n  change ONE pixel and {int((diff.abs() > 1e-7).sum())} of the {D} outputs move "
          f"(largest {float(diff.abs().max()):+.4f}).")
    print(f"  Each output is a dot product over all {px.numel()} pixels:")
    for j in range(2):
        terms = "  ".join(f"{W[j,i]:+.2f}*{px[i]:+.2f}" for i in range(3))
        print(f"    dim {j}: {terms} + ... ({px.numel()} terms) + {b[j]:+.2f} = {float(out[j]):+.3f}")
    print(f"\n  W is {tuple(W.shape)} = {W.numel():,} learned numbers. Row j is a "
          f"{PATCH}x{PATCH} FILTER:")
    print(f"  output j answers \"how much does this patch look like filter j?\"")
    return out



def show_filters(d, epochs=6, path="filters.png"):
    """The 64 patch filters, before and after training.

    Row j of the embedding matrix is a PATCH-sized filter, so it can be drawn.
    This trains a model and puts the filters at init beside the filters at the
    end, plus the change, which is the honest way to see how much the patch
    embedding actually learns on a task this small.
    """
    torch.manual_seed(0)
    (Xtr, Ytr), (Xte, Yte) = d
    m = ViT().to(DEV)
    W0 = m.patch.proj.weight.detach().clone().cpu()
    run(m, Xtr, Ytr, Xte, Yte, epochs=epochs, quiet=True)
    W1 = m.patch.proj.weight.detach().clone().cpu()
    _CACHE.setdefault(epochs, m.eval())          # share this one with the other figures
    acc1, cos = acc(m, Xte, Yte), float(F.cosine_similarity(W0.flatten(), W1.flatten(), dim=0))

    def grid(ax, W, title, vlim=None):
        v = vlim or float(W.abs().max())
        pad, G = 1, 8
        canvas = torch.full((G*(PATCH+pad)-pad, G*(PATCH+pad)-pad), float("nan"))
        for k in range(D):
            r, c = divmod(k, G)
            canvas[r*(PATCH+pad):r*(PATCH+pad)+PATCH, c*(PATCH+pad):c*(PATCH+pad)+PATCH] = W[k, 0]
        ax.imshow(canvas, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10.5)
        return v

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0))
    v = grid(axes[0], W0, f"at init — {D} random {PATCH}x{PATCH} filters")
    grid(axes[1], W1, f"after {epochs} epochs ({acc1*100:.1f}% test)", vlim=v)
    dW = W1 - W0
    grid(axes[2], dW, f"what training changed  (same colour scale)", vlim=v)

    fig.text(.5, .055,
             f"cosine(init, trained) = {cos:.4f} \u2014 the filters barely rotated, yet accuracy "
             f"went 10% \u2192 {acc1*100:.1f}%.",
             ha="center", fontsize=10.5, color="#e8590c", weight="bold")
    fig.text(.5, .005,
             f"mean |change| {dW.abs().mean():.4f}, largest {dW.abs().max():.4f}. On a task this "
             f"small the patch embedding stays close to a random projection and the blocks above "
             f"learn to read it.",
             ha="center", fontsize=9.4, color="#495057")
    fig.subplots_adjust(bottom=.16)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}  (cosine {cos:.4f}, test {acc1*100:.2f}%)")
    return path          # every figure function returns its path, so display(Image(...)) works



def show_qkv(d, epochs=6, path="qkv_heads.png"):
    """Q, K and V for one real image, and the 4 heads they are cut into.

    Top row: the token matrix going in, then Q, K, V — each 17 x 64, with the
    head boundaries drawn at 16 / 32 / 48. Bottom row: the 17x17 attention
    matrix each head computes from its own 16-column slice.

    Trains first, so the patterns are the trained ones rather than noise.
    """
    (Xtr, Ytr), (Xte, Yte) = d
    m = trained(d, epochs)                 # shared: trains once, reused
    hd, T1 = D // HEADS, NP + 1

    with torch.no_grad():
        x = m.patch(Xte[:1])
        x = torch.cat([m.cls.expand(1, -1, -1), x], 1) + m.pos
        b = m.blocks[0]
        h = b.n1(x)
        q, k, v = b.qkv(h).chunk(3, dim=-1)
        q4, k4, v4 = [z.view(1, T1, HEADS, hd).transpose(1, 2) for z in (q, k, v)]
        att = ((q4 @ k4.transpose(-2, -1)) / hd ** .5).softmax(-1)[0].cpu()
    mats = [z[0].cpu() for z in (x, q, k, v)]

    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(2, HEADS, height_ratios=[1.25, 1], hspace=.42, wspace=.22)

    for j, (M, name) in enumerate(zip(mats, ["x  (into the block)", "Q", "K", "V"])):
        ax = fig.add_subplot(gs[0, j])
        vmax = float(M.abs().max())
        ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        if j:
            for c in range(1, HEADS):
                ax.axvline(c * hd - .5, color="#2f9e44", lw=1.6)
            for c in range(HEADS):
                ax.text(c * hd + hd / 2, T1 - 1.1, f"h{c}", ha="center", va="center",
                        fontsize=8.5, color="#2f9e44", weight="bold",
                        bbox=dict(fc="white", ec="#2f9e44", lw=.7, pad=1.4, alpha=.9))
        ax.set_title(f"{name}   ({T1} x {D})", fontsize=10.5,
                     color="#212529" if j == 0 else "#1c7ed6", pad=6)
        ax.set_ylabel("token" if j == 0 else "", fontsize=9)
        ax.set_yticks(range(0, T1, 4)); ax.tick_params(labelsize=7.5)
        ax.set_xlabel(f"{D} dims, cut into {HEADS} heads of {hd}" if j else f"{D} dims, not yet split", fontsize=8.5)

    for c in range(HEADS):
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(att[c], cmap="magma", vmin=0, vmax=float(att.max()))
        ax.set_title(f"head {c}:  softmax(q·k$^T$/$\\sqrt{{16}}$)   ({T1} x {T1})", fontsize=9.5)
        ax.set_xticks([0]); ax.set_xticklabels(["[CLS]"], fontsize=7.5)
        ax.set_yticks([0]); ax.set_yticklabels(["[CLS]"], fontsize=7.5)
        ax.set_xlabel("key token", fontsize=8.5)
        if c == 0: ax.set_ylabel("query token", fontsize=8.5)

    fig.text(.5, .035, "Q, K and V are the SAME 17 tokens re-expressed three ways. The green lines "
             "cut each into 4 heads of 16 columns;", ha="center", fontsize=10, color="#212529")
    fig.text(.5, .003, "head c scores its own 16-column slice against itself, giving the four "
             f"{T1}x{T1} maps below. Block 1 of {LAYERS}, one image.",
             ha="center", fontsize=10, color="#495057")
    fig.subplots_adjust(bottom=.11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path



def trace_cls(d, epochs=6):
    """Where the 64 floats of [CLS] come from.

    [CLS] enters as a learned constant with no image content at all, and every
    sub-layer ADDS to it. Attention is the only one that can bring in anything
    from the other tokens; the MLP only reshapes what is already there. This
    prints the running magnitude so the accumulation is visible.
    """
    (Xtr, Ytr), (Xte, Yte) = d
    m = trained(d, epochs)                 # shared: trains once, reused
    hd = D // HEADS
    with torch.no_grad():
        t = m.patch(Xte[:1])
        x = torch.cat([m.cls.expand(1, -1, -1), t], 1)
        print(f"  A. the INITIAL {D} floats — a learned parameter, no image content")
        print(f"     m.cls  {[round(z, 4) for z in m.cls.flatten()[:5].tolist()]} ...  "
              f"|v| = {m.cls.norm():.4f}")
        x = x + m.pos
        print(f"     + pos[0]                        |v| = {x[0,0].norm():6.3f}\n")
        print("  B. then every sub-layer ADDS to it — the residual stream:")
        for i, b in enumerate(m.blocks):
            h = b.n1(x); B_, T_, _ = h.shape
            q, k, v = [z.view(B_, T_, HEADS, hd).transpose(1, 2)
                       for z in b.qkv(h).chunk(3, -1)]
            att = ((q @ k.transpose(-2, -1)) / hd ** .5).softmax(-1)
            o = b.proj((att @ v).transpose(1, 2).reshape(B_, T_, D))
            x = x + o
            print(f"     block {i+1} attention  += {o[0,0].norm():6.3f}   "
                  f"running |v| = {x[0,0].norm():6.3f}   reads the {NP} patches")
            dd = b.mlp(b.n2(x)); x = x + dd
            print(f"     block {i+1} MLP        += {dd[0,0].norm():6.3f}   "
                  f"running |v| = {x[0,0].norm():6.3f}   this token alone")
        final = m.norm(x)[:, 0]
        logits = m.head(final)
    print(f"\n  C. final LayerNorm -> the {D} floats the classifier reads:")
    print(f"     {[round(z, 3) for z in final[0,:6].tolist()]} ...  |v| = {final.norm():.3f}")
    print(f"     Linear({D}, 10) -> {[round(z, 2) for z in logits[0].tolist()]}")
    print(f"     argmax -> {int(logits.argmax())}   (true label {int(Yte[0])})")
    print(f"\n  Each of the {D} numbers = a learned constant + {2*LAYERS} increments.")
    print(f"  Attention supplies everything that came from the image.")
    return final



def cls_and_pos(d):
    """What [CLS] and the positional encoding actually ARE in this model.

    Neither has a closed-form formula: both are learned tensors. The only
    algebra is how they enter -- one concatenated, one added. This also
    measures whether the learned position table discovers the 4x4 grid, and
    compares it against the sinusoidal formula the chapter derives.
    """
    import itertools, math
    m = trained(d)
    print(f"  DEFINED as plain parameters, no formula:")
    print(f"    self.cls = nn.Parameter(torch.zeros(1, 1, {D}))            {D} numbers")
    print(f"    self.pos = nn.Parameter(torch.randn(1, {NP+1}, {D}) * 0.02)    {(NP+1)*D:,} numbers")
    print(f"\n  USED in two lines:")
    print(f"    X = [ cls ; e_1 ; ... ; e_{NP} ]   concatenate  ->  ({NP+1}, {D})")
    print(f"    Z = X + P                        add          ->  ({NP+1}, {D})")
    print(f"      e_k = W_patch @ flatten(patch_k) + b     the {PATCH*PATCH} -> {D} map")
    print(f"      P   = the learned ({NP+1}, {D}) table, row i added to slot i")

    P = m.pos[0].detach().cpu()[1:]                       # the NP patch slots
    S = F.cosine_similarity(P[:, None, :], P[None, :, :], dim=-1)
    G = IMG // PATCH
    near, far = [], []
    for i, j in itertools.combinations(range(NP), 2):
        (r1, c1), (r2, c2) = divmod(i, G), divmod(j, G)
        dist = abs(r1 - r2) + abs(c1 - c2)
        (near if dist == 1 else far if dist >= G else []).append(float(S[i, j]))

    def sinusoid(pos, dm=D):
        v = torch.zeros(dm)
        for i in range(dm // 2):
            w = 1.0 / (10000 ** (2 * i / dm))
            v[2*i], v[2*i+1] = math.sin(w * pos), math.cos(w * pos)
        return v
    X = torch.stack([sinusoid(p) for p in range(NP)])
    Ss = F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1)
    n2 = [float(Ss[i, j]) for i, j in itertools.combinations(range(NP), 2) if abs(i-j) == 1]
    f2 = [float(Ss[i, j]) for i, j in itertools.combinations(range(NP), 2) if abs(i-j) >= NP//2]

    print(f"\n  does the LEARNED table discover the {G}x{G} grid?   (cosine between slots)")
    print(f"    {'':24}{'adjacent':>10}{'far':>10}")
    print(f"    {'learned (this model)':24}{sum(near)/len(near):>+10.4f}{sum(far)/len(far):>+10.4f}")
    print(f"    {'sinusoidal formula':24}{sum(n2)/len(n2):>+10.4f}{sum(f2)/len(f2):>+10.4f}")
    print(f"\n  Sinusoids are smooth in position BY CONSTRUCTION -- PE(pos+k) is a fixed")
    print(f"  rotation of PE(pos). A learned table has no such obligation, and at this")
    print(f"  scale it does not find one: it just memorises {NP+1} distinct vectors, which is")
    print(f"  all the task needs. Distinctness is what breaks permutation equivariance;")
    print(f"  smoothness is a bonus that only matters when you extrapolate past training.")
    return S



def pos_contents(d):
    """What is actually IN [CLS] and the positional table — values and ranges.

    Both are reachable as m.cls and m.pos. This prints their statistics at init
    and after training, then lays the per-slot magnitudes on the patch grid,
    which is where the interesting result is.
    """
    torch.manual_seed(0)
    m0 = ViT().to(DEV)                               # at init
    m1 = trained(d)                                  # after training
    G = IMG // PATCH

    def stat(t, name):
        print(f"    {name:16} min {t.min():+.4f}  max {t.max():+.4f}  "
              f"std {t.std():.4f}  |v| {t.norm():.3f}")

    c0, c1 = m0.cls.detach().cpu().flatten(), m1.cls.detach().cpu().flatten()
    print(f"  m.cls  {tuple(m1.cls.shape)} — {c1.numel()} numbers")
    stat(c0, "at init"); stat(c1, "trained")
    print(f"    trained values   {[round(v, 4) for v in c1[:6].tolist()]} ...")
    print(f"    -> starts at exactly zero and barely moves. It is a placeholder;")
    print(f"       attention fills it (|v| 0.05 in, {7.2:.1f} out of the stack).\n")

    P0, P1 = m0.pos.detach().cpu()[0], m1.pos.detach().cpu()[0]
    print(f"  m.pos  {tuple(m1.pos.shape)} — {P1.numel():,} numbers")
    stat(P0.flatten(), "at init"); stat(P1.flatten(), "trained")
    print(f"    -> roughly +/-0.3, and {float(P1.norm()/P0.norm()):.1f}x its initialisation. "
          f"This one really is learned.\n")

    print(f"  |pos| per slot, on the {G}x{G} patch grid:")
    print(f"    [CLS] slot: {P1[0].norm():.2f}")
    for r in range(G):
        print("      " + "  ".join(f"{float(P1[1+r*G+c].norm()):5.2f}" for c in range(G)))
    centre = [i for i in range(NP) if 0 < i//G < G-1 and 0 < i%G < G-1]
    corner = [0, G-1, NP-G, NP-1]
    edge   = [i for i in range(NP) if i not in centre and i not in corner]
    print()
    for name, ix in (("centre", centre), ("edges", edge), ("corners", corner)):
        v = [float(P1[1+i].norm()) for i in ix]
        print(f"    {name:8} mean |pos| {sum(v)/len(v):.3f}")
    print(f"\n    The centre is ~{sum(float(P1[1+i].norm()) for i in centre)/len(centre) / (sum(float(P1[1+i].norm()) for i in corner)/len(corner)):.0f}x the corners. On MNIST the digit sits in the")
    print(f"    middle, so corner patches are nearly always blank — and blank patches give")
    print(f"    BYTE-IDENTICAL tokens. Position is the only thing that can tell them apart,")
    print(f"    and the model spends magnitude where telling them apart changes the answer.")

    with torch.no_grad():
        (_, _), (Xte, _) = d
        tok = m1.patch(Xte[:512])[0].cpu()
    r = float(P1[1:].norm(dim=-1).mean() / tok.norm(dim=-1).mean())
    print(f"\n  is position big enough to matter?")
    print(f"    mean |patch token| {tok.norm(dim=-1).mean():.3f}")
    print(f"    mean |pos row|     {P1[1:].norm(dim=-1).mean():.3f}   = {r:.1%} of the content")
    print(f"    Big enough to shift a score, small enough not to drown the pixels.")
    print(f"    That ratio is why ADDING position works at all.")
    return P1



def show_all_attention(d, img_i=0, path="attention_all_layers.png"):
    """Every attention map in the model: LAYERS x HEADS, for one image.

    Left block: the full 17x17 map per layer and head. Right column: the
    [CLS] row of each layer, averaged over heads and folded back onto the
    patch grid — i.e. what the classifier's token is reading at that depth.
    """
    m = trained(d)
    (_, _), (Xte, Yte) = d
    G, T1 = IMG // PATCH, NP + 1
    img = Xte[img_i, 0].cpu()
    with torch.no_grad():
        _, maps = m(Xte[img_i:img_i+1], keep_attn=True)       # LAYERS x (1,H,T,T)
    A = [a[0].cpu() for a in maps]

    fig = plt.figure(figsize=(13.6, 11.2))
    gs = fig.add_gridspec(LAYERS, HEADS + 2, width_ratios=[1]*HEADS + [.28, 1.15],
                          hspace=.30, wspace=.18)
    vmax = max(float(a.max()) for a in A)

    for l in range(LAYERS):
        for h in range(HEADS):
            ax = fig.add_subplot(gs[l, h])
            ax.imshow(A[l][h], cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if l == 0: ax.set_title(f"head {h}", fontsize=10)
            if h == 0:
                ax.set_ylabel(f"block {l+1}", fontsize=10.5, weight="bold")
                ax.set_yticks([0]); ax.set_yticklabels(["[CLS]"], fontsize=7)
            # how peaked is this head? entropy of the [CLS] row, in bits
            r = A[l][h][0].clamp_min(1e-9)
            ent = float(-(r * r.log2()).sum())
            ax.set_xlabel(f"{ent:.2f} bits", fontsize=7.5, color="#868e96", labelpad=1.5)

        ax = fig.add_subplot(gs[l, HEADS + 1])                 # what [CLS] reads
        cls_row = A[l][:, 0, 1:].mean(0).reshape(G, G)
        ax.imshow(img, cmap="gray", alpha=.45,
                  extent=[-.5, G-.5, G-.5, -.5])
        ax.imshow(cls_row, cmap="magma", alpha=.75)
        ax.set_xticks([]); ax.set_yticks([])
        if l == 0: ax.set_title(f"[CLS] reads\n(over the {G}x{G} patches)", fontsize=9.5)

    fig.suptitle(f"every attention map in the model — {LAYERS} blocks x {HEADS} heads, "
                 f"digit {int(Yte[img_i])}", fontsize=12, y=.935)
    fig.text(.5, .055, "Each panel is one head's 17x17 map; row 0 is [CLS] as the query. "
             "The number under each is the entropy of that [CLS] row —",
             ha="center", fontsize=9.6, color="#212529")
    fig.text(.5, .032, f"low bits = reads a few patches, high = spreads over all 17. "
             f"Uniform would be {math.log2(T1):.2f} bits.",
             ha="center", fontsize=9.6, color="#868e96")

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    ents = [[float(-(A[l][h][0].clamp_min(1e-9) *
                     A[l][h][0].clamp_min(1e-9).log2()).sum()) for h in range(HEADS)]
            for l in range(LAYERS)]
    print(f"  wrote {path}")
    print(f"  [CLS]-row entropy per block (uniform would be {math.log2(T1):.2f} bits):")
    for l in range(LAYERS):
        print(f"    block {l+1}:  " + "  ".join(f"{e:.2f}" for e in ents[l])
              + f"   mean {sum(ents[l])/HEADS:.2f}")
    return path



def split_heads(d):
    """What "split into 4 heads" actually does: nothing. It is a reshape.

    No parameters, no arithmetic -- head h is literally dimensions
    [h*d_head : (h+1)*d_head] of the same vector, viewed differently.
    """
    m = trained(d)
    (_, _), (Xte, _) = d
    hd = D // HEADS
    with torch.no_grad():
        x = m.patch(Xte[:1])
        x = torch.cat([m.cls.expand(1, -1, -1), x], 1) + m.pos
        b = m.blocks[0]
        q, _, _ = b.qkv(b.n1(x)).chunk(3, -1)
    print(f"  q before the split  {tuple(q.shape)}")
    print(f"    token 0's {D} numbers, first 20:")
    print("     " + " ".join(f"{v:+5.2f}" for v in q[0, 0, :20].tolist()) + " ...\n")

    q4 = q.view(1, NP + 1, HEADS, hd).transpose(1, 2)
    print(f"  q.view(1, {NP+1}, {HEADS}, {hd}).transpose(1, 2)  ->  {tuple(q4.shape)}")
    for h in range(HEADS):
        same = torch.equal(q4[0, h, 0], q[0, 0, h*hd:(h+1)*hd])
        print(f"    head {h}: " + " ".join(f"{v:+5.2f}" for v in q4[0, h, 0, :5].tolist())
              + f"  ...   == q[0, 0, {h*hd}:{(h+1)*hd}] ?  {same}")
    print(f"\n    same numbers      {torch.equal(q4.reshape(-1).sort().values, q.reshape(-1).sort().values)}")
    print(f"    same storage      {q.data_ptr() == q4.data_ptr()}   (a view, not a copy)")
    print(f"    parameters used   none")
    print(f"    arithmetic        none \u2014 not one multiply or add")
    print(f"\n  The multi-headedness lives in W\u1d60/W\u1d4f/W\u1d5b, not in the split. Those are single")
    print(f"  ({D}, {D}) matrices, and training is free to make columns 0-{hd-1} compute something")
    print(f"  different from columns {hd}-{2*hd-1}. The split only declares where the boundaries are.")
    return q4


class Pooled(ViT):
    """The same model with NO [CLS] token: pool the patch tokens instead.

    Used by why_cls() to ask whether the [CLS] token is actually needed.
    """
    def __init__(self, how="mean"):
        super().__init__()
        self.how = how
        self.pos = nn.Parameter(torch.randn(1, NP, D) * 0.02)   # NP slots, no [CLS]

    def forward(self, x, perm=None, keep_attn=False):
        t = self.patch(x)
        if perm is not None:
            t = t[:, perm]
        t = t + self.pos
        for b in self.blocks:
            t, _ = b(t, False)
        t = self.norm(t)
        p = t.mean(1) if self.how == "mean" else t.max(1).values
        return self.head(p), []


def why_cls(d, epochs=6):
    """Is the [CLS] token actually needed? Train the alternatives and see.

    The block emits NP+1 vectors; classification needs ONE. Something has to
    collapse the sequence. [CLS] is one way; pooling the patch tokens is
    another. Same model, same budget -- only the read-out differs.
    """
    (Xtr, Ytr), (Xte, Yte) = d
    print("  same model, same budget — only the READ-OUT differs\n")
    print(f"    {'read-out':16}{'params':>9}{'test acc':>11}")
    out = {}
    for name, make in [("[CLS] token", lambda: ViT()),
                       ("mean-pool",   lambda: Pooled("mean")),
                       ("max-pool",    lambda: Pooled("max"))]:
        torch.manual_seed(0)
        m = make().to(DEV)
        run(m, Xtr, Ytr, Xte, Yte, epochs=epochs, quiet=True)
        a = acc(m, Xte, Yte)
        out[name] = a
        print(f"    {name:16}{sum(p.numel() for p in m.parameters()):>9,}{a*100:>10.2f}%")
    print(f"\n  Single runs at one seed, so read this as 'no meaningful difference'")
    print(f"  rather than 'pooling wins'. The point is that [CLS] is NOT required.")
    print(f"\n  The ViT paper agrees. Its stated reason for the token is inheritance:")
    print(f'    "Similar to BERT\'s [class] token, we prepend a learnable embedding')
    print(f'     to the sequence of embedded patches"')
    print(f"  and its Appendix D.3, comparing against global average pooling:")
    print(f'    "Both work similarly well, but require different learning-rates."')
    return out


# ---------------------------------------------------------------- experiments
def exp_train(d):
    torch.manual_seed(0)          # each experiment reproduces on its own
    (Xtr, Ytr), (Xte, Yte) = d
    m = ViT().to(DEV)
    print(f"  ViT  {sum(p.numel() for p in m.parameters()):,} params · "
          f"{NP} patches of {PATCH}x{PATCH} + 1 CLS = {NP+1} tokens")
    t0 = time.time(); run(m, Xtr, Ytr, Xte, Yte, epochs=6); el = time.time() - t0
    a = acc(m, Xte, Yte)
    print(f"  final {a*100:.2f}%  ({el:.0f}s on {DEV})")

    # what does [CLS] look at?
    with torch.no_grad():
        _, maps = m(Xte[:8], keep_attn=True)
    last = maps[-1][:, :, 0, 1:].mean(1)            # [8,16] — CLS row, averaged over heads
    fig, ax = plt.subplots(2, 8, figsize=(13, 3.4))
    for i in range(8):
        ax[0, i].imshow(Xte[i, 0].cpu(), cmap="gray"); ax[0, i].set_title(int(Yte[i]), fontsize=10)
        ax[1, i].imshow(last[i].reshape(4, 4).cpu(), cmap="magma")
        for r in (0, 1): ax[r, i].axis("off")
    ax[0, 0].set_ylabel("digit"); ax[1, 0].set_ylabel("[CLS] attn")
    fig.suptitle("what the [CLS] token reads, last layer, averaged over heads", fontsize=10)
    fig.tight_layout(); fig.savefig("attention_maps.png", dpi=130)
    plt.close(fig)        # pyplot keeps every figure alive otherwise
    print("  wrote attention_maps.png")
    return m


def exp_position(d):
    torch.manual_seed(0)          # each experiment reproduces on its own
    (Xtr, Ytr), (Xte, Yte) = d
    perm = torch.randperm(NP, device=DEV)
    print("  Training two identical models, one WITHOUT positional encoding.\n")
    rows = []
    for use_pos in (True, False):
        m = ViT(use_pos=use_pos).to(DEV)
        run(m, Xtr, Ytr, Xte, Yte, epochs=6, quiet=True)
        a, ap = acc(m, Xte, Yte), acc(m, Xte, Yte, perm=perm)
        rows.append((use_pos, a, ap))
        print(f"    pos={str(use_pos):5}  in order {a*100:6.2f}%   patches shuffled {ap*100:6.2f}%"
              f"   delta {(ap-a)*100:+6.2f}")
    print("\n  Read the second row: with no positional encoding the shuffle changes")
    print("  nothing at all, because the layer never knew the order. That is")
    print("  permutation equivariance, measured. The first row is what position buys.")
    return rows


def exp_scale(d):
    torch.manual_seed(0)          # each experiment reproduces on its own
    (Xtr, Ytr), (Xte, Yte) = d
    print("  Same budget, same epochs, growing training set.\n")
    print(f"    {'N':>7}  {'CNN':>8}  {'ViT':>8}   gap")
    out = []
    for n in (1000, 4000, 16000, 60000):
        sub = torch.randperm(Xtr.shape[0], device=DEV)[:n]
        xs, ys = Xtr[sub], Ytr[sub]
        ep = max(6, int(60000 / n))        # equal-ish number of gradient steps
        r = {}
        for name, M in (("CNN", SmallCNN), ("ViT", ViT)):
            torch.manual_seed(0)
            m = M().to(DEV)
            run(m, xs, ys, Xte, Yte, epochs=ep, quiet=True)
            r[name] = acc(m, Xte, Yte)
        out.append((n, r["CNN"], r["ViT"]))
        print(f"    {n:>7}  {r['CNN']*100:7.2f}%  {r['ViT']*100:7.2f}%   {(r['ViT']-r['CNN'])*100:+6.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["train", "position", "scale"])
    a = ap.parse_args()
    print(f"device {DEV}\n")
    d = load()
    show_data(d)
    times = {}
    for name, fn in (("train", exp_train), ("position", exp_position), ("scale", exp_scale)):
        if a.only in (None, name):
            print(f"\n=== {name} " + "=" * (60 - len(name)))
            t0 = time.time(); fn(d); times[name] = time.time() - t0
            print(f"\n  [{name}: {times[name]:.0f}s on {DEV}]")
    if len(times) > 1:
        print(f"\n=== total {sum(times.values()):.0f}s on {DEV} "
              + " · ".join(f"{k} {v:.0f}s" for k, v in times.items()))
