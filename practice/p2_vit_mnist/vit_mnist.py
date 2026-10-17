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
import argparse, os, time, urllib.request, numpy as np
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
    return pack(d["x_train"], d["y_train"]), pack(d["x_test"], d["y_test"])


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
    for name, fn in (("train", exp_train), ("position", exp_position), ("scale", exp_scale)):
        if a.only in (None, name):
            print(f"\n=== {name} " + "=" * (60 - len(name)))
            fn(d)
