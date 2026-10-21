# p2 — a transformer that reads handwritten digits

An image is not a sequence. A Vision Transformer makes it one: cut the 28×28 image into 7×7
patches, project each to a vector, and call that the sequence — 16 tokens, plus a `[CLS]` token
whose final vector the classifier reads. Everything after that is chapter 01 unchanged.

```bash
python vit_mnist.py              # the three experiments, ~60 s on an M-series Mac
python vit_mnist.py --only scale # just one
```

No torchvision — MNIST arrives as one 11 MB `.npz`, so the file runs anywhere torch does.

The notebook (`vit_mnist.ipynb`) is the guided version, in four parts: **1 · what goes in**
(pixels → patches → the real 16×64 matrix → why 64), **2 · what the model is** (architecture,
batching, shapes, Q/K/V and the 4 heads, `[CLS]` and position), **3 · how it learns** (the loop,
the loss curve, whether the patch filters move), and **4 · the three measurements** below.
Figures that need a trained model share one, so the whole notebook runs in ~2 minutes.

The notebook is **committed with its outputs already in it**, so you can read every result
without running anything. Each explanation renders inside a **click-to-expand block** — a real
`<details>` widget, so it works the same in Jupyter, Colab and on GitHub. Parts 1-3 arrive
collapsed — run all, then expand only what you want to look
at. The three measurements in part 4 stay open, since they are the point. Click the margin left
of an output to expand it (Jupyter), or use the arrow (Colab/Lab).

`rebuild.sh` re-runs the notebook and checks the result (images embedded, no error outputs, no
empty cells, the fold blocks present). Run it after any edit to `vit_mnist.py` or to a cell —
otherwise the committed outputs go stale.

The architecture figure opens the transformer block rather than drawing it as a box: every line
of the flow with its shape, and orange arrows on the two steps that consume Q, K and V.

```
x                  (17, 64)        what arrives
Wq Wk Wv           3 x (64, 64)    learned
Q  K  V            each (17, 64)   = 4 heads x (17, 16)
Q.K^T / sqrt(16)   A (4, 17, 17)   <- Q and K used HERE
softmax            A' same shape
A' @ V             (4, 17, 16)     <- V used HERE
merge + W^O        (17, 64)
x + attn(x)                        residual
x + MLP(x)                         64->256->64, per token
```

Each of Q, K, V now carries its own equation — `Q = x @ Wq` with `Wq` a learned `(64, 64)` —
rather than being lumped on one line.

`split_heads()` answers what "split into 4 heads" does: **nothing**. It is a reshape. Head *h* is
literally dimensions `[16h : 16h+16]` of the same vector — same numbers, same storage, no
parameters, not one multiply. The multi-headedness lives in `Wq`/`Wk`/`Wv`, which training is
free to make compute something different per column block; the split only declares the
boundaries.

Q and K are consumed at `Q.K^T` and never appear again; V is consumed at `A' @ V`. Hence the line
at the foot of the block: **Q and K only make scores, only V carries content** — which is also why
`d_v` need not equal `d_k`.

`show_all_attention()` draws every map in the model — 4 blocks x 4 heads — plus a column folding
each block's `[CLS]` row back onto the 4x4 patch grid over the digit, so you can see what the
classifier's token reads at each depth. By block 2 it has found the stroke. The bright vertical
stripes in blocks 2-3 are tokens that *every* query reads from. Entropy of the `[CLS]` row
(uniform = 4.09 bits) goes 3.69 → 3.07 → 2.97 → 3.83: attention sharpens in the middle and
spreads again at the end.

**`[CLS]` and the positional encoding have no closed-form formula** — both are plain learned
tensors, entering as one concatenation and one addition:

```python
self.cls = nn.Parameter(torch.zeros(1, 1,  64))          #    64 numbers
self.pos = nn.Parameter(torch.randn(1, 17, 64) * 0.02)   # 1,088 numbers

X = torch.cat([cls, patch_tokens], 1)   # (1,64) onto (16,64) -> (17,64)  CONCATENATED
Z = X + pos                             # (17,64) + (17,64)     -> (17,64)  ADDED
```

So this model does **not** use the sinusoidal `PE(pos,2i) = sin(pos/10000^(2i/d))` the chapter
derives — ViT and BERT both use a learned table, and so does this. `cls_and_pos()` measures the
difference rather than asserting it, as cosine between position slots:

| | adjacent | far |
|---|---:|---:|
| learned (this model) | −0.0695 | −0.0003 |
| sinusoidal formula | +0.9662 | +0.6685 |

`pos_contents()` then shows what is actually in them. `[CLS]` starts at exactly zero and barely
moves (±0.02, |v| = 0.05) — a placeholder attention fills. Position really is learned (±0.3, 3.4x
its init), and its per-slot magnitudes are **not uniform**: centre 2x2 mean 0.879, edges 0.408,
corners 0.226. The centre is ~4x the corners, because on MNIST the digit sits in the middle and
corner patches are nearly always blank — and blank patches give byte-identical tokens, so
position is the only thing that can distinguish them. The model spends magnitude where
distinguishing changes the answer.

That also corrects the cosine measurement above: the table finds no structure in **direction**,
but it does in **magnitude**. Finally, position is 15% of the content it is added to
(|pos| 0.48 vs |token| 3.12) — enough to shift a score, not enough to drown the pixels, which is
why adding works.

A sinusoid is smooth in position **by construction** (`PE(pos+k)` is a fixed rotation of
`PE(pos)`); a learned table is not obliged to be, and at this scale finds no such structure.
That is a property of 16 slots and 60,000 images, not a law — large ViTs do develop 2-D structure
in theirs. It also clarifies what experiment 2 depends on: that the 17 rows are **distinct**, not
that they are smooth.

## The point: three chapter claims, measured

### 1 · `train` — it works, and the attention maps are readable

```
ViT  204,298 params · 16 patches of 7x7 + 1 CLS = 17 tokens
final 95.24%  (11s on mps)
```

`attention_maps.png` shows which patches `[CLS]` reads. They are not uniform — on a `1`
the mass runs up the stroke, on a `0` it sits around the ring. **Attention weights are
one of the few interpretable things in a neural network**, because they are a
distribution over *inputs* rather than over hidden features.

### 2 · `position` — permutation equivariance, to the digit

Two identical models, one trained with the positional encoding switched off. Then shuffle
the patch order at test time:

| | in order | patches shuffled | delta |
|---|---|---|---|
| **with** position | 95.51 % | 84.57 % | −10.94 |
| **without** position | 92.28 % | 92.28 % | **+0.00** |

That exact zero is the chapter's claim made concrete: *attention cannot see order*. The
model with no positional encoding is not merely robust to the shuffle — it is
**mathematically incapable of noticing it**, because the only thing it ever received was
a set. The 3.2-point gap in the first column is what position buys you.

### 3 · `scale` — a prior is worth most when data is scarce

Same parameter budget, same number of gradient steps, growing training set:

| N | CNN | ViT | gap |
|---:|---:|---:|---:|
| 1 000 | 92.57 % | 73.20 % | −19.37 |
| 4 000 | 94.42 % | 82.61 % | −11.81 |
| 16 000 | 96.45 % | 87.64 % | −8.81 |
| 60 000 | 98.35 % | 95.47 % | **−2.88** |

**Read the gap column, not the accuracy columns.** The CNN's hard-coded locality is free
knowledge handed over before training starts, and it is worth **19 points at N = 1 000 and
2.9 points at N = 60 000**. The transformer has to *learn* that neighbouring patches are
related, and the more data it gets the less that costs it.

> **No crossover here, and that is not a failed experiment.** The ViT never overtakes the
> CNN on MNIST, because all of MNIST is inside the small-data regime — the crossover in
> the ViT paper happens somewhere past 100 M images. What a laptop can show you is the
> *slope*: the prior paying off less and less. That slope is the whole argument.

## Things worth trying

- Change `PATCH` to 4 (49 tokens) or 14 (4 tokens). Where does accuracy break, and is the
  cost compute or information?
- Set `LAYERS = 1`. How much does one round of mixing lose?
- In `exp_position`, shuffle the **pixels** inside each patch instead of the patches. Why
  does that hurt the no-position model, when shuffling patches did not?
- The `PatchEmbed` layer is a `Conv2d` with `kernel_size == stride`. Make them differ, so
  patches overlap — you have just reintroduced a locality prior. Does the N = 1 000 number
  move toward the CNN's?
