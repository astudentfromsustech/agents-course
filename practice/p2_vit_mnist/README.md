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

Q and K are consumed at `Q.K^T` and never appear again; V is consumed at `A' @ V`. Hence the line
at the foot of the block: **Q and K only make scores, only V carries content** — which is also why
`d_v` need not equal `d_k`.

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
