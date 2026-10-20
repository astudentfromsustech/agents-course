# p2 — a transformer that reads handwritten digits

An image is not a sequence, so the first question is how it becomes one. A Vision
Transformer cuts the 28×28 image into 7×7 patches, flattens each into a vector, and
calls *that* the sequence — 16 tokens, plus a `[CLS]` token whose final vector the
classifier reads. Everything after that is chapter 01 unchanged.

```bash
python vit_mnist.py              # all three experiments, ~50 s on an M-series Mac
python vit_mnist.py --only scale # just one
```

No torchvision — MNIST arrives as one 11 MB `.npz`, so the file runs anywhere torch does.

Running it writes two figures. `data_and_patches.png` first — the digits, one of them cut on
the 7×7 grid, and the same 16 patches laid out as the strip the model actually receives. **The
transformer never sees the 4×4 arrangement**, only those tokens in order, which is the whole
setup for experiment 2.

`trace_shapes()` walks one batch through the model, printing every tensor on the way — every
shape from a live tensor, so it cannot drift from the code. It is the fastest way to see that
the 17x17 attention matrix is the only place information crosses between positions, and that
the block shape `(B, 17, 64)` never changes, which is why blocks stack.

`show_training()` opens up the loop that `exp_train` hides: every one of the 1 410 gradient
steps, not just the six epoch-end printouts, plus what cross-entropy actually measures on a
single digit (`p(7)` 0.077 → 0.998, loss 2.56 → 0.00). It also explains why the printed loss
jitters — that number is only the *last* batch's, one of 235.

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
