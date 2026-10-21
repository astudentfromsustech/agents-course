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
x                (17,64)
Q = x @ Wq       (17,64) @ (64,64) = (17,64)            Wq learned
K = x @ Wk       (17,64) @ (64,64) = (17,64)            Wk learned
V = x @ Wv       (17,64) @ (64,64) = (17,64)            Wv learned
split 4 heads    (17,64) -> (4,17,16)                   a RESHAPE - no maths
Q @ K^T/sqrt16   (4,17,16) @ (4,16,17) = (4,17,17)      <- Q,K used HERE
softmax          (4,17,17) unchanged                    each row sums to 1
A' @ V           (4,17,17) @ (4,17,16) = (4,17,16)      <- V used HERE
merge @ W^O      (17,64) @ (64,64) = (17,64)            W^O learned
x + attn(x)      (17,64)                                residual
x + MLP(x)       (17,64)                                64->256->64
```

Every operand carries its shape, so the transpose is visible: `(4,17,16) @ (4,16,17)` is why the
inner dimensions meet at 16 and the result is `17x17` rather than `16x16` — the step that turns
content into scores. `A' @ V` reverses it, back to `(4,17,16)`. The four `(64,64)` matrices are
the block's attention parameters: 4 x 4,096 = 16,384 of its 49,792, the rest being the MLP.

`conv_or_not()` addresses the obvious objection to `Conv2d`/`kernel_size`/`stride` appearing in a
transformer. It rebuilds PatchEmbed with no convolution — `unfold` then matmul — and the outputs
are **bit-identical** (max difference 0.00). The Conv2d is a fused GPU kernel doing the cutting
and the matmul in one call, nothing more. Of the three things that make a CNN a CNN — shared
weights, overlapping receptive fields, a hierarchy of growing fields — this layer has only the
first, and shared weights are no more convolutional than one embedding table serving all word
positions. The locality prior `exp_scale` measures comes from overlap and stacking, both absent.

`does_multihead_help()` asks whether multi-head beats single-head, which changing `HEADS` tests
cleanly since it adds **no parameters**. Across 5 seeds: 1 head 94.73 %, 4 heads 95.34 % —
**+0.61 points, se 0.25, distinguishable**. Real but small. (A 3-seed version of the same test
reports "not distinguishable" at 1.7 se, and a single run had 2 heads beating 4 — the effect is
small enough that underpowered comparisons flip the conclusion.) Why it helps: the softmax budget
— one head's weights sum to 1, so two competing needs must share it and arrive downstream as a
blend nothing can un-mix. Why only a little here: on MNIST few relations compete, so the budget
rarely binds; in language a token needs its subject and antecedent and syntactic head at once.

`split_heads()` answers what "split into 4 heads" does: **nothing**. It is a reshape, and the
formula is just a relabelling — `Q[h][t][j] = Q[t][16h + j]`, equivalently
`Q[h] = x @ Wq[:, 16h : 16h+16]` (verified). So "4 heads" is 4 separate `(64,16)` projections
stored as one `(64,64)` matrix. Head *h* is
literally dimensions `[16h : 16h+16]` of the same vector — same numbers, same storage, no
parameters, not one multiply. The multi-headedness lives in `Wq`/`Wk`/`Wv`, which training is
free to make compute something different per column block; the split only declares the
boundaries.

Q and K are consumed at `Q.K^T` and never appear again; V is consumed at `A' @ V`. Hence the line
at the foot of the block: **Q and K only make scores, only V carries content** — which is also why
`d_v` need not equal `d_k`.

`how_heads_differ()` answers what the maps only suggest, measured over 1,024 images. Three axes,
the first a negative result: the heads do **not** specialise by position — uniform attention gives
a mean offset of 1.50 and all 16 heads measure 1.47-1.53 with preferred direction ~(0,0), so none
learned a fixed offset, even though the CNN-equivalence construction needs exactly that. They
differ instead in how much **content** drives them (block 1 head 2 draws nearly the same map for
every digit at 0.21; block 3 reaches 0.94 — content-dependence rises with depth) and in what
**kind of patch** they read (block 1 head 0 is indifferent to ink at -0.002, head 3 seeks it at
+0.539; by block 4 all four converge to +0.22-0.30). Most specialised in block 1, least in block 4.

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

`why_cls()` asks whether the [CLS] token is even needed. The block emits 17 vectors and
classification needs one, so something must collapse the sequence — [CLS], or pooling the patch
tokens. Same model, same budget, only the read-out differs:

| read-out | params | test acc |
|---|---:|---:|
| `[CLS]` token | 204,298 | 95.24 % |
| mean-pool | 204,234 | 95.62 % |
| max-pool | 204,234 | 95.96 % |

**No** — pooling matches or beats it. (Single runs at one seed: read this as "no meaningful
difference", not "pooling wins".) The ViT paper's own reason for the token is inheritance, not
superiority — *"Similar to BERT's [class] token, we prepend a learnable embedding to the sequence
of embedded patches"* — and its Appendix D.3 concludes of CLS vs global average pooling:
*"Both work similarly well, but require different learning-rates."* So it is a convention, and a
defensible one, not a necessity.

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

## 5 · One image, end to end

`predict()` draws the whole journey for a single digit in eight panels: input, the 16 patches,
the token matrix, the final `[CLS]` vector, what `[CLS]` reads at each block, the 10 logits, the
softmax, and the answer. Run it on a failure too — `V.predict(d, img_i=18)` shows a loopy `3`
called a `5` at 92.6 % confidence, with `3` not even the runner-up. Worth seeing, because a high
softmax probability means *far from the decision boundary*, not *likely correct*; nothing in the
architecture measures its own uncertainty.

## Things worth trying

- Change `PATCH` to 4 (49 tokens) or 14 (4 tokens). Where does accuracy break, and is the
  cost compute or information?
- Set `LAYERS = 1`. How much does one round of mixing lose?
- In `exp_position`, shuffle the **pixels** inside each patch instead of the patches. Why
  does that hurt the no-position model, when shuffling patches did not?
- The `PatchEmbed` layer is a `Conv2d` with `kernel_size == stride`. Make them differ, so
  patches overlap — you have just reintroduced a locality prior. Does the N = 1 000 number
  move toward the CNN's?
