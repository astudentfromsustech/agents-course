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

`show_tokens()` draws the real `16 x 64` matrix for one digit — every row a token, 64 numbers
wide. The eight blank patches come out **byte-identical**, because a blank patch is 49 copies of
the same background value and its token is `W @ (constant) + bias` wherever it sat. That makes
the position argument concrete: at this stage token 0 and token 15 are indistinguishable.

`show_filters()` answers "when are those weights learned?" — in the same loop as everything
else. Row j of the embedding matrix is a 7x7 filter, so it can be drawn: at init, after six
epochs, and the difference on the same colour scale. The third panel is nearly blank.
`cosine(init, trained) = 0.99` while accuracy goes 10% -> 95%, so on a task this small the patch
embedding stays close to a random projection and the blocks above learn to read it. That is a
finding about this setup, not a general law.

The architecture figure now also marks every learned group in orange (3,200 patch + 1,152
CLS/position + 199,168 blocks + 778 head = 204,298) and names what has no parameters at all:
Q.K', the scaling, softmax, @V, the residual adds, the [CLS] read-out.

`is_it_padding()` settles whether 49 -> 64 is a transformation or zero-padding: no output
equals any pixel, none is zero, and changing ONE pixel moves all 64 outputs. `W` is 64 rows of
49 weights and row j is a 7x7 filter, so output j answers "how much does this patch look like
filter j?".

The patch projection here is not a teaching simplification — it is line-for-line what timm and
the paper authors' own JAX code do (`nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
stride=patch_size)` then `flatten(2).transpose(1,2)`), only with different numbers. Their
comment says it best: *"We can merge s2d+emb into a single conv; it's the same."*

`why_64()` answers why a token is 64-d when the patch is 49 pixels: 49 is forced by the patch
size, 64 is chosen. Measured rather than asserted — the embedding matrix is `64 x 49`, so its
rank is at most 49, and the singular values of 64,000 real tokens fall off a cliff at exactly
index 49. The spare 15 directions start empty; they are scratch space the blocks write into,
since every sub-layer adds to the residual stream rather than replacing it.

`batch_check()` answers the question the architecture diagram raises: training is batched (256
images at once), so do they contaminate each other? Measured, not assumed — image 0's logits
are the same to 4e-07 whether it travels with 255 companions or alone, because the attention
tensor is 256 *separate* 17x17 matrices and nothing contracts the batch axis. The one place the
batch genuinely combines is the loss, which averages over all 256.

`draw_architecture()` draws the model the way CNN papers draw one — slabs left to right, the
digit at one end and the ten classes at the other. Worth comparing against a CNN diagram: a
CNN's slabs shrink (28->24->12->8->4) because every conv and pool trades resolution for
channels, while the ViT's stays `17 x 64` from the first block to the last. No pooling, no
stride, nothing shrinks.

`draw_flow()` is that same trace as a diagram: the whole path down the left, one block opened
up on the right, with the 17x17 attention matrix marked as the only step where information
crosses between positions.

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
