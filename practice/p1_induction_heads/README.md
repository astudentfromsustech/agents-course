# p1 · watch multi-head attention learn an algorithm

p0 built the attention forward pass by hand. This trains a tiny transformer until it
discovers a circuit, then **measures which head learned what**. ~19 s on the Mac's GPU.

**Two ways to run it, same code:**

| | |
|---|---|
| [open in Colab](https://colab.research.google.com/github/astudentfromsustech/agents-course/blob/main/practice/p1_induction_heads/induction.ipynb) | Google's GPU, nothing installed |
| `bash ../start_local_runtime.sh` | Colab's interface driving **this Mac's GPU** — see [RUNNING.md](../RUNNING.md) |

```bash
conda activate courses_self-attention
python induction.py            # or just the script: trains, scores, writes loss.png + heads.png
```

The notebook's first cell prints which device it found, and nothing else changes between them.

Environment (created for this): conda env `courses_self-attention`, Python 3.12, torch 2.14.1
on Metal/MPS. The model is 41k parameters — small enough to read, big enough to be real.

## The task

A random block, repeated to fill the sequence, **with the block length drawn fresh for every
example**:

```
[ 7 3 9 1 | 7 3 9 1 | 7 3 ... ]   period 4
[ 2 8 5 | 2 8 5 | 2 8 5 | ... ]   period 3
```

The first pass through a block is unpredictable. Everything after it is determined — if you can
do one thing: *I am at token 7; where did 7 appear before; what came after it there.*

**The random period is load-bearing, and it is the methodological point of the exercise.** My
first version used a fixed period, and all four heads scored 0.83–0.93 — a beautiful result that
was completely wrong. With a fixed period a head can score perfectly by always looking 31
positions back, matching no tokens at all. The metric was measuring a positional reflex and
calling it induction. Randomising the period removes the shortcut; the control column proves it
is gone.

## What it reports

```
            induction  best fixed   at      what this head is
            (content)  (position)  offset
  L1H0       0.471      0.071       23    INDUCTION HEAD — matches on content
  L1H1       0.406      0.057       31    INDUCTION HEAD — matches on content
  L0H1       0.019      0.360        2    positional — always looks 2 back
  L0H0       0.001      0.954        1    positional — always looks 1 back  ← previous-token head
```

chance ≈ 0.030. The middle column is the control: the best score any *fixed* offset could get on
the same batch. **A head only earns the name if it beats its own shortcut.**

## What you should take from it

**The circuit has two parts, in two layers, and neither works alone.**

| layer 0 | a **previous-token head** writes "what came before me" into each position |
| layer 1 | an **induction head** reads that, matches the current token against it, and copies what followed |

That composition is the reason this task needs two layers. It is also the reason multi-head
matters: these are different jobs, learned by different heads, running side by side without
mixing — which is exactly the property p0 made you prove.

`heads.png` shows it directly: L0H0 is a clean one-step diagonal; both L1 heads light up a
stripe parallel to the dashed line the moment the block starts repeating.

`loss.png` shows the other thing worth seeing — **loss does not fall smoothly.** It stalls near
0.4 for a few hundred steps while the circuit has not formed, then drops two orders of magnitude
once it does. Plot it on a linear axis and the plateau disappears; that is why the axis is log.

## Things worth trying

- Set the period fixed again and watch the control column expose the shortcut.
- Drop to `L = 1` layer. It should fail — one layer cannot compose the two steps.
- Raise `H` to 4 heads. Do you get more induction heads, or redundant ones?
- Delete the previous-token head's contribution at test time and see the induction heads break.
