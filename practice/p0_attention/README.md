# p0 · self-attention and multi-head, by hand

For after you have watched the lectures and still want the mechanism in your hands.
Pure `numpy`, no PyTorch, no GPU, no downloads.

```bash
python3 attention_lab.py              # implement the TODOs, run until green
python3 attention_lab.py --solutions  # see what passing looks like
```

**You are checked against your own slow loop, not a library.** `attention_loop()` is given:
the definition written out longhand, one output vector at a time. Your job is the matrix
form, and if the two disagree, the matrix form is wrong. No black box anywhere.

## What it makes you prove

| Part | The thing you cannot fake |
|---|---|
| 1 | Your vectorised attention equals the longhand version, weights and all |
| 2 | **Why `1/√d_k` exists** — measured, not asserted: attention entropy stays ~3.7 as `d_k` goes 4→256 with the scaling, and collapses 2.74→0.22 without it. The softmax hardens into an argmax as the model widens |
| 3 | **Heads do not mix.** Perturb head 2's key projection; heads 0 and 1 must come out bit-identical. This is the one thing people get wrong when they write multi-head from memory |
| 4 | **What masking buys.** Rewrite tokens 4–5 entirely; outputs for positions 0–3 must not move by one bit |
| 5 | Masking works in the matrix form too, not just the loop |

Parts 3 and 4 are the ones worth doing slowly. They are not shape checks — they are
*behavioural* tests, and they fail loudly for an implementation that looks right and is not.

## If you get stuck

Run `--solutions` to see green, then read `solutions.py`. Reading the reference after
genuinely trying is learning; reading it first is not.

## Then

- `01_self_attention.html` §multi-head — the slide this exercise is testing
- [Transformer Puzzles](https://github.com/srush/Transformer-Puzzles) — the next question:
  not *how* attention computes, but *what* it can compute
