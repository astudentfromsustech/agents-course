#!/usr/bin/env python3
"""
Self-attention and multi-head attention, by hand — with a harness that checks you.

    python3 attention_lab.py              # run the checks against YOUR code
    python3 attention_lab.py --solutions  # run them against the reference

The reference you are checked against is not a library. It is attention_loop() below:
the slow, obvious, one-output-at-a-time version. If your matrix form disagrees with it,
the matrix form is wrong. That is the whole method.
"""
import sys
import numpy as np

# NumPy 2.x on Apple's Accelerate BLAS raises spurious "divide by zero / overflow
# encountered in matmul" warnings: the BLAS kernels set the FP error flags even when
# every result is finite and exact (verified here against hand-computed dot products
# to 1.8e-15). Nothing in this file legitimately divides by zero, so the noise is
# silenced rather than left to look like a bug in your implementation.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

rng = np.random.default_rng(0)
SOLVED = "--solutions" in sys.argv
if SOLVED:
    from solutions import attention_matrix, multihead, softmax


# ─────────────────────────────────────────────────────────────────────────────
# GIVEN — the ground truth. Read it; it is the definition, written out longhand.
# ─────────────────────────────────────────────────────────────────────────────
def attention_loop(Q, K, V, mask=None):
    """One output vector at a time, no matrix tricks. n = tokens, d_k = key dim."""
    n, d_k = Q.shape
    out = np.zeros((n, V.shape[1]))
    A = np.zeros((n, n))
    for i in range(n):                       # for each query position
        scores = np.array([Q[i] @ K[j] / np.sqrt(d_k) for j in range(n)])
        if mask is not None:
            scores = np.array([s if mask[i, j] else -np.inf
                               for j, s in enumerate(scores)])
        s = scores - scores.max()
        w = np.exp(s) / np.exp(s).sum()      # softmax over the n keys
        A[i] = w
        out[i] = sum(w[j] * V[j] for j in range(n))   # weighted sum of values
    return out, A


# ─────────────────────────────────────────────────────────────────────────────
# YOUR TURN
# ─────────────────────────────────────────────────────────────────────────────
if not SOLVED:

    def softmax(x, axis=-1):
        """TODO 0. Numerically stable softmax. Subtract the max before exp — ask
        yourself what exp(1000) does before you skip that."""
        raise NotImplementedError("TODO 0: softmax")

    def attention_matrix(Q, K, V, mask=None):
        """TODO 1. The same thing as attention_loop, with no Python loop at all.
        Return (output, attention_weights) — both, so the harness can check the
        weights as well as the result.

        Shapes: Q,K,V are (n, d_k). scores is (n, n). output is (n, d_v).
        Where mask is False, the score must become -inf BEFORE the softmax.
        """
        raise NotImplementedError("TODO 1: attention_matrix")

    def multihead(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
        """TODO 2. Project X once into Q, K, V of width d_model, cut each into
        n_heads slices of width d_model/n_heads, attend INSIDE each slice, then
        concatenate and project through Wo.

        The thing to get right, and the thing the harness tests: head h's queries
        may only ever meet head h's keys. Nothing crosses between heads until Wo.
        """
        raise NotImplementedError("TODO 2: multihead")


# ─────────────────────────────────────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"\n        {detail}" if detail and not cond else ""))
    return bool(cond)


def main():
    n, d_k, d_model, n_heads = 6, 8, 12, 3
    Q, K, V = rng.normal(size=(n, d_k)), rng.normal(size=(n, d_k)), rng.normal(size=(n, d_k))
    ok = []

    print("\n1 · matrix form == longhand form")
    out_m, A_m = attention_matrix(Q, K, V)
    out_l, A_l = attention_loop(Q, K, V)
    ok.append(check("output matches the loop", np.allclose(out_m, out_l),
                    f"max abs diff {np.abs(out_m - out_l).max():.3e}"))
    ok.append(check("attention weights match", np.allclose(A_m, A_l)))
    ok.append(check("every row of A sums to 1", np.allclose(A_m.sum(axis=1), 1)))
    ok.append(check("every weight is non-negative", (A_m >= 0).all()))

    print("\n2 · why the 1/sqrt(d_k) is there")
    # Dot products of random vectors grow like sqrt(d_k). Without the division the
    # scores get bigger as the model gets wider, and softmax hardens into argmax.
    # Entropy of each attention row is the measurement: ln(64) = 4.159 is uniform,
    # 0 is "all mass on one token".
    def entropy(p):
        return float((-(p * np.log(p + 1e-12)).sum(-1)).mean())

    m = 64
    print(f"        {'d_k':>5} {'H scaled':>10} {'H unscaled':>12}   (uniform = {np.log(m):.3f})")
    Hs, Hu = [], []
    for d in (4, 16, 64, 256):
        q, k = rng.normal(size=(m, d)), rng.normal(size=(m, d))
        sc = q @ k.T
        hs, hu = entropy(softmax(sc / np.sqrt(d), -1)), entropy(softmax(sc, -1))
        Hs.append(hs); Hu.append(hu)
        print(f"        {d:5} {hs:10.3f} {hu:12.3f}")
    ok.append(check("scaled attention stays spread out as d_k grows",
                    max(Hs) - min(Hs) < 0.3,
                    f"scaled entropy wandered: {Hs}"))
    ok.append(check("unscaled attention collapses toward argmax as d_k grows",
                    Hu[-1] < Hu[0] / 3,
                    f"expected a collapse, got {Hu}"))

    print("\n3 · multi-head — heads must not mix")
    X = rng.normal(size=(n, d_model))
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_model)) * 0.3 for _ in range(3))
    Wo = rng.normal(size=(d_model, d_model)) * 0.3
    base = multihead(X, Wq, Wk, Wv, Wo, n_heads)
    ok.append(check("output keeps d_model width", base.shape == (n, d_model), f"got {base.shape}"))

    # perturb ONLY head 2's key projection; head 0 and 1 must be untouched
    d_head = d_model // n_heads
    Wk2 = Wk.copy()
    Wk2[:, 2 * d_head:3 * d_head] += 5.0
    pert = multihead(X, Wq, Wk2, Wv, np.eye(d_model), n_heads)   # Wo=I so slices stay separable
    basep = multihead(X, Wq, Wk, Wv, np.eye(d_model), n_heads)
    head01_same = np.allclose(pert[:, :2 * d_head], basep[:, :2 * d_head])
    head2_moved = not np.allclose(pert[:, 2 * d_head:], basep[:, 2 * d_head:])
    ok.append(check("changing head 2's keys leaves heads 0-1 identical", head01_same,
                    "your heads are sharing keys — each head must attend only inside its own slice"))
    ok.append(check("changing head 2's keys does change head 2", head2_moved))

    print("\n4 · causal mask — the past cannot see the future")
    mask = np.tril(np.ones((n, n), dtype=bool))
    o1, A1 = attention_matrix(Q, K, V, mask)
    ok.append(check("upper triangle of A is exactly zero", np.allclose(np.triu(A1, 1), 0)))
    V2 = V.copy(); V2[4:] += 99.0                 # rewrite the future
    o2, _ = attention_matrix(Q, K, V2, mask)
    ok.append(check("positions 0-3 unchanged when tokens 4-5 are rewritten",
                    np.allclose(o1[:4], o2[:4]),
                    "this is the single property masking exists to give you"))

    print("\n5 · masked matrix form still == masked longhand")
    o_loop, _ = attention_loop(Q, K, V, mask)
    ok.append(check("masked output matches the loop", np.allclose(o1, o_loop)))

    print(f"\n{sum(ok)}/{len(ok)} checks passed"
          + ("  — all green.\n" if all(ok) else "  — read the FAIL lines above.\n"))
    return 0 if all(ok) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotImplementedError as e:
        print(f"\n  {e}\n  Implement it in attention_lab.py, then run again.")
        print("  Stuck? `python3 attention_lab.py --solutions` runs the same checks")
        print("  against the reference, so you can see what passing looks like.\n")
        sys.exit(1)
