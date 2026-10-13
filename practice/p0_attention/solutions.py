"""Reference implementations. Open this AFTER you have tried attention_lab.py."""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # shift: exp overflow guard
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention_matrix(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    A = softmax(scores, axis=-1)
    return A @ V, A


def multihead(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    n, d_model = X.shape
    d_head = d_model // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                  # (n, d_model)
    outs = []
    for h in range(n_heads):
        sl = slice(h * d_head, (h + 1) * d_head)      # this head's slice only
        o, _ = attention_matrix(Q[:, sl], K[:, sl], V[:, sl], mask)
        outs.append(o)
    return np.concatenate(outs, axis=-1) @ Wo
