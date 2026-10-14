#!/usr/bin/env python3
"""
Self-attention, one step at a time, with numbers small enough to check by hand.

    python3 walkthrough.py          # pause at each step (Enter to continue)
    python3 walkthrough.py --all    # run straight through
    python3 walkthrough.py --step 5 # jump to one step

4 tokens, d_model = 6, 2 heads of width 3. Every matrix prints in full. At several
points you are asked to verify one number yourself before the script confirms it —
do that, at least once. It is the difference between reading and knowing.
"""
import sys
import numpy as np

np.set_printoptions(precision=2, suppress=True, linewidth=120)
ALL = "--all" in sys.argv
ONLY = None
if "--step" in sys.argv:
    ONLY = int(sys.argv[sys.argv.index("--step") + 1])

TOK = ["The", "cat", "sat", "down"]
n, d_model, H = 4, 6, 2
d_head = d_model // H
step_no = 0


def step(title, body):
    global step_no
    step_no += 1
    if ONLY is not None and step_no != ONLY:
        body(silent=True)
        return
    print(f"\n{'─'*74}\nSTEP {step_no} · {title}\n{'─'*74}")
    body(silent=False)
    if not ALL and ONLY is None:
        try:
            input("\n    [Enter] ")
        except EOFError:
            pass


def show(name, M, note=""):
    print(f"\n  {name}   shape {M.shape}{'   ' + note if note else ''}")
    for i, row in enumerate(np.atleast_2d(M)):
        label = f"{TOK[i]:>5} │" if M.shape[0] == n and len(TOK) == n else "      │"
        print(f"   {label} {row}")


S = {}

# ── 1 ────────────────────────────────────────────────────────────────────────
def s1(silent):
    rng = np.random.default_rng(1)
    X = np.round(rng.normal(size=(n, d_model)), 1)
    S["X"] = X
    if silent: return
    print("\n  Four tokens. Each is already a vector — embedding happened before attention.")
    show("X  (input)", X, "one row per token, 6 numbers each")
    print("\n  Attention never sees words. It sees these rows, and asks which rows")
    print("  should influence which other rows.")
step("the input", s1)

# ── 2 ────────────────────────────────────────────────────────────────────────
def s2(silent):
    rng = np.random.default_rng(2)   # seeded -> reproducible; the 2 is just a label
    scale = 0.5                      # try 3.0 and watch step 7 collapse onto one token
    # three SEPARATE draws -- Wq = Wk = Wv would make every query match its own key
    Wq, Wk, Wv = (np.round(rng.normal(size=(d_model, d_model)) * scale, 1) for _ in range(3))
    S.update(Wq=Wq, Wk=Wk, Wv=Wv)
    if silent: return
    print("\n  Three learned matrices. These are the ONLY parameters in attention.")
    print("  Same matrices applied to every token — attention has no per-position weights.")
    show("W_q", Wq); show("W_k", Wk); show("W_v", Wv)
    print("\n  Reading that one line:")
    print("    default_rng(2)        seeded, so these numbers are reproducible. 2 is arbitrary.")
    print("    rng.normal(...)       a real model LEARNS these; random stands in for 'trained'.")
    print("    size=(d_model,d_model)  X (4,6) @ W (6,6) -> (4,6): width in, same width out.")
    print(f"    * {scale}                 keeps dot products small — see step 5. Try 3.0.")
    print("    np.round(...,1)       cosmetic, so you can check the arithmetic by hand.")
    print("    for _ in range(3)     THREE different draws. Wq=Wk=Wv would make every")
    print(f"                          query its own key — here W_q == W_k is {np.array_equal(Wq, Wk)}.")
step("the three weight matrices", s2)

# ── 3 ────────────────────────────────────────────────────────────────────────
def s3(silent):
    Q, K, V = S["X"] @ S["Wq"], S["X"] @ S["Wk"], S["X"] @ S["Wv"]
    S.update(Q=Q, K=K, V=V)
    if silent: return
    print("\n  Each token produces three vectors from its own row of X:")
    print("    query  = what am I looking for?")
    print("    key    = what do I advertise?")
    print("    value  = what do I pass on if chosen?")
    print("\n  @ is MATRIX MULTIPLICATION (np.matmul), not elementwise — that is *.")
    print(f"    X @ W_q   {S['X'].shape} @ {S['Wq'].shape} -> {(S['X'] @ S['Wq']).shape}"
          "   inner dims must agree")
    print(f"    X * X     elementwise, {(S['X'] * S['X']).shape} — a different operation entirely")
    print("\n  Row i of the result depends on row i of X and NOTHING else. No mixing")
    print("  happens here; that is why the same three matrices serve every position.")
    show("Q", Q); show("K", K); show("V", V)
    X, Wq, Wk = S["X"], S["Wq"], S["Wk"]
    sym = lambda M: np.allclose(M, M.T)
    print("\n  WHY THREE PROJECTIONS, and not just X?")
    print("    Score X against itself and relevance is fixed — embedding similarity,")
    print("    forever. The projections make relevance LEARNABLE.")
    print("    And W_q must differ from W_k, or the scores come out SYMMETRIC:")
    print(f"      no projection   X @ X.T           symmetric? {sym(X @ X.T)}")
    print(f"      tied W_q=W_k    (XWq)(XWq).T      symmetric? {sym((X@Wq) @ (X@Wq).T)}")
    print(f"      separate        (XWq)(XWk).T      symmetric? {sym((X@Wq) @ (X@Wk).T)}")
    St, Sr = (X@Wq) @ (X@Wq).T, (X@Wq) @ (X@Wk).T
    print(f"      tied:     cat->The {St[1,0]:6.2f}  The->cat {St[0,1]:6.2f}   forced equal")
    print(f"      separate: cat->The {Sr[1,0]:6.2f}  The->cat {Sr[0,1]:6.2f}   free to differ")
    print("    Relevance is directional — an adjective seeks its noun, not the reverse.")
    print("    V is separate again: the key is an address, the value is the payload.")

    hand = S["X"][1] @ S["Wq"]
    print(f"\n  CHECK: row 'cat' of Q is X[1] @ W_q")
    print(f"         by hand → {np.round(hand,2)}")
    print(f"         from Q  → {np.round(Q[1],2)}   match: {np.allclose(hand, Q[1])}")
step("project X into Q, K, V", s3)

# ── 4 ────────────────────────────────────────────────────────────────────────
def s4(silent):
    scores = S["Q"] @ S["K"].T
    S["scores"] = scores
    if silent: return
    print("\n  Every query meets every key. scores[i,j] = how much token i wants token j.")
    show("scores = Q @ Kᵀ", scores, "rows = queries, cols = keys")
    i, j = 1, 0
    hand = float(np.dot(S["Q"][i], S["K"][j]))
    print(f"\n  CHECK: scores[{i},{j}] is the dot product of query '{TOK[i]}' with key '{TOK[j]}'")
    print(f"         by hand → {hand:.2f}      from matrix → {scores[i,j]:.2f}"
          f"   match: {np.isclose(hand, scores[i,j])}")
    print("\n  This is the only place information moves between tokens. Everything")
    print("  else in a transformer processes each position alone.")
step("scores — every query against every key", s4)

# ── 5 ────────────────────────────────────────────────────────────────────────
def s5(silent):
    sc = S["scores"] / np.sqrt(d_model)
    S["scaled"] = sc
    if silent: return
    print(f"\n  Divide by √d_k = √{d_model} = {np.sqrt(d_model):.2f}.")
    show("before", S["scores"]); show("after ÷√d_k", sc)
    print(f"\n  Spread shrinks: max|score| {np.abs(S['scores']).max():.2f} → {np.abs(sc).max():.2f}")
    print("  Harmless here. At d_k=256 the unscaled version saturates softmax into a")
    print("  hard argmax — attention_lab.py step 2 measures exactly that.")
step("scale by √d_k", s5)

# ── 6 ────────────────────────────────────────────────────────────────────────
def s6(silent):
    mask = np.tril(np.ones((n, n), dtype=bool))
    masked = np.where(mask, S["scaled"], -np.inf)
    S.update(mask=mask, masked=masked)
    if silent: return
    print("\n  Causal mask: token i may not look at anything after i.")
    show("mask (True = allowed)", mask.astype(int))
    show("scores after masking", masked, "-inf becomes 0 after softmax")
    print("\n  Row 'The' can see only itself. Row 'down' can see everything.")
step("mask the future", s6)

# ── 7 ────────────────────────────────────────────────────────────────────────
def s7(silent):
    x = S["masked"] - S["masked"].max(-1, keepdims=True)
    A = np.exp(x) / np.exp(x).sum(-1, keepdims=True)
    S["A"] = A
    if silent: return
    print("\n  Softmax each row: raw scores become weights that sum to 1.")
    show("A  (attention weights)", A)
    print(f"\n  row sums → {A.sum(1)}   all 1.0: {np.allclose(A.sum(1), 1)}")
    print("\n  Read row 'sat': those three numbers are how much 'sat' draws from")
    print(f"  The/cat/sat → {np.round(A[2][:3],2)}. That row IS 'which words are relevant'.")

    sm = lambda s: (np.exp(s - s.max(-1, keepdims=True))
                    / np.exp(s - s.max(-1, keepdims=True)).sum(-1, keepdims=True))
    sc = S["scaled"]
    print("\n  exp() makes every value positive and turns differences into RATIOS:")
    print("    a score 1 higher earns e≈2.72x the weight, 2 higher earns 7.4x.")
    print("    That is what makes it a SOFT argmax rather than an average.")
    print(f"\n    row sums    {A.sum(1)}   each query spends a budget of 1")
    print(f"    column sums {np.round(A.sum(0),2)}   NOT 1 — nothing normalises a column")
    print(f"\n    softmax(s) == softmax(s+100)?  {np.allclose(sm(sc), sm(sc+100))}   shift-invariant,")
    print("      which is why subtracting the row max is free — and necessary, since")
    print("      exp(1000) overflows to inf and inf/inf is nan.")
    print("    But NOT scale-invariant — this is the √d_k story from step 5:")
    for f in (0.25, 1, 4):
        tag = "flatter" if f < 1 else ("as computed" if f == 1 else "sharper, toward argmax")
        print(f"      softmax({f:>4} x s)  peak weight {sm(sc*f).max():.3f}   {tag}")
    print(f"\n    masked entries exactly zero? {np.all(A[np.triu_indices(n,1)] == 0)}  because exp(-inf)=0")
    print("      — which is why the mask is applied to SCORES, before softmax.")
step("softmax → attention weights", s7)

# ── 8 ────────────────────────────────────────────────────────────────────────
def s8(silent):
    out = S["A"] @ S["V"]
    S["out"] = out
    if silent: return
    print("\n  Each output row = weighted average of the VALUE rows, using that row's weights.")
    show("output = A @ V", out)
    i = 2
    hand = sum(S["A"][i, j] * S["V"][j] for j in range(n))
    print(f"\n  CHECK: output row '{TOK[i]}' built by hand as a weighted sum of V:")
    for j in range(i + 1):
        print(f"           {S['A'][i,j]:.2f} × V[{TOK[j]}]")
    print(f"         by hand → {np.round(hand,2)}")
    print(f"         from A@V → {np.round(S['out'][i],2)}   match: {np.allclose(hand, out[i])}")
    print("\n  That is self-attention, complete. One token's output is a blend of the")
    print("  other tokens' values, mixed in proportions the model chose.")
step("output — the weighted sum", s8)

# ── 9 ────────────────────────────────────────────────────────────────────────
def s9(silent):
    Q, K, V = S["Q"], S["K"], S["V"]
    outs, As = [], []
    for h in range(H):
        sl = slice(h * d_head, (h + 1) * d_head)
        sc = Q[:, sl] @ K[:, sl].T / np.sqrt(d_head)
        sc = np.where(S["mask"], sc, -np.inf)
        e = np.exp(sc - sc.max(-1, keepdims=True))
        a = e / e.sum(-1, keepdims=True)
        As.append(a); outs.append(a @ V[:, sl])
    S.update(As=As, outs=outs)
    if silent: return
    print(f"\n  Same Q,K,V — now cut into {H} slices of width {d_head} and attended SEPARATELY.")
    for h in range(H):
        print(f"\n  ── head {h}: columns {h*d_head}:{(h+1)*d_head} only ──")
        show(f"  A (head {h})", As[h])
    print("\n  Two different opinions about which words are relevant, from the same input.")
    print("  Head 0's queries never touched head 1's keys — that separation is the")
    print("  whole of multi-head, and p0's harness tests exactly it.")
step("multi-head — split and attend separately", s9)

# ── 10 ───────────────────────────────────────────────────────────────────────
def s10(silent):
    cat = np.concatenate(S["outs"], axis=-1)
    rng = np.random.default_rng(3)
    Wo = np.round(rng.normal(size=(d_model, d_model)) * 0.5, 1)
    if silent: return
    print("\n  Concatenate the heads back to full width, then one more learned matrix.")
    show("concat(head0, head1)", cat, f"{d_head} + {d_head} = {d_model}")
    show("final = concat @ W_o", cat @ Wo)
    print("\n  W_o is where the heads finally mix — and the only place they do.")
    print("\n  That is one attention layer, end to end. A transformer stacks this with")
    print("  a feed-forward block and a residual add, N times. Nothing else is new.")
step("concatenate and project", s10)

# ── 11 ───────────────────────────────────────────────────────────────────────
def s11(silent):
    if silent: return
    X, Wq, Wk, Wv, mask = S["X"], S["Wq"], S["Wk"], S["Wv"], S["mask"]

    def attend(Xin):
        q, k, v = Xin @ Wq, Xin @ Wk, Xin @ Wv
        sc = np.where(mask, q @ k.T / np.sqrt(d_model), -np.inf)
        e = np.exp(sc - sc.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True)) @ v

    A_ctx, B_ctx = X.copy(), X.copy()
    B_ctx[0] = B_ctx[2] = -X[0]          # same token 1, different neighbours
    oa, ob = attend(A_ctx)[1], attend(B_ctx)[1]

    print("\n  What you bought is CONTEXT. Before attention a token's vector is the")
    print("  same wherever it appears. After, it carries its neighbourhood.")
    print(f"\n    token 1's INPUT row identical in both contexts?  {np.allclose(A_ctx[1], B_ctx[1])}")
    print(f"    token 1's OUTPUT row identical?                 {np.allclose(oa, ob)}")
    print(f"      context A -> {np.round(oa, 2)}")
    print(f"      context B -> {np.round(ob, 2)}")
    print(f"      differing by {np.abs(oa - ob).max():.2f} at the largest coordinate")
    print("\n  What you attach decides the task; attention does not change:")
    print("    generate      last position -> linear -> vocab logits -> sample   (GPT)")
    print("    classify      pool -> linear -> classes")
    print("    tag tokens    each position -> linear -> label")
    print("    retrieve      pool -> one vector, compared by cosine  (chapter 05)")
    print("    enc -> dec    encoder outputs become the K and V a decoder attends to")
    print("\n  None of it is text-specific: image patches, amino acids, genomic bins.")
    print("  Attention needs a set of vectors and a reason to relate them.")
    print("\n  And why stack N layers? One layer = one round of looking around.")
    print("  Stacking lets information travel in HOPS. p1 shows it forming: a")
    print("  previous-token head in layer 0 feeding an induction head in layer 1,")
    print("  neither working alone. That composition is the reason for depth.")
step("what the output is for", s11)

print("\n" + "─" * 74)
print("Next:  python3 attention_lab.py      — now write it yourself")
print("       python3 walkthrough.py --step 7   — revisit any step\n")
