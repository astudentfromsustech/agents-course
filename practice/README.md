# practice

Runnable exercises. Each isolates one mechanism and finishes in a sitting.

| | | needs |
|---|---|---|
| [`p0_attention`](p0_attention/) | self-attention and multi-head **by hand**, checked against your own longhand loop | `numpy` only |
| [`p1_induction_heads`](p1_induction_heads/) | train a tiny transformer until it **discovers a circuit**, then score every head | conda env `courses_self-attention` (torch + MPS) |
| [`p2_vit_mnist`](p2_vit_mnist/) | a **ViT reads handwritten digits** — then measure permutation equivariance and what a CNN's prior is worth | conda env `courses_self-attention` (torch + MPS) |

Later exercises (`p3`–`p7`, the agent track) are specified in each chapter and written when
reached — see `../PLAN.md`.

## Environment

p0 needs nothing. p1 and p2 need the GPU env:

```bash
conda create -n courses_self-attention python=3.12
conda activate courses_self-attention
pip install torch matplotlib      # p2 needs no torchvision: MNIST is one .npz
python -c "import torch; print(torch.backends.mps.is_available())"   # expect True
```

On an M-series Mac this gives Metal/MPS acceleration with no CUDA and no extra setup.

## Where it runs

Colab, or this Mac, or Colab's interface driving this Mac's GPU — same notebook either way.
See [RUNNING.md](RUNNING.md). The first cell of each notebook prints the device it found.
