# Where the notebooks run

Same notebook, three places. **Nothing in the code changes** — the first cell detects where it is
and prints the device it found.

| | compute | setup | good for |
|---|---|---|---|
| **Colab** | Google's GPU (free tier: usually a T4) | none — click the badge | anywhere, any device, zero install |
| **Colab UI + your Mac** | **this Mac's GPU** (M5 Max, 40 cores) | one command, below | Colab's interface, your hardware, no session limits, nothing uploaded |
| **Local Jupyter** | this Mac's GPU | `jupyter lab` | offline, fastest to start |

The banner cell prints which you got:

```
frontend : local Jupyter
compute  : mps  (Apple arm GPU (Metal))
torch    : 2.14.1
```

## Colab's interface, your Mac's GPU

```bash
bash practice/start_local_runtime.sh
```

It prints a URL with a token. In Colab: **Connect ▾ → "Connect to a local runtime"** → paste it.
The notebook now executes on this machine — your files, your GPU, no upload, no session timeout.

🔴 **This needs `notebook<7`.** The bridge package `jupyter_http_over_ws` imports
`notebook.base.handlers`, which Notebook 7 removed, so a current install fails with
`ModuleNotFoundError: No module named 'notebook.base'`. The env is pinned to 6.5.7 and the start
script refuses to launch on anything else rather than failing confusingly later.

## One-time environment

```bash
conda create -y -n courses_self-attention python=3.12
conda activate courses_self-attention
pip install torch matplotlib 'notebook<7' jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
```

`p0` needs none of this — it is pure numpy and runs on the system python.

## Which to use

**Colab for `p0`** — it is numpy, compute is irrelevant, and the browser is the least friction.

**Either for `p1`** — it trains in ~18 s on the Mac's GPU. Colab's T4 is comparable at this size.
Use the local runtime when you want to keep the figures, edit alongside the repo, or avoid
re-installing on every fresh Colab session.
