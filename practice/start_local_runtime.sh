#!/usr/bin/env bash
# Serve this folder's notebooks from THIS Mac, and let Colab's UI connect to it.
# Colab frontend + your GPU. Nothing is uploaded; the kernel runs here.
#
#   bash practice/start_local_runtime.sh
#
# Then in Colab: Connect ▾ → "Connect to a local runtime" → paste the URL printed below.
set -euo pipefail

ENV_BIN="$HOME/.conda/envs/courses_self-attention/bin"
PORT="${PORT:-8888}"

if [[ ! -x "$ENV_BIN/jupyter" ]]; then
  echo "The conda env is missing. Create it once:"
  echo "  conda create -y -n courses_self-attention python=3.12"
  echo "  conda activate courses_self-attention"
  echo "  pip install torch matplotlib 'notebook<7' jupyter_http_over_ws"
  echo "  jupyter serverextension enable --py jupyter_http_over_ws"
  exit 1
fi

# notebook<7 is required: jupyter_http_over_ws imports notebook.base, which Notebook 7 removed.
ver=$("$ENV_BIN/python" -c "import notebook;print(notebook.__version__)")
case "$ver" in
  6.*) ;;
  *) echo "notebook $ver will not work with the Colab bridge — need 6.x:"
     echo "  pip install 'notebook<7'"; exit 1 ;;
esac

echo "serving $(cd "$(dirname "$0")" && pwd) on port $PORT"
echo "paste the http://localhost:$PORT/?token=… URL below into Colab's local-runtime dialog"
echo
exec "$ENV_BIN/jupyter" notebook \
  --notebook-dir="$(cd "$(dirname "$0")" && pwd)" \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port="$PORT" --NotebookApp.port_retries=0 --no-browser
