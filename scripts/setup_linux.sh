#!/usr/bin/env bash
# One-shot environment setup for Linux + NVIDIA GPU.
#
#   scripts/setup_linux.sh                 # venv + PyTorch + TripoSR + Blender 4.2 LTS
#   CUDA_TAG=cu124 scripts/setup_linux.sh  # pick the PyTorch CUDA build for your driver
#   WITH_INSTANTMESH=1 scripts/setup_linux.sh
#
# Prerequisites: NVIDIA driver (nvidia-smi works), git, Python 3.10/3.11,
# and the CUDA toolkit (nvcc) matching CUDA_TAG for building torchmcubes.
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu128}"            # cu128 covers RTX 20xx-50xx; cu124 for older drivers
PYTHON="${PYTHON:-python3.11}"
BLENDER_VERSION="${BLENDER_VERSION:-4.2.23}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p third_party

echo "==> GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "WARNING: nvidia-smi not found"

echo "==> Python venv (.venv, $PYTHON)"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

echo "==> PyTorch ($CUDA_TAG)"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python - <<'EOF'
import torch
print("torch", torch.__version__, "CUDA", torch.version.cuda, "available:", torch.cuda.is_available())
EOF

echo "==> Pipeline requirements"
pip install -r requirements.txt

echo "==> TripoSR"
if [ ! -d third_party/TripoSR ]; then
  git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR
fi
# torchmcubes compiles a CUDA extension against the installed torch.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
grep -v torchmcubes third_party/TripoSR/requirements.txt | grep -v '^gradio' > third_party/triposr-req.txt
pip install -r third_party/triposr-req.txt

if [ "${WITH_INSTANTMESH:-0}" = "1" ]; then
  echo "==> InstantMesh (separate venv: its pins conflict with TripoSR)"
  if [ ! -d third_party/InstantMesh ]; then
    git clone --depth 1 https://github.com/TencentARC/InstantMesh third_party/InstantMesh
  fi
  "$PYTHON" -m venv third_party/InstantMesh/.venv
  third_party/InstantMesh/.venv/bin/pip install --upgrade pip wheel
  third_party/InstantMesh/.venv/bin/pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
  third_party/InstantMesh/.venv/bin/pip install -r third_party/InstantMesh/requirements.txt
  echo "Set reconstruction.instantmesh_python: third_party/InstantMesh/.venv/bin/python"
fi

echo "==> Blender ${BLENDER_VERSION}"
if ! command -v blender >/dev/null && [ ! -x "third_party/blender/blender" ]; then
  series="${BLENDER_VERSION%.*}"
  url="https://download.blender.org/release/Blender${series}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  curl -L "$url" | tar -xJ -C third_party
  mv "third_party/blender-${BLENDER_VERSION}-linux-x64" third_party/blender
fi
BLENDER_BIN="$(command -v blender || echo "$ROOT/third_party/blender/blender")"
"$BLENDER_BIN" --version | head -1

echo "==> Local LLMs (Ollama)"
if command -v ollama >/dev/null; then
  ollama pull qwen2.5:14b-instruct
  ollama pull qwen2.5vl:7b
else
  echo "Ollama not installed (https://ollama.com) - or point llm/vision at vLLM / LM Studio / Claude."
fi

if [ ! -f config.yaml ]; then
  sed -e "s#^  executable: null.*#  executable: ${BLENDER_BIN}#" config.example.yaml > config.yaml
  echo "Wrote config.yaml"
fi

echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  python main.py --check --config config.yaml"
echo "  python main.py --config config.yaml --prompt 'A functional MP5 submachine gun' --enable_rigging"
