#!/usr/bin/env bash
# macOS setup for Day 20 lab. Apple Silicon (M1–M4) gets Metal-accelerated llama.cpp;
# Intel Macs fall back to CPU. Both work with vLLM-CPU in Docker.
set -euo pipefail

cd "$(dirname "$0")/.."
LAB_ROOT="$(pwd)"

echo "==> Day 20 lab setup (macOS)"
echo "    repo: $LAB_ROOT"

# 1. Homebrew sanity check
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew is required. Install from https://brew.sh first." >&2
  exit 1
fi

# 2. Make sure Xcode CLT is present (Metal SDK ships with it)
if ! xcode-select -p >/dev/null 2>&1; then
  echo "==> Installing Xcode Command Line Tools (interactive)"
  xcode-select --install || true
fi

# 3. cmake is needed if pip falls through to source build
brew list cmake >/dev/null 2>&1 || brew install cmake

# 4. Python virtualenv
PYTHON="${PYTHON:-python3}"
if [[ ! -d .venv ]]; then
  echo "==> Creating .venv with $PYTHON"
  uv venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m ensurepip

echo "==> Upgrading pip"
python -m pip install --upgrade pip wheel >/dev/null

echo "==> Installing Python deps from requirements.txt"
pip install -r requirements.txt

# 5. llama-cpp-python with Metal on Apple Silicon, plain CPU on Intel
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
  echo "==> Building llama-cpp-python with Metal (Apple Silicon)"
  CMAKE_ARGS="-DGGML_METAL=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
else
  echo "==> Installing prebuilt llama-cpp-python (Intel Mac, CPU only)"
  pip install --upgrade llama-cpp-python
fi

# 6. Probe + download model
python 00-setup/detect-hardware.py
python 00-setup/download-model.py

echo
echo "==> Setup complete. Activate with: source .venv/bin/activate"
echo "==> Then proceed to: 01-llama-cpp-quickstart/README.md"
