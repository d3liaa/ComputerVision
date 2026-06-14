#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/bosca/Desktop/ComputerVision"
ENV="$HOME/cv_pytorch_env"
BUILD="$HOME/nksr_build"
MAMBA="$ROOT/.wsl_tools/bin/micromamba"

if [ ! -x "$MAMBA" ]; then
  mkdir -p "$ROOT/.wsl_tools"
  curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$ROOT/.wsl_tools/micromamba.tar.bz2"
  tar -xjf "$ROOT/.wsl_tools/micromamba.tar.bz2" -C "$ROOT/.wsl_tools"
fi

if [ ! -x "$ENV/bin/python" ]; then
  "$MAMBA" create -y -p "$ENV" -c nvidia/label/cuda-12.6.3 -c conda-forge \
    python=3.10 pip cuda-nvcc cuda-cudart-dev libcublas-dev libcusparse-dev \
    libcusolver-dev eigen zlib ninja git cmake gxx_linux-64
fi

"$ENV/bin/python" -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
"$ENV/bin/python" -m pip install python-pycg pykdtree numpy scipy trimesh plyfile \
  ninja GitPython tqdm omegaconf matplotlib

rm -rf "$BUILD"
cp -a "$ROOT/nksr" "$BUILD"

cd "$BUILD/package"
rm -rf build
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export GIT_PYTHON_GIT_EXECUTABLE="/usr/bin/git"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

"$ENV/bin/python" -m pip install --no-build-isolation .

cd "$ROOT"
"$ENV/bin/python" run_nksr_scanner.py \
  --input point_cloud_suzanne.ply \
  --output mesh_suzanne_nksr.ply \
  --repo "$BUILD" \
  --device cuda \
  --detail-level 1.0 \
  --mise-iter 1
