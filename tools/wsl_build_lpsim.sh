#!/usr/bin/env bash
# Build LPSim (external/LPSim) inside WSL. Sources stay on the Windows side; the build tree
# lives on the Linux filesystem for speed.
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../tools/wsl_build_lpsim.sh [build_dir]
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/external/LPSim"
BUILD="${1:-/root/lpsim-build}"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CXX_FLAGS=-w
make -C "$BUILD" -j"$(nproc)"
ls -la "$BUILD/lpsim"
