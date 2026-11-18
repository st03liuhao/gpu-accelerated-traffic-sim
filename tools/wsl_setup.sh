#!/usr/bin/env bash
# One-time WSL (Ubuntu 24.04) setup for building LPSim with the RL lane-change model.
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../tools/wsl_setup.sh
# Installs: build tools, Qt5, Boost, CUDA toolkit 12.8 (WSL repo), and pandana's
# contraction-hierarchies routing library (libchrouting.so) under /usr/include/pandana.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "== apt packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential cmake git wget ca-certificates pkg-config python3 python3-pip \
  qtbase5-dev libqt5opengl5-dev libglew-dev libboost-all-dev libomp-dev

echo "== CUDA toolkit (WSL-Ubuntu repo: toolkit only, the driver comes from Windows)"
if ! command -v /usr/local/cuda/bin/nvcc >/dev/null; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update -qq
  apt-get install -y -qq cuda-toolkit-12-8
fi
/usr/local/cuda/bin/nvcc --version | tail -1

echo "== pandana contraction hierarchies -> /usr/include/pandana/src/libchrouting.so"
if [ ! -f /usr/include/pandana/src/libchrouting.so ]; then
  rm -rf /usr/include/pandana
  git clone -q --depth 1 https://github.com/UDST/pandana.git /usr/include/pandana
  cd /usr/include/pandana/src
  g++ -O3 -fPIC -shared -fopenmp -std=c++11 -w -I. -Icontraction_hierarchies/src \
    accessibility.cpp graphalg.cpp contraction_hierarchies/src/libch.cpp \
    -o libchrouting.so
fi
ls -la /usr/include/pandana/src/libchrouting.so
echo "== done"
