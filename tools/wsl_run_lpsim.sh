#!/usr/bin/env bash
# Run the WSL-built LPSim on a network.
#   wsl -d Ubuntu-24.04 -u root -- bash tools/wsl_run_lpsim.sh <network_dir> [limit_people] [start_hr] [end_hr] [build_dir]
# Writes outputs into <build_dir>/run_<network name>/.
set -euo pipefail
NET="$(realpath "$1")"
LIMIT="${2:-5000}"
START="${3:-5}"
END="${4:-12}"
BUILD="${5:-/root/lpsim-build}"
RUN="$BUILD/run_$(basename "$NET")"
rm -rf "$RUN"          # never read a previous run's outputs
mkdir -p "$RUN"
cp "$BUILD/lpsim" "$RUN/lpsim"
cat > "$RUN/command_line_options.ini" <<EOF
[General]
GUI=false
USE_CPU=false
NETWORK_PATH=$NET/
USE_JOHNSON_ROUTING=false
USE_SP_ROUTING=true
USE_PREV_PATHS=false
LIMIT_NUM_PEOPLE=$LIMIT
ADD_RANDOM_PEOPLE=false
NUM_PASSES=1
TIME_STEP=${LPSIM_DT:-0.1}
START_HR=$START
END_HR=$END
OD_DEMAND_FILENAME=${LPSIM_OD:-od_demand.csv}
SHOW_BENCHMARKS=true
REROUTE_INCREMENT=0
NUM_GPUS=1
EOF
# LPSIM_DT: simulation step (default 0.1 s; the RL policy decides every 0.5 s)
# optional IDM parameters (LPSim defaults otherwise): LPSIM_A, LPSIM_B, LPSIM_T, LPSIM_S0
# e.g. the training env's dynamics: LPSIM_A=1.5 LPSIM_B=2.0 LPSIM_T=1.2 LPSIM_S0=6.38 (s0 + 5 m vehicle length)
for kv in "a:${LPSIM_A:-}" "b:${LPSIM_B:-}" "T:${LPSIM_T:-}" "s_0:${LPSIM_S0:-}"; do
  if [ -n "${kv#*:}" ]; then echo "${kv%%:*}=${kv#*:}" >> "$RUN/command_line_options.ini"; fi
done
# LPSIM_RL=1: RL lane-change policy instead of LPSim's rule
if [ -n "${LPSIM_RL:-}" ]; then echo "RL_LANE_CHANGE=true" >> "$RUN/command_line_options.ini"; fi
cd "$RUN"
export LD_LIBRARY_PATH=/usr/include/pandana/src:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
time ./lpsim
ls -la "$RUN"
