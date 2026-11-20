// Lane-changing policy inference for LPSim's per-vehicle CUDA kernel.
//
// Include this header in cuda_traffic_sim.cu after generating lc_policy_weights.h with
//   python -m lcrl.export_cuda --policy runs/<run>/policy.pt --out cuda/lc_policy_weights.h
//
// One thread evaluates one vehicle: ~9.0k multiply-adds, weights in __constant__ memory
// (every lane of a warp reads the same weight at the same time -> broadcast).
#pragma once
#include "lc_policy_weights.h"

// Lateral actions
#define LC_LAT_KEEP 0   // keep lane / continue the ongoing change
#define LC_LAT_LEFT 1   // start (or continue) a change to lane-1; aborts an ongoing right change
#define LC_LAT_RIGHT 2  // start (or continue) a change to lane+1; aborts an ongoing left change

// Acceleration offset (m/s^2) added to the IDM acceleration for each longitudinal action.
__device__ __forceinline__ float lcLonDelta(int lon) {
  const float d[LC_NUM_LON] = LC_LON_DELTAS;
  return d[lon];
}

// obs:     LC_OBS_DIM features, built exactly as lcrl/env.py:LaneChangeEnv.observe()
// latMask: lateral actions allowed by the safety shield (LC_LAT_KEEP must be allowed)
// lat/lon: greedy actions
__device__ inline void lcPolicyAct(const float* __restrict__ obs, const bool latMask[LC_NUM_LAT],
                                   int& lat, int& lon) {
  float h1[LC_HIDDEN], h2[LC_HIDDEN];
#pragma unroll 4
  for (int j = 0; j < LC_HIDDEN; ++j) {
    float s = lc_b1[j];
    for (int i = 0; i < LC_OBS_DIM; ++i) s += lc_W1[j * LC_OBS_DIM + i] * obs[i];
    h1[j] = tanhf(s);
  }
#pragma unroll 4
  for (int j = 0; j < LC_HIDDEN; ++j) {
    float s = lc_b2[j];
    for (int i = 0; i < LC_HIDDEN; ++i) s += lc_W2[j * LC_HIDDEN + i] * h1[i];
    h2[j] = tanhf(s);
  }
  float best = -1e30f;
  lat = LC_LAT_KEEP;
  lon = 0;
  for (int j = 0; j < LC_NUM_LAT + LC_NUM_LON; ++j) {
    float s = lc_b3[j];
    for (int i = 0; i < LC_HIDDEN; ++i) s += lc_W3[j * LC_HIDDEN + i] * h2[i];
    if (j < LC_NUM_LAT) {
      if (latMask[j] && s > best) { best = s; lat = j; }
      if (j == LC_NUM_LAT - 1) best = -1e30f;
    } else if (s > best) {
      best = s;
      lon = j - LC_NUM_LAT;
    }
  }
}
