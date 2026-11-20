// Neighbour search on per-lane sorted vehicle lists (lcrl/neighbours.py).
//
// The vehicles of every lane list (one list per lane for a freeway segment, one per
// (link, lane) for an intersection) are sorted by (x, slot) in PyTorch first. One thread
// then handles one (env, vehicle i, lane k):
//   - binary search for i's position in lane k's list;
//   - leader = next entry, follower = previous entry;
//   - queue count / speed sum: scan forward while within i's information range;
//   - queue cost: scan backward while within 60 m;
//   - best gap: scan both ways while within i's information range.
// Cost per thread is O(log n + vehicles within range) instead of O(N). Results equal the
// dense [E, N, N, K] version in IntersectionEnv._tables / LaneChangeEnv._tables, including
// tie-breaking (smallest slot index first); only float sums may differ in the last bits.
// ATen only: torch/extension.h does not compile under nvcc + MSVC; the Python binding
// is in neighbours.cpp.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {

// first entry p in [s, t) with (x_p, slot_p) >= (xq, q)
__device__ __forceinline__ int lower_bound_xs(const float* ex, const int* es, int s, int t, float xq, int q) {
  while (s < t) {
    int m = (s + t) >> 1;
    bool less = (ex[m] < xq) || (ex[m] == xq && es[m] < q);
    if (less) s = m + 1; else t = m;
  }
  return s;
}

// first entry p in [s, t) with x_p >= xq
__device__ __forceinline__ int lower_bound_x(const float* ex, int s, int t, float xq) {
  while (s < t) {
    int m = (s + t) >> 1;
    if (ex[m] < xq) s = m + 1; else t = m;
  }
  return s;
}

__global__ void pass1_kernel(
    const int* __restrict__ ent_slot, const float* __restrict__ ent_x,   // [E, M] sorted lists
    const int* __restrict__ seg_start, const int* __restrict__ seg_end,  // [E, L]
    const float* __restrict__ x, const float* __restrict__ info,         // [E, N]
    const float* __restrict__ v, const float* __restrict__ loss,         // [E, N]
    const int* __restrict__ qbase,                                       // [E, N] list id of lane 0, -1: none
    const int* __restrict__ chain_leg, const float* __restrict__ chain_a,// [E, N] (ix only)
    const int* __restrict__ box_leg, const int* __restrict__ box_lane,   // [E, N] (ix only)
    int has_chain, int E, int N, int K, int M, int L, float INF,
    float* __restrict__ lead_dx, int64_t* __restrict__ lead_idx, float* __restrict__ link_lead_dx,
    float* __restrict__ lag_dx, int64_t* __restrict__ lag_idx,
    float* __restrict__ count, float* __restrict__ speed_sum, float* __restrict__ queue_cost) {
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (int64_t)E * N * K) return;
  const int k = tid % K;
  const int i = (tid / K) % N;
  const int e = tid / ((int64_t)K * N);
  const int ei = e * N + i;
  const float xi = x[ei];
  float ldx = INF, gdx = INF, cnt = 0.f, ssum = 0.f, qc = 0.f;
  int64_t lidx = 0, gidx = 0;
  const int base = qbase[ei];
  if (base >= 0) {
    const int lid = base + k;
    const int s = seg_start[e * L + lid], t = seg_end[e * L + lid];
    const float* ex = ent_x + (int64_t)e * M;
    const int* es = ent_slot + (int64_t)e * M;
    const int lb = lower_bound_xs(ex, es, s, t, xi, i);
    const int ub = (lb < t && es[lb] == i) ? lb + 1 : lb;      // first entry ahead of i
    if (ub < t) { ldx = ex[ub] - xi; lidx = es[ub]; }
    if (lb > s) {                                              // nearest behind; smallest slot on ties
      const float xb = ex[lb - 1];
      const int p = lower_bound_x(ex, s, lb, xb);
      gdx = -(xb - xi); gidx = es[p];
    }
    const float rng = info[ei];
    for (int p = ub; p < t; ++p) {                             // queue ahead within range
      const float d = ex[p] - xi;
      if (!(d < rng)) break;
      cnt += 1.f;
      ssum += v[e * N + es[p]];
    }
    for (int p = lb - 1; p >= s; --p) {                        // slow queue held up within 60 m
      const float d = ex[p] - xi;
      if (!(d > -60.f)) break;
      const int j = es[p];
      if (v[e * N + j] < 3.f) qc += loss[e * N + j];
    }
  }
  const int o = ei * K + k;
  link_lead_dx[o] = ldx;
  if (has_chain && chain_leg[ei] >= 0) {                       // approach vehicle behind box vehicles
    const float a = chain_a[ei];
    const int leg = chain_leg[ei];
    float cdx = INF; int64_t cidx = 0;
    for (int j = 0; j < N; ++j) {
      if (box_leg[e * N + j] == leg && box_lane[e * N + j] == k) {
        const float c = a + x[e * N + j];
        if (c < cdx) { cdx = c; cidx = j; }
      }
    }
    if (cdx < ldx) { ldx = cdx; lidx = cidx; }
  }
  lead_dx[o] = ldx; lead_idx[o] = lidx;
  lag_dx[o] = gdx; lag_idx[o] = gidx;
  count[o] = cnt; speed_sum[o] = ssum; queue_cost[o] = qc;
}

__global__ void pass2_kernel(
    const int* __restrict__ ent_slot, const float* __restrict__ ent_x,
    const int* __restrict__ seg_start, const int* __restrict__ seg_end,
    const float* __restrict__ x, const float* __restrict__ info, const int* __restrict__ qbase,
    const float* __restrict__ gap,                                       // [E, N, K] min(lead_gap, info)
    int E, int N, int K, int M, int L,
    float* __restrict__ best_gap, float* __restrict__ best_pos) {
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (int64_t)E * N * K) return;
  const int k = tid % K;
  const int i = (tid / K) % N;
  const int e = tid / ((int64_t)K * N);
  const int ei = e * N + i;
  const float xi = x[ei], rng = info[ei];
  float best = -1.f, bdx = 0.f, bg = 0.f;
  int bj = -1;
  const int base = qbase[ei];
  if (base >= 0) {
    const int lid = base + k;
    const int s = seg_start[e * L + lid], t = seg_end[e * L + lid];
    const float* ex = ent_x + (int64_t)e * M;
    const int* es = ent_slot + (int64_t)e * M;
    const int lb = lower_bound_xs(ex, es, s, t, xi, i);
    auto visit = [&](int p) {
      const int j = es[p];
      const float g = gap[((int64_t)e * N + j) * K + k];
      if (bj < 0 || g > best || (g == best && j < bj)) { best = g; bj = j; bdx = ex[p] - xi; bg = g; }
    };
    for (int p = lb; p < t; ++p) {                             // i itself and ahead
      if (!(fabsf(ex[p] - xi) < rng)) break;
      visit(p);
    }
    for (int p = lb - 1; p >= s; --p) {                        // behind
      if (!(fabsf(ex[p] - xi) < rng)) break;
      visit(p);
    }
  }
  const int o = ei * K + k;
  if (bj < 0 || best < 0.f) { best_gap[o] = rng; best_pos[o] = 0.f; }
  else { best_gap[o] = best; best_pos[o] = bdx + bg * 0.5f; }
}

inline const int* I(const at::Tensor& t) { return t.numel() ? t.data_ptr<int>() : nullptr; }
inline const float* F(const at::Tensor& t) { return t.numel() ? t.data_ptr<float>() : nullptr; }

}  // namespace

std::vector<at::Tensor> pass1(at::Tensor ent_slot, at::Tensor ent_x, at::Tensor seg_start,
                                 at::Tensor seg_end, at::Tensor x, at::Tensor info, at::Tensor v,
                                 at::Tensor loss, at::Tensor qbase, at::Tensor chain_leg,
                                 at::Tensor chain_a, at::Tensor box_leg, at::Tensor box_lane,
                                 int64_t K, double INF) {
  const int E = x.size(0), N = x.size(1), M = ent_x.size(1), L = seg_start.size(1);
  auto fo = x.options();
  auto io = x.options().dtype(at::kLong);
  auto lead_dx = at::empty({E, N, K}, fo), link_lead_dx = at::empty({E, N, K}, fo);
  auto lag_dx = at::empty({E, N, K}, fo), count = at::empty({E, N, K}, fo);
  auto speed_sum = at::empty({E, N, K}, fo), queue_cost = at::empty({E, N, K}, fo);
  auto lead_idx = at::empty({E, N, K}, io), lag_idx = at::empty({E, N, K}, io);
  const int64_t n = (int64_t)E * N * K;
  const int threads = 256;
  const int has_chain = chain_leg.numel() > 0;
  pass1_kernel<<<(n + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      I(ent_slot), F(ent_x), I(seg_start), I(seg_end), F(x), F(info), F(v), F(loss), I(qbase),
      I(chain_leg), F(chain_a), I(box_leg), I(box_lane), has_chain, E, N, (int)K, M, L, (float)INF,
      lead_dx.data_ptr<float>(), lead_idx.data_ptr<int64_t>(), link_lead_dx.data_ptr<float>(),
      lag_dx.data_ptr<float>(), lag_idx.data_ptr<int64_t>(), count.data_ptr<float>(),
      speed_sum.data_ptr<float>(), queue_cost.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {lead_dx, lead_idx, link_lead_dx, lag_dx, lag_idx, count, speed_sum, queue_cost};
}

std::vector<at::Tensor> pass2(at::Tensor ent_slot, at::Tensor ent_x, at::Tensor seg_start,
                                 at::Tensor seg_end, at::Tensor x, at::Tensor info, at::Tensor qbase,
                                 at::Tensor gap) {
  const int E = x.size(0), N = x.size(1), K = gap.size(2), M = ent_x.size(1), L = seg_start.size(1);
  auto best_gap = at::empty({E, N, K}, x.options()), best_pos = at::empty({E, N, K}, x.options());
  const int64_t n = (int64_t)E * N * K;
  const int threads = 256;
  pass2_kernel<<<(n + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      I(ent_slot), F(ent_x), I(seg_start), I(seg_end), F(x), F(info), I(qbase), F(gap), E, N, K, M, L,
      best_gap.data_ptr<float>(), best_pos.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {best_gap, best_pos};
}
