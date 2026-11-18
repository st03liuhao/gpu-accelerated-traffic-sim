// Parity test: runs lcPolicyAct on observations dumped by tests/test_cuda_parity.py and
// writes the chosen actions back, so they can be compared with the PyTorch policy.
//   test_lc_policy <obs.bin> <mask.bin> <out.bin> <count>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "lc_policy.cuh"

__global__ void kernel_act(const float* obs, const unsigned char* mask, int* out, int n) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  bool m[LC_NUM_LAT];
  for (int k = 0; k < LC_NUM_LAT; ++k) m[k] = mask[p * LC_NUM_LAT + k] != 0;
  int lat, lon;
  lcPolicyAct(obs + (size_t)p * LC_OBS_DIM, m, lat, lon);
  out[2 * p] = lat;
  out[2 * p + 1] = lon;
}

static std::vector<char> readAll(const char* path, size_t bytes) {
  std::vector<char> buf(bytes);
  FILE* f = fopen(path, "rb");
  if (!f || fread(buf.data(), 1, bytes, f) != bytes) { fprintf(stderr, "read %s failed\n", path); exit(1); }
  fclose(f);
  return buf;
}

int main(int argc, char** argv) {
  if (argc != 5) { fprintf(stderr, "usage: %s obs.bin mask.bin out.bin count\n", argv[0]); return 1; }
  int n = atoi(argv[4]);
  auto obs = readAll(argv[1], sizeof(float) * LC_OBS_DIM * n);
  auto mask = readAll(argv[2], LC_NUM_LAT * n);
  float* d_obs; unsigned char* d_mask; int* d_out;
  cudaMalloc(&d_obs, obs.size()); cudaMalloc(&d_mask, mask.size()); cudaMalloc(&d_out, sizeof(int) * 2 * n);
  cudaMemcpy(d_obs, obs.data(), obs.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice);
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  kernel_act<<<(n + 127) / 128, 128>>>(d_obs, d_mask, d_out, n);
  cudaEventRecord(t1);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }
  float ms; cudaEventElapsedTime(&ms, t0, t1);
  std::vector<int> out(2 * n);
  cudaMemcpy(out.data(), d_out, sizeof(int) * 2 * n, cudaMemcpyDeviceToHost);
  FILE* f = fopen(argv[3], "wb"); fwrite(out.data(), sizeof(int), 2 * n, f); fclose(f);
  printf("vehicles=%d kernel_ms=%.3f\n", n, ms);
  return 0;
}
