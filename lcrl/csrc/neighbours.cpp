// Python binding for the sorted-list neighbour search (neighbours.cu).
#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> pass1(at::Tensor ent_slot, at::Tensor ent_x, at::Tensor seg_start, at::Tensor seg_end,
                              at::Tensor x, at::Tensor info, at::Tensor v, at::Tensor loss, at::Tensor qbase,
                              at::Tensor chain_leg, at::Tensor chain_a, at::Tensor box_leg, at::Tensor box_lane,
                              int64_t K, double INF);
std::vector<at::Tensor> pass2(at::Tensor ent_slot, at::Tensor ent_x, at::Tensor seg_start, at::Tensor seg_end,
                              at::Tensor x, at::Tensor info, at::Tensor qbase, at::Tensor gap);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pass1", &pass1, "leaders, followers, queue counts and costs on sorted lane lists");
  m.def("pass2", &pass2, "best visible gap per lane on sorted lane lists");
}
