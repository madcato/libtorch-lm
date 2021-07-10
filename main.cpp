#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "CUDA not available";
  }
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
