#include <iostream>
#include <map>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include "include/positional_encoding.hpp"
#include "include/token_embedding.hpp"
#include "include/seq2seq_transformer.hpp"

using namespace std;

const string SOURCE_LANGUAGE = "de";
const string TARGET_LANGUAGE = "en";

// Place-holders
map<string,int> token_transform = {};
map<string,int> vocab_transform = {};

void pytorch_version() {
  std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;
}

torch::Device cuda_available() {
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "CUDA not available";
  }
  return device;
}

int main() {
  pytorch_version();
  torch::Device device = cuda_available();

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  printf("Loading tensor");
  std::vector<torch::Tensor> train;
  torch::load(train, "./data/src.pt");
  // torch::jit::script::Module module = torch::jit::load("./data/src.pt");
}
