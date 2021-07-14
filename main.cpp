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

std::vector<char> readFile(const char* filename)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);

    // Stop eating new lines in binary mode!!!
    file.unsetf(std::ios::skipws);

    // get its size:
    std::streampos fileSize;

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // reserve capacity
    std::vector<char> vec;
    vec.reserve(fileSize);

    // read the data:
    vec.insert(vec.begin(),
               std::istream_iterator<char>(file),
               std::istream_iterator<char>());

    return vec;
}

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
  auto tensora = torch::ones({3, 4});
  // torch::load(tensora, "./data/src.pt");
  std::vector<char> f = readFile("./data/src.pt");
  torch::IValue x = torch::pickle_load(f);
  std::cout << x << "\n";
  torch::Tensor tensorb = x.toTensor();
  for(auto a = 0 ; a < tensorb.dim() ; a++) {
    std::cout << "Dim " << a << ": " << tensorb.size(a) << "\n";
  }

  // torch::jit::script::Module container = torch::jit::load("../data/train.pt");

  // // Load values by name
  // torch::Tensor a = container.get_attribute("src").toTensor();
  // std::cout << a.shape() << "\n";

  // torch::Tensor b = container.get_attribute("tgt").toTensor();
  // std::cout << b.shape() << "\n";
}
