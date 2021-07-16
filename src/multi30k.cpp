#include <torch/torch.h>
#include <iostream>

#include "../include/multi30k.hpp"

using namespace torch::indexing;

constexpr uint32_t kTrainSize = 60000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kImageMagicNumber = 2051;
constexpr uint32_t kTargetMagicNumber = 2049;
constexpr uint32_t kImageRows = 28;
constexpr uint32_t kImageColumns = 28;
constexpr const char* kTrainSourceFilename = "src.pt";
constexpr const char* kTrainTargetFilename = "tgt.pt";
constexpr const char* kTestSourceFilename = "test_src.pt";
constexpr const char* kTestTargetFilename = "test_tgt.pt";

std::string join_paths(std::string head, const std::string& tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}

torch::Tensor readFile(const std::string& filename)
{
    // open the file:
    std::ifstream file(filename, std::ios::binary);
    TORCH_CHECK(file, "Error opening images file at ", filename);

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

    torch::IValue x = torch::pickle_load(vec);
    torch::Tensor tensor = x.toTensor();
    tensor = tensor.transpose(0,1);
    return tensor;
}

torch::Tensor read_sources(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainSourceFilename : kTestSourceFilename);
  torch::Tensor tensor = readFile(path);
  return tensor;
}

torch::Tensor read_targets(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainTargetFilename : kTestTargetFilename);
  torch::Tensor tensor = readFile(path);
  return tensor;
}

MULTI30KImpl::MULTI30KImpl(const std::string& root, Mode mode)
    : sources_(read_sources(root, mode == Mode::kTrain)),
      targets_(read_targets(root, mode == Mode::kTrain)),
      mode(mode) {}

torch::data::Example<> MULTI30KImpl::get(size_t index) {
  // std::cout << "Index: " << index << std::endl;
  // torch::Tensor tensor = sources_[index];
  // std::cout << "Tensor dims: " << tensor.dim() << std::endl;
  // std::cout << tensor << std::endl;
  // tensor = targets_[index];
  // std::cout << "Tensor dims: " << tensor.dim() << std::endl;
  // std::cout << tensor << std::endl;
  return {sources_[index], targets_[index]};
}

size_t MULTI30KImpl::features() const {
  return sources_.size(1);
}

c10::optional<size_t> MULTI30KImpl::size() const {
  return sources_.size(0);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
bool MULTI30KImpl::is_train() const noexcept {
  return sources_.size(0) == kTrainSize;
}

const torch::Tensor& MULTI30KImpl::sources() const {
  exit(0);
  return sources_;
}

const torch::Tensor& MULTI30KImpl::targets() const {
  exit(0);
  return targets_;
}
