#ifndef MULTI30K_HPP_
#define MULTI30K_HPP_

#include <torch/torch.h>

/// The MULTI30K dataset.
class TORCH_API MULTI30KImpl : public torch::data::Dataset<MULTI30KImpl> {
 public:
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the MNIST dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
  explicit MULTI30KImpl(const std::string& root, Mode mode = Mode::kTrain);

  /// Returns the `Example` at the given `index`.
  torch::data::Example<> get(size_t index) override;

  /// Returns the size of each sample.
  size_t features() const;

  /// Returns the size of the dataset.
  c10::optional<size_t> size() const override;

  /// Returns true if this is the training subset of MNIST.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  bool is_train() const noexcept;

  /// Returns all sources stacked into a single tensor.
  const torch::Tensor& sources() const;

  /// Returns all targets stacked into a single tensor.
  const torch::Tensor& targets() const;

 private:
  torch::Tensor sources_, targets_;
  Mode mode;
};

TORCH_MODULE(MULTI30K);

#endif  // MULTI30K_HPP_