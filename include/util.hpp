#ifndef UTIL_HPP_
#define UTIL_HPP_

const int64_t UNK_IDX = 0;
const int64_t PAD_IDX = 1;
const int64_t BOS_IDX = 2;
const int64_t EOS_IDX = 3;

torch::Tensor generate_square_subsequent_mask(int64_t sz, const torch::Device& device);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> create_mask(const torch::Tensor& src, const torch::Tensor& tgt, const torch::Device& device);

#endif  // UTIL_HPP_
