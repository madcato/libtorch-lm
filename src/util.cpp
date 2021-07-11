#include <torch/torch.h>
#include <tuple>
#include "../include/util.hpp"

torch::Tensor generate_square_subsequent_mask(int64_t sz, const torch::Device& device) {
    torch::Tensor mask = (at::triu(at::ones({sz, sz}, device)) == 1).transpose(0, 1);
    mask = mask.to(torch::kFloat64).masked_fill(mask == 0, -std::numeric_limits<double>::infinity()).masked_fill(mask == 1, 0.0f);
    return mask;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> create_mask(const torch::Tensor& src, const torch::Tensor& tgt, const torch::Device& device) {
    int64_t src_seq_len = src.size(0);
    int64_t tgt_seq_len = tgt.size(0);

    torch::Tensor tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device);
    torch::Tensor src_mask = torch::zeros({src_seq_len, src_seq_len}, device).to(at::kBool);

    torch::Tensor src_padding_mask = (src == PAD_IDX).transpose(0, 1);
    torch::Tensor tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1);
    return std::make_tuple(src_mask, tgt_mask, src_padding_mask, tgt_padding_mask);
}