#include <torch/torch.h>

torch::Tensor generate_square_subsequent_mask(int64_t sz, torch::Device device) {
    torch::Tensor mask = (at::triu(at::ones({sz, sz}, device)) == 1).transpose(0, 1);
    mask = mask.to(torch::kFloat64).masked_fill(mask == 0, -std::numeric_limits<double>::infinity()).masked_fill(mask == 1, 0.0f);
    return mask;
}
