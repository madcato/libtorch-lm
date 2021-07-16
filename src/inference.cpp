#include <torch/torch.h>
#include "../include/seq2seq_transformer.hpp"
#include "../include/util.hpp"

torch::Tensor greedy_decode(Seq2SeqTransformer& model, torch::Tensor& src, torch::Tensor src_mask, int64_t max_len, int64_t start_symbol, const torch::Device& device) {
    src = src.to(device);
    src_mask = src_mask.to(device);

    torch::Tensor memory = model->encode(src, src_mask);
    torch::Tensor ys = torch::ones({1, 1}).fill_(start_symbol).toType(torch::kLong).to(device);
    for(int64_t i = 0 ; i < max_len ; i++) {
        memory = memory.to(device);
        torch::Tensor tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).toType(torch::kBool)).to(device);
        torch::Tensor out = model->decode(ys, memory, tgt_mask);
        out = out.transpose(0, 1);
        torch::Tensor prob = model->generator(out.index({Slice(None, None), -1}));
        std::tuple<torch::Tensor, torch::Tensor> tuple = torch::max(prob, /*dim=*/1);
        torch::Tensor next_word = std::get<1>(tuple);
        int64_t next_word_int = next_word.item<int64_t>();

        ys = torch::cat({ys, torch::ones({1, 1}).toType(src.scalar_type()).fill_(next_word)}, /*dim=*/0);
        if (next_word_int == EOS_IDX) {
            break;
        }
    }
    return ys;
}
