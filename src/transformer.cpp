#include <torch/torch.h>
#include "../include/util.hpp"
#include "../include/seq2seq_transformer.hpp"
#include "../include/transformer.hpp"

Seq2SeqTransformer create_transformer(torch::Device& device,
                                             int64_t num_encoder_layers,
                                             int64_t num_decoder_layers,
                                             int64_t emb_size,
                                             int64_t nhead,
                                             int64_t src_vocab_size,
                                             int64_t tgt_vocab_size,
                                             int64_t dim_feedforward = 512,
                                             double dropout = 0.1) {
    torch::manual_seed(0);
    Seq2SeqTransformer transformer = Seq2SeqTransformer(num_encoder_layers,
                                                              num_decoder_layers,
                                                              emb_size,
                                                              nhead,
                                                              src_vocab_size,
                                                              tgt_vocab_size,
                                                              dim_feedforward,
                                                              dropout);
    for(auto p: transformer->parameters()) {
        if (p.dim() > 1) {
            torch::nn::init::xavier_uniform_(p);
        }
    }

    transformer->to(device);

    return transformer;
}

torch::nn::CrossEntropyLoss create_loss() {
    return torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().ignore_index(PAD_IDX));
}

torch::optim::Adam create_optimizer(const std::vector<at::Tensor>& transformer_parameters) {
    return torch::optim::Adam(transformer_parameters, torch::optim::AdamOptions().lr(0.0001).betas(std::make_tuple(0.9, 0.98)).eps(1e-9));
}
