#ifndef TRANSFORMER_HPP_
#define TRANSFORMER_HPP_

#include <torch/torch.h>

Seq2SeqTransformer create_transformer(torch::Device& device,
                                             int64_t num_encoder_layers,
                                             int64_t num_decoder_layers,
                                             int64_t emb_size,
                                             int64_t nhead,
                                             int64_t src_vocab_size,
                                             int64_t tgt_vocab_size,
                                             int64_t dim_feedforward,
                                             double dropout);

torch::nn::CrossEntropyLoss create_loss();

torch::optim::Adam create_optimizer(const std::vector<at::Tensor>& transformer_parameters);

#endif  // TRANSFORMER_HPP_