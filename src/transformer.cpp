// torch.manual_seed(0)

// SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
// TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
// EMB_SIZE = 512
// NHEAD = 8
// FFN_HID_DIM = 512
// BATCH_SIZE = 128
// NUM_ENCODER_LAYERS = 3
// NUM_DECODER_LAYERS = 3

// transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
//                                  NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

// for p in transformer.parameters():
//     if p.dim() > 1:
//         nn.init.xavier_uniform_(p)

// transformer = transformer.to(DEVICE)

// loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

// optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

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

torch::optim::Adam create_optimizer(const std::vector<torch::optim::OptimizerParamGroup>& transformer_parameters) {
    return torch::optim::Adam(transformer_parameters, torch::optim::AdamOptions().lr(0.0001).betas(std::make_tuple(0.9, 0.98)).eps(1e-9));
}
