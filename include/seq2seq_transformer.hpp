#ifndef SEQ2SEQ_TRANSFORMER_HPP_
#define SEQ2SEQ_TRANSFORMER_HPP_

#include <torch/torch.h>
#include "positional_encoding.hpp"
#include "token_embedding.hpp"

using namespace torch::indexing;

// Seq2Seq Network
class Seq2SeqTransformerImpl: public torch::nn::Module {
    public:
    Seq2SeqTransformerImpl(int64_t num_encoder_layers,
                      int64_t num_decoder_layers,
                      int64_t emb_size,
                      int64_t nhead,
                      int64_t src_vocab_size,
                      int64_t tgt_vocab_size,
                      int64_t dim_feedforward = 512,
                      double dropout = 0.1) {
        transformer = torch::nn::Transformer(torch::nn::TransformerOptions()
                        .d_model(emb_size)
                        .nhead(nhead)
                        .num_encoder_layers(num_encoder_layers)
                        .num_decoder_layers(num_decoder_layers)
                        .dim_feedforward(dim_feedforward)
                        .dropout(dropout));
        generator = torch::nn::Linear(torch::nn::LinearOptions(emb_size, tgt_vocab_size));
        src_tok_emb = TokenEmbedding(src_vocab_size, emb_size);
        tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size);
        positional_encoding = PositionalEncoding(emb_size, dropout);
    }

    torch::Tensor forward(torch::Tensor src,
                          torch::Tensor trg,
                          torch::Tensor src_mask,
                          torch::Tensor tgt_mask,
                          torch::Tensor src_padding_mask,
                          torch::Tensor tgt_padding_mask,
                          torch::Tensor memmory_key_padding_mask) {
        torch::Tensor src_emb = positional_encoding(src_tok_emb(src));
        torch::Tensor tgt_emb = positional_encoding(tgt_tok_emb(trg));
        const torch::Tensor& memory_mask = {};
        torch::Tensor outs = transformer(src_emb, tgt_emb, src_mask, tgt_mask, memory_mask, //at::empty({}),
                                         src_padding_mask, tgt_padding_mask, memmory_key_padding_mask);
        return generator(outs);
    } 

    torch::nn::Transformer transformer = nullptr;
    torch::nn::Linear generator = nullptr;
    TokenEmbedding src_tok_emb = nullptr;
    TokenEmbedding tgt_tok_emb = nullptr;
    PositionalEncoding positional_encoding = nullptr;
};

TORCH_MODULE(Seq2SeqTransformer);

#endif  // SEQ2SEQ_TRANSFORMER_HPP_
