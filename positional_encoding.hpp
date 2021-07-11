#ifndef PPOSITIONAL_ENCODING_HPP_
#define PPOSITIONAL_ENCODING_HPP_

#include <cmath>
#include <torch/torch.h>

using namespace torch::indexing;

class PositionalEncoding: torch::nn::Module {
    PositionalEncoding(int64_t emb_size, double dropout, int64_t maxlen = 5000) {
        torch::Tensor den = at::exp(- torch::arange(0, emb_size, 2) * log(10000) / emb_size);
        torch::Tensor pos = torch::arange(0, maxlen).reshape({maxlen, 1});
        pos_embedding = torch::zeros({maxlen, emb_size});
        pos_embedding.index_put_({Slice(None, None), Slice(0, None, 2)}, at::sin(pos * den));
        pos_embedding.index_put_({Slice(None, None), Slice(1, None, 2)}, at::cos(pos * den));

        this->dropout = torch::nn::Dropout(torch::nn::DropoutOptions(dropout));
        register_buffer("pos_embedding", pos_embedding);
    }

    torch::Tensor forward(torch::Tensor token_embedding) {
        return this->dropout(token_embedding + pos_embedding.index({None, token_embedding.size(0), Slice(None, None)}));
    } 

    torch::nn::Dropout dropout;
    torch::Tensor pos_embedding;
};

#endif  // PPOSITIONAL_ENCODING_HPP_
