#ifndef TOKEN_EMBEDDING_HPP_
#define TOKEN_EMBEDDING_HPP_

#include <cmath>
#include <torch/torch.h>

using namespace torch::indexing;

// helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TORCH_API TokenEmbeddingImpl: public torch::nn::Module {
    public:
    TokenEmbeddingImpl(int64_t vocab_size, int64_t emb_size) {
        embedding = register_module("embedding", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, emb_size)));
        this->emb_size_square = sqrt(emb_size);
    }

    torch::Tensor forward(torch::Tensor tokens) {
        return embedding(tokens.to(torch::kInt64)) * emb_size_square;
    } 

    torch::nn::Embedding embedding = nullptr;
    int64_t emb_size_square;
};

TORCH_MODULE(TokenEmbedding);

#endif  // TOKEN_EMBEDDING_HPP_
