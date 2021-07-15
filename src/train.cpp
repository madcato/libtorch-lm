#include <torch/torch.h>
#include "../include/seq2seq_transformer.hpp"
#include "../include/multi30k.hpp"
#include "../include/util.hpp"

using namespace torch::indexing;

double train_epoch(Seq2SeqTransformer& model, torch::optim::Adam& optimizer, const torch::Device& device) {
    // model->train();
    double losses = 0.0;
    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
      MULTI30K("./data/")->map(
          torch::data::transforms::Stack<>()),
      /*batch_size=*/164);

    for (auto& batch : *data_loader) {
        torch::Tensor src = batch.data.to(device);
        torch::Tensor tgt = batch.target.to(device);

        src = src.transpose(0, 1);
        tgt = tgt.transpose(0, 1);

        torch::Tensor tgt_input = tgt.index({Slice(None, -1, None)});

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> mask = create_mask(src, tgt_input, device);

        torch::Tensor src_mask = std::get<0>(mask);
        torch::Tensor tgt_mask = std::get<1>(mask);
        torch::Tensor src_padding_mask = std::get<2>(mask);
        torch::Tensor tgt_padding_mask = std::get<3>(mask);
        torch::Tensor logits = model->forward(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask);
        return 0.0;
        // optimizer.zero_grad();
    }

    return 0.0;
}

// for(auto i = 0 ; i < src.dim() ; i++) {
//             std::cout << "Dim: " << i << std::endl;
//             std::cout << "Size: " << src.size(i) << std::endl;
//             for(auto t = 0 ; t < src.size(i) ; t++) {
//                 std::cout << src[i].is_cuda() << std::endl;
//             }
//         }

//         for(auto i = 0 ; i < tgt.dim() ; i++) {
//             std::cout << "Dim: " << i << std::endl;
//             std::cout << "Size: " << tgt.size(i) << std::endl;
//             for(auto t = 0 ; t < tgt.size(i) ; t++) {
//                 std::cout << tgt[i].is_cuda() << std::endl;
//             }
//         }


// std::cout << "src0 " << src.size(0) << std::endl;
//         std::cout << "tgt0 " << tgt.size(0) << std::endl;

//         std::cout << "src1 " << src.size(1) << std::endl;
//         std::cout << "tgt1 " << tgt.size(1) << std::endl<< std::endl<< std::endl;