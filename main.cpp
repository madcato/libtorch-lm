#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include "include/positional_encoding.hpp"
#include "include/token_embedding.hpp"
#include "include/seq2seq_transformer.hpp"
#include "include/train.hpp"
#include "include/transformer.hpp"
#include "include/multi30k.hpp"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

const string SOURCE_LANGUAGE = "de";
const string TARGET_LANGUAGE = "en";

// Place-holders
map<string,int> token_transform = {};
map<string,int> vocab_transform = {};

void pytorch_version() {
  std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;
}

torch::Device cuda_available() {
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "CUDA not available";
  }
  return device;
}

int main() {
  pytorch_version();
  torch::Device device = cuda_available();

  int64_t num_encoder_layers = 3;
  int64_t num_decoder_layers = 3;
  int64_t emb_size = 512;
  int64_t nhead = 8;
  int64_t src_vocab_size = 19215;  // TODO: calculate this
  int64_t tgt_vocab_size = 10838;  // TODO: calculate this
  int64_t dim_feedforward = 512;
  double dropout = 0.1;
  Seq2SeqTransformer transformer = create_transformer(device,
                                                      num_encoder_layers,
                                                      num_decoder_layers,
                                                      emb_size,
                                                      nhead,
                                                      src_vocab_size,
                                                      tgt_vocab_size,
                                                      dim_feedforward,
                                                      dropout);
  
  torch::nn::CrossEntropyLoss loss_fn = create_loss();

  auto parameters = transformer->parameters();
  torch::optim::Adam optimizer = create_optimizer(parameters);

  transformer->to(device);

  const int NUM_EPOCHCS = 18;

  for(auto epoch = 1 ; epoch <= NUM_EPOCHCS ; epoch++) {
    auto t1 = high_resolution_clock::now();
    double train_loss = train_epoch(transformer, optimizer, loss_fn, device);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    double val_loss = evaluate(transformer, loss_fn, device);
    printf("Epoch: %i, Train loss: %f, Val loss: %f Epoch time: %li\n", epoch, train_loss, val_loss, ms_int.count());
    // torch::save(net, "net.pt");
  }
  
   // Create a multi-threaded data loader for the MNIST dataset.
  // auto data_loader = torch::data::make_data_loader(
  //     MULTI30K("./data/")->map(
  //         torch::data::transforms::Stack<>()),
  //     /*batch_size=*/64);
  
  // MULTI30K multi30k = MULTI30K("./data/");

  // for(auto i = 0 ; i < 20 ; i++) {
  //   torch::data::Example<> example = multi30k->get(i);
  //   auto data = example.data;
  //   auto target = example.target;
  //   std::cout << target << std::endl;
  // }
}
