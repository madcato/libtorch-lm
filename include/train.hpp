#ifndef TRAIN_HPP_
#define TRAIN_HPP_

double train_epoch(Seq2SeqTransformer& model, torch::optim::Adam& optimizer, torch::nn::CrossEntropyLoss& loss_fn, const torch::Device& device);
double evaluate(Seq2SeqTransformer& model, torch::nn::CrossEntropyLoss& loss_fn, const torch::Device& device);

#endif  // TRAIN_HPP_