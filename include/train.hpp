#ifndef TRAIN_HPP_
#define TRAIN_HPP_

double train_epoch(Seq2SeqTransformer& model, torch::optim::Adam& optimizer, const torch::Device& device);

#endif  // TRAIN_HPP_