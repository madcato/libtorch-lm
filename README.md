# libtorch-LM

This is a lenguaje model made in c++ with libtorch.

This code is a partial adaptation of the PyTorch tutorial: [Language translation with nn.Transformer and torchtext](https://pytorch.org/tutorials/beginner/translation_transformer.html)

Only mode, training and evaluation is translated to C++. Data downloading, preparation and inference is not ported.

## Requirments

- CMake
- Libtorch
- NVIDIA drivers
- CUDA
- cudnn

## Build

Build with `rake`

Clean with `rake clean`

## Download and prepare data

1. `cd prepare`
2. `python3 -m spacy download de_core_news_sm`
3. `python3 -m spacy download en_core_web_sm`
4. `python3 multi30k.py`
   This command create two files on `./data` dir: `src.pt` and `tgt.pt` with the source and target data prepared for training. Also create `test_src.pt` and `test_tgt.pt` with the source and target data prepared for training.
5. Create directory `model`. pt files with the trained model will be saved here.
6. Execute training: `./build/libtorch-lm`
