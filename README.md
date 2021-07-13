# libtorch-LM

This is a lenguaje model made in c++ with libtorch.

## Requirments

- CMake
- Libtorch
- NVIDIA drivers
- CUDA
- cudnn

## Build

Build with `rake`

Clena with `rake clean`

## Downlaod and prepare data

1. `cd prepare`
2. `python3 -m spacy download de_core_news_sm`
3. `python3 -m spacy download en_core_web_sm`
4. `python3 multi30k.py`
   This command create two files on `./data` dir: `src.pt` and `tgt.pt` with the source and target data prepared for training. Also create `test_src.pt` and `test_tgt.pt` with the source and target data prepared for training.


## Next setp

- [ ] Write a method to dump Multi30K to a csv file
- [ ] Investigate clases mnist.cpp, mnist.py, 
- [ ] Multi30K.py
- [ ] Dataset (python and c++) 
- [ ] Iterators
- [ ] Create a multi30k.cpp
