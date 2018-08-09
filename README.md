# Machine-Translation

An attention-based neural machine translation model.[Luong et al., 2015]

## About
This software is a python3 implementation of attention-based neural machine translation (NMT) model proposed by [Luong et al., 2015]. As a neural network toolkit, it mainly uses Chainer.

## Demo

## VS. 

## Requirement

- python3

To run the models, you need following packages

- os
- numpy
- chainer
- pickle
- argparse
- tqdm
- time
- collections
- itertools
- warnings

If you use GPU, you also need 

- cupy

If you want to output attention weight heat map during test, you also need

- matplotlib
- seaborn
- PyPDF2

## Usage

First, you need to preprocess a raw parallel corpus using Preprocess.py. 

```
python Preprocess.py -src_train source_train_path -tgt_train target_train_path -save_name save_name 

```

This outputs "data/data_nama.data" and "data/data_nama.vocab_dict". 

If you also provide development data (highly recommended), specify the data path here


```
python Preprocess.py -src_train source_train_path -tgt_train target_train_path -save_name save_name -src_dev source_development_path -tgt_dev target_development_path
```

By enabling -output_vocab, it also outputs source and target vocabulary lists.


After processing the data, you can train an NMT model as follows:

```
python Preprocess.py -src_train source_train_path -tgt_train target_train_path -save_name save_name -src_dev source_development_path -tgt_dev target_development_path
```



## Install

## Author