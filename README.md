# Machine-Translation

An attention-based neural machine translation model.[Luong et al., 2015]

## About
This software is a python3 implementation of attention-based neural machine translation (NMT) model proposed by [Luong et al., 2015]. As a neural network toolkit, it mainly uses Chainer.

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

First, you need to preprocess a raw parallel corpus using Preprocess.py. To run the file, you specify one of these three pairs of options:

- -src_min and -tgt_min (default: 5)
- -srcV and -tgtV (default: None)
- -src_vocab and -tgt_vocab (default: None)

The first options specify the size of source and target vocabulary. The second options specify the cutoff word frequency; that is, only the words that appear at least the cutoff times in train data are included in vocabulary. In the last options, you can pass the paths of vocabulary files (.txt) that contain a word in vocabulary line by line.

To preprocess train and development files, run the followings

```
python Preprocess.py -src_train source_train_path -tgt_train target_train_path -src_dev source_development_path -tgt_dev target_development_path -save_name save_name
```

This outputs "data/save_name.data" and "data/save_name.vocab_dict". These files are to be used during training/testing.


If you do not have development data, you can omit src_dev and tgt_dev options; however, it is highly recommended to prepare these data to train models efficiently.


By enabling -output_vocab, it also outputs source and target vocabulary txt files.


After processing the data, you can train an NMT model as follows:

```
python train.py -save_name save_name -epoch_size 10 -opt_type Adam -gpuid 0 
```

If you omit -gpuid option, CPU is used instead. This file outputs a model in "Results" directory at every epoch. Enable -remove_models option to keep only the model that has achieved the best perplexity on development data. For other options, use -h option and see usage messages.


Once training is done, you can select a model and translate test data as follows:

```
python translate.py -model model_path -src_test test_data_path -gpuid 0 -beam_size 5
```

This outputs translations as "model_path.translation.txt". 

If you want to output top k translation candidates for each source sentence, set the option -k larger than 0 (default:0), and it produces another file "model_path.translation_topk.txt". 

If you want to output attention weight heat map, enable -out_attention_weight. Then, another file "model_path.attn_W.pdf" is generated that illustrates attention weight for each translation.

## Author
Takashi Wada
## Reference
Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. "Effective approaches to attentionbased neural machine translation." In Empirical Methods in Natural Language Processing (EMNLP), pages 1412–1421, Lisbon, Portugal. Association for Computational Linguistics.