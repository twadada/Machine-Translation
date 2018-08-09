import pickle
import argparse
import os
from models.preprocessing import Dataset,Build_Vocab, Register_wordID, Vocab_dict

# src_train=/cl/work/takashi-w/ASPEC-JE/train/train-1.txt.tok.low.clean.ja
# src_dev=/cl/work/takashi-w/ASPEC-JE/train/train-1.txt.tok.low.clean.ja
# tgt_train=/cl/work/takashi-w/ASPEC-JE/train/train-1.txt.tok.low.clean.en
# tgt_dev=/cl/work/takashi-w/ASPEC-JE/train/train-1.txt.tok.low.clean.en
# python Preprocess2.py -src_train $src_train -tgt_train $tgt_train

parser = argparse.ArgumentParser()

parser.add_argument(
    '-src_train',
    type=str,
    default=None,
    required=True,
    help='source train data path')
parser.add_argument(
    '-src_dev',
    type=str,
    default=None,
    help='source development data path')
parser.add_argument(
    '-tgt_train',
    type=str,
    default=None,
    required=True,
    help='target train data path')
parser.add_argument(
    '-tgt_dev',
    default=None,
    type=str,
    help='target development data path')

parser.add_argument(
    '-srcV',
    default=None,
    type=int,
    help='size of source vocaburary')
parser.add_argument(
    '-src_min',
    default=5,
    type=int,
    help='cutoff frequency of source words: a vocabulary list contains words that appear at least this times in source train data')
parser.add_argument(
    '-tgtV',
    default=None,
    type=int,
    help='size of target vocaburary')
parser.add_argument(
    '-tgt_min',
    default=5,
    type=int,
    help='cutoff frequency of traget words: a vocabulary list contains words that appear at least this times in target train data')

parser.add_argument(
    '-src_vocab',
    default=None,
    type=str,
    help='source vocabulary txt file')

parser.add_argument(
    '-tgt_vocab',
    default=None,
    type=str,
    help='target vocabulary txt file')

parser.add_argument(
    '-data_name',
    default="default",
    type=str,
    help='data name')
parser.add_argument(
    '-output_vocab',
    action='store_true',
    help='output vocabulary txt file'
)

opt = parser.parse_args()

if __name__ == '__main__':
    if (opt.data_name == "default"):
        if (opt.srcV != None or opt.tgtV != None or opt.src_min != 5 or opt.tgt_min != 5):
            raise Exception("this is not the default settting")

    vocab_srctgt = [opt.src_vocab, opt.tgt_vocab]
    train_srctgt = [opt.src_train, opt.tgt_train]
    dev_srctgt =   [opt.src_dev, opt.tgt_dev]
    V_srctgt =     [opt.srcV, opt.tgtV]
    V_minfreq_srctgt = [opt.src_min, opt.tgt_min]

    if(None in dev_srctgt):
        if(dev_srctgt[0] is not None or dev_srctgt[1] is not None):
            raise Exception("When you provide development data, both source and target corpora must be given.")
        else:
            print("no development data are provided; it is highly recommended to provide one for model selection")
    dataset = Dataset()
    vocab_dict = Vocab_dict()
    for i in range(2):#process src and tgt corpus

        vocab_file = vocab_srctgt[i]
        if (vocab_file is not None):
            print("Vocabulary file is given; vocab_size and vocab min freq are ignored.")
            vocab = open(vocab_file, 'r').read().splitlines()
            while ('' in vocab):  # if empty element exists
                vocab.remove('')
        else:
            print("Build vocablary list from train data.")
            vocab = Build_Vocab(train_srctgt[i], V_srctgt[i], V_minfreq_srctgt[i])

        vocab2id, id2vocab = Register_wordID(vocab)
        vocab_dict.register_dict(vocab2id,id2vocab)
        dataset.load_data(train_srctgt[i], dev_srctgt[i], vocab2id)

    dataset.add_EOS_BOS() #add EOS and BOS to tgt_lines


    assert len(dataset.lines_id[0]) == len(dataset.lines_id[1])
    if(None not in dev_srctgt):
        assert len(dataset.lines_id_dev[0]) == len(dataset.lines_id_dev[1])

    print("srcV: ", dataset.V_size[0])
    print("tgtV: ", dataset.V_size[1])
    print("train data size: ", len(dataset.lines_id[0]))
    if (None not in dev_srctgt):
        print("dev data size:", len(dataset.lines_id_dev[0]))


    if not os.path.exists("data"):
        os.makedirs("data")

    with open("data/"+opt.data_name+ ".data", mode='wb') as f:
        pickle.dump(dataset, f)
        f.close()

    with open("data/"+opt.data_name+ ".vocab_dict", mode='wb') as f:
        pickle.dump(vocab_dict, f)
        f.close()

    if(opt.output_vocab):
        with open("data/"+opt.data_name+ ".src_vocab.txt", "w") as f:
            for i in range(len(vocab_dict.id2vocab[0])):
                f.write((vocab_dict.id2vocab[0][i]) + "\n")

        with open("data/"+opt.data_name+ ".tgt_vocab.txt", "w") as f:
            for i in range(len(vocab_dict.id2vocab[1])):
                f.write((vocab_dict.id2vocab[1][i]) + "\n")
