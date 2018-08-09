from collections import Counter, OrderedDict
from itertools import chain
import warnings
import numpy as np

def Convert_word2id(file, vocab2id):
    lines_id = []
    lines = []
    sentence_len = []
    for line in open(file):
        line = line.strip('\n')
        lines.append(line)
        line = line.split()
        sentence_len.append(len(line))

        for i, w in enumerate(line):
            try:
                word_id = vocab2id[w]
                line[i] = word_id
            except KeyError:  # if not in vocab
                line[i] = 1  # 1 = word id of UNK

        lines_id.append(np.array(line).astype(np.int32))

    sentence_len = np.array(sentence_len).astype(np.int32)

    return lines, lines_id,sentence_len

def Register_wordID(vocab_list):
    #vocab_list: list of vocabulary
    #return two dictionaries that convert word2id and vice versa

    assert type(vocab_list) == list

    vocab = ["</s>", "UNK"]
    vocab2id = {}
    id2vocab = {}
    vocab =vocab + vocab_list

    for word in vocab:
            vocab2id[word] = len(vocab2id)
            id2vocab[len(id2vocab)] = word

    assert vocab2id["</s>"] == 0
    assert vocab2id["UNK"] == 1

    return vocab2id, id2vocab

def Build_Vocab(train_file, vocab_size, freq_threshold):
    #build a list of vocab from train corpus
    #return: vocab list and vocab size

    with open(train_file) as f:
        wordfreq = Counter(chain.from_iterable(map(str.split, f)))
        wordfreq = OrderedDict(wordfreq.most_common()) #sort words by frequency

    if (vocab_size != None):
        print("vocab size is given: ",vocab_size) #extract top K most frequent words
        print("vocab freq min is ignored") #extract top K most frequent words

        if (len(wordfreq) < vocab_size):
            warnings.warn("Your Specified vocab size is larger than the total number of the vocabulary")
            vocab = list(wordfreq.keys())
        else:
            vocab = list(wordfreq.keys())[:vocab_size]

    else:  # if vocab size is not given, use threshold
        print("vocab min freq is given", freq_threshold)

        words = np.array(list(wordfreq.keys()))
        freq = np.array(list(wordfreq.values()))
        idx = freq >= freq_threshold
        vocab = words[idx].tolist()

    return vocab

class Dataset():

    def __init__(self):

        self.V_size = []

        self.lines = []
        self.lines_id = []
        self.lengths = []

        self.lines_dev = []
        self.lines_id_dev = []
        self.lengths_dev = []

    def load_data(self, train_file, dev_file, vocab2id):
        self.V_size.append(len(vocab2id))
        self.register_dataset(train_file, vocab2id, True)
        if (dev_file is not None):
            self.register_dataset(dev_file, vocab2id, False)

    def register_dataset(self, corpus, vocab2id, train_flag):
        lines, lines_id, sent_length = Convert_word2id(corpus, vocab2id)
        if(train_flag):
            self.lines.append(lines)
            self.lines_id.append(lines_id)
            self.lengths.append(sent_length)
        else:
            self.lines_dev.append(lines)
            self.lines_id_dev.append(lines_id)
            self.lengths_dev.append(sent_length)


    def add_EOS_BOS(self):

        for i in range(len(self.lines[1])):
            self.lines[1][i]   = np.append(np.append("<s>",self.lines[1][i]),"</s>")
            self.lines_id[1][i] = np.append(np.append([0],self.lines_id[1][i]),[0]).astype(np.int32)

        self.lengths[1] +=1

        if(self.lines_dev != []):
            for i in range(len(self.lines_dev[1])):
                self.lines_dev[1][i] = np.append(np.append("<s>", self.lines_dev[1][i]), "</s>")
                self.lines_id_dev[1][i] = np.append(np.append([0], self.lines_id_dev[1][i]), [0]).astype(np.int32)
            self.lengths_dev[1] += 1

class Vocab_dict():
    def __init__(self):

        self.vocab2id = []
        self.id2vocab = []

    def register_dict(self,vocab2_id,id2vocab):
        self.vocab2id.append(vocab2_id)
        self.id2vocab.append(id2vocab)
