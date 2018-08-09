import os
import pickle
import argparse
import chainer
from models.preprocessing import Convert_word2id
from tqdm import tqdm
from train import NMT  # Import Foo into main_module's namespace explicitly

# src_test=/cl/work/takashi-w/ASPEC-JE/test/test.txt.tok.low.ja
# model=default.encoder_decoder_epoch3
# python translate.py -model $model -src_test $src_test


parser = argparse.ArgumentParser()

parser.add_argument(
    '-gpuid',
    type=int,
    default=-1,
    help='gpuid; -1 means using cpu (default)')
parser.add_argument(
    '-src_test',
    type=str,
    required=True,
    help='source test data path')
parser.add_argument(
    '-model',
    type=str,
    required=True,
    help='model path')
parser.add_argument(
    '-beam_size',
    type=int,
    default=1,
    help='beam size: 1 means greedy decoding (default)')

parser.add_argument(
    '-normalize',
    action='store_true',
    help='normalize decoding probability by length')

parser.add_argument(
    '-out_attention_weight',
    action='store_true',
    help='output attention weight heatmap')

parser.add_argument(
    '-k',
    type=int,
    default=0,
    help='output additional txt file that lists top k translation candidates for each soruce sentence'
         '\n\n; set k larger than 0 to enable this option; (defalut: disabled)')


class Translator(chainer.Chain):
    def __init__(self, encoder, decoder, id2vocab, model):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.id2vocab = id2vocab
            self.model = model

    def translate_base(self, s_id, s_lengths, beam_size, normalize, *args):
        EOS_id = 0
        translation_best_list = []
        translation_all_list = []
        attention_weight_list = []
        for k in tqdm(range(len(s_id))):
            h_last, c_last, hs = self.encoder([s_id[k]], [s_lengths[k]])  # (bt * s_len * enc_size)
            translation_best, prob_best, translation, translation_prob, attention_p_out = self.decoder.translate(h_last, c_last, hs, beam_size, EOS_id,
                                                                            normalize, *args)
            translation_best_list.append(translation_best)
            translation_all_list.append(translation)
            attention_weight_list.append(attention_p_out[0])

        return translation_best_list,translation_all_list, attention_weight_list


    def translate(self, s_id, s_lengths, beam_size, normalize,
                    out_attention_weight, k, *args):

        translation_best_list, translation_all_list, attention_weight_list \
            = self.translate_base(s_id, s_lengths, beam_size, normalize,*args)

        #save outputs
        translation_word = translator.save_transation(translation_best_list)

        if (k):
            if k > beam_size:
                raise Exception("k must be smaller than beam size")
            print("save top k translation candidates")
            translator.save_topk_translations(translation_all_list, k)

        if (out_attention_weight):
            print("save attention weight heatmap")
            translator.save_attn_weight(s_id, translation_word, attention_weight_list)


    def save_transation(self, translation_list):
        translation_word = []
        tgt_id2vocab = self.id2vocab[1]
        for i in range(len(translation_list)):

            translation = translation_list[i]
            translation = " ".join([tgt_id2vocab[word_id] for word_id in translation])
            translation_word.append(translation)
            with open(self.model + ".translation.txt", "a") as output:
                output.write(translation + "\n")
        return translation_word

    def save_topk_translations(self, topk_translation, k):
        tgt_id2vocab = self.id2vocab[1]
        for i in range(len(topk_translation)):
            translation_list = topk_translation[i][0:k]
            for sentence in translation_list:
                translation = " ".join([tgt_id2vocab[word_id] for word_id in sentence])
                with open(self.model + ".translation_top"+str(k)+".txt", "a") as output:
                    output.write(translation + "\n")
            with open(self.model + ".translation_top"+str(k)+".txt", "a") as output:
                output.write('\n')

    def save_attn_weight(self, s_id, translation_word, attn_wight):

        src_id2vocab = self.id2vocab[0]
        for k in range(len(attn_wight)):
            attn_wight_tmp = attn_wight[k]  # t_len, s_len
            translation = translation_word[k].split() + ["<\s>"]
            attn_wight_tmp = round(attn_wight_tmp, 2)  # s_len * bt * 1 * 5 (= window +1)
            x_labels = [src_id2vocab[s_id[k][j]] for j in range(len(s_id[k]))]
            y_labels = translation
            plt.figure(figsize=(len(x_labels) * 0.2, len(y_labels) * 0.2))
            ax = sns.heatmap(attn_wight_tmp,
                             cbar=False,
                             vmin=0, vmax=1,
                             cmap="Reds")
            ax.set_xticklabels(x_labels, rotation=90)
            ax.set_yticklabels(y_labels, rotation=0)
            plt.tight_layout()
            plt.savefig(self.model+ ".attn_W" + str(k) + ".pdf")
            plt.close('all')

        merger = PdfFileMerger() #merge all pdfs into one file
        for k in range(len(s_id)):
            merger.append(open(self.model+ ".attn_W" + str(k) + ".pdf", 'rb'))
            os.remove(self.model+ ".attn_W" + str(k) + ".pdf")
        with open(self.model+ ".attn_W.pdf", 'wb') as fout:
            merger.write(fout)


if __name__ == '__main__':
    opt = parser.parse_args()
    if(opt.k):
        import matplotlib
        matplotlib.use('Agg')
        font = {'family': 'IPAexGothic'}
        matplotlib.rc('font', **font)
        import seaborn as sns
        import matplotlib.pyplot as plt
        from PyPDF2 import PdfFileMerger

    print("beam_size: ",opt.beam_size)
    print("normalize: ",opt.normalize)
    file = open(opt.model, 'rb')
    model = pickle.load(file)
    test_lines, test_lines_id, test_sent_length = Convert_word2id(opt.src_test, model.vocab2id[0])
    model.to_cpu()

    if opt.gpuid >= 0:
        chainer.cuda.get_device_from_id(opt.gpuid).use()
        to_gpu = chainer.cuda.to_gpu
        model.to_gpu()
        ####numpy->cupy####
        for i in range(len(test_lines_id)):
            test_lines_id[i] = to_gpu(test_lines_id[i])  # list of xp.array

    translator = Translator(model.encoder,model.decoder, model.id2vocab, opt.model)
    translator.translate(test_lines_id, test_sent_length, opt.beam_size, opt.normalize,
                        opt.out_attention_weight, opt.k)
