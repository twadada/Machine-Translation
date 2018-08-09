#!/usr/bin/env python
# -*- coding: utf-8 -*-
#python train.py -data_name default -epoch_size 7 -opt_type Adam -remove_models -gpuid 1
import chainer
from chainer import cuda
import chainer.functions as F
from train_base import  Train
from models.NMT_models import  create_encoder, create_decoder,np
import pickle
import argparse
from train_option import global_train_parser, Read_global_options


class NMT(chainer.Chain):

    def __init__(self, encoder, decoder):
        super().__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder



    def __call__(self, s_id, s_lengths, BOS_t_id, t_lengths, *args):
        h_last, c_last, hs = self.encoder(s_id, s_lengths)
        _,_,_, softmax_score = self.decoder(h_last,c_last, hs, BOS_t_id, t_lengths, *args)
        return softmax_score

    def Calc_loss(self,t_id_EOS, softmax_score,*args):
        batch_size = len(t_id_EOS)
        t_id_EOS = F.pad_sequence(t_id_EOS, padding=-1).reshape(-1) #(bt * max_len) * 1
        loss = F.sum(F.softmax_cross_entropy(softmax_score, t_id_EOS, reduce="no", ignore_label=-1))/batch_size
        return loss

    def Register_vocab(self,vocab2id, id2vocab):
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[global_train_parser])
    opt = parser.parse_args()
    Read_global_options(opt)
    np.random.seed(opt.seed)

    file = open("data/" + opt.data_name + ".data", 'rb')
    dataset = pickle.load(file)

    file = open("data/" + opt.data_name + ".vocab_dict", 'rb')
    vocab_dict = pickle.load(file)

    use_gpu = opt.gpuid >= 0

    if use_gpu:
        xp = cuda.cupy
        chainer.cuda.get_device_from_id(opt.gpuid).use()
        to_gpu = chainer.cuda.to_gpu
        ####numpy->cupy####
        for i in range(len(dataset.lines_id[0])):
            dataset.lines_id[0][i] = to_gpu(dataset.lines_id[0][i])  # list of xp.array
            dataset.lines_id[1][i] = to_gpu(dataset.lines_id[1][i])  # list of xp.array

    else:
        xp = np

    chainer.config.train = True
    folder_name = "Results/"

    file_subname =  opt.data_name + "." +opt.encoder+"_"+opt.decoder

    print("Save model as: ",file_subname)

    encoder = create_encoder(opt.encoder, dataset.V_size[0], opt.n_layer, opt.src_emb_size, opt.enc_size, opt.dr_rate)
    decoder = create_decoder(opt.decoder, dataset.V_size[1], opt.n_layer, opt.tgt_emb_size, opt.dec_size, opt.dr_rate)

    model = NMT(encoder, decoder)
    model.Register_vocab(vocab_dict.vocab2id, vocab_dict.id2vocab)

    if use_gpu:
        model.to_gpu()

    train = Train(dataset, opt, folder_name, file_subname)
    train.set_optimiser(opt.opt_type, opt.learning_rate, opt.enable_decay, opt.decay, opt.start_decay_at)
    train.main(model,opt.epoch_size, opt.batch_size,
             opt.shuffle_batch_idx, opt.early_stopping, opt.remove_models)
