# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import chainer
import warnings
import chainer.functions as F
import chainer.links as L
import models.sub_models as sub_models

def create_decoder(decoder_name,*args):
    if decoder_name == 'decoder':
        decoder = Decoder(*args)
    else:
        raise Exception(decoder_name + " not defined")
    return decoder

def create_encoder(encoder_name,*args):
    if encoder_name == 'encoder':
        encoder = Encoder_BiLSTM_Concat(*args)
    else:
        raise Exception(encoder_name + " not defined")
    return encoder

class Encoder_Base(chainer.ChainList):
    def __init__(self):
        super().__init__()

    def __call__(self, s_id, s_lengths):

        emb = self[0](s_id, s_lengths)
        h_last, c_last, hs = self[1](emb)
        return h_last, c_last, hs

class Encoder_BiLSTM_Concat(Encoder_Base):
    def __init__(self, srcV, n_layer, emb_size, enc_size, dr_rate, *args):
        super().__init__()
        self.add_link(sub_models.Embed(srcV, emb_size))
        self.add_link(sub_models.BiLSTM_Concat(n_layer, emb_size, int(enc_size/2), dr_rate))

class Decoder(chainer.Chain):

    def __init__(self, tgtV, n_layer, emb_size, dec_size, dr_rate, *args):
        super().__init__()
        self.dr_rate = dr_rate
        self.tgtV = tgtV
        with self.init_scope():
            self.init_param = chainer.initializers.Uniform(0.1)
            self.W = L.Linear(dec_size, dec_size, initialW=self.init_param, nobias=True)  # no bias when calcuating attention score
            self.Wc = L.Linear(2 * dec_size, dec_size, initialW=self.init_param, nobias=True)
            self.Ws = L.Linear(dec_size, tgtV, initialW=self.init_param, initial_bias=self.init_param)
            self.emb_t = L.EmbedID(tgtV, emb_size, ignore_label=-1, initialW=self.init_param)
            self.decoder_lstm = L.NStepLSTM(n_layer, emb_size , dec_size, dr_rate, initialW=self.init_param,
                                  initial_bias=self.init_param)

    def __call__(self, h_last,c_last, hs_padded, BOS_t_id, t_lengths, *args):
        return self.forward( h_last,c_last, hs_padded, BOS_t_id, t_lengths, *args)

    def forward(self, h_last,c_last, hs_padded, BOS_t_id, t_lengths, *args):
        '''    
       :param h_last: (n_layer * bt * dec_size)
       :param c_last: (n_layer * bt * dec_size)
       :param hs:     (bt * s_len * dec_size), list of Var
       :param context_vector: ((bt * maxlen_t) * dec_size))
       :return:softmax_score: ((bt * maxlen_t) * tgtV)
       #args not used
        '''
        h_last, c_last, ht, output_vector, attention_p = self.decode(h_last, c_last, hs_padded,BOS_t_id, t_lengths,args)
        softmax_score = self.Ws(F.dropout(output_vector, ratio=self.dr_rate))  # ((bt * maxlen_t) * tgtV)
        return h_last, c_last, ht, softmax_score

    def embed(self,BOS_t_id,*args):
        t_id_flattened = F.concat(BOS_t_id, axis=0)  # (bt * len,)
        t_id_emb = self.emb_t(t_id_flattened)  # 全ての単語のword embを得る #(bt * max_s_len) * demb
        return t_id_emb

    def decode(self,h_last, c_last, hs_padded, BOS_t_id, t_lengths ,*args):

        t_id_emb = self.embed(BOS_t_id)
        t_id_emb = list(F.split_axis(t_id_emb, np.cumsum(t_lengths[:-1]), axis=0))  # bt * max_s_len * demb

        h_last, c_last, ht = self.decoder_lstm(h_last, c_last, t_id_emb)  # ht: bt * len_t * demb(最上位層の出力 from each hidden state)
        ht_padded = F.pad_sequence(ht, padding=0)
        context_vector, attention_p = self.attention(hs_padded, ht_padded)
        context_vector_ht = F.concat([ht_padded, context_vector], axis=2)  # (bt * maxlen_t * 2dec_size)
        output_vector = F.tanh(self.Wc(F.concat(context_vector_ht, axis=0)))  # ((bt * maxlen_t) * dec_size)
        return h_last, c_last, ht, output_vector, attention_p.data

    def attention(self,hs_padded, ht_padded):
        ht_padded_W = self.W(F.concat(ht_padded, axis=0)).reshape(ht_padded.shape)  # bt * maxlen_t * demb
        hs_swap = F.swapaxes(hs_padded, 1, 2)  # bt *  demb * maxlen_s
        attn_matix = F.matmul(ht_padded_W, hs_swap)  # bt *  maxlen_t * maxlen_s
        attn_matix_sm = F.softmax(attn_matix, axis=2)  # bt *  maxlen_t * maxlen_s
        context_vector = F.matmul(attn_matix_sm, hs_padded)  # (bt *  maxlen_t * maxlen_s) * (#bt * maxlen_s * demb) = bt * maxlen_t * demb

        return context_vector, attn_matix_sm

    def broad_cast2beam_size(self, hs, beam_size, *args):

        hs = F.broadcast_to(hs,
                            (beam_size, len(hs[0]), len(hs[0][0])))  # beam_size * len_s * demb (train)

        return hs, args


    def translate(self,h_last, c_last, hs, beam_size, EOS_id, normalize, first_word = 0, *args):

        """ 
        :param h_last: (1 * n_layer * enc_size), Var
        :param c_last: (1 * n_layer * enc_size), Var
        :param hs:     (1 * s_len * enc_size),   Var
        :return: translation_best: (1 * t_len),  list
        :return: prob_best: float
        :return: translation: (beam_size * t_len), list of lists
        :return: translation_prob: (beam_size, ),  list of float
        """

        h_last, c_last, next_word_id, word_cumprob, previous_words, attention_p=\
            self.decode_first_beam(h_last, c_last, hs, beam_size, first_word, *args)

        # attention_p: 1 *  1 * s_len (bt *  maxlen_t * maxlen_s)

        translation = []
        translation_prob = []
        attention_p_list = []
        if (EOS_id in next_word_id):  # if one of the first words (tgV) is end_token
            idx = next_word_id.index(EOS_id)
            translation.append([])  # Translation = None
            translation_prob.append(word_cumprob[idx])  # P(EOS|BOS)
            #translation_prob.append(-np.inf)  # P(EOS|BOS): the sentences that start with EOS
            attention_p_list.append(attention_p[0])  # P(EOS|BOS)
            # no need to update h_last and c_last
            del next_word_id[idx]
            del previous_words[idx]
            del word_cumprob[idx]
            beam_size = beam_size - 1

        return self.decode_one_by_one(h_last, c_last, hs, beam_size, EOS_id,
                                      next_word_id, word_cumprob, previous_words,
                                      translation,translation_prob, normalize,
                                      attention_p,attention_p_list, args)

    def decode_first_beam(self, h_last, c_last, hs, beam_size_first, first_id, *args):
        xp = chainer.cuda.get_array_module(h_last)
        first_word_id = xp.array([first_id], dtype=xp.int32).reshape(1, 1)  # <BOS>, 1
        h_last, c_last, ht, output_vector ,attention_p = self.decode(h_last, c_last, hs, first_word_id, [1], *args)

        # ht = values[2]
        # attention_p: 1 *  1 * s_len (bt *  maxlen_t * maxlen_s)

        softmax_score = self.Ws(F.dropout(output_vector, ratio=self.dr_rate))  # ((bt * maxlen_t) * tgtV)
        # print("softmax_score: , ", xp.sort(softmax_score.reshape(-1).data[0:30]))

        softmax_prob = F.log(F.softmax(softmax_score, axis=1)).reshape(-1).data  # (eNv, arrray)

        next_word_id = xp.argsort(softmax_prob).astype(xp.int32)[::-1][0:beam_size_first]
        word_cumprob = softmax_prob[next_word_id]
        previous_words = [[next_word_id.tolist()[i]] for i in range(beam_size_first)]

        next_word_id = next_word_id.tolist()
        word_cumprob = word_cumprob.tolist()
        attention_p = attention_p.tolist()

        return h_last, c_last, next_word_id, word_cumprob, previous_words, attention_p


    def decode_one_by_one(self,h_last, c_last, hs, beam_size, EOS_id,
                          next_word_id, word_cumprob, previous_words,
                          translation,translation_prob, normalize,
                          attention_p_cum, attention_p_out, *args):

        h_last = F.broadcast_to(h_last,
                                    (len(h_last), beam_size,
                                     len(h_last[0][0])))  # N_layer * beam_size * demb (train)
        c_last = F.broadcast_to(c_last,
                               (len(c_last), beam_size,
                                len(c_last[0][0])))  # N_layer * beam_size * demb (train)

        #attention_p: 1 *  1 * s_len (bt *  maxlen_t * maxlen_s)
        xp = chainer.cuda.get_array_module(h_last)
        attention_p_cum = xp.array(attention_p_cum).astype(xp.float32)

        attention_p_cum = F.broadcast_to(attention_p_cum,
                               (beam_size, 1, len(attention_p_cum[0][0])))

        #attention_p: bt *  1 * s_len (bt *  maxlen_t * maxlen_s)

        if beam_size > 0:

            for repeat in range(120):
                next_word_id = xp.array(next_word_id).astype(xp.int32).reshape(-1, 1)
                word_cumprob = xp.array(word_cumprob).astype(xp.float32)

                hs_tmp, args_tmp = self.broad_cast2beam_size(hs, beam_size, *args)

                h_last_tmp, c_last_tmp, ht, output_vector, attention_p = self.decode(h_last, c_last, hs_tmp, next_word_id,
                                                                        [1] * beam_size, args_tmp)
                # ht = values[2]
                # attention_p: bt *  1 * s_len  (bt *  maxlen_t * maxlen_s)


                softmax_score = self.Ws(F.dropout(output_vector, ratio=self.dr_rate))  # ((bt * maxlen_t) * tgtV)
                # h_last = Nlayer * beam_size * demb
                softmax_prob = F.log(F.softmax(softmax_score, axis=1))  # (beam_size, eNv)


                softmax_cumprob = F.broadcast_to(word_cumprob.reshape(beam_size, 1),
                                                 softmax_prob.shape)  # (beam,1) -> (beam, tgV)
                softmax_cumprob += softmax_prob  # (beam_size, tgtV)
                next_word_id = xp.argsort(softmax_cumprob.reshape(-1).data).astype(xp.int32)[::-1][0:beam_size]  # (beam_size,)
                word_cumprob = softmax_cumprob.reshape(-1).data[next_word_id]  # (beam_size, )

                h_last_tmp = F.swapaxes(h_last_tmp, 0, 1)  # beam_size * N_layer * demb
                c_last_tmp = F.swapaxes(c_last_tmp, 0, 1)  # beam_size * N_layer * demb

                previous_index = (next_word_id // self.tgtV).tolist()
                h_last = [h_last_tmp[i] for i in previous_index]  # beam_size * N_layer * demb
                c_last = [c_last_tmp[i] for i in previous_index]  # beam_size * N_layer * demb
                previous_words_tmp = [previous_words[i] for i in previous_index]  # (beamsize ,)
                attention_p_tmp = F.stack([attention_p[i] for i in previous_index]) # (beamsize ,)
                # attention_p: bt *  1 * s_len  (bt *  maxlen_t * maxlen_s)

                attention_p_cum = F.concat([attention_p_cum,attention_p_tmp],axis= 1)

                # attention_p: bt *  t_len * s_len  (bt *  maxlen_t * maxlen_s)

                next_word_id = next_word_id % self.tgtV  # (beam_size, )
                previous_words = [previous_words_tmp[i] + [next_word_id.tolist()[i]] for i in range(beam_size)]

                eos_idx = np.where((next_word_id == EOS_id).tolist())[0]  # extract item from array

                next_word_id = next_word_id.tolist()
                word_cumprob = word_cumprob.tolist()  # (beam_size, )
                attention_p_cum = attention_p_cum.data.tolist()  # bt *  t_len * s_len

                for idx in sorted(eos_idx, reverse=True):
                    translation.append(previous_words[idx][:-1])  # beam_size * N_layer * demb
                    attention_p_out.append(attention_p_cum[idx])
                    if(normalize):
                        warnings.warn('normalizing')
                        translation_prob.append(word_cumprob[idx] / len(previous_words[idx]))
                    else:
                        translation_prob.append(word_cumprob[idx])

                    del next_word_id[idx]
                    del h_last[idx]
                    del c_last[idx]
                    del previous_words[idx]
                    del word_cumprob[idx]
                    del attention_p_cum[idx]



                beam_size = beam_size - len(eos_idx)
                hs = hs[:beam_size]  # reduce dimention

                if (beam_size == 0):  # 全ての文で一回以上eosが出た場合
                    break

                attention_p_cum = xp.array(attention_p_cum).astype(xp.float32)
                # next_word_id = xp.array(next_word_id).astype(xp.int32).reshape(-1, 1)
                # word_cumprob = xp.array(word_cumprob).astype(xp.float32)
                h_last = F.stack(h_last)  # beam_size * N_layer * demb
                c_last = F.stack(c_last)  # beam_size * N_layer * demb
                h_last = F.swapaxes(h_last, 0, 1)  # N_layer * beam_size * demb
                c_last = F.swapaxes(c_last, 0, 1)  # N_layer * beam_size * demb

                # end for
            if (beam_size != 0):
                # word_cumprob = word_cumprob.tolist()  # (beam_size, )
                attention_p_cum = attention_p_cum.tolist()  # bt *  t_len * s_len

                for idx in range(beam_size):  # idx = index for sentences longer than 120 words
                    translation.append(previous_words[idx])
                    attention_p_out.append(attention_p_cum[idx])
                    if (normalize):
                        translation_prob.append(word_cumprob[idx] / len(previous_words[idx]))
                    else:
                        translation_prob.append(word_cumprob[idx])

        translation_best = translation[np.argmax(translation_prob)]
        prob_best = np.max(translation_prob)
        idx = np.argsort(translation_prob)[::-1]  # decreasing order
        translation = [translation[i] for i in idx]
        attention_p_out = [attention_p_out[i] for i in idx]
        translation_prob = [translation_prob[i] for i in idx]

        return translation_best, prob_best, translation, translation_prob, attention_p_out
