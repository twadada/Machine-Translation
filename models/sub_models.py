# -*- coding: utf-8 -*-
#!/usr/bin/env python

import chainer
import chainer.functions as F
import chainer.links as L


class Embed(chainer.Chain):
    def __init__(self, srcV, emb_size):
        super().__init__()
        with self.init_scope():
            self.init_param = chainer.initializers.Uniform(0.1)
            self.emb_s = L.EmbedID(srcV, emb_size, ignore_label=-1, initialW=self.init_param)

    def __call__(self, s_id, s_lengths):
        #assume s_id is sorted by length
        s_len = len(s_id[0])
        s_id_flattened = F.concat(s_id, axis=0)  # (bt * s_len) * 1
        s_id_emb = self.emb_s(s_id_flattened).reshape(len(s_id),s_len,-1)  # 全ての単語のword embを得る #(bt * s_len) * demb

        #s_id_emb = list(F.split_axis(s_id_emb, np.cumsum(s_lengths[:-1]), axis=0))  # bt * len_s *demb
        return s_id_emb # bt * len_s *demb


class BiLSTM_Concat(chainer.Chain):
    def __init__(self, n_layer, emb_size, enc_size, dr_rate):
        super().__init__()

        with self.init_scope():
            self.init_param = chainer.initializers.Uniform(0.1)
            self.encoder_fwd = L.NStepLSTM(n_layer, emb_size, enc_size, dr_rate, initialW=self.init_param,
                              initial_bias=self.init_param)
            self.encoder_bkw = L.NStepLSTM(n_layer, emb_size, enc_size, dr_rate, initialW=self.init_param,
                              initial_bias=self.init_param)

    def __call__(self, s_id_emb):
        s_id_emb = F.separate(s_id_emb,axis=0)
        s_id_emb_bkw = [s_id_emb[i][::-1] for i in range(len(s_id_emb))]

        h_last_fwd, c_last_fwd, hs_fwd = self.encoder_fwd(None, None, s_id_emb)  # input all the source words
        h_last_bkw, c_last_bkw, hs_bkw = self.encoder_bkw(None, None, s_id_emb_bkw)  # input all the source words

        hs_fwd = F.stack(hs_fwd)# assume that the length is sorted
        hs_bkw = F.stack(hs_bkw)# assume that the length is sorted
        h_last, c_last, hs = self.merge_lstm(h_last_fwd, h_last_bkw,c_last_fwd,c_last_bkw,hs_fwd, hs_bkw)
        return h_last, c_last, hs

    def merge_lstm(self,h_last_fwd, h_last_bkw,c_last_fwd,c_last_bkw,hs_fwd, hs_bkw):
        h_last = F.concat([h_last_fwd, h_last_bkw], axis=2)  # 論文だと、bkwのword idx 0 の和(feed型か)
        c_last = F.concat([c_last_fwd, c_last_bkw], axis=2)  # n_layer * bt * demb
        hs_bkw = F.swapaxes(hs_bkw, 0, 1)
        hs_bkw = F.swapaxes(hs_bkw[::-1], 0, 1)
        hs = F.concat([hs_fwd, hs_bkw], axis=2)
        return h_last,c_last,hs


class LSTM_scratch(chainer.Chain):
    def __init__(self,in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        super().__init__()
        with self.init_scope():
            self.init_param = chainer.initializers.Uniform(0.1)
            self.W_forget = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param)
            self.W_input = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param )
            self.W_output = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param)
            self.W_cell = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param )


    def __call__(self, x_in,init_states, *args):
        h_state, c_state, h_all_states = self.forward(x_in,init_states, *args)[0:3]
        return h_state, c_state, h_all_states

    def forward(self, x_in,init_states, *args):
        #x_in: list of Variable, bt, s_len, in_size
        #h_state, c_state:  bt, out_size
        h_state = init_states[0]
        c_state = init_states[1]
        x_in = F.stack(x_in)  # assume x_in is sorted; bt, s_len, in_size
        x_in = F.swapaxes(x_in, 0, 1)  # s_len, bt, in_size
        h_all_states = []
        forget_gate_all = []
        for i in range(len(x_in)): #for each time step
            h_state, c_state, forget_gate = self.f_LSTM(h_state,c_state,x_in[i],
                                              self.W_forget, self.W_input, self.W_output, self.W_cell)
            h_all_states.append(h_state)
            forget_gate_all.append(forget_gate) #s_len, bt, out_size

        h_state = h_state.reshape((1,) + h_state.shape) #h_state:1 * bt *out_size
        c_state = c_state.reshape((1,) + c_state.shape) #h_state:1 * bt *out_size
        ##c_state: bt *out_size
        ##h_states:
        h_all_states = F.stack(h_all_states) #s_len, bt, out_size
        h_all_states = F.swapaxes(h_all_states,0,1)#bt, s_len, out_size

        return h_state, c_state, h_all_states, forget_gate_all

    def f_LSTM(self,h_state,c_state,x_in_ith,
               W_forget,W_input,W_output,W_cell):

        h_x_concat = F.concat([x_in_ith, h_state], axis=1)  # bt, out_size+in_size
        forget_gate = F.sigmoid(W_forget(h_x_concat))# bt, out_size
        input_gate = F.sigmoid(W_input(h_x_concat))
        output_gate = F.sigmoid(W_output(h_x_concat))
        c_tmp = F.tanh(W_cell(h_x_concat))
        # update c_state and h_state
        c_state = forget_gate * c_state + input_gate * c_tmp  #
        h_state = output_gate * F.tanh(c_state)
        return h_state, c_state, forget_gate


class BiLSTM_scratch_Concat(LSTM_scratch):
    def __init__(self,in_size, out_size):
        super().__init__(in_size, out_size)
        with self.init_scope():
            self.W_forget_bkw = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param)
            self.W_input_bkw = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param )
            self.W_output_bkw = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param)
            self.W_cell_bkw = L.Linear(in_size+out_size, out_size, initialW=self.init_param, initial_bias=self.init_param )

    def forward(self, x_in,init_states, *args):
        #x_in: list of Variable, bt, s_len, in_size
        #h_state, c_state:  bt, out_size
        h_state = init_states[0]
        c_state= init_states[1]
        h_state_bkw = init_states[2]
        c_state_bkw = init_states[3]

        x_in = F.stack(x_in)  # assume x_in is sorted; bt, s_len, in_size
        x_in = F.swapaxes(x_in, 0, 1)  # s_len, bt, in_size
        h_all_states = []
        h_all_states_bkw = []
        forget_gate_all = []
        forget_gate_all_bkw = []
        for i in range(len(x_in)): #for each time step
            h_state, c_state, forget_gate = self.f_LSTM(h_state,c_state, x_in[i],
                                              self.W_forget, self.W_input, self.W_output, self.W_cell)
            h_state_bkw, c_state_bkw, forget_gate_bkw = self.f_LSTM(h_state_bkw,c_state_bkw,x_in[-i-1],
                                              self.W_forget_bkw, self.W_input_bkw, self.W_output_bkw, self.W_cell_bkw)
            h_all_states.append(h_state)
            h_all_states_bkw.append(h_state_bkw)
            forget_gate_all.append(forget_gate)
            forget_gate_all_bkw.append(forget_gate_bkw)

        h_state = h_state.reshape((1,) + h_state.shape) #h_state:1 * bt *out_size
        c_state = c_state.reshape((1,) + c_state.shape) #h_state:1 * bt *out_size
        h_state_bkw = h_state_bkw.reshape((1,) + h_state_bkw.shape) #h_state:1 * bt *out_size
        c_state_bkw = c_state_bkw.reshape((1,) + c_state_bkw.shape) #h_state:1 * bt *out_size
        ##c_state: bt *out_size
        ##h_states:
        h_all_states = F.stack(h_all_states) #s_len, bt, out_size
        h_all_states = F.swapaxes(h_all_states,0,1)#bt, s_len, out_size
        h_all_states_bkw = F.stack(h_all_states_bkw) #s_len, bt, out_size
        h_all_states_bkw = F.swapaxes(h_all_states_bkw[::-1],0,1)#bt, s_len, out_size
        h_last, c_last, hs = self.merge_fwdbkw(h_state, h_state_bkw,c_state,c_state_bkw,h_all_states, h_all_states_bkw)

        return h_last,c_last, hs,[forget_gate_all,forget_gate_all_bkw]

    def merge_fwdbkw(self,h_last_fwd, h_last_bkw,c_last_fwd,c_last_bkw,hs_fwd, hs_bkw):
        h_last = F.concat([h_last_fwd, h_last_bkw], axis=2)  # 論文だと、bkwのword idx 0 の和(feed型か)
        c_last = F.concat([c_last_fwd, c_last_bkw], axis=2)  # n_layer * bt * demb
        hs = F.concat([hs_fwd, hs_bkw], axis=2)
        return h_last,c_last,hs
