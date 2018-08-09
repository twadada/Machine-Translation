#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import chainer
from chainer import optimizers
import time
import os
from models.minibatch_processing import Sort_batch_by_srclen, Shuffle_train_data, Generate_bacth_idx

def Update_params_default(model, dataset, index):
    s_id = [dataset.lines_id[0][i] for i in index]  # list of xp.array
    s_lengths = dataset.lengths[0][index]  # np.array
    BOS_t_id = [dataset.lines_id[1][i][:-1] for i in index]  # list of xp.array
    t_id_EOS = [dataset.lines_id[1][i][1:]  for i in index]  # list of xp.array
    t_lengths = dataset.lengths[1][index]   # np.array
    softmax_score = model(s_id, s_lengths, BOS_t_id, t_lengths)
    loss = model.Calc_loss(t_id_EOS, softmax_score)
    return loss


def Calc_Dev_loss_default(model,dataset):
    cum_loss = 0
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for k in range(len(dataset.src_lines_id_DEV)):
            softmax_score = model([dataset.lines_id_dev[0][k]],
                              [dataset.lengths_dev[0][k]],
                              [dataset.lines_id_dev[1][k][:-1]],
                              [dataset.lengths_dev[1][k]])
            loss = model.Calc_loss([dataset.lines_id_dev[1][k][1:]], softmax_score)
            cum_loss += loss.data.tolist()
    return cum_loss


class Train():
    def __init__(self, dataset, opt, folder_name, file_subname, Update_params = None, Calc_Dev_loss = None):
        os.environ['PATH'] += ':/usr/local/cuda-7.5/bin:/usr/local/cuda-7.5/bin'  # CUDAに通す
        if (not os.path.isdir(folder_name[:-1])):
            os.mkdir(folder_name[:-1])

        self.dataset = dataset
        self.folder_name = folder_name
        self.file_subname = file_subname

        if(Update_params==None):
            self.Update_params = Update_params_default #default
        else:
            self.Update_params = Update_params

        if(Calc_Dev_loss==None):
            self.Calc_Dev_loss = Calc_Dev_loss_default #default
        else:
            self.Calc_Dev_loss = Calc_Dev_loss

        self.Entropy_old = np.inf
        self.Entropy_new = np.inf
        self.min_Entropy = np.inf
        self.max_epoch = 0
        self.max_model_name = None
        self.early_stopping_count = 0
        chainer.config.use_cudnn = "always"

    def save_models(self, model, model_name, epoch, remove_models):

        if (self.min_Entropy > self.Entropy_new):#save when entropy has decreased

            #serializers.save_npz(model_name + ".npz", model)#save model
            with open(model_name, mode='wb') as f:
                model.to_cpu()
                pickle.dump(model, f)
                f.close()
                model.to_gpu()

            if(remove_models and self.max_epoch != 0):
                print("remove the previous best model")
                os.remove(self.max_model_name)

            self.min_Entropy = self.Entropy_new
            self.max_epoch = epoch
            self.early_stopping_count = 0
            self.max_model_name = model_name

        else:
            if (not remove_models):  # keep every model
                with open(model_name, mode='wb') as f:
                    model.to_cpu()
                    pickle.dump(model, f)
                    f.close()
                    model.to_gpu()

                    #serializers.save_npz(model_name + ".npz", model)
            self.early_stopping_count += 1
            print("early stopping count:", self.early_stopping_count)


    def set_optimiser(self,opt_type, lr_rate = None,enable_decay = False, decay = 1.0, start_decay_at = -1):
        if opt_type == "SGD":
            if(lr_rate is None):
                lr_rate = 1.0
            self.enable_decay = enable_decay
            self.decay = decay
            self.start_decay_at = start_decay_at
            self.optimizer = optimizers.SGD(lr_rate)

        elif opt_type == "Adam":
            if (lr_rate is None):
                lr_rate = 0.001
            print("disable lr decay")
            self.enable_decay = False
            self.optimizer = optimizers.Adam(alpha=lr_rate)

        elif opt_type == "AdaDelta":
            if (lr_rate is None):
                lr_rate = 0.95
            print("disable lr decay")
            self.enable_decay = False
            self.optimizer = optimizers.AdaDelta(rho = lr_rate)
        else:
            raise Exception("Invalid optimizer type" + opt_type)

    def main(self,model,epoch_size, batch_size,
             shuffle_batch_idx = False,
             early_stopping = False,remove_models =False):

        chainer.config.train = True
        self.optimizer.setup(model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))
        #modelを構築
        self.dataset = Shuffle_train_data(self.dataset)
        self.dataset = Sort_batch_by_srclen(self.dataset)
        batch_idx_list = Generate_bacth_idx(self.dataset, batch_size)
        batch_idx_list = np.random.permutation(batch_idx_list)      #shuffle batch order

        use_dev_data = self.dataset.lines_id_dev != [] #check whether dataset contains dev data or not

        print ("epoch start")
        for epoch in range(1, epoch_size+1): #for each epoch
            print("epoch: ",epoch)
            if (self.enable_decay and ((self.Entropy_old < self.Entropy_new) or (epoch >= self.start_decay_at))):
                print("update SGD lr")
                self.optimizer.lr = self.optimizer.lr * self.decay

            self.Entropy_old = self.Entropy_new
            cumloss = 0

            if(shuffle_batch_idx):
                batch_idx_list = Generate_bacth_idx(self.dataset, batch_size)  #shuffle batch index

            batch_idx_list = np.random.permutation(batch_idx_list) #shuffle batch order
            start = time.time()
            step = 0
            for bt_idx in batch_idx_list:
                if (step == int(len(batch_idx_list)/2)):
                    print("half way through")
                step += 1
                model.cleargrads()
                loss = self.Update_params(model, self.dataset, bt_idx)
                loss.backward()
                self.optimizer.update()
                cumloss = cumloss + loss.data.tolist()

            #end of epoch
            cumloss = cumloss/len(batch_idx_list)
            elapsed_time = time.time() - start
            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
            print("loss: ",cumloss)
            print(self.folder_name +  self.file_subname+"_epoch" + str(epoch))

            if(use_dev_data):
                self.Entropy_new = self.Calc_Dev_loss(model, self.dataset)
                self.Entropy_new = self.Entropy_new / float(len(self.dataset.src_lines_id_DEV))
                print("cross entropy", self.Entropy_old, self.Entropy_new)
                model_name = self.folder_name + self.file_subname + \
                             "_epoch" + str(epoch) + "_ppl_" + str(round(self.Entropy_new, 3))

                self.save_models(model, model_name, epoch, remove_models)
            else:
                model_name = self.folder_name + self.file_subname + "_epoch" + str(epoch)

                with open(model_name, mode='wb') as f:
                    model.to_cpu()
                    pickle.dump(model, f)
                    f.close()
                    model.to_gpu()

                    # serializers.save_npz(model_name + ".npz", model)

            if(early_stopping and self.early_stopping_count==3 ):
                break

        print("finish training")
        if (use_dev_data):
            file = open(self.max_model_name, 'rb')
            model = pickle.load(file)
            with open(self.max_model_name + "_best_model", mode='wb') as f:
                model.to_cpu()
                pickle.dump(model, f)
                f.close()
                model.to_gpu()

            if (remove_models):
                os.remove(self.max_model_name)

