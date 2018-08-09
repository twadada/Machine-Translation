import argparse
import numpy as np
import warnings

def Read_global_options(opt):

    if(opt.enc_size != opt.dec_size and opt.enc_size != 2 * opt.dec_size):
        raise Exception("encoder size must be the same size of decoder")

    #print opt data
    print("src_emb_size", opt.src_emb_size)
    print("tgt_emb_size", opt.tgt_emb_size)
    print("enc_hidden_size", opt.enc_size)
    print("dec_hidden_size", opt.dec_size)

    if(opt.src_emb_size!=opt.tgt_emb_size):
        warnings.warn("src_embsize and tgt_embsize are different!")


global_train_parser = argparse.ArgumentParser(add_help=False)

global_train_parser.add_argument(
    '-gpuid',
    default=-1,
    type=int,
    help='GPU ID')

global_train_parser.add_argument(
        '-dr_rate',
        default=0.3,
        type=float,
        help='dropout rate applied to decoder')

global_train_parser.add_argument(
        '-n_layer',
        default=1,
        type=int,
        help='number of layers')

global_train_parser.add_argument(
        '-src_emb_size',
        default=512,
        type=int,
        help='source embedding size')
global_train_parser.add_argument(
        '-tgt_emb_size',
        default=512,
        type=int,
        help='target embedding size')
global_train_parser.add_argument(
        '-enc_size',
        default=512,
        type=int,
        help='encoder hidden state size')
global_train_parser.add_argument(
        '-dec_size',
        default=512,
        type=int,
        help='decoder hidden state size')
global_train_parser.add_argument(
        '-batch_size',
        default=64,
        type=int,
        help='batch size')

global_train_parser.add_argument(
    '-epoch_size',
    default=10,
    type=int,
    help='epoch size')

global_train_parser.add_argument(
    '-decoder',
    default="decoder",
    type=str,
    help='decoder model name')

global_train_parser.add_argument(
    '-encoder',
    default="encoder",
    type=str,
    help='encoder model name')

global_train_parser.add_argument(
    '-data_name',
    type=str,
    help='data name')

global_train_parser.add_argument(
    '-opt_type',
    default="SGD",
    choices=['SGD', 'Adam', 'AdaDelta'],
    help='optimizer (SGD, Adam, or AdaDelta)')

global_train_parser.add_argument(
    '-learning_rate',
    default=None,
    type=float,
    help='SGD learning rate')

global_train_parser.add_argument(
    '-enable_decay',
    action='store_true',
    help='enable SGD learning rate decay')

global_train_parser.add_argument(
    '-decay',
    default=0.7,
    type=float,
    help='SGD learning rate decay rate')

global_train_parser.add_argument(
    '-start_decay_at',
    default=np.inf,
    type=int,
    help='which epoch to start decay at (defalut:inf)')

global_train_parser.add_argument(
    '-shuffle_batch_idx',
    action='store_true',
    help='create a new mini batch at every epoch')

global_train_parser.add_argument(
    '-early_stopping',
    action='store_true',
help='enable early stopping: stop training when the best perplexiy on development data has not been updated 3 epochs consequtively')

global_train_parser.add_argument(
    '-remove_models',
    action='store_true',
    help='only keep the model that has achieved the best perplexity on the development data')

global_train_parser.add_argument(
    '-seed',
    default=0,
    type=int,
    help='random seed')

