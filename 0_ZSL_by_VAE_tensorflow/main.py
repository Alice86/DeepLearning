#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE model training
    Run this file for VZSL training
    if continue from a pretrained checkpoint: rename the file under checkpoint as "data_path_feature_path"
"""

import os
import scipy.misc
import numpy as np
import tensorflow as tf

from model_VZSL import VAE
from utils import *
import argparse

parser = argparse.ArgumentParser(conflict_handler='resolve')

parser.add_argument("--epoch", type=int, default=120, help="Epoch to train [100]")
parser.add_argument("--learning_rate", type=float, default=6e-4, help="Uper bound of learning rate for adam [0.0005]")
parser.add_argument("--beta1", type=float, default=0.3, help="Momentum term of adam [0.5]")
parser.add_argument("--hidden_num", type=int, default=1000, help="Num. of nodes for encoder/decoder MLP [1000]")
parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout parameter to avoid overfitting [0.8]")

parser.add_argument("--lamb", type=float, default=1, help="Parameter lambda controlling the extent of regularization [1]")
parser.add_argument("--M", type=int, default=85, help="Num. of attributrs [85]")
parser.add_argument("--S", type=int, default=40, help="Num. of seen classes [40]")
parser.add_argument("--U", type=int, default=10, help="Num. of unseen classes [10]")
parser.add_argument("--z_dim", type=int, default=100, help="Dimension of the latent [100]")
parser.add_argument("--x_dim", type=int, default=2048, help="Dimension of feature vector. [2048]")

parser.add_argument("--batch_size", type=int, default=256, help="The size of training mini-batch [256]")
parser.add_argument("--test_size", type=int, default=7913, help="The size of testing samples [6985][7913]")

parser.add_argument("--data_path", type=str, default="AWA2", help="Directory name of dataset AWA2]")
parser.add_argument("--feature_path", type=str, default="ResNet_feature", help="Directory name of feature [ResNet_featuree]")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")

parser.add_argument("--is_train", type=bool, default=True, help="True for raining, False for testing [False]")

config = parser.parse_args()

def main(_):

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    with tf.Session() as sess:
        
        vae = VAE(sess, config)

        if config.is_train:
            vae.train(config)
        else:
            vae.load(config)


if __name__ == '__main__':
    tf.app.run()