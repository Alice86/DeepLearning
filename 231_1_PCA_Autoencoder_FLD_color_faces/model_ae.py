#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-encoding model
"""

import tensorflow as tf
import cv2
import numpy as np
from datetime import datetime
import math
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from utils import *
from mywarper import *

# log = lambda *args: print(datetime.now().strftime('%H:%M:%S'), ':', *args)
# def select_device(use_gpu=True):
#     from tensorflow.python.client import device_lib
#     log(device_lib.list_local_devices())
#     device = '/device:GPU:0' if use_gpu else '/CPU:0'
#     log('Using device: ', device)
#     return device
# device = select_device(use_gpu=False)

class AE(object):
    def __init__(self, sess, l_dim=10, z_dim=50, x_dim=3, batch_size=100, test_size=200, image_size=128, keep_prob=0.8, checkpoint_dir='check_point'):
        """
        Args for model:
            sess: TensorFlow session
            config:
        """
        self.sess = sess
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.l_dim = l_dim
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.keep_prob = keep_prob
        
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.build_model()
        
    def encoder_lms(self, x, reuse=False, train=True):
        with tf.variable_scope("encoder_lms", reuse=reuse) as scope:
          
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer(1.0)
            b_init = tf.zeros_initializer()

            x_flat = tf.reshape(x, [-1, 68*2])
            w0 = tf.get_variable('w0', [68*2, 100], initializer=w_init)
            b0 = tf.get_variable('b0', [100], initializer=b_init)
            h0 = tf.matmul(x_flat, w0) + b0
            h0 = tf.nn.leaky_relu(h0)
            #h0 = tf.nn.dropout(h0, self.keep_prob)

            w1 = tf.get_variable('w1', [100, self.l_dim], initializer=w_init)
            b1 = tf.get_variable('b1', [self.l_dim], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.leaky_relu(h1)
            #h1 = tf.nn.dropout(h1, self.keep_prob)

            #wo = tf.get_variable('wo', [10, self.l_dim], initializer=w_init)
            #bo = tf.get_variable('bo', [self.l_dim], initializer=b_init)
            #z = tf.matmul(h1, wo) + bo           
        
        return h1

    def encoder_img(self, x, reuse=False, train=True):
        with tf.variable_scope("encoder_img", reuse=reuse) as scope:

            h0 = tf.contrib.layers.conv2d(x, 16, 5, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            h1 = tf.contrib.layers.conv2d(h0, 32, 3, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            #h1 = tf.nn.dropout(h1, self.keep_prob)

            h2 = tf.contrib.layers.conv2d(h1, 64, 3, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            #h2 = tf.nn.dropout(h2, self.keep_prob)

            h3 = tf.contrib.layers.conv2d(h2, 128, 3, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            #h3 = tf.nn.dropout(h3, self.keep_prob)
            
            h_flat = tf.contrib.layers.flatten(h3)            
            #hf = tf.contrib.layers.fully_connected(h_flat, 50, activation_fn=tf.nn.leaky_relu)

            z = tf.contrib.layers.fully_connected(h_flat, self.z_dim, activation_fn=tf.nn.leaky_relu)
            
        return z

    def decoder_lms(self, z, reuse=False, train=True):   
        with tf.variable_scope("decoder_lms", reuse=reuse) as scope:

            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer(1.0)
            b_init = tf.zeros_initializer()
            
            w0 = tf.get_variable('w0', [self.l_dim, 100], initializer=w_init)
            b0 = tf.get_variable('b0', [100], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.leaky_relu(h0)
            #h0 = tf.nn.dropout(h0, self.keep_prob)

            w1 = tf.get_variable('w1', [100, 68*2], initializer=w_init)
            b1 = tf.get_variable('b1', [68*2], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.sigmoid(h1)
            #h1 = tf.nn.dropout(h1, self.keep_prob)
            
            x = tf.reshape(h1, [-1, 68, 2])
            
        return x

    def decoder_img(self, z, reuse=False, train=True):   
        with tf.variable_scope("decoder_img", reuse=reuse) as scope:
            size = z.get_shape()[0]
            
            hf = tf.contrib.layers.fully_connected(z, 128*8*8, activation_fn=tf.nn.leaky_relu)
            h0 = tf.reshape(hf, [-1, 8, 8, 128])        
            #z_patch = tf.tile(tf.reshape(z, [-1, 1, 1, self.z_dim]), [1,8,8,1])       
            
            h1 = tf.contrib.layers.conv2d_transpose(h0, 64, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu) 
            #h1 = tf.nn.dropout(h1, self.keep_prob)

            h2 = tf.contrib.layers.conv2d_transpose(h1, 32, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu) 
            #h2 = tf.nn.dropout(h2, self.keep_prob)

            h3 = tf.contrib.layers.conv2d_transpose(h2, 16, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu)  
            #h3 = tf.nn.dropout(h3, self.keep_prob)

            x = tf.contrib.layers.conv2d_transpose(h3, self.x_dim, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
            
        return x
    
    def build_model(self):

        '''Placeholders: train/test, latent sample'''
        self.img = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.x_dim], name="image")
        self.lms = tf.placeholder(tf.float32, [None, 68, 2], name="landmark")
        self.sample_img = tf.placeholder(tf.float32, [None, self.z_dim], name="sample_img")
        self.sample_lms = tf.placeholder(tf.float32, [None, self.l_dim], name="sample_lms")
        
        '''Training'''
        # Landmark
        latent_lms = self.encoder_lms(self.lms, reuse=False, train=True) 
        recon_lms = self.decoder_lms(latent_lms, reuse=False, train=True) 
        self.lms_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(recon_lms-self.lms), axis=2)))        

        # Image
        latent_img = self.encoder_img(self.img, reuse=False, train=True) 
        recon_img = self.decoder_img(latent_img, reuse=False, train=True) 
        self.img_loss = tf.reduce_mean(tf.square(recon_img-self.img)) 

        '''Testing'''
        self.latent_lms = self.encoder_lms(self.lms, reuse=True, train=False) 
        self.recon_lms = self.decoder_lms(self.latent_lms, reuse=True, train=False) 
        
        self.latent_img = self.encoder_img(self.img, reuse=True, train=False) 
        self.recon_img = self.decoder_img(self.latent_img, reuse=True, train=False) 

        self.lms_err =  tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.recon_lms-self.lms), axis=2)))
        self.img_err =  tf.reduce_mean(tf.square(self.recon_img-self.img)) 
        
        '''Sample'''
        self.sampled_lms = self.decoder_lms(self.sample_lms, reuse=True, train=False) 
        self.sampled_img = self.decoder_img(self.sample_img, reuse=True, train=False)        
        
        self.saver = tf.train.Saver() 
        
    def train_lms(self, num_epoch=150, learning_rate=5e-3):
       
        '''Optimizer (adam)'''
        self.optim_lms = tf.train.AdamOptimizer(learning_rate).minimize(self.lms_loss)

        '''Load data'''
        landmarks = np.load("/content/drive/My Drive/landmarks.npy")
        # normalize
        landmarks = landmarks/128
        print('Landmark shape: {}'.format(np.shape(landmarks)))

        np.random.seed(1)
        test_idx = np.random.choice(landmarks.shape[0], self.test_size, replace=False)
        self.test_idx = test_idx
        train_idx = [i for i in range(landmarks.shape[0]) if i not in test_idx]
        test_lms = landmarks[test_idx,:]
        train_lms = landmarks[train_idx,:]
        self.mean_lms = train_lms.mean(0)
        
        '''Initialize variables'''
        self.sess.run(tf.global_variables_initializer())
        
        '''Training loop'''
        counter = 0

        # Training iterations
        print("Start training")
        for epoch in range(num_epoch):
            batch_num = (landmarks.shape[0]-self.test_size) // self.batch_size
            
            for batch in range(batch_num):
                counter += 1
                
                # mini-batch data
                batch_idx = train_idx[batch*self.batch_size:(batch+1)*self.batch_size]
                batch_lms = landmarks[batch_idx]
                
                _, lms_loss = self.sess.run([self.optim_lms, self.lms_loss],
                                            feed_dict={self.lms: batch_lms})
                                
            # Each 5 epoch: predict and monitor loss and accuracy
            if epoch % 100 == 0:
                lms_err = self.sess.run(self.lms_err, feed_dict={self.lms: test_lms})
                print("Landmark training: Epoch[%2d] Iter[%3d] lms_loss: %.3f; Testing error:  %.3f" % (epoch, counter, lms_loss, lms_err))

        recon_lms, latent_test = self.sess.run([self.recon_lms, self.latent_lms],feed_dict={self.lms: test_lms})                                    
        latent_lms = self.sess.run(self.latent_lms,feed_dict={self.lms: train_lms}) 

        # interpolate
        #target = 8
        k = 2
        #latent_target = latent_test[target]       
        variances = np.var(latent_lms, axis=0)
        latent_idx = np.argsort(variances)[::-1][:k]
        means = latent_lms.mean(axis=0)
        stds = np.sqrt(variances[latent_idx])
        
        recon_target = np.tile(means, [k*10,1])        
        for i,idx in enumerate(latent_idx):
            mean, std = means[idx], stds[i]
            grid = np.linspace(mean-std,mean+std,10)            
            recon_target[i*10:(i+1)*10, idx] = grid
        
        inter_lms = self.sess.run(self.sampled_lms, feed_dict={self.sample_lms: recon_target})

        return recon_lms, self.mean_lms, inter_lms #, target_recon, latent_test #*(self.lmax-self.lmin+1e-12)+self.lmin
            
    def train_img(self, num_epoch=300, learning_rate=7e-4):
        '''Optimizer (adam)'''
        self.optim_img = tf.train.AdamOptimizer(learning_rate).minimize(self.img_loss)

        '''Load data'''
        
        if os.path.isfile("/content/drive/My Drive/image_warp.npy"):
            image_warp = np.load("/content/drive/My Drive/image_warp.npy")
        else:
            images = np.load('/content/drive/My Drive/images.npy')/255    
            image_warp = np.array(list(map(warp, images, landmarks, [self.mean_lms*128]*landmarks.shape[0])))
            np.save("/content/drive/My Drive/image_warp.npy", image_warp)
        
        print('Image shape: {}'.format(np.shape(image_warp)))

        test_idx = self.test_idx        
        train_idx = [i for i in range(image_warp.shape[0]) if i not in test_idx]

        '''Initialize variables'''
        self.sess.run(tf.global_variables_initializer())
        
        '''Training loop'''
        counter = 0
        
        # Training iterations
        print("Start training")
        for epoch in range(num_epoch):
            batch_num = (image_warp.shape[0]-self.test_size) // self.batch_size
            
            for batch in range(batch_num):
                counter += 1
                
                # mini-batch data
                batch_idx = train_idx[batch*self.batch_size:(batch+1)*self.batch_size]
                batch_img = image_warp[batch_idx]

                _, img_loss = self.sess.run([self.optim_img, self.img_loss],
                                            feed_dict={self.img: batch_img})
                
            # Each 5 epoch: predict and monitor loss and accuracy
            if epoch % 50 == 0:
                img_err = self.sess.run(self.img_err, feed_dict={self.img: image_warp[test_idx]})
                print("Image Training: Epoch[%2d] Iter[%3d] img_loss: %.5f; Testing error: %.5f" % (epoch, counter, img_loss, img_err))
                                    
        towarp_img = self.sess.run(self.recon_img, feed_dict={self.img: image_warp[test_idx]})   
        latent_img = self.sess.run(self.latent_img, feed_dict={self.img: image_warp[train_idx]})   

        # interpolate
        k=4        
        variances = np.var(latent_img, axis=0)
        latent_idx = np.argsort(variances)[::-1][:k]
        means, stds = latent_img.mean(axis=0), np.sqrt(variances[latent_idx])
        
        recon_target = np.tile(means, [k*10,1])
        for i,idx in enumerate(latent_idx):
            mean, std = means[idx], stds[i]
            grid = np.linspace(mean-2*std,mean+2*std,10)            
            recon_target[i*10:(i+1)*10, idx] = grid
        
        inter_img = self.sess.run(self.sampled_img, feed_dict={self.sample_img: recon_target})                        

        return towarp_img, inter_img

if __name__ == '__main__':
    with tf.Session() as sess:
        autoencoder = AE(sess, l_dim=10, z_dim=50, x_dim=3) 
        recon_lms, mean_lms, inter_lms = autoencoder.train_lms(num_epoch=900, learning_rate=1e-3)
        towarp_img, inter_img = autoencoder.train_img(num_epoch=500, learning_rate=5e-4)

    recon_img = np.array(list(map(warp, towarp_img[:20], [mean_lms*128]*20, recon_lms[:20]*128)))
    fig2_1a = plot_img(recon_img, 4,5,'Reconstruction by AutoEncoder')
    fig2_1a.savefig('fig/Fig2_1a')

    fig2_2a = plot_img(inter_img, 4, 10, 'Interpolation on 4 latent variables')
    fig2_2a.savefig('fig/Fig2_2a')

    inter_lms_img = np.array(list(map(warp, [recon_img[8]]*20, [mean_lms*128]*20, inter_lms*128)))
    fig2_2b = plot_img(inter_lms_img, 2, 10, 'Interpolation on 2 latent landmarks for a random image')
    fig2_2b.savefig('fig/Fig2_2b')

    fig2_2c = plot_lms(inter_lms, 2, 10, 'Warping plot - Interpolation on 2 latent landmarks')
    fig2_2c.savefig('fig/Fig2_2c')
