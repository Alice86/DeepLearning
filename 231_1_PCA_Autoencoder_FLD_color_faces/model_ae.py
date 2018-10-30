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
    def __init__(self, sess, l_dim=10, z_dim=50, x_dim=3, learning_rate=7e-4, batch_size=100, test_size=200, image_size=128, keep_prob=0.8, checkpoint_dir='check_point'):
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

        self.learning_rate = learning_rate
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
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.zeros_initializer()

            x_flat = tf.reshape(x, [-1, 68*2])
            w0 = tf.get_variable('w0', [68*2, 100], initializer=w_init)
            b0 = tf.get_variable('b0', [100], initializer=b_init)
            h0 = tf.matmul(x_flat, w0) + b0
            h0 = tf.nn.leaky_relu(h0)
            # h0 = tf.nn.dropout(h0, self.keep_prob)

            w1 = tf.get_variable('w1', [100, 10], initializer=w_init)
            b1 = tf.get_variable('b1', [10], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.leaky_relu(h1)
            # h1 = tf.nn.dropout(h1, self.keep_prob)

            wo = tf.get_variable('wo', [10, self.l_dim], initializer=w_init)
            bo = tf.get_variable('bo', [self.l_dim], initializer=b_init)
            z = tf.matmul(h1, wo) + bo           
        return z

    def encoder_img(self, x, reuse=False, train=True):
        with tf.variable_scope("encoder_img", reuse=reuse) as scope:

            h0 = tf.contrib.layers.conv2d(x, 16, 5, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)

            h1 = tf.contrib.layers.conv2d(h0, 32, 3, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)

            h2 = tf.contrib.layers.conv2d(h1, 64, 3, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)

            h3 = tf.contrib.layers.conv2d(h2, 128, 3, 2, padding='SAME', activation_fn=tf.nn.leaky_relu)
            
            h_flat = tf.contrib.layers.flatten(h3)            
            #hf = tf.contrib.layers.fully_connected(h_flat, 50, activation_fn=tf.nn.leaky_relu)

            z = tf.contrib.layers.fully_connected(h_flat, self.z_dim, activation_fn=tf.nn.leaky_relu)
            
        return z

    def decoder_lms(self, z, reuse=False, train=True):   
        with tf.variable_scope("decoder_lms", reuse=reuse) as scope:

            # initializers
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.zeros_initializer()
            
            w0 = tf.get_variable('w0', [self.l_dim, 100], initializer=w_init)
            b0 = tf.get_variable('b0', [100], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.leaky_relu(h0)
            # h0 = tf.nn.dropout(h0, self.keep_prob)

            w1 = tf.get_variable('w1', [100, 68*2], initializer=w_init)
            b1 = tf.get_variable('b1', [68*2], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.sigmoid(h1)
            # h1 = tf.nn.dropout(h1, self.keep_prob)

            x = tf.reshape(h1, [-1, 68, 2])
        return x

    def decoder_img(self, z, reuse=False, train=True):   
        with tf.variable_scope("decoder_img", reuse=reuse) as scope:
            size = z.get_shape()[0]
            
            hf = tf.contrib.layers.fully_connected(z, 128*8*8, activation_fn=tf.nn.leaky_relu)
            h0 = tf.reshape(hf, [-1, 8, 8, 128])        
            #z_patch = tf.tile(tf.reshape(z, [-1, 1, 1, self.z_dim]), [1,8,8,1])       
            #h0 = tf.contrib.layers.conv2d_transpose(z_patch, 128, [8, 8], stride=1, padding='SAME', activation_fn=tf.nn.leaky_relu)  

            h1 = tf.contrib.layers.conv2d_transpose(h0, 64, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu)  

            h2 = tf.contrib.layers.conv2d_transpose(h1, 32, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu)  

            h3 = tf.contrib.layers.conv2d_transpose(h2, 16, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.leaky_relu)  

            x = tf.contrib.layers.conv2d_transpose(h3, self.x_dim, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
            
        return x
    
    def build_model(self):
        '''
        Define the model and loss for training
            Define placeholders: image features, correspondinglabels, classes involved (seen/unseen)
            Build the vae training model: encoding (KL matrix) + decoding (feature reconstruction)
            Compute the loss in three parts
            Build the testing model: input testing examples to the trained model, eval top-1 accuracy
        '''

        '''Placeholders: attributes, train, test'''
        self.img = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.x_dim], name="image")
        self.lms = tf.placeholder(tf.float32, [None, 68, 2], name="landmark")
        
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
        
        self.saver = tf.train.Saver() 
        
    def train_lms(self, num_epoch=150, learning_rate=5e-3):
       
        '''Optimizer (adam)'''
        self.optim_lms = tf.train.AdamOptimizer(learning_rate).minimize(self.lms_loss)

        '''Load data'''
        # load data
        # images = np.load('/content/drive/My Drive/images.npy')
        landmarks = np.load("/content/drive/My Drive/landmarks.npy")
        # normalize
        lmin, lmax = landmarks.min(), landmarks.max()
        landmarks = (landmarks-lmin)/(lmax-lmin+1e-12)
        # print('Image shape: {}'.format(np.shape(images)))
        print('Landmark shape: {}'.format(np.shape(landmarks)))
        np.random.seed(1)
        test_idx = np.random.choice(landmarks.shape[0], self.test_size, replace=False)
        train_idx = [i for i in range(landmarks.shape[0]) if i not in test_idx]
        test_lms = landmarks[test_idx,:]
        train_lms = landmarks[train_idx,:]
        
        '''Initialize variables'''
        self.sess.run(tf.global_variables_initializer())
        
        '''Training loop'''
        counter = 0

        # load checkpoint if exist
        
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
            if epoch % 30 == 0:
                lms_err = self.sess.run(self.lms_err, feed_dict={self.lms: test_lms})
                print("Landmark training: Epoch[%2d] Iter[%3d] lms_loss: %.3f; Testing error:  %.3f" % (epoch, counter, lms_loss, lms_err))

        recon_lms = self.sess.run(self.recon_lms,feed_dict={self.lms: test_lms})                                    
        latent_lms = self.sess.run(self.latent_lms,feed_dict={self.lms: train_lms}) 
        
        return recon_lms*(lmax-lmin+1e-12)+lmin, latent_lms
            
    def train_img(self, num_epoch=300, learning_rate=7e-4):
        '''Optimizer (adam)'''
        self.optim_img = tf.train.AdamOptimizer(learning_rate).minimize(self.img_loss)

        '''Load data'''
        # load data
        landmarks = np.load("/content/drive/My Drive/landmarks.npy")
        np.random.seed(1)
        test_idx = np.random.choice(landmarks.shape[0], self.test_size, replace=False)
        train_idx = [i for i in range(landmarks.shape[0]) if i not in test_idx]
        train_lms = landmarks[train_idx,:]
        mean_lms = train_lms.mean(0)
        if os.path.isfile("/content/drive/My Drive/image_warp.npy"):
            image_warp = np.load("/content/drive/My Drive/image_warp.npy")
        else:
            images = np.load('/content/drive/My Drive/images.npy')/255    
            image_warp = np.array(list(map(warp, images, landmarks, [mean_lms]*landmarks.shape[0])))
            np.save("/content/drive/My Drive/image_warp.npy", image_warp)
        
        print('Image shape: {}'.format(np.shape(image_warp)))
        print('Landmark shape: {}'.format(np.shape(landmarks)))

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
                batch_img = image_warp[batch_idx]

                _, img_loss = self.sess.run([self.optim_img, self.img_loss],
                                            feed_dict={self.img: batch_img})
                
            # Each 5 epoch: predict and monitor loss and accuracy
            if epoch % 20 == 0:
                img_err = self.sess.run(self.img_err, feed_dict={self.img: image_warp[test_idx]})
                print("Image Training: Epoch[%2d] Iter[%3d] img_loss: %.5f; Testing error: %.5f" % (epoch, counter, img_loss, img_err))
                                    
        towarp_img = self.sess.run(self.recon_img, feed_dict={self.img: image_warp[test_idx]})   
        latent_img = self.sess.run(self.latent_img, feed_dict={self.img: image_warp[train_idx]})   

        recon_img = np.array(list(map(warp, towarp_img, [mean_lms]*self.test_size, recon_lms)))
        
        return recon_img, latent_img

if __name__ == '__main__':
    with tf.Session() as sess:
        autoencoder = AE(sess, l_dim=10, z_dim=50, x_dim=3, learning_rate=6e-3, batch_size=100) 
        recon_lms, latent_lms = autoencoder.train_lms(num_epoch=300, learning_rate=7e-4)
        recon_img, latent_img = autoencoder.train_img(num_epoch=300, learning_rate=7e-4)
    images = np.load('/content/drive/My Drive/images.npy')

np.random.seed(1)
test_idx = np.random.choice(images.shape[0], 200, replace=False)
test_img = images[test_idx]/255
err = ((recon_img-images[test_idx])**2).mean()
print('Testing error: %.5f' % err)
fig2_1a = plot_img(recon_img[:20], 4,5,'Reconstructed faces')
fig2_1b = plot_img(test_img[:20], 4,5,'Original faces')
fig2_1a.savefig('fig/Fig2_1a')
fig2_1b.savefig('fig/Fig2_1b')
