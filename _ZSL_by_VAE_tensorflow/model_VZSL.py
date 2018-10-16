#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE model and training
     class VAE(sess, config):
         __init__
         encoder(): recognition model/approximate gaussian posterior distribution q_phi
         prior(): class-conditioned latent gaussian prior p_psi
         decoder(): reconstruction model/decoding gaussian distribution p_theta
         build_model(): Model structure and loss
         train(): read data, training flow, monitor training
         save(): save checkpoint during training
         load(): load checkpoint to continue on previous training
"""

from __future__ import division
import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import *

class VAE(object):
    def __init__(self, sess, config):
        """
        Args for model:
            sess: TensorFlow session
            config:
                lamb: Parameter lambda controlling the extent of regularization. [1]
                M: Num. of attributrs. [85]
                S: Num. of seen classes. [40]
                U: Num. of unseen classes. [10]
                z_dim: Dimension of latent vectors. [100]
                x_dim: Dimension of feature vector. [4096]
                batch_size: Size of batch specified before training [256].
                test_size: Size of test data. [6810]
                hidden_num: Num. of nodes for encoder/decoder MLP. [1000]
                keep_prob: Dropout parameter to avoid overfitting [0.8]
                data_path: Directory name of dataset. [data/AWA1]
                feature_path: Directory name of extracted feature from image. [VGG19_feature]
                checkpoint_dir: Directory name to save the checkpoints [checkpoint]
        """
        self.sess = sess
        
        self.lamb = config.lamb
        self.M = config.M
        self.S = config.S
        self.U = config.U
        self.z_dim = config.z_dim
        self.x_dim = config.x_dim

        self.test_size= config.test_size
        self.batch_size = config.batch_size
        self.hidden_num = config.hidden_num
        self.keep_prob = config.keep_prob
        
        self.data_path = config.data_path
        self.feature_path = config.feature_path
        self.checkpoint_dir = config.checkpoint_dir
        
        self.build_model()
        
    def encoder(self, features, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            """
            Define encoder network structure [recognition q(z|x)]:
            2-layer gaussian MLP (1000 nodes, relu activation, 0.8 dropout): 
                feature - fc1 - relu1 - fc2 - relu2 - output
            """
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)
    
            # 2-layer
            w0 = tf.get_variable('w0', [features.get_shape()[1], self.hidden_num], initializer=w_init) 
            b0 = tf.get_variable('b0', [self.hidden_num], initializer=b_init)
            h0 = tf.matmul(features, w0) + b0
            h0 = tf.nn.relu(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)
    
            w1 = tf.get_variable('w1', [self.hidden_num, self.hidden_num], initializer=w_init)
            b1 = tf.get_variable('b1', [self.hidden_num], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.relu(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)
    
            # output layer
            wo = tf.get_variable('wo', [self.hidden_num, self.z_dim*2], initializer=w_init)
            bo = tf.get_variable('bo', [self.z_dim*2], initializer=b_init)
            output = tf.matmul(h1, wo) + bo
            
            mean_q = output[:, :self.z_dim]
            std_q = tf.exp(output[:, self.z_dim:])
        
        return mean_q, std_q

    def prior(self, attributes, reuse=False, train=True):
        with tf.variable_scope("prior", reuse=reuse) as scope:
            """
            Define prior distribution [p(z|a)]:
                attibute-specific latent gaussian 
                unknown parameters to be learned: w_mean, w_std (M*z_dim)
            """
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer(1.0)
            b_init = tf.constant_initializer(0.) 

            # prior distribution
            w = tf.get_variable('w', [self.M, self.z_dim*2], initializer=w_init)
            b = tf.get_variable('b', [self.z_dim*2], initializer=b_init)
            output = tf.matmul(attributes, w)+b
            mean_p = output[:, :self.z_dim]
            std_p = tf.sqrt(tf.exp(output[:, self.z_dim:]))

        return mean_p, std_p # (S+U) * z_dim
 

    def decoder(self, z, reuse=False, train=True):   
        with tf.variable_scope("decoder", reuse=reuse) as scope:
            """
            Define decoder network structure [p(x|z)]:
                same as encoder, gaussian MLP for continuous data (Bernoulli for binary)
            """
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.) 
            
            # 2 layer
            w0 = tf.get_variable('w0', [self.z_dim, self.hidden_num], initializer=w_init)
            b0 = tf.get_variable('b0', [self.hidden_num], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.relu(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)
    
            w1 = tf.get_variable('w1', [self.hidden_num, self.hidden_num], initializer=w_init)
            b1 = tf.get_variable('b1', [self.hidden_num], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.relu(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)      

            # output layer
            wo = tf.get_variable('wo', [self.hidden_num, self.x_dim*2], initializer=w_init)
            bo = tf.get_variable('bo', [self.x_dim*2], initializer=b_init)
            output = tf.nn.sigmoid(tf.matmul(h1, wo) + bo)

            mean_d = output[:, :self.x_dim]
            std_d = tf.exp(output[:, self.x_dim:])
        
        return mean_d, std_d
    
    def build_model(self):
        '''
        Define the model and loss for training
            Define placeholders: image features, correspondinglabels, classes involved (seen/unseen)
            Build the vae training model: encoding (KL matrix) + decoding (feature reconstruction)
            Compute the loss in three parts
            Build the testing model: input testing examples to the trained model, eval top-1 accuracy
        '''

        '''Placeholders: attributes, train, test'''
        self.A = tf.placeholder(tf.float32, [self.S+self.U, self.M], name="attribites")

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name="features")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")
        self.classes = tf.placeholder(tf.int32, [None], name="classes")
        
        '''Training'''
        # Encoding
        mean_q, std_q = self.encoder(features=self.x, reuse=False, train=True) # batch_size * z_dim
        # prior parameter learning
        mean_p, std_p = self.prior(attributes=self.A, reuse=False, train=True) # (S+U) * z_dim
        
        KL = [] # KL_div for each obs., each class (batch_size * S+U)
        # KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q) + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
        for i in range(self.batch_size):
            KL.append(0.5*tf.reduce_sum(tf.log(tf.square(std_p)+1e-32) \
                                    - tf.log(tf.square(std_q[i,:])+1e-32) - 1 \
                                    + tf.square(std_q[i,:])/(tf.square(std_p)+1e-32) \
                                    + tf.square(mean_p-mean_q[i,:])/(tf.square(std_p)+1e-32), axis=1) )
        KL = tf.stack(KL)
        
        # Decoding: sample from posterior and reconstruct
        z = tf.random_normal(tf.shape(mean_q), mean_q, std_q, dtype=tf.float32) # batch_size*z_dim
        mean_d, std_d = self.decoder(z=z, reuse=False, train=True)         
        
        '''Loss = reconstruction error + KL loss + margin regularizer'''
        # reconstruction error (log gaussian prob)
        self.recon_loss = 0.5*tf.reduce_mean(tf.reduce_sum( tf.log(tf.square(std_d)+1e-32) \
                                + tf.square(self.x - mean_d)/(tf.square(std_d)+1e-32), axis=1)) # 
        # tf.log(2*np.pi) +
        # KL loss
        indices = tf.stack([tf.range(self.batch_size), self.labels-1], axis=1)
        self.kl_loss = tf.reduce_mean(tf.gather_nd(KL, indices))
        # margin regularizer
        self.margin_loss = tf.reduce_mean(tf.reduce_logsumexp(-tf.gather(KL, self.classes-1, axis=1), axis=1))        
        
        self.loss =  self.kl_loss+self.lamb*self.margin_loss + self.recon_loss

        '''Testing'''
        self.mean_q, self.std_q = self.encoder(features=self.x, reuse=True, train=False)
        self.mean_p, self.std_p = self.prior(attributes=self.A, reuse=True, train=False)
        KL = [] # test_size * (S+U)
        for i in range(self.test_size): 
            KL.append( 0.5*tf.reduce_sum(tf.log(tf.square(self.std_p)+1e-32) \
                        - tf.log(tf.square(self.std_q[i,:])+1e-32) - 1 \
                        + tf.square(self.std_q[i,:])/(tf.square(self.std_p)+1e-32) \
                        + tf.square(self.mean_p-self.mean_q[i,:])/(tf.square(self.std_p)+1e-32), axis=1) )
        self.KL = tf.stack(KL) # test_size * (s+u)
        # make prediction as the smallest KL_div
        self.predict = tf.argmin(tf.gather(self.KL, self.classes-1, axis=1), axis=1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.gather(self.classes, self.predict), self.labels), tf.float32))

        self.saver = tf.train.Saver() 
    
    def train(self, config):
        '''
        Construct the iterative training flow
            Obtimizer: adam
            Load data and split train/test: use functions from module 'utils'
            Initialize, train for mini-batch, make prediction on testing 
            Monitor: loss, top-1 accuracy
        '''
       
        '''Optimizer (adam)'''
        self.optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)

        '''Load data'''
        # extracted features, attibutes, train/test split
        data_path = os.path.join("data", self.data_path)
        features, labels = Load_feature(os.path.join(data_path, self.feature_path))
        features = normalize(features, 0, 1)
        attributes = Attibute_process(data_path)
        attributes = attributes*(attributes>0)
        attributes /= np.max(attributes)
        # print(np.mean(features), np.mean(attributes))
        
        class_label = Class_label_indexing(data_path)
        train_index, test_index, train_class, test_class = Split(data_path, labels, class_label)
        test_class = sorted(test_class)
        train_size, test_size = len(train_index), len(test_index)
        print('Training size: {}'.format(train_size), 'Testing size: {}'.format(test_size))
        print('Testing classes:', test_class)
        
        '''Initialize variables'''
        self.sess.run(tf.global_variables_initializer())
        
        '''Training loop'''
        counter = 0
        self.losses, self.recons, self.kls, self.margins, self.accs = [], [], [], [], []

        # load checkpoint if exist
        self.load()
        
        # Training iterations
        print("Start training")
        for epoch in range(config.epoch):
            np.random.shuffle(train_index)
            batch_num = train_size // self.batch_size
            
            for num in range(batch_num):
                counter += 1
                
                # mini-batch data
                batch_idx = train_index[num*self.batch_size:(num+1)*self.batch_size]
                batch_feature = features[batch_idx,:]
                batch_label = labels[batch_idx]

                _, step_loss, recon_loss, kl_loss, margin_loss = self.sess.run([self.optim, self.loss, self.recon_loss, self.kl_loss, self.margin_loss],
                                     feed_dict={self.x: batch_feature,
                                                self.labels: batch_label,
                                                self.A: attributes,
                                                self.classes: train_class})
                # Monitor the start point        
                if counter == 1:
                    print("Training: Epoch[%2d] Iter[%3d] step_loss: %.3f, recon_loss: %.3f, kl_loss: %.3f, margin_loss: %.3f" % (epoch, counter, step_loss, recon_loss, kl_loss, margin_loss))
                    pred, acc = self.sess.run([self.predict, self.accuracy],
                                             feed_dict={self.x: features[test_index,:],
                                                        self.labels: labels[test_index],
                                                        self.A: attributes,
                                                        self.classes: test_class})
                    print("Testing: Epoch[%2d] Iter[%3d] Top-1 accuracy: %.5f" % (epoch, counter, acc))
                    
                    self.losses.append(step_loss)
                    self.recons.append(recon_loss)
                    self.kls.append(kl_loss)
                    self.margins.append(margin_loss)
                    self.accs.append(acc)


            # Each epoch: predict and monitor loss and accuracy
            pred, acc = self.sess.run([self.predict, self.accuracy],
                 feed_dict={self.x: features[test_index,:],
                            self.labels: labels[test_index],
                            self.A: attributes,
                            self.classes: test_class})
            
            self.losses.append(step_loss)
            self.recons.append(recon_loss)
            self.kls.append(kl_loss)
            self.margins.append(margin_loss)
            self.accs.append(acc)
                        
            if np.mod(epoch, 5) == 0: 
            # if np.mod(counter, 200) == 1 
                print("Training: Epoch[%2d] Iter[%3d] step_loss: %.3f, recon_loss: %.3f, kl_loss: %.3f, margin_loss: %.3f" % (epoch, counter, step_loss, recon_loss, kl_loss, margin_loss))

            # if np.mod(counter, 200) == 1 
                print("Testing: Epoch[%2d] Iter[%3d] Top-1 accuracy: %.5f" % (epoch, counter, acc))
            # print(pred)                    
                
            # save checkpoint
            # if or num == batch_num-1: # np.mod(counter, 1000) == 1 
            self.save(counter)
            
                    
        # Plot
        print(np.mean(self.accs[-20:]), np.std(self.accs[-20:]))
        plt.title('Training Accuracy')
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy')
        plt.plot(self.accs)
        plt.savefig('Acc.png')
        plt.close()
        
        plt.title('Training Accuracy')
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy')
        plt.plot(self.losses, label = 'loss')
        plt.plot(self.recons, label = 'recon loss')
        plt.plot(self.kls, label = 'kl loss')
        plt.plot(self.margins, label = 'margin loss')
        plt.legend()
        plt.savefig('Loss.png')
        plt.close()
        
    '''
    Checkpoint load and save
        continue previous training from saved checkpoint
    '''
    def save(self, step):
        model_name = "ZSL.model"
        model_dir = "%s_%s"%(self.data_path, self.feature_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_path):
            print("Creating checkpoint at {}".format(checkpoint_path))
            os.makedirs(checkpoint_path)
        self.saver.save(self.sess, os.path.join(checkpoint_path, model_name), global_step=step)

    def load(self):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s"%(self.data_path, self.feature_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_path, ckpt_name))
            print(" [*] Load SUCCESS...")
            return True
        else:
            print(" [*] Checkerpoint not found...")
            return False
