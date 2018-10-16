# -*- coding: utf-8 -*-
import os
from glob import glob
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datasets import *


class DesNet(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.batch_size = flags.batch_size

        self.data_path = os.path.join('./Image', flags.dataset_name)
        self.output_dir = flags.output_dir

        self.log_dir = os.path.join(self.output_dir, 'log')
        self.sample_dir = os.path.join(self.output_dir, 'sample')
        self.model_dir = os.path.join(self.output_dir, 'checkpoints')

        if tf.gfile.Exists(self.log_dir):
            tf.gfile.DeleteRecursively(self.log_dir)
        tf.gfile.MakeDirs(self.log_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.build_model(flags)

    def descriptor(self, inputs, is_training=True, reuse=False):
        ####################################################
        # Define network structure for descriptor.
        # Recommended structure:
        # conv1: channel 64 kernel 4*4 stride 2
        # conv2: channel 128 kernel 2*2 stride 1
        # fc: channel output 1
        # conv1 - bn - relu - conv2 - bn - relu -fc
        ####################################################
        with tf.variable_scope('des', reuse=reuse):
            
            w1 = tf.get_variable('w1', [4, 4, inputs.get_shape()[-1], 64],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            b1 = tf.get_variable('b1', [64], initializer=tf.constant_initializer(0.0))
            h1 = tf.nn.bias_add(tf.nn.conv2d(inputs, w1, strides=[1, 2, 2, 1], padding='SAME'), b1)
            # padding: 
            h1b = tf.contrib.layers.batch_norm(h1, decay=0.9, updates_collections=None, epsilon=1e-5, 
                                               scale=True, is_training=is_training, scope='bn1')
            # bn
            h1r = tf.maximum(h1b, 0)

            w2 = tf.get_variable('w2', [2, 2, 64, 128],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            b2 = tf.get_variable('b2', [128], initializer=tf.constant_initializer(0.0))
            h2 = tf.nn.bias_add(tf.nn.conv2d(h1r, w2, strides=[1, 1, 1, 1], padding='SAME'), b2)
            h2b = tf.contrib.layers.batch_norm(h2, decay=0.9, updates_collections=None, epsilon=1e-5, 
                                               scale=True, is_training=is_training, scope='bn2')
            h2r = tf.maximum(h2b, 0)

            h3 = tf.layers.dense(tf.layers.flatten(h2r), 1, name="fc3")
            # dense and flatten

            return h3 

    def Langevin_sampling(self, samples, flags):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated samples.
        ####################################################        
        def step(Y, i):
            sigma, delta = flags.ref_sig, flags.delta # reference distri., learning rate for langevin
            grad = tf.gradients(self.descriptor(Y, reuse=True), Y, name='grad')[0]
            energy = Y/(sigma**2) - grad # output[0]
            noise = tf.random_normal(shape=Y.get_shape(), name='noise')
            Y = Y - 0.5*(delta**2)*energy + delta*noise
            i += 1
            return Y, i
        
        cond = lambda Y, i: tf.less(i, flags.T) # i -= 1
        body = lambda Y, i: step(Y, i)
        
        with tf.name_scope("langevin"):
            i = tf.constant(0)
            output, i = tf.while_loop(cond, body, [samples, i])
        
        return output

    def build_model(self, flags):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        # define placeholder in model, feed data in training
        self.real_image = tf.placeholder(tf.float32,
                shape=[self.batch_size, flags.image_size, flags.image_size, 3], name="real") 
        self.sample_image = tf.placeholder(tf.float32,
                shape=[self.batch_size, flags.image_size, flags.image_size, 3], name="sample") 
        
        self.score_real = self.descriptor(self.real_image, reuse=False)
        self.score_sample = self.descriptor(self.sample_image, reuse=True)
        
        self.train_loss = tf.subtract(tf.reduce_mean(self.score_sample, axis=0), tf.reduce_mean(self.score_real, axis=0))
        
        # tf.reduce_mean(self.score_real-tf.norm(self.real_image, ord=2)/2/(flags.ref_sig**2), axis=0) - tf.reduce_mean(self.score_sample-tf.norm(self.sample_image, ord=2)/2/(flags.ref_sig**2), axis=0)
        # self.recon_loss = tf.reduce_mean(tf.norm(self.real_image - self.sample_image, ord=2, axis=0) )
                # tf.contrib.metrics.streaming_mean: metrics computed on dynamically valued Tensors
                # "value_tensor", idempotent operation returns the current value of the metric
                # "update_op", operation that accumulates the information from the current value of the Tensors being measured
                
        self.loss = self.train_loss - (tf.reduce_mean(
            tf.norm(self.real_image, ord=2, axis=0)-tf.norm(self.sample_image, ord=2, axis=0)))/2/(flags.ref_sig**2)  # +reference
        
        self.sampler = self.Langevin_sampling(self.sample_image, flags)
        
        # optimization
        self.vars = [var for var in tf.trainable_variables() if 'des' in var.name] # par in des net
        self.optim = tf.train.AdamOptimizer(
                flags.learning_rate, beta1=flags.beta1).minimize(self.loss,var_list=self.vars)
        
        
        #self.grads = self.optim.compute_gradients(self.train_loss, var_list=self.vars)
       
        #self.cache = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) 
        #              for var in self.vars]
        #self.reset = [var.assign(tf.zeros_like(var)) for var in self.cache]
        #self.update = [self.cache[i].assign_add(gv[0]) 
        #                                         for i, gv in enumerate(self.grads)]
        #self.apply = self.optim.apply_gradients([(tf.divide(self.cache[i], 2), gv[1]) 
        #                                         for i, gv in enumerate(self.grads)])
        

    def train(self, flags):
        # Prepare training data, scale is [0, 255]
        train_data = DataSet(self.data_path, image_size=flags.image_size)
        train_data = train_data.to_range(0, 255)

        saver = tf.train.Saver(max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        self.sess.graph.finalize()

        print(" Start training ...")

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # synthesized images in self.sample_dir,
        # loss in self.log_dir (using writer).
        ####################################################
        
        data_len = train_data.shape[0]
        batch_idxs = data_len // self.batch_size
        
        t_loss_log = []
        # r_loss_log = []
        counter = 0
        
        #idx = 0
        #batch_images = train_data[idx * self.batch_size:(idx+1)*self.batch_size]
        #mean_image = batch_images.mean(axis=(1,2), keepdims=1)\
        #                       .repeat(train_data.shape[1], axis=1).repeat(train_data.shape[2], axis=2)
        #sample_image = self.sess.run(self.sampler, feed_dict={self.fake_image: mean_image})      
        
        for epoch in range(flags.epoch):            

            for idx in range(batch_idxs):
                counter += 1
                batch_images = train_data[idx*self.batch_size:(idx+1)*self.batch_size]
                mean_image = batch_images.mean(axis=(1,2), keepdims=1)\
                               .repeat(train_data.shape[1], axis=1).repeat(train_data.shape[2], axis=2)
                
                sample_image = self.sess.run(self.sampler, feed_dict={self.sample_image: mean_image})
                                
                _, t_loss = self.sess.run([self.optim, self.train_loss], feed_dict={
                                        self.real_image: batch_images,
                                        self.sample_image: sample_image})
                # self.sess.run(self.apply)
                
                #batch_images = train_data[idx*self.batch_size:min(data_len, (idx+1)*self.batch_size)]
                #mean_image = batch_images.mean(axis=(1,2), keepdims=1)\
                #               .repeat(train_data.shape[1], axis=1).repeat(train_data.shape[2], axis=2)
                #sample_image = self.sess.run(self.sampler, feed_dict={self.fake_image: mean_image})
                
                t_loss_log.append(t_loss)
                # r_loss_log.append(r_loss)
                                
            if epoch % 100 == 0:
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                save_images(sample_image, "%s/output-%03d.png" % (self.sample_dir, epoch))

                print("Epoch[%2d], train_loss: %.6f" % (epoch, t_loss))
            
        plt.plot(t_loss_log)
        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig("%s/loss.png" % (self.sample_dir))
        plt.clf()
                