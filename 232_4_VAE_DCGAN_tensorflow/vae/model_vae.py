from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
from skimage import io
import matplotlib.pyplot as plt

from ops import *
from utils import *
import random
import numpy as np

class VAE(object):
    def __init__(self, sess, image_size=28,
                 batch_size=100, sample_size=100, output_size=28,
                 z_dim=5, c_dim=1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            image_size: The size of input image.
            batch_size: The size of batch. Should be specified before training.
            sample_size: (optional) The size of sampling. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [28]
            z_dim: (optional) Dimension of latent vectors. [5]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [1]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def encoder(self, image, reuse=False, train=True):
        with tf.variable_scope("encoder", reuse=reuse) as scope:
            #######################################################
            # TODO: Define encoder network structure here. op.py
            # includes some basic layer functions for you to use.
            # Please use batch normalization layer after conv layer.
            # And use 'train' argument to indicate the mode of bn.
            # The output of encoder network should have two parts:
            # A mean vector and a log(std) vector. Both of them have
            # the same dimension with latent vector z.
            #######################################################
#            z_dim = self.z_dim
#            d_1 = tf.layers.conv2d(image, 32, [4, 4], strides=(2, 2), padding='same')
#            d_1 = lrelu(batch_norm(d_1, train=train, name="bn1"))
#            d_2 = tf.layers.conv2d(d_1, 64, [4, 4], strides=(2, 2), padding='same')
#            d_2 = lrelu(batch_norm(d_2, train=train, name="bn2"))
#            d_3 = tf.layers.conv2d(d_2, 128, [4, 4], strides=(2, 2), padding='same')
#            # d_3 = lrelu(batch_norm(d_3, train=train, name="bn3"))
#            d_4 = linear(tf.layers.flatten(d_3), z_dim * 2)
#            mean = d_4[:, :z_dim]
#            std = d_4[:, z_dim:]
            h1r = lrelu(conv2d(image, 32, name='e_cv1'))
            
            h2 = conv2d(h1r, 64, name='e_cv2')
            h2r = lrelu(batch_norm(h2, train=train, name="e_bn2"))
            
            h3 = conv2d(h2r, 8, name='e_cv3')
             #h3r = lrelu(batch_norm(h3, train=train, name="e_bn3"))
            
             # h4r = lrelu(linear(tf.reshape(h2r, [self.batch_size, -1]), 200, scope='e_fc4'))
            
            h5 = linear(tf.reshape(h3, [self.batch_size, -1]), self.z_dim*2, scope="e_fc5")
            
            mean = h5[:,:self.z_dim]
            sd = h5[:,self.z_dim:]
                    
            return mean, sd
            #######################################################
            #                   end of your code
            #######################################################


    def decoder(self, z, reuse=False, train=True):
        with tf.variable_scope("decoder", reuse=reuse):
            #######################################################
            # TODO: Define decoder network structure here. The size
            # of output should match the size of images. To make the
            # output pixel values in [0,1], add a sigmoid layer before
            # the output. Also use batch normalization layer after
            # deconv layer, and use 'train' argument to indicate the
            # mode of bn layer. Note that when sampling images using
            # trained model, you need to set train='False'.
            #######################################################
#            z = tf.reshape(z, [-1, 1, 1, z._shape[1]])
#            g_1 = tf.layers.conv2d_transpose(z, 128, [4, 4], strides=(1, 1), padding='valid')
#            g_1 = lrelu(batch_norm(g_1, train=train, name="bn1"))
#            g_2 = tf.layers.conv2d_transpose(g_1, 64, [4, 4], strides=(1, 1), padding='valid')
#            g_2 = lrelu(batch_norm(g_2, train=train, name="bn2"))
#            g_3 = tf.layers.conv2d_transpose(g_2, 32, [4, 4], strides=(2, 2), padding='same')
#            g_3 = lrelu(batch_norm(g_3, train=train, name="bn3"))
#            g_4 = tf.layers.conv2d_transpose(g_3, 1, [4, 4], strides=(2, 2), padding='same')
#            return tf.nn.sigmoid(g_4)
             # h1r = lrelu(linear(z, 100, 'd_fc1'))
             # z = tf.reshape(z, [-1, 1, 1, z._shape[1]])
            
             h2 = linear(z, 7*7*self.c_dim*8, 'd_fc2')
             h2 = tf.reshape(h2, [-1, 7, 7, self.c_dim*8])
             # h2f = deconv2d(h2, [self.batch_size, 7, 7, self.c_dim*8], name='d_cv2')
             # h2r = lrelu(batch_norm(h2f, train=train, name="d_bn2"))
            
             h3 = deconv2d(h2, [self.batch_size, 14, 14, self.c_dim*32], name='d_cv3')
             h3r = lrelu(batch_norm(h3, train=train, name="d_bn3"))
            
             h4 = deconv2d(h3r, [self.batch_size, 28, 28, self.c_dim*16], name='d_cv4')
             #h4r = lrelu(batch_norm(h4, train=train, name="d_bn4"))
            
             h5 = deconv2d(h4, [self.batch_size, 28, 28, self.c_dim], d_h=1, d_w=1, name='d_cv5')
            
             img = tf.nn.sigmoid(h5)
             return img
            #######################################################
            #                   end of your code
            #######################################################

    def build_model(self):
        #######################################################
        # TODO: In this build_model function, define inputs,
        # operations on inputs and loss of VAE. For input,
        # you need to define it as placeholders. Remember loss
        # term has two parts: reconstruction loss and KL divergence
        # loss. Save the loss as self.loss. Use the
        # reparameterization trick to sample z.
        #######################################################
        self.input_image = tf.placeholder(tf.float32, \
             [self.batch_size, self.image_size, self.image_size, self.c_dim], name="input_image")        
        #self.output_image = tf.placeholder(tf.float32, \
        #     [self.batch_size, self.image_size, self.image_size, self.c_dim], name="output_image")        
        self.z = tf.placeholder(tf.float32, \
             [self.batch_size, self.z_dim], name="z")   
        
        #try:
        self.mean, self.sd = self.encoder(self.input_image, reuse=False, train=True) #why not reuse
        # epsilon = tf.random_normal(tf.stack([tf.shape(mean)[0], self.z_dim]))
        # z  = mean + tf.multiply(epsilon, tf.exp(sd))
        z = tf.random_normal(tf.shape(self.mean), self.mean, tf.exp(self.sd), dtype=tf.float32)
        self.output_image = self.decoder(z, reuse=False, train=True)
        
        self.sample = self.decoder(self.z, reuse=True, train=False)
        #except ValueError:
        #    mean, sd = self.encoder(self.input_image, reuse=False, train=True)
        #    epsilon = tf.random_normal(tf.stack([tf.shape(mean)[0], self.z_dim])) 
        #    z  = mean + tf.multiply(epsilon, tf.exp(sd))
        #    self.output_image = self.decoder(z, reuse=False, train=True)
            
        # logp(x|z)
        x = tf.reshape(self.input_image, [self.batch_size, -1])
        y = tf.reshape(self.output_image, [self.batch_size, -1])
        self.cr_en = tf.reduce_mean(-tf.reduce_sum((x * tf.log(y + 1e-12) + (1. - x) * tf.log(1. - y + 1e-12)), 1))
        # cr_en_loss = tf.reduce_mean(cr_en)
        # d_kl(q(z|x)||p(z)) z~normal
        self.kl_div = tf.reduce_mean(- 0.5 * tf.reduce_sum(1.0 + 2.0 * self.sd - tf.square(self.mean) - tf.square(tf.exp(self.sd)), 1))

        #kl_div = -0.5 * tf.reduce_sum(1.0 + 2.0 * tf.log(sd + 1e-10) \
        #                              - tf.square(mean) - tf.square(sd), 1)
        
        self.loss = self.cr_en+self.kl_div

        #######################################################
        #                   end of your code
        #######################################################
        self.saver = tf.train.Saver() # ???

    def train(self, config):
        """Train VAE"""
        # load MNIST dataset
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        data = mnist.train.images
        data = data.astype(np.float32)
        data_len = data.shape[0]
        data = np.reshape(data, [-1, 28, 28, self.c_dim])

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        try:
            self.sess.run(tf.global_variables_initializer())
        except:
            tf.initialize_all_variables().run()

        #start_time = time.time()
        counter = 1

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        sample_dir = os.path.join(config.sample_dir, config.dataset)
        if not os.path.exists(config.sample_dir):
            os.mkdir(config.sample_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        
        self.c = []
        self.k = []
        for epoch in xrange(config.epoch):
            batch_idxs = min(data_len, config.train_size) // config.batch_size
            for idx in xrange(0, batch_idxs):
                counter += 1
                batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size, :]
                #######################################################
                # TODO: Train your model here, print the loss term at
                # each training step to monitor the training process.
                # Print reconstructed images and sample images every
                # config.print_step steps. Sample z from standard normal
                # distribution for sampling images. You may use function
                # save_images in utils.py to save images.
                #######################################################
                # sample_z = np.random.normal(0,1,[self.batch_size, self.z_dim]).astype(np.float32)
                #smpl_idx = np.random.choice(data_len, self.batch_size, replace=False)
                #sample_inputs = data[smpl_idx, :]

                mean, sd = self.sess.run([self.mean, self.sd], {self.input_image: batch_images})
#                plt.hist(np.reshape(mean, [500, -1]))
#                plt.hist(np.reshape(sd, [500, -1]))
#                if idx == 0:
#                    plt.show("hist.png")
                # z = np.random.normal(mean, np.exp(sd), [self.batch_size, self.z_dim]).astype(np.float32)

                _, step_loss, recon_image, closs, kloss = self.sess.run([optim, self.loss, self.output_image, self.cr_en, self.kl_div],
                                     feed_dict={self.input_image: batch_images})

                self.c.append(closs)
                self.k.append(kloss)
                
                print("Epoch[%2d] Batch[%3d/%3d] step_loss: %.8f" % (epoch, idx, batch_idxs, step_loss))
#                self.decoder.reuse_variables()
#                print(counter, tf.get_variable("decoder/d_cv3/w"))
                
                # reconstructed image
                if np.mod(counter, 50) == 2:
                    save_images(recon_image[0:100,], [10, 10], './{}/recon_{:02d}_{:02d}.png'.format(config.sample_dir, epoch, idx))
                    save_images(batch_images[0:100,], [10, 10], './{}/orig_{:02d}_{:02d}.png'.format(config.sample_dir, epoch, idx))

                
                # sample image
                if np.mod(counter, 50) == 2:
                    sample_z = np.random.normal(0, 1, [1000, self.z_dim]).astype(np.float32)
                    samples = self.sess.run(self.sample,
                                            feed_dict={self.z:sample_z})
                    # samples = ( samples - samples.min() ) / (samples.max()-samples.min())
                    save_images(samples[0:100,], [10, 10], './{}/train_{:02d}_{:02d}.png'.format(config.sample_dir, epoch, idx))
                                   
                #######################################################
                #                   end of your code
                #######################################################
                if np.mod(counter, 500) == 2 or (epoch == config.epoch-1 and idx == batch_idxs-1):
                    self.save(config.checkpoint_dir, counter)
        
        plt.figure(2)
        plt.title('Training Loss')
        plt.xlabel('training step')
        plt.ylabel('loss')
        plt.plot(self.c)
        plt.plot(self.k)
                
    def save(self, checkpoint_dir, step):
        model_name = "mnist.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False
