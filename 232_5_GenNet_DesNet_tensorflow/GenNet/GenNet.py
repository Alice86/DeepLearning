from __future__ import division

import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from ops import *
from datasets import *


class GenNet(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.delta = config.delta
        self.sigma = config.sigma
        self.sample_steps = config.sample_steps
        self.z_dim = config.z_dim

        self.num_epochs = config.num_epochs
        self.data_path = os.path.join(config.data_path, config.category)
        self.log_step = config.log_step
        self.output_dir = os.path.join(config.output_dir, config.category)

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

        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32, name="latent") 
        self.real_image = tf.placeholder(
                        shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name="real")

        self.build_model()

    def generator(self, inputs, reuse=False, is_training=True):
        ####################################################
        # Define the structure of generator, you may use the
        # generator structure of DCGAN. ops.py defines some
        # layers that you may use.
        ####################################################
        def batch_norm(input, epsilon=1e-5, momentum=0.9, train=True, name="batch_norm"):
            return tf.contrib.layers.batch_norm(input, decay=momentum,
                                        updates_collections=None, epsilon=epsilon,
                                        scale=True, is_training=is_training, scope=name)

        with tf.variable_scope('gen', reuse=reuse):
            h1r = tf.expand_dims(tf.expand_dims(inputs, 1), 1)

            h2 = tf.layers.conv2d_transpose(h1r, 3*128, [4, 4], strides=(2, 2), padding='valid', name='g_cv2')
            h2r = leaky_relu(batch_norm(h2, train=is_training, name='g_bn2'))

            h3 = tf.layers.conv2d_transpose(h2r, 3*64, [5, 5], strides=(2, 2), padding='same', name='g_cv3')
            h3r = leaky_relu(batch_norm(h3, train=is_training, name='g_bn3'))

            h4 = tf.layers.conv2d_transpose(h3r, 3*32, [5, 5], strides=(2, 2), padding='same', name='g_cv4')
            h4r = leaky_relu(batch_norm(h4, train=is_training, name='g_bn4'))

            h5 = tf.layers.conv2d_transpose(h4r, 3*16, [5, 5], strides=(2, 2), padding='same', name='g_cv5')
            h5r = leaky_relu(batch_norm(h5, train=is_training, name='g_bn5'))

            h6 = tf.layers.conv2d_transpose(h5r, 3, [5, 5], strides=(2, 2), padding='same', name='g_cv6')

            return tf.nn.tanh(h6)


    def langevin_dynamics(self, z, Y):
        ####################################################
        # Define Langevin dynamics sampling operation.
        # To define multiple sampling steps, you may use
        # tf.while_loop to define a loop on computation graph.
        # The return should be the updated z.
        ####################################################
        def step(z, Y, i):
            sigma = self.sigma
            delta = self.delta
            fz = self.generator(z, reuse=True)
            grad = tf.gradients(0.5/(sigma**2)*tf.norm(Y - fz, ord=2, axis=0), z, name='grad_z')[0]
            energy = - grad - z
            noise = tf.random_normal(shape=tf.shape(z), name='noise')
            z = z  + delta*noise + 0.5*(delta**2)*energy
            i += 1
            return z, Y, i
        
        cond = lambda z, Y, i: tf.less(i, self.sample_steps)
        body = lambda z, Y, i: step(z, Y, i)
        
        with tf.name_scope("langevin"):
            i = tf.constant(0)
            output, _, _ = tf.while_loop(cond, body, [z, Y, i])
        
            return output


    def build_model(self):
        ####################################################
        # Define the learning process. Record the loss.
        ####################################################
        
        self.train_z = self.generator(self.z, reuse = False)
        self.gen_z = self.generator(self.z, reuse = True)
        
        # loss
        self.train_loss = tf.reduce_sum(0.5/(self.sigma**2)*tf.norm(self.real_image - self.train_z, ord=2, axis=0) )
        
        self.sampler = self.langevin_dynamics(self.z, self.real_image)
        
        # optimizer
        self.vars = [var for var in tf.trainable_variables() if 'gen' in var.name]
        self.optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1).minimize(self.train_loss, var_list=self.vars)
        
        

    def train(self):
        # Prepare training data
        train_data = DataSet(self.data_path, image_size=self.image_size)
        train_data = train_data.to_range(-1, 1)

        num_batches = int(math.ceil(len(train_data) / self.batch_size))
        summary_op = tf.summary.merge_all()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)
        writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.graph.finalize()

        print('Start training ...')

        ####################################################
        # Train the model here. Print the loss term at each
        # epoch to monitor the training process. You may use
        # save_images() in ./datasets.py to save images. At
        # each log_step, record model in self.model_dir,
        # reconstructed images and synthesized images in
        # self.sample_dir, loss in self.log_dir (using writer).
        ####################################################
        data_len = train_data.shape[0]
        
        t_loss_log = []
        counter = 0
        sample = np.zeros((self.batch_size, self.z_dim))
        
        for epoch in range(self.num_epochs):
            batch_idxs = data_len // self.batch_size

            for idx in range(batch_idxs):
                counter += 1
                batch_images = train_data[idx * self.batch_size:min(data_len, (idx+1)*self.batch_size)]               
                
                _, t_loss = self.sess.run([self.optim, self.train_loss], feed_dict={
                                        self.real_image: batch_images,
                                        self.z: sample})
                
                sample = self.sess.run(self.sampler, feed_dict={self.z: sample,
                                                                self.real_image: batch_images})                
                
                t_loss_log.append(t_loss)
                
                
            if epoch % 50 == 0:
                print("Epoch[%2d], train_loss: %.6f" % (epoch, t_loss))
                # recon
                recon = self.sess.run(self.gen_z, {self.z: sample})
                # random image
                z = np.random.normal(size=[64, self.z_dim])
                gen = self.sess.run(self.gen_z, {self.z: z})
                # interpolation
                grid = np.linspace(-2,2,8)
                inter_z = np.array(np.meshgrid(grid, grid)).reshape(2,-1).T
                inter = self.sess.run(self.gen_z, {self.z: inter_z})
                save_images(inter, "%s/interpolation-%03d.png" % (self.sample_dir, epoch))
                
                # save image
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                save_images(gen, "%s/generation-%03d.png" % (self.sample_dir, epoch))
                save_images(recon, "%s/reconstruction-%03d.png" % (self.sample_dir, epoch))
                
        
        plt.plot(t_loss_log)
        plt.title('Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig("%s/loss.png" % (self.sample_dir))
        plt.clf()

