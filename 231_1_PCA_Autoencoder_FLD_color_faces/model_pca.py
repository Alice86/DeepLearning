#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA
"""

import numpy as np
import cv2
from skimage import color
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import *
from mywarper import *

class face_PCA(object):
    def __init__(self, z_dim=50, l_dim=10, image_size=128, num_landmark=68):
        """
        Args for model:
        """
        self.z_dim = z_dim
        self.l_dim = l_dim
        self.image_size = image_size
        self.num_landmark = num_landmark
        
    def image_to_data(self, images):
        # rgb to hsv
        img_hsv = np.array([color.rgb2hsv(img) for img in images])
        # extract the V value
        data = np.reshape(img_hsv[:,:,:,2], [-1, self.image_size*self.image_size])
        return data, img_hsv[:,:,:,0:2]

    def data_to_image(self, data, reserve):
        img_v = np.reshape(data, [-1, self.image_size, self.image_size, 1])
        img_hsv = np.concatenate((reserve, img_v), axis = 3)
        img_rgb = np.array([color.hsv2rgb(img) for img in img_hsv])
        return img_rgb
    
    def pca_face(self, train_img, num_plot=None, plot=True):
        train_data, _ = self.image_to_data(train_img)
        if not num_plot:
          num_plot=self.z_dim
        pca = PCA(n_components=num_plot)
        pca.fit(train_data)
        fig = None
        if plot:
            eigenvectors = pca.components_
            eigen_face = np.reshape(eigenvectors, [num_plot, self.image_size, self.image_size])
            fig = plot_img(eigen_face, 1, num_plot, 'Eigen Faces')
        return pca, fig

    def pca_landmark(self, train_lms, num_plot=None, plot=True):
        train_flat = np.reshape(train_lms, [-1, self.num_landmark*2])
        if not num_plot:
          num_plot=self.l_dim
        pca = PCA(n_components=num_plot)
        pca.fit(train_flat)
        fig = None
        if plot:
            eigenvectors = pca.components_
            eigen_lms = np.reshape(eigenvectors, [num_plot, self.num_landmark, 2]) + train_lms.mean(axis=0)
            fig = plot_lms(eigen_lms, 2, num_plot/2, 'Eigen Landmarks')
        return pca, fig 
        
    def reconstruction_img(self, samples, train_img, pca=None):
        if pca == None:
            pca, _ = self.pca_face(train_img, None, False)
        # recon
        data, reserve = self.image_to_data(samples)
        reduced = pca.transform(data)
        recon = pca.inverse_transform(reduced)
        recon_img = self.data_to_image(recon, reserve)
        # intensity 
        #errs = [(cv2.cvtColor(np.int32(s), cv2.COLOR_RGB2GRAY)-cv2.cvtColor(np.int32(r), cv2.COLOR_RGB2GRAY))**2 
        #    for s, r in zip(samples, recon_img)]
        errs = (recon-data)**2
        err = np.mean(errs)
        return recon_img, err
 
    def reconstruction_lms(self, lms, train_lms, pca=None):
        if pca == None:
            pca, _ = self.pca_landmark(train_lms, None, False)
        # recon
        flatten = np.reshape(lms,[-1, self.num_landmark*2])
        reduced = pca.transform(flatten)
        recon = pca.inverse_transform(reduced)
        recon_lms = np.reshape(recon, [-1, self.num_landmark, 2])
        errs = np.sqrt(((recon_lms - lms)**2).sum(axis=2))
        err = errs.mean()
        # err = errs.mean()
        return recon_lms, err  

    def warp_pca_face(self, train_img, train_lms, pca_lms):
        if not pca_lms:
            pca_lms, _ = self.pca_landmark(train_lms, None, False)
        train_warp = np.array(list(map(warp, train_img, train_lms, [train_lms.mean(axis=0)]*train_lms.shape[0])))
        warp_pca_img, _ = self.pca_face(train_warp, None, False)
        return warp_pca_img, pca_lms
      
    def reconstruction(self, train_img, test_img, train_lms, test_lms, pca_lms, warp_pca_img):
        if not pca_lms:
            pca_lms, _ = self.pca_landmark(train_lms, None, False)
        if not warp_pca_img:
            warp_pca_img, _ = self.warp_pca_face(train_img, train_lms, pca_lms)          
        # pca landmark
        recon_lms, err_lms = self.reconstruction_lms(test_lms, train_lms, pca_lms)
        # warp and eigen image
        train_mean = [train_lms.mean(axis=0)]*train_lms.shape[0]
        train_warp = np.array(list(map(warp, train_img, train_lms, train_mean)))
        # test_mean = [test_lms.mean(axis=0)]*test_lms.shape[0]
        test_warp = np.array(list(map(warp, test_img, test_lms, train_mean)))
        recon_warp, err_img = self.reconstruction_img(test_warp, train_warp, warp_pca_img) 
        recon_img = np.array(list(map(warp, recon_warp, train_mean, recon_lms)))
        errs = (self.image_to_data(recon_img)[0]-self.image_to_data(test_img)[0])**2
        err = errs.mean()
        print(err)
        return recon_img, err, err_lms, err_img
      
    def sample(self, train_img, train_lms, sample_size, pca_lms, warp_pca_img, plot = True):
        if not pca_lms:
            pca_lms, _ = self.pca_landmark(train_lms, None, False)
        if not warp_pca_img:
            warp_pca_img, _ = self.warp_pca_face(train_img, train_lms, pca_lms)
        lms_vectors, lms_values = pca_lms.components_, pca_lms.explained_variance_
        img_vectors, img_values = warp_pca_img.components_, warp_pca_img.explained_variance_
        
        mean_lms = [train_lms.mean(axis=0)]*sample_size
        weight_lms = np.concatenate([np.random.normal(0,np.sqrt(value),[sample_size,1]) for value in lms_values], axis=1)
        sample_lms = np.reshape(np.dot(weight_lms, lms_vectors), [sample_size,self.num_landmark,2]) + train_lms.mean(axis = 0)
        
        train_data, img_hs = self.image_to_data(train_img)
        mean_img = train_data.mean(axis=0)
        weight_img = np.concatenate([np.random.normal(0,np.sqrt(value),[sample_size,1]) for value in img_values], axis=1)
        sample_data = np.dot(weight_img, img_vectors) + mean_img
        # sample_img = self.data_to_image(sample_data, np.array([img_hs[4]]*sample_size))
        sample_img = np.reshape(sample_data, [sample_size, self.image_size, self.image_size, 1])

        sample_warp = np.array(list(map(warp, sample_img, mean_lms, sample_lms)))
        fig = None
        if plot:
            fig = plot_img(sample_warp, 5, sample_size//5, 'Sample Images')
        return sample_warp, fig    

if __name__ == '__main__':
    # load data
    images = np.load("images.npy")
    landmarks = np.load("landmarks.npy")
    labels = np.load("labels.npy")
    train_img, test_img, train_lms, test_lms, train_idx, test_idx = split(200, images, landmarks)
    # print(np.shape(train_lms.mean(axis=0, keepdims=True)))
    print('Data loaded', np.shape(images))

    # Model
    face_pca = face_PCA(z_dim=50, l_dim=10)

    # 1-1. plot eigen faces
    pca_img, fig1_1_1 = face_pca.pca_face(train_img, num_plot=10, plot = True)
    fig1_1_1.savefig('fig/Fig1_1_1.png')
    plt.close()

    # 1-2. plot 10 reconstruction
    samples = test_img[:10]
    recon_img, _ = face_pca.reconstruction_img(samples, train_img, pca_img)
    plot_image = np.concatenate((recon_img, samples), axis = 0)
    fig1_1_2 = plot_img(plot_image,2,10,'Reconstructed v.s original faces')
    fig1_1_2.savefig('fig/Fig1_1_2.png')
    plt.close()

    # 1-3. plot error on latent dim
    comps = list(range(0,51,5))
    comps[0] = 1

    img_errs = [face_PCA(z_dim=n_comp).reconstruction_img(test_img, train_img, None)[1] for n_comp in comps]
    plt.title('Reconstruction Error over Num. of Components')
    plt.xlabel('Num. of Components')
    plt.ylabel('Reconstruction Error')
    plt.plot(comps, img_errs)
    plt.savefig('fig/Fig1-1-3')
    plt.close()
    print('Eigen face done')
    
    # 2-1. plot eigen landmark
    pca_lms, fig1_2_1 = face_pca.pca_landmark(train_lms, num_plot=10, plot=True)
    fig1_2_1.savefig('fig/Fig1_2_1.png')
    plt.close()

    # 2-2. plot error on latent dim
    lms_errs = [face_PCA(l_dim=n_comp).reconstruction_lms(test_lms, train_lms, None)[1] for n_comp in comps]
    plt.title('Landmark Reconstruction Error over Num. of Components')
    plt.xlabel('Num. of Components')
    plt.ylabel('Reconstruction Error')
    plt.plot(comps, lms_errs)
    plt.savefig('fig/Fig1_2_2')
    plt.close()
    print('Eigen landmark done')

    """ Geometry + Appearance"""
    #3-1. plot reconstruction
    warp_pca_img, _ = face_pca.warp_pca_face(train_img, train_lms, pca_lms)
    sample_img = test_img[:20]
    sample_lms = test_lms[:20]
    recon_img, err, err_lms, err_img = face_pca.reconstruction(train_img, sample_img, train_lms, sample_lms, pca_lms, None)
    fig1_3_1a = plot_img(recon_img, 4,5,'Reconstructed faces')
    fig1_3_1b = plot_img(sample_img, 4,5,'Original faces')
    fig1_3_1a.savefig('fig/Fig1_3_1a')
    fig1_3_1b.savefig('fig/Fig1_3_1b')

    #3-2 plot errs on latent dim
    errs, lmss, imgs = [], [], []
    for n_comp in comps:
        recon_img, err, err_lms, err_img = face_PCA(z_dim=n_comp).reconstruction(train_img, test_img, train_lms, test_lms, pca_lms, None)
        errs += [err]
        lmss += [err_lms]
        imgs += [err_img]

    plt.title('Reconstruction Errors over Num. of Components')
    plt.xlabel('Num. of Components')
    plt.ylabel('Reconstruction Error')
    plt.plot(comps, errs, label = 'total', color='black')
    plt.savefig('fig/Fig1_3_2a')
    plt.close()

    plt.title('Geometry loss over Num. of Components')
    plt.xlabel('Num. of Components')
    plt.ylabel('Error')
    plt.plot(comps, lmss)
    plt.savefig('fig/Fig1_3_2b')
    plt.close()

    plt.title('Appearance loss over Num. of Components')
    plt.xlabel('Num. of Components')
    plt.ylabel('Error')
    plt.plot(comps, imgs)
    plt.savefig('fig/Fig1_3_2c')
    plt.close()
    print('Eigen warp face done')

    #3-2 sample image
    _, fig1_3_2 = face_pca.sample(train_img, train_lms, 50, pca_lms, None, plot = True)
    fig1_3_2.savefig('fig/Fig_1_3_2')
    print('Sample face done')

    #FLD: