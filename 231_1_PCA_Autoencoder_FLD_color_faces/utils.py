# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:26:24 2018

@author: dell
"""
import cv2
import glob
import numpy as np
import scipy.io
from skimage import color
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def read_data():
    ## image
    female_images = np.array([cv2.imread(file) for file in glob.glob('female_images/*jpg')])
    male_images = np.array([cv2.imread(file) for file in glob.glob('male_images/*jpg')])
    # cv2 read as BGR, to RGB
    female_images = female_images[:,:,:,::-1]
    male_images = male_images[:,:,:,::-1]
    images = np.concatenate((female_images, male_images), axis = 0)

    ## landmark
    female_landmarks = np.array([scipy.io.loadmat(file)['lms'] for file in glob.glob('female_landmarks/*mat')])
    male_landmarks = np.array([scipy.io.loadmat(file)['lms'] for file in glob.glob('male_landmarks/*mat')])
    landmarks = np.concatenate((female_landmarks, male_landmarks), axis = 0)
    
    ## labels
    gender_label = [0] * female_images.shape[0] + [1] * male_images.shape[0]
    
    print('Female samples: {}'.format(np.shape(female_images)[0]), 'Male samples: {}'.format(np.shape(male_images)[0]))
    # (588, 128, 128, 3) (412, 128, 128, 3)
    print('Image shape: {}'.format(images.shape))
    print('Landmark shape: {}'.format(landmarks.shape))
    np.save('images.npy', images)
    np.save('landmarks.npy', landmarks)
    np.save('labels.npy', gender_label)
    print('Data saved to npy')
    # plt.imshow(images[0])
    # plt.show()
    # print(np.shape(images))   (1000, 128, 128, 3)
    return 

def split(testing_size, images, landmarks):
    n = images.shape[0]
    np.random.seed(1)
    test_idx = np.random.choice(n, 200, replace=False, )
    train_idx = [i for i in list(range(n)) if i not in test_idx]
    train_img = images[train_idx,:]
    test_img = images[test_idx,:]
    train_lms = landmarks[train_idx,:]
    test_lms = landmarks[test_idx,:]
    return train_img, test_img, train_lms, test_lms, train_idx, test_idx

def plot_img(images,Nh,Nc, title = None):
    fig = plt.figure(figsize=(Nc, Nh))
    plt.clf()
    if title:
      plt.suptitle(title, fontsize=12)

    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)
    for i, image in enumerate(images[0:Nh*Nc]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if len(image.shape) == 2 or image.shape[2] == 1:
          image = np.reshape(image, np.shape(image)[:2])
          immin=image.min()
          immax=image.max()
          image=(image-immin)/(immax-immin+1e-8)
          plt.imshow(image, cmap ='gray')
        else:
          immin=image.min()
          immax=image.max()
          image=(image-immin)/(immax-immin+1e-8)
          plt.imshow(image)
    return fig 

def plot_lms(lms,Nh,Nc, title = None):
    fig = plt.figure(figsize=(Nc, Nh))
    plt.clf()
    if title:
      plt.suptitle(title, fontsize=16)
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)
    for i, lm in enumerate(lms):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.plot(lm[:,0], -lm[:,1], '.')
    return fig 

if __name__ == '__main__':
    read_data()