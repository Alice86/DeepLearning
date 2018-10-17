"""
Functions for data manipulation
"""
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import math
from PIL import Image
import scipy.misc

import json
from time import gmtime, strftime

'''Load data from .txt files'''

def Load_feature(path):
    if os.path.isfile(os.path.join(path, "features.npy")) and os.path.isfile(os.path.join(path, "labels.npy")):
        print('Loading from .npy: {}'.format(path))
        features = np.load(os.path.join(path, "features.npy"))
        labels = np.load(os.path.join(path, "labels.npy"))
    else:
        print('Loading from .txt: {}'.format(path))
        features = np.loadtxt(os.path.join(path, "features.txt"), dtype=float)
        labels = np.loadtxt(os.path.join(path, "labels.txt"), dtype=int)
        data = (features, labels)
        np.save(os.path.join(path, "features.npy"), features)
        np.save(os.path.join(path, "labels.npy"), labels)
    print('Data loaded, shape: {}'.format(features.shape))
    return features, labels

def Attibute_process(path):
    attributes = np.loadtxt(os.path.join(path, "attributes.txt"), dtype=float)
    print('Attribute vector shape: {}'.format(attributes.shape))
    return attributes # G*M

def Class_label_indexing(path):
    classes = np.loadtxt(os.path.join(path, "classes.txt"), dtype=str, usecols=[1])
    class_labeling = {k:v for v, k in zip(list(range(1,len(classes)+1)),classes)}
    return class_labeling

def Split(path, labels, class_label):
    test = np.loadtxt(os.path.join(path, "test_classes.txt"), dtype=str)
    test_class = [class_label[key] for key in test]
    train_class = [l for l in range(1, len(class_label)+1) if l not in test_class]
    test_index = [i for i,l in enumerate(labels) if l in test_class]
    train_index = [i for i in range(len(labels)) if i not in test_index]
    return train_index, test_index, train_class, test_class

def normalize(x, low_bound, up_bound):
    min_val = x.min(axis=0)
    max_val = x.max(axis=0)
    a = (up_bound-low_bound)/(max_val-min_val)
    b = up_bound - a*max_val
    return a*x+b

'''Load raw image data (Some codes from UCLA STAT M232A course project templates)'''

IMG_EXTENSIONS = ['.jpeg', '.png']

class Image_process(object):
    def __init__(self, data_path, image_length=224, image_width=224):
        self.root_dir = data_path
        self.imgList = [f for f in os.listdir(data_path) if any(f.endswith(ext) for ext in IMG_EXTENSIONS)]
        self.imgList.sort()
        self.image_length = image_length
        self.image_width = image_width
        self.images = np.zeros((len(self.imgList), image_length, image_width, 3)).astype(float)
        print('Loading dataset: {}'.format(data_path))

        for i in range(len(self.imgList)):
            image = Image.open(os.path.join(self.root_dir, self.imgList[i])).convert('RGB')
            image = Image.resize((self.image_length, self.image_width))
            image = np.array(image).astype(float)
            self.images[i] = image
        print('Data loaded, shape: {}'.format(self.images.shape))

    def raw_data(self):
        return self.images

    def normalize(self, low_bound, up_bound, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        min_val = self.images.min()
        max_val = self.images.max()
        image_to_range = low_bound + (self.images - min_val) / (max_val - min_val) * (up_bound - low_bound)
        return (image_to_range - mean) / std

'''Image operations (From UCLA STAT M232A course project templates)'''

def save_images(images, file_name, space=0, mean_img=None):
    scipy.misc.imsave(file_name, merge_images(images, space, mean_img))

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return imread(image_path, is_grayscale), image_size, is_crop, resize_w

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

