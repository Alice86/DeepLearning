#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing AWA1 VGG19 features
    Merge features and label classes based on .txt files 
    Obtain 'features.txt' and 'labels.txt' to be used in the model
"""

import numpy as np 
import os
import glob

def preprocessing():
    
    if os.path.isfile('features.txt') and os.path.isfile('labels.txt'):
        print('Feature has been processed.')
    else:
        print('Processing feature...')
        classes = np.loadtxt("classes.txt", dtype=str, usecols=[1])
        class_labeling = {k:v for v, k in zip(list(range(1,len(classes)+1)),classes)}
        # folder_list = glob.glob('feature/**')
        f, l = [], []
        for folder, label in class_labeling.items():
        # for i, folder in enumerate(folder_list):
            for file in glob.glob(os.path.join(*['feature', folder, "*.txt"])):
                l.append(label)
                new = np.loadtxt(file)
                f.append(new.reshape([1,-1]))   
        features = np.concatenate(f, axis=0)
        print('Saving feature as txt')
        np.savetxt('features.txt', features, fmt='%.8e')
        np.savetxt('labels.txt', l, fmt='%d')
    
if __name__ == '__main__':
    preprocessing()