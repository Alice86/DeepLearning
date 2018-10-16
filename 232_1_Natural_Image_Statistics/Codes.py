#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 232A - Project 1
 Author: Jiayu WU
 Date : 0114018
 Description: Codes for Project 1

"""
import os
os.getcwd() 
os.chdir("/Users/alice/Documents/Projects/232/project1") 

# Problem 1

## import image
from PIL import Image
im1 = Image.open('natural_scene_1.jpg')
im1.show()

## convert image to grey level 
im1_grey = im1.convert('L')
im1_grey.show() 

## convert intensity from uint8 to float32
import numpy as np
im1_array = np.array(im1_grey,'f')
print (im1_array.shape,im1_array.dtype)

## re-scale the intensity to [0,31]
#max = np.max(im1_array.reshape(1600*1200,1))
#min = np.min(im1_array.reshape(1600*1200,1))
#r = (max - min) / 32
im1_l = im1_array
for i in range(1200):
    for j in range(1600):
        im1_l[i,j] = int(im1_array[i,j]/8)
        
## convolve the images with a gradient filter against the horizontal axis
im1_diff = im1_l[:,1:1600] - im1_l[:,0:1599] 
        
## plot the histogram H(z) [-31, +31]
from matplotlib import pyplot as plt 
H = np.histogram(im1_diff,bins=list(range(-31,33)))
x = H[1][0:-1]
y = H[0]/np.sum(H[0])
plt.plot(x, y)
# plt.hist(im1_diff,bins=list(range(-31,32))) why so slow...
plt.title("Figure 1-1. Histgram for the horizontal difference") 
plt.show()

## log-plot logH(z)
#H_log = np.asfarray(H[0])
#for i in range(H_log.shape[0]):
#    if H_log[i] == 0:
#        H_log[i] = 1
plt.plot(x, np.log(y))
plt.title("Figure 1-2. Log-histogram for the horizontal difference") 
plt.show()

# mean, variance, and kurtosis for this histogram
from scipy.stats import kurtosis
l=im1_diff.reshape(1200*1599,1)
mean_h = np.mean(l)
var_h = np.var(l)
kts_h = kurtosis(l)
## -0.00777
## 6.1359
## 6.57756615

## fit the histogram to a Generalized Gaussian distribution
from math import gamma
sse = 100
params = [0,0]
for sigma in np.linspace(0.01,2.01,201):
    for gamm in np.linspace(0.01,2.01,201):
        y_hat = (gamm/2*sigma*gamma(1/gamm))*np.exp(-np.abs(x/sigma)**gamm)
        sse_new = np.sum((y_hat-y)**2)
        if sse_new < sse:
            sse = sse_new
            params = sigma, gamm
print(params)


from scipy.stats import gennorm
params = gennorm.fit(l)
## gamma = params[0] = 0.1578
#x=H[1][1:]-0.5
#y=H[0]
#sse = 1000000000000
#params = [0,0,0]
#for amp in np.linspace(1027700,1027900,501):
#    for sigma in np.linspace(0.75,0.8,51):
#        for gamma in np.linspace(0.7,0.75,51):
#            y_hat = amp*np.exp(-np.abs(x/sigma)**gamma)
#            sse_new = np.sum((y_hat-y)**2)
#            if sse_new < sse:
#                sse = sse_new
#                params = amp,sigma, gamma
#print(params)
### amp = 1027783.6
### sigma = 0.784
### gamma = 0.711

#def gaussian(x, sigma, gamma):
#    return gamma/(2*sigma*scipy.special.gamma(1/gamma)) * exp(-(x/sigma)**gamma)
#def gaussian(x, amp, gamma):
#    return amp*np.exp(-(np.abs(x)**gamma))
#from scipy.optimize import curve_fit
#init_vals = [0.5, 0.5]
#x=H[1][1:]-0.5
#y=H[0]
## y=np.log(H_log)
#params, covar = curve_fit(gaussian, x, y, p0=init_vals)
#print (params)
#def gaus(x, sigma, gamma):
#    return np.exp(-np.abs(x/sigma)**gamma)
#from scipy.optimize import curve_fit
#init_vals = [30, 30]
#x=H[1][1:]-0.5
#y=np.log(H_log)
## y=np.log(H_log)
#params, covar = curve_fit(gaus, x, y, p0=init_vals)
#print (params)
x1 = np.linspace(-31,31,1000)
y1 = (params[1]/2*params[0]*gamma(1/params[1]))*np.exp(-np.abs(x1/params[0])**params[1])
#pdf = gennorm.pdf(x1,params[0],loc=params[1],scale=params[2])
plt.plot(x, y, label="Histogram")
plt.plot(x1, y1, label="Fitted generalized gaussian")
plt.title("Figure 3. Fitted generalized gaussian distribution") 
plt.legend()
plt.show()

# Impose on the plot the Gaussian distribution using the mean and the variance above
from scipy.stats import norm
pdf_n = norm.pdf(x,mean_h,var_h)
plt.plot(x, y, label="Histogram")
plt.plot(x, pdf_n, label="Fitted gaussian")
plt.title("Figure 4-1. Fitted gaussian distribution") 
plt.legend()
plt.show()

# Impose the log-Gaussian distribution
plt.plot(x, np.log(y),label = "Log-histogram")
plt.plot(x, np.log(pdf_n), label = "Fitted log-gaussian")
plt.title("Figure 4-2. Fitted log-gaussian distribution") 
plt.legend()
plt.show()

# Down-sample image by a 2 * 2 average
# im1_l.shape
def down_sample(im,level):
    n=im.shape[0]
    m=im.shape[1]
    dim=np.zeros((int(n/level), int(m/level)))
    for j in range(0,m,level):
        for i in range(0,n,level):
            dim[int(i/level),int(j/level)] = np.mean(list(im[i:int(i+level),j:int(j+level)]))
    return dim
im1_1 = down_sample(im1_l,2)
diff1 = im1_1[:,1:800] - im1_1[:,0:799] 
H1 = np.histogram(diff1,bins=list(range(-31,33)))
im1_2 = down_sample(im1_1,2)
diff2 = im1_2[:,1:400] - im1_2[:,0:399] 
H2 = np.histogram(diff2,bins=list(range(-31,33)))
im1_3 = down_sample(im1_2,2)
diff3 = im1_3[:,1:200] - im1_3[:,0:199] 
H3 = np.histogram(diff3,bins=list(range(-31,33)))

plt.plot(x, y,label = "Original")
h1 = H1[0]/np.sum(H1[0])
plt.plot(x, h1, label = "1st down-sampling")
h2 = H2[0]/np.sum(H2[0])
plt.plot(x, h2, label = "2nd down-sampling")
h3 = H3[0]/np.sum(H3[0])
plt.plot(x, h3,label = "3rd down-sampling")
plt.legend()
plt.title("Figure 5-1. Histogram for the horizontal difference in 3 times down-sampling") 
plt.show()    

plt.plot(x, np.log(y),label = "Original")
plt.plot(x, np.log(h1),label = "1st down-sampling")
plt.plot(x, np.log(h2),label = "2nd down-sampling")
plt.plot(x, np.log(h3),label = "3rd down-sampling")
plt.legend()
plt.title("Figure 5-2. Log-histogram for the horizontal difference in 3 times down-sampling") 
plt.show()    

#Synthesize a uniform noise image
np.random.seed(7)
noise = np.random.uniform(0,31,(1200,1600))
diffn = noise[:,1:1600] - noise[:,0:1599] 
Hn = np.histogram(diffn,bins=list(range(-31,33)))
ln = diffn.reshape(1200*1599,1)
mean_n = np.mean(ln)
var_n = np.var(ln)
kts_n = kurtosis(ln)
## mean_n = 0.000267
## var_n = 160.2248
## kts_n = -0.60037344


yn = Hn[0]/np.sum(Hn[0])
from scipy.stats import norm
pdf_norm = norm.pdf(x,mean_n,var_n)

plt.plot(x, yn, label = "Histogram")
plt.plot(x, pdf_norm, label = "Fitted gaussian")
plt.legend()
plt.title("Figure 6-1. Histogram and fitted gaussian for the uniform noise image") 
plt.show()    

plt.plot(x, np.log(yn),label = "Log-histogram")
plt.plot(x, np.log(pdf_norm), label = "Fitted log-gaussian")
plt.legend()
plt.title("Figure 6-2. Log-histogram and fitted gaussian for the uniform noise image") 
plt.show()    

