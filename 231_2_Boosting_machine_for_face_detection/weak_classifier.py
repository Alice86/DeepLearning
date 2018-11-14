from abc import ABC, abstractmethod
import numpy as np
from joblib import Parallel, delayed
from utils import *

class Weak_Classifier(ABC):
	#initialize a harr filter with the positive and negative rects
	#rects are in the form of [x1, y1, x2, y2] 0-index
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		self.id = id
		self.plus_rects = plus_rects
		self.minus_rects = minus_rects
		self.num_bins = num_bins
		self.activations = None

	#take in one integrated image and return the value after applying the image
	#integrated_image is a 2D np array
	#return value is the number BEFORE polarity is applied
	def apply_filter2image(self, integrated_image):
		pos = 0
		for rect in self.plus_rects:
			rect = [int(n) for n in rect]
			pos += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		neg = 0
		for rect in self.minus_rects:
			rect = [int(n) for n in rect]
			neg += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		return pos - neg
	
		
	#take in a list of integrated images and calculate values for each image
	#integrated images are passed in as a 3-D np-array
	#calculate activations for all images BEFORE polarity is applied
	#only need to be called once
	def apply_filter(self, integrated_images):
		values = []
		for idx in range(integrated_images.shape[0]):
			values.append(self.apply_filter2image(integrated_images[idx, ...]))
		if (self.id + 1) % 100 == 0:
			print('Weak Classifier No. %d has finished applying' % (self.id + 1))
		return values
	
	#using this function to compute the error of
	#applying this weak classifier to the dataset given current weights
	#return the error and potentially other identifier of this weak classifier
	#detailed implementation is up you and depends
	#your implementation of Boosting_Classifier.train()
	@abstractmethod
	def calc_error(self, weights, labels):
		pass
	
	@abstractmethod
	def predict_image(self, integrated_image):
		pass

class Ada_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.polarity = None
		self.threshold = None

	def calc_error(self, weights, labels):
		######################
		hist, thresholds = np.histogram(self.activations, self.num_bins)
		sort_idx = np.argsort(self.activations)
		sort_labels = labels[sort_idx]
		sort_weights = weights[sort_idx]
		
		n = labels.shape[0]
		c_idx = 0 
		eps = 1
		for i, threshold in enumerate(thresholds):
			polarity = 1
			h = np.ones(n)    
			# neg_idx = (self.activations < threshold)
			h[:c_idx] = -1
			eps_new = (sort_weights*(sort_labels!=h)).sum()
			if eps_new > 0.5:
				polarity = -1
				h *= -1
				eps_new = (sort_weights*(sort_labels!=h)).sum() 
			if eps_new < eps:
				eps = eps_new.copy()
				self.polarity = polarity
				self.threshold = threshold
			if i < self.num_bins:
				c_idx += hist[i]		
		return eps
		
	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		return self.polarity * np.sign(value - self.threshold)

class Real_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.thresholds = None #this is different from threshold in ada_weak_classifier, think about it
		self.bin_pqs = None
		self.train_assignment = None

	def calc_error(self, weights, labels):
		######################
		hist, self.thresholds = np.histogram(self.activations, self.num_bins)
		sort_idx = np.argsort(self.activations)
		sort_labels = labels[sort_idx]
		sort_weights = weights[sort_idx]
		
		n = labels.shape[0]
		s = 0
		self.bin_pqs = np.zeros((2, self.num_bins+2))
		self.train_assignment = np.zeros((n))
		pq = sort_weights*sort_labels
		for i in range(self.num_bins):
			hs = hist[i]
			ps = sum([p for p in pq[s:s+hs] if p>0])
			qs = sum([-q for q in pq[s:s+hs] if q<0])
			self.bin_pqs[0,i+1] = ps
			self.bin_pqs[1,i+1] = qs 
			self.train_assignment[sort_idx[s:s+hs]] = 0.5 * np.log((ps+1e-12)/(qs+1e-12))		
			s += hs
		eps = (np.sign(self.train_assignment)!=labels).mean() #z=2*sum([np.sqrt(self.bin_pqs[0]*self.bin_pqs[1])]) # 
		return eps

	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		bin_idx = np.sum(self.thresholds < value)
		return 0.5 * np.log((self.bin_pqs[0, bin_idx]+1e-12)/(self.bin_pqs[1, bin_idx]+1e-12))

def main():
	plus_rects = [(1, 2, 3, 4)]
	minus_rects = [(4, 5, 6, 7)]
	num_bins = 50
	id=0
	ada_hf = Ada_Weak_Classifier(id, plus_rects, minus_rects, num_bins)
	real_hf = Real_Weak_Classifier(id, plus_rects, minus_rects, num_bins)

if __name__ == '__main__':
	main()
