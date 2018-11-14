import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style='Ada'):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.load_trained_wcs('Ada_chosen_wcs.pkl')
			self.visualizer.strong_classifier_scores, self.visualizer.weak_classifier_accuracies = {}, {}
			_, wcs = zip(*self.chosen_wcs)
			self.weak_classifiers = [Real_Weak_Classifier(wc.id, wc.plus_rects, wc.minus_rects, self.num_bins) for wc in wcs]
			#[Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins) for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		
		else: 
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			
			wc_activations = np.array(wc_activations)
			
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		
		for i,wc in enumerate(self.weak_classifiers):
			wc.activations = wc_activations[i, :]
		
		return wc_activations
	
	#select weak classifiers to form a strong classifier
	#after training, by calling self.sc_function(), a prediction can be made
	#self.chosen_wcs should be assigned a value after self.train() finishes
	#call Weak_Classifier.calc_error() in this function
	#cache training results to self.visualizer for visualization
	def train(self, save_dir = None, continuing=False):
		######################
		n = self.labels.shape[0]
		strong_score = np.zeros((n))
		# init
		self.chosen_wcs = []
		weights = np.ones((n))/n

		if os.path.exists(save_dir):
			self.load_trained_wcs(save_dir)
			print('Trained model loaded')
			if not continuing:
				return			

		i = -1
		if continuing:
			if os.path.exists('new_chosen_wcs.pkl'):
				self.load_trained_wcs('new_chosen_wcs.pkl')
				return
			pre_chosen = self.chosen_wcs.copy()
			self.chosen_wcs = []
			self.visualizer.strong_classifier_scores = {}
			self.visualizer.weak_classifier_accuracies = {}
			for i, (_, wc) in enumerate(pre_chosen):
				wc.activations = wc.apply_filter(self.data)
				eps = wc.calc_error(weights, self.labels)
				alpha = 0.5*np.log((1-eps)/eps)
				self.chosen_wcs += [[alpha, wc]]
				wc_pred = wc.polarity * np.sign(wc.activations-wc.threshold)
				score = alpha * wc_pred

				weights *= np.exp(-self.labels*score)
				weights /= weights.sum()

				strong_score += score
				acc = (np.sign(strong_score)==self.labels).mean()
				print('Load trained weak classifier %d with accuracy %.5f' %(i, acc) )
				self.visualizer.strong_classifier_scores[i] = strong_score.copy()
		
		for epoch in range(self.num_chosen_wc+1):

			if self.style == 'Ada':
				if self.num_cores == 1:
					wc_epss = [wc.calc_error(weights, self.labels) for wc in self.weak_classifiers]
				else:
					wc_epss = Parallel(n_jobs = self.num_cores)(delayed(wc.calc_error)(weights, self.labels) for wc in self.weak_classifiers)
					
				if epoch in self.visualizer.top_wc_intervals:
					sort_eps = np.array(wc_epss)[np.argsort(wc_epss)]
					accs = 1-sort_eps[:1000]
					self.visualizer.weak_classifier_accuracies[epoch] = accs

				choose = np.argmin(wc_epss)
				wc = self.weak_classifiers[choose]
				eps = wc.calc_error(weights, self.labels)

				alpha = 0.5*np.log((1-eps)/eps)
				self.chosen_wcs += [[alpha, wc]]

				wc_pred = wc.polarity * np.sign(wc.activations-wc.threshold)
				score = alpha * wc_pred
			
			if self.style == 'Real':
				# if self.num_cores == 1:
				# 	wc_epss = [wc.calc_error(weights, self.labels) for wc in self.weak_classifiers]
				# else:
				# 	wc_epss = Parallel(n_jobs = self.num_cores)(delayed(wc.calc_error)(weights, self.labels) for wc in self.weak_classifiers)
					
				# if epoch in self.visualizer.top_wc_intervals:
				# 	sort_eps = np.array(wc_epss)[np.argsort(wc_epss)]
				# 	accs = 1-sort_eps[:1000]
				# 	self.visualizer.weak_classifier_accuracies[epoch] = accs
										
				# choose = np.argmin(wc_epss)
				# wc = self.weak_classifiers[choose]
				wc = self.weak_classifiers[epoch]
				eps = wc.calc_error(weights, self.labels)
				self.chosen_wcs += [wc]
				score = wc.train_assignment
								
			weights *= np.exp(-self.labels*score)
			weights /= weights.sum()

			strong_score += score
			acc = (np.sign(strong_score)==self.labels).mean()
			self.visualizer.strong_classifier_scores[epoch+i+1] = strong_score.copy()
		
			if epoch % 2 == 0:
				print('Training epoch %d completed with eps %3f acc %.3f' % (epoch, eps, acc) )				

		if continuing:
			save = (self.chosen_wcs, self.visualizer.strong_classifier_scores, self.visualizer.weak_classifier_accuracies)
			pickle.dump(save, open('new_chosen_wcs.pkl', 'wb'))
			return 

		elif save_dir is not None:
			save = (self.chosen_wcs, self.visualizer.strong_classifier_scores, self.visualizer.weak_classifier_accuracies)
			pickle.dump(save, open(save_dir, 'wb'))
			return				

	def sc_function(self, image):
		if self.style == 'Ada':
			return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])	
		if self.style == 'Real':
			return np.sum([np.array([wc.predict_image(image) for wc in self.chosen_wcs])])

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs, self.visualizer.strong_classifier_scores, self.visualizer.weak_classifier_accuracies = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, name, scale_step = 10, neg=False):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		# train_predicts = []
		# for idx in range(self.data.shape[0]):
		# 	train_predicts.append(self.sc_function(self.data[idx, ...]))
		# print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		scales = 1 / np.linspace(1, 8, scale_step)


		if os.path.exists('patches_%s.npy' %name):
			patches = np.load('patches_%s.npy' %name)
			patch_xyxy = np.load('patch_position_%s.npy' %name)
			print('Patches loaded')
		else:
			patches, patch_xyxy = image2patches(scales, gray_img)
			np.save('patches_%s.npy' %name, patches)
			np.save('patch_position_%s.npy' %name, patch_xyxy)
			print('Patches saved')	
		
		if os.path.exists('detected_%s.npy' %name):
			predicts = np.load('detected_%s.npy' %name)
			pos_predicts_xyxy = np.load('detected_patches_%s.npy' %name)
			print('Detection loaded')
		else: 
			print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
			predicts = [self.sc_function(patch) for patch in tqdm(patches)]
			print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
			pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
			np.save('detected_%s.npy' %name, predicts)
			np.save('detected_patches_%s.npy' %name, pos_predicts_xyxy)
			print('Detection saved')		

		if pos_predicts_xyxy.shape[0] == 0:
			print('No faces in the image')
			return 
		
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		
		print('after nms num. of faces:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 1) #gree rectangular with line width 3
		
		if neg:
			if os.path.exists('neg_detected_%s.npy' %name):
				predicts = np.load('neg_detected_%s.npy' %name)
				pos_predicts_xyxy = np.load('neg_detected_patches_%s.npy' %name)
				print('Neg_mining detection loaded')
			else: 
				print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
				predicts = [self.sc_function(patch) for patch in tqdm(patches)]
				print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
				pos_predicts_xyxy = np.array([list(patch_xyxy[idx]) + [score] for idx, score in enumerate(predicts) if score > 0])
				np.save('neg_detected_%s.npy' %name, predicts)
				np.save('neg_detected_patches_%s.npy' %name, pos_predicts_xyxy)
				print('Neg_mining detection saved')		
						
			xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
			
			print('Neg-mining after nms num. of faces:', xyxy_after_nms.shape[0])
			for idx in range(xyxy_after_nms.shape[0]):
				pred = xyxy_after_nms[idx, :]
				cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 0, 255), 1) #gree rectangular with line width 3

		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		patches, patch_xyxy = image2patches(scales, gray_img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = np.array([self.sc_function(patch) for patch in tqdm(patches)])
		print(predicts.shape)
		wrong_patches = patches[(predicts > 0)]
		print(wrong_patches.shape)

		return wrong_patches

	def visualize(self):
		self.visualizer.filters = self.chosen_wcs
		self.visualizer.labels = self.labels
		self.visualizer.path = self.style
		
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		
		if self.style == 'Ada':
			self.visualizer.draw_wc_errors()
			self.visualizer.draw_haar()
			
		self.visualizer.draw_strong_errors()
		
