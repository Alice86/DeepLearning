import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import *
from utils import *

def main():
	'''Adanoost'''
	# flag for debugging
	flag_subset = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = '%s_wc_activations.npy'%boosting_type if not flag_subset else '%s_wc_activations_subset.npy'%boosting_type
	chosen_wc_cache_dir = '%s_chosen_wcs.pkl'%boosting_type if not flag_subset else '%s_chosen_wcs_subset.pkl'%boosting_type

	# data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))

	# number of bins for boosting
	num_bins = 25

	# number of cpus for parallel computing
	num_cores = 8  #use 1 when debugging
	
	# create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	# create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100]) if not flag_subset else  Visualizer([10, 20], [1, 10, 20])
	
	# create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	# calculate filter values for all training images
	# start = time.clock()
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	# end = time.clock()
	# print('%f seconds for activation calculation' % (end - start))

	boost.train(chosen_wc_cache_dir)

	boost.visualize()
	print("Visualization completed")
	
	# Detection
	# for i in range(2):
	# 	name = 'Face_%d' %(i+1)
	# 	original_img = cv2.imread('./Testing_Images/%s.jpg' %name)
	# 	result_img = boost.face_detection(img=original_img, name=name)
	# 	cv2.imwrite('%s/Result_img_%s.png' % (boosting_type, name), result_img)

	'''Negtive Mining'''
	# mining for new dataset
	if os.path.exists('new_data.npy'):
		new_data = np.load('new_data.npy')
	else:
		new_data = []
		for i in range(3):
			name = 'Non_Face_%d' %(i+1)
			if os.path.exists('neg_patches_%s.npy'%name):
				neg_patches = np.load('neg_patches_%s.npy'%name)
			else:
				neg_img = cv2.imread('./Testing_Images/%s.jpg' %name)
				neg_patches = boost.get_hard_negative_patches(neg_img)
				np.save('neg_patches_%s.npy'%name, new_data)
			new_data += [neg_patches]

			new_data = np.concatenate(new_data, axis=0)
			np.save('new_data.npy',new_data)
	n = len(new_data)
	print('Mined negative examples: %d' %n)
	
	new_data = np.concatenate((data, new_data),0)
	new_labels = np.concatenate((labels,-1*np.ones((n))),0)

	# 'new_wc_activations.npy' & cat
	# if not os.path.exists('new_wc_activations.npy'):
	# 	ac_pre = np.load(act_cache_dir)
	# 	if os.path.exists('new_wc_activations.npy'):
	# 		ac_new = np.load('new_wc_activations.npy')
	# 		print(ac_pre.shape, ac_new.shape)
	# 	else:	
	# 		new_labels = -1*np.ones((n))		
	# 		boost_mine = Boosting_Classifier(filters, new_data, new_labels, training_epochs, num_bins, drawer, num_cores, boosting_type)
	# 		boost_mine.calculate_training_activations('new_wc_activations.npy', 'new_wc_activations.npy')
	# 	np.save('cat_wc_activations.npy', np.concatenate((ac_pre, ac_new),1))

	# run from previous
	drawer = Visualizer([10, 20, 50, 100, 150, 200], [1, 10, 20, 50, 100, 150, 200])
	cont = True
	boost_new = Boosting_Classifier(filters, new_data, new_labels, training_epochs, num_bins, drawer, num_cores, boosting_type, cont)
	boost_new.calculate_training_activations('cat_wc_activations.npy', 'cat_wc_activations.npy')
	boost_new.train(chosen_wc_cache_dir)
	
	boost_new.visualize()
	print("Visualization completed")

	# Detection
	# for i in range(2):
	# 	name = 'Face_%d' %(i+1)
	# 	original_img = cv2.imread('./Testing_Images/%s.jpg' %name)
	# 	result_img = boost_new.face_detection(img=original_img, name=name, neg=True)
	# 	cv2.imwrite('%s/Result_img_%s.png' % (boosting_type, name), result_img)

	'''Realboost'''
	training_epochs = 100
	flag_subset = False
	boosting_type = 'Real'
	act_cache_dir = '%s_wc_activations.npy'%boosting_type if not flag_subset else '%s_wc_activations_subset.npy'%boosting_type
	chosen_wc_cache_dir = '%s_chosen_wcs.pkl'%boosting_type if not flag_subset else '%s_chosen_wcs_subset.pkl'%boosting_type
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])

	realboost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	# start = time.clock()
	realboost.calculate_training_activations(act_cache_dir, act_cache_dir)
	# end = time.clock()
	# print('%f seconds for activation calculation' % (end - start))

	realboost.train(chosen_wc_cache_dir)

	realboost.visualize()
	print("Visualization completed")

	# Detection
	for i in range(1):
		name = 'real_Face_%d' %(i+1)
		nm = 'Face_%d' %(i+1)
		img = cv2.imread('./Testing_Images/%s.jpg' % nm)
		predicts = np.load('detected_%s.npy' %name)
		patch_xyxy = np.load('patch_position_%s.npy' %name)
		pos_predicts_xyxy = np.array([list(patch_xyxy[idx]) + [score] for idx, score in enumerate(predicts) if score > 0])
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		count = 0
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			if pred[4]>0:
				cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 1)
				count += 1
		cv2.imwrite('neg_Result_img_%s.png'%name, img)
		print(count)

		# original_img = cv2.imread('./Testing_Images/%s.jpg' %nm)
		# result_img = realboost.face_detection(img=original_img, name=name)
		# cv2.imwrite('%s/Result_img_%s.png' % (boosting_type, nm), result_img)

if __name__ == '__main__':
	main()
