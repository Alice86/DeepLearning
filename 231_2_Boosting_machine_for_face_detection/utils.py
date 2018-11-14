import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pdb

#image is a 2D np array
def integrate_image(image):
	return np.cumsum(np.cumsum(image, axis = 0), axis = 1)

#images is a 3D np array: a value at each pixel as the sum of above to the left
def integrate_images(images):
	ii = np.zeros(images.shape)
	for idx, i in enumerate(images):
		ii[idx, ...] = integrate_image(i)
	return ii

#generate Haar Filters: [for a rec] sum of pixels in an area (kernel 1) - sum of pixels in another area (kernel -1)
#return a list of tuples, whose first element is a list of plus rect
#and the second element is a list of minus rect
#each rect is a tuple of (x1, y1, x2, y2)
def generate_Haar_filters(w_min, h_min, w_max, h_max, image_w, image_h, short = False):
	#A type: horizonal
	filters_A = []
	for w in range(w_min, w_max + 1, 2):
		for h in range(h_min, h_max + 1):
			for x in range(image_w - w):
				for y in range(image_h - h):
					filters_A.append(([(x, y, x + w / 2 - 1, y + h - 1)],
						[(x + w / 2, y, x + w - 1, y + h - 1)]))
	#B type: vertical
	filters_B = []
	for w in range(w_min, w_max + 1):
		for h in range(h_min, h_max + 1, 2):
			for x in range(image_w - w):
				for y in range(image_h - h):
					filters_B.append(([(x, y, x + w - 1, y + h / 2 - 1)],
						[(x, y + h / 2, x + w - 1, y + h - 1)]))
	#C type: middle
	filters_C = []
	for w in range(w_min + (3 - w_min % 3), w_max + 1, 3):
		for h in range(h_min, h_max + 1):
			for x in range(image_w - w):
				for y in range(image_h - h):
					filters_C.append(([(x, y, x + w / 3 - 1, y + h - 1), (x + 2 * w / 3, y, x + w - 1, y + h - 1)],
						[(x + w / 3, y, x + 2 * w / 3 - 1, y + h - 1)]))
	#D type: diagonal
	filters_D = []
	for w in range(w_min, w_max + 1, 2):
		for h in range(h_min, h_max + 1, 2):
			for x in range(image_w - w):
				for y in range(image_h - h):
					filters_D.append(([(x, y, x + w / 2 - 1, y + h / 2 - 1), (x + w / 2, y + h / 2, x + w - 1, y + h - 1)],
						[(x + w /2, y, x + w - 1, y + h / 2 - 1), (x, y + h / 2, x + w / 2 - 1, y + h - 1)]))
	filters = []
	if short:
		partial_number = 250
		filters = filters_A[0: partial_number] + filters_B[0: partial_number] +\
				  filters_C[0: partial_number] + filters_D[0: partial_number]
	else:
		filters = filters_A + filters_B + filters_C + filters_D
	return filters

def read_images(data_dir, w, h):
	files = os.listdir(data_dir)
	fimages = [os.path.join(data_dir, f) for f in files if f[-3:] == 'bmp']
	images = np.zeros((len(fimages), h, w))
	for i, fi in enumerate(fimages):
		images[i, ...] = cv2.imread(fi, cv2.IMREAD_GRAYSCALE)
	return images

def load_data(pos_data_dir, neg_data_dir, image_w, image_h, subset = False):
	if os.path.exists('raw_data.npy'):
		pos_data, neg_data = np.load('raw_data.npy')
	else:
		pos_data = read_images(pos_data_dir, image_w, image_h)
		neg_data = read_images(neg_data_dir, image_w, image_h)
		np.save('raw_data.npy', (pos_data, neg_data))
	
	if subset:
		pos_data = pos_data[0: 100, ...]
		neg_data = neg_data[0: 100, ...]
	else:
		neg_data = neg_data[0: 25356, ...]
	pos_labels = np.ones(pos_data.shape[0])
	neg_labels = -1 * np.ones(neg_data.shape[0])

	data = np.concatenate((pos_data, neg_data), axis = 0)
	labels = np.concatenate((pos_labels, neg_labels))
	assert(data.shape[0] == labels.shape[0])
	print('Load in %d images, %d faces, %d non-faces' % (data.shape[0], pos_data.shape[0], neg_data.shape[0]))
	return data, labels

def main():
	i = np.random.randint(5, size=(3, 3, 3))
	print(i)
	print(integrate_images(i))
	filters = generate_Haar_filters(4, 4, 16, 16, 16, 16)
	print(len(filters))

if __name__ == '__main__':
	main()
