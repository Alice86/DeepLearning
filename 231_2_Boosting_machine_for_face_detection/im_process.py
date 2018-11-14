from sklearn.feature_extraction import image as IMG
import numpy as np
import cv2
from utils import integrate_images

#extract patches from the image for all scales of the image
#return the INTEGREATED images and the coordinates of the patches
def image2patches(scales, image, patch_w = 16, patch_h = 16):
	all_patches = np.zeros((0, patch_h, patch_w))
	all_x1y1x2y2 = []
	for s in scales:
		simage = cv2.resize(image, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
		height, width = simage.shape
		print('Image shape is: %d X %d' % (height, width))
		patches = IMG.extract_patches_2d(simage, (patch_w, patch_h)) # move along the row first

		total_patch = patches.shape[0]
		row_patch = (height - patch_h + 1)
		col_patch = (width - patch_w + 1)
		assert(total_patch == row_patch * col_patch)
		scale_xyxy = []
		for pid in range(total_patch):
			y1 = pid / col_patch
			x1 = pid % col_patch
			y2 = y1 + patch_h - 1
			x2 = x1 + patch_w - 1
			scale_xyxy.append([int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)])
		all_patches = np.concatenate((all_patches, patches), axis = 0)
		all_x1y1x2y2 += scale_xyxy
	return integrate_images(normalize(all_patches)), all_x1y1x2y2

# non-maximum suppression:return a vector of prediction (0/1) after nms, same length as scores
#input: [x1, y1, x2, y2, score], threshold used for nms
#output: [x1, y1, x2, y2, score] after nms
def nms(xyxys, overlap_thresh):
	######################
	areas = (xyxys[:,2] - xyxys[:,0] + 1) * (xyxys[:,3] - xyxys[:,1] + 1)
	# sort according to the bottom-right corner 
	all_idx = np.argsort(xyxys[:,4])
	
	picks = []
	while len(all_idx) > 0:
		last = len(all_idx)-1
		print('non-suppressed remain %d' %last)
		temp_pick = all_idx[last]
		suppress = [last]
		for pos in range(last):
			i = all_idx[pos]
			xx1 = max(xyxys[i][0], xyxys[temp_pick][0])
			yy1 = max(xyxys[i][1], xyxys[temp_pick][1])
			xx2 = min(xyxys[i][2], xyxys[temp_pick][2])
			yy2 = min(xyxys[i][3], xyxys[temp_pick][3])
			
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			overlap = float(w*h) / min(areas[i], areas[temp_pick])
 			
			if overlap > overlap_thresh:
				suppress.append(pos)
		
		picks.append(temp_pick)
		all_idx = np.delete(all_idx, suppress, 0)
	
	return xyxys[np.array(picks)]

def normalize(images):
	standard = np.std(images)
	images = (images - np.min(images)) / (np.max(images) - np.min(images))
	return images

def main():
	original_img = cv2.imread('Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	scales = 1 / np.linspace(1, 10, 46)
	patches, patch_xyxy = image2patches(scales, original_img)
	print(patches.shape)
	print(len(patch_xyxy))
if __name__ == '__main__':
	main()
