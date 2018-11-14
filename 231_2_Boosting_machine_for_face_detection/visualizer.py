import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.labels = None
		self.filters = None
		self.path = None
	
	def draw_histograms(self):
		for t in self.histogram_intervals:
			scores = self.strong_classifier_scores[t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			#bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, 100, alpha=0.5, label='Faces')
			plt.hist(neg_scores, 100, alpha=0.5, label='Non-Faces')
			#plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
			#plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers' % t)
			plt.savefig('%s/histogram_%d.png' % (self.path, t))

	def draw_rocs(self):
		plt.figure()
		for t in self.histogram_intervals:
			scores = self.strong_classifier_scores[t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('%s/ROC Curve.png' % self.path)

	def draw_wc_errors(self):
		plt.figure()
		for t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[t]
			plt.plot(1-accuracies, label = 'After %d Selection' % t)
		plt.ylabel('Error')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Error')
		plt.legend(loc = 'upper right')
		plt.savefig('%s/Weak Classifier Error.png' % self.path)

	def draw_strong_errors(self):
		errs = []
		plt.figure()
		for t in self.strong_classifier_scores:
			score = self.strong_classifier_scores[t]
			err = (np.sign(score)!=self.labels).mean()
			errs += [err]
		plt.plot(self.strong_classifier_scores.keys(), errs)
		plt.ylabel('Errors')
		plt.xlabel('Training epoch')
		plt.title('Strong Classifier Error in 100 Epochs')
		plt.savefig('%s/Strong Classifier Error'% self.path)

	def draw_haar(self, num=20):
		alphas, _ = zip(*self.filters)
		choose = np.argsort(alphas)[::-1][:num]

		fig = plt.figure()
		plt.clf()
		plt.suptitle('Top %d Haar Filters' %num , fontsize=12)
		gs = gridspec.GridSpec(5, num//5)
		gs.update(wspace=0.3, hspace=0.3)
		for i, idx in enumerate(choose):
			filter = self.filters[idx]
			weight, plus, minus = filter[0], filter[1].plus_rects[0], filter[1].minus_rects[0]
			image = np.ones((16,16))/2
			# print(image.shape, plus, minus)
			image[int(plus[0]):int(plus[2])+1, int(plus[1]):int(plus[3])+1] = 1
			image[int(minus[0]):int(minus[2])+1, int(minus[1]):int(minus[3])+1] = 0
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.title('weight: %6f' %weight, fontsize=5)
			plt.imshow(image, cmap ='gray')
		fig.savefig('%s/Top %d Haar Filters.png' %(self.path, num))


if __name__ == '__main__':
	main()
