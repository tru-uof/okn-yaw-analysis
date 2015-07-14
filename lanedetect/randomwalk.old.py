import copy
import cv
import sys
import numpy as np
from steerable import AngleFilter

def to_grayscale(orig_image):
	image = cv.CreateMat(orig_image.rows, orig_image.cols,
			cv.CV_8UC1)
	cv.CvtColor(orig_image, image, cv.CV_RGB2GRAY)
	return image


def preprocess(orig_image):
	#smooth_kernel = (5, 5)
	#morph_kernel = (3, 7)

	smooth_kernel = (5, 5)
	morph_kernel = (7, 7)

	
	threshold = 20

	image = cv.CreateMat(orig_image.rows, orig_image.cols,
			cv.CV_8UC1)
	cv.CvtColor(orig_image, image, cv.CV_RGB2GRAY)
	cv.Smooth(image, image, cv.CV_BLUR, *smooth_kernel)

	img_open = cv.CloneMat(image)
	img_close = cv.CloneMat(image)

	kernel = cv.CreateStructuringElementEx(
			morph_kernel[0], morph_kernel[1],
			morph_kernel[0]/2, morph_kernel[1]/2,
			cv.CV_SHAPE_RECT)

	cv.MorphologyEx(image, img_open, None,
		kernel,cv.CV_MOP_OPEN)

	cv.MorphologyEx(image, img_close, None,
		kernel,cv.CV_MOP_CLOSE)

	cv.AbsDiff(img_open, img_close, image)

	cv.MorphologyEx(image, image, cv.CloneMat(image),
		kernel, cv.CV_MOP_CLOSE)
	
	binary = cv.CreateMat(image.rows, image.cols,
			cv.CV_8UC1)

	cv.Threshold(image, binary, threshold, 255,
		cv.CV_THRESH_BINARY)

	return binary

from scipy.stats import pearsonr
class LaneHypothesis(object):
	def __init__(self, position, momentum, weight=1.0):
		self.positions = []
		self.positions.append(position)
		self.weight = weight
		self.prev_data = None
		self.momentum = momentum
		self.dirfilter = None

	def _feat_probability(self, feat):
		std = 0.1
		return scipy.stats.norm.pdf(1.0 - feat, 0.0, scale=std)

	@property
	def position(self):
		return self.positions[-1]

	def evidence(self, data):
		if self.dirfilter is None:
			self.dirfilter = AngleFilter(data.shape)
		
		c = self.dirfilter.get_component_values(data, mode='valid')[0][0]
		feat = self.dirfilter.basis.max_response_value(c)/115.0
		#feat = pearsonr(data.flatten(), self.prev_data.flatten())[0]
		#prob = self._feat_probability(lightness*sobeldiff)
		return feat

	def prior(self, prev):
		disp = self.position - prev.position
		self.momentum = 0.5*disp + 0.5*prev.momentum
		prob = scipy.stats.norm.pdf(disp, loc=self.momentum, scale=20.0)
		return prob

def prune_particles(particles, n_particles):
	pruned = []
	positions = []
	position_weights = {}
	for p in particles:
		pos = p.position
		if pos not in position_weights:
			position_weights[pos] = []
		position_weights[pos].append(p.weight)
		#if positions.count(p.position) < n_same:
		#	pruned.append(p)
		#	positions.append(p.position)

	position_weights = [(p, np.mean(w))
		for p, w in position_weights.iteritems()]
	cumweight = sum(zip(*position_weights)[1])
	n_bins = len(position_weights)
	position_n = dict((p, round(w/cumweight*n_bins+1))
		for p, w in position_weights)
	
	for p in particles:
		if len(pruned) >= n_particles: break
		pos = p.position
		if position_n[pos] <= 0: continue
		pruned.append(p)
		position_n[pos] -= 1
	
	return pruned
		
import scipy
def lane_filter(frame, prior_loc, blocksize=(9,9), disp=(5, 5), branch=3, orig=None):
	frame = np.array(frame)
	#frame = np.mean(frame, axis=2)
	
	if disp is None:
		disp = blocksize

	def normalize_weights(particles):
		cum = sum([p.weight for p in particles])
		for p in particles:
			p.weight /= cum

	def span(origin, width):
		return slice(origin-width/2, origin+width/2)

	particles = [LaneHypothesis(p, -10) for p in \
			range(blocksize[0]/2, frame.shape[1]-blocksize[0]/2, disp[0])]

	#particles = particles[len(particles)/2:]

	#weight_prior = scipy.stats.norm(loc=prior_loc, scale=10.0)
	#for p in particles:
	#	p.weight = weight_prior.pdf(p.position)
	
	#normalize_weights(particles)

	n_particles = len(particles)

	#normalize_weights(particles)
	
	rng = range(frame.shape[0]-blocksize[0]/2, disp[1]+blocksize[0]/2, -disp[1])
	#rng = rng[:len(rng)*2/3]
	import pylab
	for i, row in enumerate(rng):
		new_particles = []
		rowspan = span(row, blocksize[1])
		particles = prune_particles(particles, n_particles)
		for p in particles:
			#pylab.scatter(p.position, row)
			#cv.Circle(orig, (p.position, row), 3, (255, 0, 0))
			colspan = span(p.position, blocksize[0])
			data = frame[rowspan, colspan]
			
			evidence = p.evidence(data)

			cv.Rectangle(orig,
				(colspan.start, rowspan.start),
				(colspan.stop, rowspan.stop), (255, 0, 0))
				

			
			age = len(p.positions)
			#p.weight = p.weight*(age/float(age+1)) + evidence*(1/float(age+1))
			p.weight *= evidence
			for new_disp in (-disp[0]*3, -disp[0]*2, -disp[0], 0, disp[0], disp[0]*2, disp[0]*3):
				new_pos = p.position + new_disp
				if new_pos < blocksize[0]/2 or \
					new_pos > frame.shape[1]-blocksize[0]/2:
					continue
				new_p = copy.deepcopy(p)
				new_p.positions.append(new_pos)
				new_p.weight *= new_p.prior(p)
				new_particles.append(new_p)
		
		new_particles.sort(key=lambda p: p.weight, reverse=True)
		particles = new_particles
		#particles = new_particles[:n_particles]
		
		#particles = new_particles
		#pylab.hist([p.weight for p in particles])
		#pylab.show()
		#max_weight = particles[0].weight
		#particles = [p for p in new_particles if p.weight > 0.2]
		#print particles[0].weight
		normalize_weights(particles)
		winner = particles[0]

	for p in particles[:1]:
		for row, pos in zip(rng, p.positions):
			cv.Circle(orig, (pos, row), 1, (0, 0, 255))
	

	#print particles[0].weight
	return particles[0].positions
	"""
	import pylab
	
	rows = range(frame.shape[0], 0, -blocksize[1])
	print len(rows)
	print len(particles[0].positions)
	pylab.plot(particles[20].positions, rows)
	pylab.imshow(frame)
	
	pylab.show()
	print [p.weight for p in particles]
	"""
			
import scipy.ndimage
def preproc_videotest():
	cap = cv.CaptureFromFile(sys.argv[1])
	if len(sys.argv) > 2:
		crop = map(int, sys.argv[2:])
		xo, yo, xe, ye = crop
	else:
		crop = None

	start = 400
	for i in range(1000000):
		frame = cv.QueryFrame(cap)
		# This is slowww
		frame = cv.GetMat(frame)
		if crop:
			frame = np.array(frame)
			frame = frame[yo:yo+ye, xo:xo+xe]
			frame = cv.fromarray(frame)
		orig = cv.CloneMat(frame)
		
		frame = np.array(to_grayscale(frame), dtype=float)
		frame /= 255.0
		#cv.Canny(frame, frame, 50.0, 220.0)
		lane = lane_filter(frame, start, orig=orig)
		start = lane[0]
		cv.ShowImage('orig', orig)
		#cv.ShowImage('proc', frame)
		cv.WaitKey(10)
	#cv.WaitKey(0)

def preproc_imagetest():
	cv.LoadImageM(sys.argv[1])
	#cv.ShowImage('orig', image)
	image = preprocess(image)
	cv.ShowImage('preproc', image)
	cv.WaitKey(0)

if __name__ == '__main__':
	preproc_videotest()
