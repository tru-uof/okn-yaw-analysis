import sys
import pylab
import numpy as np
from steerable import AngleFilter
import scipy.cluster
import scipy.stats
import itertools
import curvemodel
from tru.video_capture import VideoFrames

def interval_probability(a, b, cdf):
	return cdf(b) - cdf(a)

def discretized_probability(center, accuracy, cdf):
	return interval_probability(center-accuracy/2.0, center+accuracy/2.0, cdf)

exp_pdf = lambda x,var=30: 1.0/(var*np.sqrt(2*np.pi))*np.exp(-x**2/(2*var**2))
class LaneHypothesis(object):
	init_x = 450
	skip_p = scipy.stats.norm.pdf(30, loc=0, scale=30)
	curvchangecdf = scipy.stats.norm(0, 1.0).cdf
	
	def __init__(self, position, angle, parent=None, weight=None):
		self.position = position
		self.angle = angle
		self.parent = parent

		if weight is None:
			x, y = self.position
			diff = np.linalg.norm(np.subtract((x, 3*y), (self.init_x, 0)))
			weight = scipy.stats.norm.pdf(diff, loc=0, scale=30)
		
		self.weight = weight
		
		self.stopped = False

	def mean_weight(self):
		weights = [p.weight for p in walk_particle(self)]
		prod = np.product(weights)
		return prod**(1/float(len(weights)))

	def spawn_next(self, candidates):
		children = []
		
		for (x, y), angle, weight in candidates:

			#angle = np.arctan2(y-self.position[1], x-self.position[0])
		
			pred_angle = 0.5*angle + 0.5*self.angle
			exp_delta = (y-self.position[1])/np.tan(pred_angle)
			exp_pos = self.position[0]+exp_delta

			exp_diff = (x - exp_pos)
			x = exp_pos
	
			exp_diff_p = exp_pdf(exp_diff)
			
			child = LaneHypothesis((x, y), angle, self, exp_diff_p)
			children.append(child)
		
		skip = LaneHypothesis(self.position, self.angle, self, self.skip_p)
		children.append(skip)
		return children
	
def walk_particle(p):
	while p is not None:
		yield p
		p = p.parent

class HypothesisTracker(object):
	def spawn_hypotheses(self, particles, candidates):
		new_paths = []
		for pos, a, weight in candidates:
			new = LaneHypothesis(pos, a)
			new_paths.append(new)
		
		continuations = []
		for p in particles:
			if p.stopped:
				if len(list(walk_particle(p))) < 4:
					particles.remove(p)
				continue
			p.stopped = True
			new_paths.extend(p.spawn_next(candidates))

		particles.extend(new_paths)
		
		particles.sort(key=lambda p: p.mean_weight(), reverse=True)
		retained = []

		while len(particles) > 0:
			p = particles.pop(0)
			retained.append(p)
			n = 5
			lineage = list(walk_particle(p))
			if len(lineage) <= n:
				continue
			parent = lineage[n]
			for cand in particles:
				lineage = list(walk_particle(cand))
				linpos = [l.position for l in lineage]
				if parent.position in linpos:
					particles.remove(cand)
		
		particles = retained[:5]
		return particles

def detwalk(image, prev_start):
	blocksize = (9, 9)

	filt = AngleFilter(blocksize)
	
	rows = range(0, image.shape[0]-blocksize[1], blocksize[1]/2)
	
	tracker = HypothesisTracker()
	particles = []
	
	for i, row in enumerate(rows):
		candidates = []
		d = image[row:row+blocksize[1]]
		comp = filt.get_component_values(d, mode='valid')[0]
		angles = filt.basis.max_response_angle(comp.T)
		values = filt.basis.max_response_value(comp.T)

		#valid = values > 1.5
		valid = np.argsort(values)[::-1][:5]
		indices = np.array(range(len(angles)), dtype=float)
		
		angles = np.array(angles[valid])
		feats = np.array((indices[valid]))
		weights = np.array((values[valid]))
		if len(feats) < 2: continue
		feats = feats.reshape((len(feats), 1))
		clusters = scipy.cluster.hierarchy.fclusterdata(feats, 3, criterion='distance')
		for c in np.unique(clusters):
			x = np.mean(feats[clusters == c])+blocksize[0]/2.0
			y = row+blocksize[1]/2.0
			#pylab.scatter(x, y)
			candidates.append(((x, y),
					np.median(angles[clusters == c]),
					np.mean(values[clusters == c])))
		candidates.sort(key=lambda x: x[-1])
		particles = tracker.spawn_hypotheses(particles, candidates)

	particles.sort(key=lambda p: p.mean_weight(), reverse=True)
	return particles

def frames(video, start=0.0, crop=None):
	the_frames = VideoFrames(video)
	the_frames.seek(start)

	if crop:
		xo, yo, xe, ye = crop

	for ts, frame in the_frames:
		frame = np.array(frame, dtype=float)
		frame /= 255.0
		orig_frame = frame
		if crop:
			frame = frame[yo:yo+ye, xo:xo+xe]
		frame = frame[::-1]
		frame = np.mean(frame, axis=2)
		frame = frame**10

		#if prev_frame is not None:
		#	frame = 0.5*prev_frame + 0.5*frame
		
		yield frame, orig_frame, ts

if __name__ == '__main__':
	#np.seterr(invalid='raise')
	import time
	print 'fuu'
	pylab.ion()

	crop = sys.argv[5:]
	crop = map(int, crop)

	start = float(sys.argv[2])
	end = float(sys.argv[3])
	init_x = int(sys.argv[4])

	prev_time = time.time()

	def posconvert((x, y)):
		x = x + crop[0]
		y = crop[1] + crop[3] - y
		return x, y
	
	from matplotlib.patches import Rectangle
	cliprect = Rectangle(crop[:2], *crop[2:], fc='none')
	prev_winner = []
	for image, orig_image, ts in frames(sys.argv[1], start, crop):
		if ts > end:
			break
		LaneHypothesis.init_x = init_x
		particles = detwalk(image, prev_winner)

		if len(particles) > 0:
			pos = [p.position for p in walk_particle(particles[0])]
			pos = pos[::-1]
		else:
			pos = None

		tp = None
		pos = np.array(pos)
		tpi = np.argmin(pos[:,0])
		if pos is not None and pos[tpi,0] < pos[-1,0]:
			x, y = pos[tpi]
			x += crop[0]
			y = crop[1] + crop[3] - y
			tp = x, y
			print "%f,%f,%f"%(ts, x, y)
		else:
			print "%f,%f,%f"%(ts, np.nan, np.nan)
		
		"""
		pylab.cla()
		pylab.gca().add_patch(cliprect)
		pylab.imshow(orig_image[:,:,::-1])
		if tp is not None:
			pylab.scatter(*tp)
		
		if pos is not None:
			#color = 'blue' if conf > 0.01 else 'red'
			color = 'blue'
			pylab.plot(*zip(*map(posconvert, pos)), color=color)
		pylab.draw()
		pylab.pause(0.01)
		"""
		
