import randomwalk
import pylab
import cv
import sys
import numpy as np

def to_grayscale(orig_image):
	image = cv.CreateMat(orig_image.rows, orig_image.cols,
			cv.CV_8UC1)
	cv.CvtColor(orig_image, image, cv.CV_RGB2GRAY)
	return image

def get_block(center, size):
	x = slice(center[0]-size[0]/2, center[0]+size[0]/2)
	y = slice(center[1]-size[1]/2, center[1]+size[1]/2)
	return x, y

def test_single_particle(frame, initial_x):
	n_branches = 5
	displacement = (6, 6)
	block_size = (10, 10)

	pylab.gray()
	pylab.imshow(frame)

	particle = randomwalk.LaneHypothesis(initial_x)
	particles = [particle]
	
	rows = range(frame.shape[0]-block_size[1]/2,
			displacement[1]+block_size[1]/2,
			-displacement[1])
	
	def is_in_frame(xs, ys):
		if xs.start < 0: return False
		if xs.stop > frame.shape[1]: return False
		if ys.start < 0: return False
		if ys.stop > frame.shape[0]: return False

		return True

	def next_particles(p, row):
		shift = n_branches/2*displacement[0]
		shifts = range(-shift, shift+1, displacement[0])
		next_positions = [s+p.position for s in shifts]
		for pos in next_positions:
			new = next_particle(p, pos, row)
			if new is None: continue
			yield new

	def next_particle(p, pos, row):
		xs, ys = get_block((pos, row), block_size)
		if not is_in_frame(xs, ys): return None
		new = p.spawn(pos)
		pylab.plot((p.position, pos), (row, row-displacement[1]), color='blue')
		return new


	for i, row in enumerate(rows):
		new_particles = []
		for p in particles:
			new_particles.extend(next_particles(p, row))
		if i >= 3: break
		particles = new_particles
	pylab.show()

def videotest():
	cap = cv.CaptureFromFile(sys.argv[1])
	if len(sys.argv) > 2:
		crop = map(int, sys.argv[2:])
		xo, yo, xe, ye = crop
	else:
		crop = None
	
	for i in range(1000000):
		frame = cv.QueryFrame(cap)
		frame = cv.GetMat(frame)
		if crop:
			frame = np.array(frame)
			frame = frame[yo:yo+ye, xo:xo+xe]
			frame = cv.fromarray(frame)
		orig = cv.CloneMat(frame)
		
		frame = np.array(to_grayscale(frame))
		test_single_particle(frame, 440)

if __name__ == '__main__':
	videotest()
