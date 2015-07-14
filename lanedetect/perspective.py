import itertools
import cv
import numpy as np

def stupid_affine(img, M):
	trans = np.zeros(img.shape)
	trans[:,:,3] = 1.0

	for i, j in itertools.product(*map(range, img.shape[:2])):
		ref = np.dot(M, (i, j, 1.0))
		ref /= ref[2]
		ref = ref.astype(np.int)[:2]
		try:
			if np.any(ref < 0): continue
			trans[i, j, :] = img[ref[0], ref[1], :]
		except IndexError:
			continue
	return trans

def get_homography_matrix(points):
	target_points = []
	for bottom, top in points:
		target_points.append(bottom)
		target_points.append([bottom[0], top[1]])

	
	reference_points = list(itertools.chain(*points))
	H = cv.fromarray(np.zeros((3, 3)))
	
	#import pylab
	#for r, t in zip(target_points, reference_points):
	#	print zip(r, t)
	#	pylab.plot(*zip(r, t))
	#pylab.show()
	

	#reference_points = cv.fromarray(np.array(reference_points, dtype=float))
	#target_points = cv.fromarray(np.array(target_points, dtype=float))
	#cv.FindHomography(
	#	reference_points, target_points, H)
	#return H

	reference_points = map(tuple, reference_points)
	target_points = map(tuple, target_points)
	cv.GetPerspectiveTransform(reference_points, target_points, H)
	
	return H

def plot_perspective_lines(image_path, lines):
	import pylab
	
	image = pylab.imread(image_path)
	pylab.imshow(image)

	for line in lines:
		pylab.plot(*zip(*line))

	pylab.show()

def demo_perspective_lines(argv):
	import re
	image = argv[0]
	line_strings = argv[1:]
	
	lines = []
	matcher = re.compile(r'(\d+),(\d+):(\d+),(\d+)')
	def line_parse(s):
		m = matcher.match(s)
		coords = map(int, m.groups())
		coords = np.array([coords[:2], coords[2:]])
		return coords

	lines = map(line_parse, line_strings)

	H = get_homography_matrix(lines)
	
	
	
	import pylab

	#orig = pylab.imread(image)
	orig = cv.LoadImageM(image)
	
	#for bottom, top in lines:
	#	cv.Line(orig, tuple(bottom), tuple(top), [255,0,0])
	
	cv.ShowImage('img', orig)
	T = cv.CloneMat(orig)
	
	cv.WarpPerspective(orig, T, H, cv.CV_INTER_LINEAR)
	cv.ShowImage('T', T)
	
	cv.WaitKey(0)
	
	return
	
	

def main():
	import sys
	demo_perspective_lines(sys.argv[1:])


if __name__ == '__main__': main()
