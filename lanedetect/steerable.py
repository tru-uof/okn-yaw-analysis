import numpy as np
import itertools
import pylab
import sys
from scipy import ndimage
import scipy.signal
from sepconv import decompose_separable_kernel, separable_convolution

def generate_kernel(size, func):
	kernel = np.zeros(size[::-1])
	#pixels = itertools.product(*map(range, kernel.shape))
	for y, x in np.ndindex(kernel.shape):
		xc = ((x+0.5)/float(kernel.shape[1])-0.5)*4
		yc = ((y+0.5)/float(kernel.shape[0])-0.5)*4
		kernel[y, x] = func(xc, yc)
	return kernel

class SecondDegreeGaussian(object):
	@classmethod
	def generate_bank(cls, kernel_size):
		#a = lambda x, y: 0.9213*(2*x**2 - 1)*np.exp(-(x**2+y**2))
		#b = lambda x, y: 1.843*x*y*np.exp(-(x**2+y**2))
		#c = lambda x, y: a(y, x)
		var = 1.0
		xx = lambda x, y: -(2.0*x**2/var - 1)*(2.0/var)*np.exp(-(x**2+y**2)/var)
		xy = lambda x, y: 4*x*y/var**2*np.exp(-(x**2+y**2)/var)
		yy = lambda x, y: xx(y, x)
	
		kernels = [generate_kernel(kernel_size, f) for f in (xx, xy, yy)]
		return kernels

	coefficients = [
			lambda a: np.sin(a)**2,
			lambda a: 2*np.cos(a)*np.sin(a),
			lambda a: np.cos(a)**2
			]

	def __init__(self, kernel_size):
		self.kernels = SecondDegreeGaussian.generate_bank(kernel_size)
		self.separate_kernels = map(decompose_separable_kernel, self.kernels)
		self.directions = [[1.0, 0.0], [1.0/np.sqrt(2)]*2, [0.0, 1.0]]
		self.angles = np.arctan2(*zip(*self.directions))

	def blending(self, comp, angle):
		return self.coefficients[comp](angle)
	
	def kernel_at_angle(self, angle):
		kernel = np.zeros(self.kernels[0].shape)
		for i, k in enumerate(self.kernels):
			kernel += self.blending(i, angle)*k
		return kernel

	def response_at_angle(self, comp, angle):
		# Some premature optimization
		s = np.sin(angle)
		c = np.cos(angle)
		return comp[0]*s**2 + 2*comp[1]*c*s + comp[2]*c**2

	def max_response_angle(self, co):
		xx, xy, yy = co
		
		A = np.sqrt(xx**2 + yy**2 - 2*xx*yy + 4*xy**2)
		max_response = np.arctan2(( xx - yy + A)*0.5, xy)
		
		return max_response

	def max_response_value(self, co):
		return self.response_at_angle(co, self.max_response_angle(co))

	def min_response_angle(self, co):
		xx, xy, yy = co
		
		A = np.sqrt(xx**2 + yy**2 - 2*xx*yy + 4*xy**2)
		min_response = np.arctan2(( xx - yy - A)*0.5, xy)
		
		return min_response

	def min_response_value(self, co):
		return self.response_at_angle(co, self.min_response_angle(co))
	
	def get_component_values(self, data, **kwargs):
		# TODO: Use the separability
		filt = lambda kernel: separable_convolution(data, kernel, **kwargs)
		#filt = lambda kernel: scipy.signal.fftconvolve(data, kernel, **kwargs)
		components = np.dstack([filt(kernel)
			for kernel in self.separate_kernels])
		return components


class AngleFilter(object):
	def __init__(self, kernel_size):
		self.basis = SecondDegreeGaussian(kernel_size)

	def get_component_values(self, data, **kwargs):
		return self.basis.get_component_values(data, **kwargs)

	def directions(self, data, **kwargs):
		components = self.get_component_values(data, **kwargs)
		directions = np.zeros(list(components.shape[:2])+[len(self.basis.directions[0])])
		basis_directions = np.array(self.basis.directions)
		for y, x in np.ndindex(directions.shape[:2]):
			directions[y, x, :] = np.dot(basis_directions.T, components[y, x])
		return directions

	def block_angle_value(self, components, angle):
		return sum(c*self.basis.coefficients[i](angle)
			for i, c in enumerate(components))


def test_optimal_angle_filter():
	image = pylab.imread(sys.argv[1])
	image = np.mean(image, axis=2)
	image = image[::-1]
	pylab.gray()

	f = AngleFilter((7, 7))

	components = f.get_component_values(image, mode='same')

	angles = np.radians(range(0, 180, 1))
	def best_angle_value(c):
		values = f.basis.response_at_angle(c, angles)
		return angles[np.argmax(values)], np.max(values)

	for y, x in np.ndindex(components.shape[:2]):
		if y%4 != 0: continue
		if x%4 != 0: continue
		maxval = f.basis.max_response_value(components[y,x])
		if maxval < 2: continue
		maxang_an = f.basis.max_response_angle(components[y,x])
		maxang, maxval = best_angle_value(components[y,x])
		pylab.scatter(x, y)
		dy = -5.0
		dx, dy = np.array((np.cos(maxang), np.sin(maxang)))*10
		#dx = np.tan(maxang_an)*dy
		#d = d/np.linalg.norm(d)*3
		pylab.arrow(x, y, dx, dy, color='blue')

		#d = np.array((-np.sin(maxang_an), -np.cos(maxang_an)))
		#d = d/np.linalg.norm(d)*3
		#pylab.arrow(x, y, d[0], d[1], color='green')

	
		#pylab.plot(x, y, '.')

	pylab.imshow(image, origin="lower")
	#pylab.xlim(0, components.shape[1])
	#pylab.ylim(components.shape[0], 0)
	pylab.show()
	return
	
	#pylab.subplot(1,2,1)
	#pylab.imshow(image)
	#pylab.subplot(1,2,2)
	filtered = np.zeros(image.shape)
	for y, x in np.ndindex(components.shape[:2]):
		maxval = f.basis.max_response_value(components[y,x])
		minval = f.basis.min_response_value(components[y,x])
		#print maxval, minval
		filtered[y,x] = maxval
		#filtered[y, x] = best_angle_value(components[y,x])
	#pylab.hist(filtered.flatten())
	pylab.imshow(filtered > 3)
	pylab.show()

def test_angle_filter():
	import time
	image = pylab.imread(sys.argv[1])
	image = np.mean(image, axis=2)
	image = image[:20, :20]
	image = -image
	image = image[::-1]
	pylab.gray()


	angle_filter = AngleFilter((9, 9))
	
	
	angles = np.radians(range(0, 180, 1))
	def best_angle_value(c):
		values = angle_filter.basis.response_at_angle(c, angles)
		return angles[np.argmin(values)], np.min(values)
	image = np.zeros((90, 90))
	for i, angle in enumerate(range(0, 180, 1)):
		image = angle_filter.basis.kernel_at_angle(np.radians(angle))
	
		components = angle_filter.get_component_values(image, mode='valid')[0][0]
		cangle = np.degrees(angle_filter.basis.min_response_angle(components))
		eangle = np.degrees(best_angle_value(components)[0])

		if cangle < 0: cangle += 180
		
		#pylab.scatter(angle, cangle-angle)
		#pylab.scatter(i, angle, color='green')
		pylab.scatter(i, eangle, color='blue')
		pylab.scatter(i, cangle, color='red')
		#pylab.scatter(i, angle_filter.basis.max_response_value(components), color='red')
		#pylab.scatter(i, best_angle_value(components)[1], color='blue')
		#pylab.scatter(i, angle_filter.basis.response_at_angle(components, np.radians(angle)), color='green')
	pylab.show()
	
	image = angle_filter.basis.kernel_at_angle(np.radians(60))
	pylab.ion()
	origin = (np.array(image.shape)/2.0)[::-1]
	max_angle = (None, None)
	for angle in range(0, 180, 5):
		kernel = angle_filter.basis.kernel_at_angle(np.radians(angle))
		filtered = scipy.signal.convolve(image, kernel, mode='valid')
		
		#print filtered
		pylab.cla()
		pylab.imshow(image, origin='lower')
		pylab.imshow(kernel, alpha=0.9, origin='lower')
		if filtered[0][0] > max_angle[0]:
			max_angle = (filtered[0][0], angle)

			direction = [np.cos(np.radians(angle)), np.sin(np.radians(angle))]
			direction = np.array(direction)/np.linalg.norm(direction)*10
	
		pylab.plot(*zip(origin, origin+direction))
		
		#print (filtered[0][0], angle)
		components = angle_filter.get_component_values(image, mode='valid')[0][0]
		
		#print filtered[0][0], sum([c*angle_filter.basis.coefficients[i](np.radians(angle)) for i, c in enumerate(components)])
		print filtered[0][0], angle_filter.basis.response_at_angle(components, np.radians(angle))
		 
		
		pylab.draw()
	
	print max_angle
	print np.degrees(angle_filter.basis.max_response_angle(components))
	pylab.ioff()
	components = angle_filter.get_component_values(image, mode='valid')[0][0]
	pylab.figure()
	for angle in range(0, 180, 1):
		pylab.scatter(angle, sum([c*angle_filter.basis.coefficients[i](np.radians(angle)) for i, c in enumerate(components)]))
	pylab.show()

	return
	angle = max_angle[1]
	direction = [np.sin(np.radians(angle)), -np.cos(np.radians(angle))]
	direction = np.array(direction)/np.linalg.norm(direction)*10
	print max_angle
	
	#angle = angle_filter.angles(image, mode='valid')[0][0][0]
	#print np.degrees(angle)
	pylab.imshow(image)
	pylab.plot(*zip(origin, origin+direction))
	pylab.ioff()
	pylab.show()
	return

	pylab.figure()
	pylab.imshow(image)
	
	components = angle_filter.get_component_values(image, mode='valid')[0][0]
	
	#for i, c in enumerate(components):
	#	angle = c**2*np.array(angle_filter.basis.angles[i])
	#	print "Angle", angle
	#	#pylab.plot(*zip(origin, origin+direction))

	return
	pylab.imshow(image)
	pylab.scatter(*origin)
	pylab.plot(*zip(origin, origin+direction))

	
	
	pylab.show()
	#for kernel, direction in angle_filter.basis:
	#	filtered = scipy.signal.convolve2d(image, kernel, mode='valid')
	#	print float(filtered)
	#	pylab.figure()
	#	pylab.imshow(filtered)
	#	pylab.plot(*zip([0, 0], direction))

	#pylab.show()

if __name__ == '__main__':
	test_angle_filter()
	
