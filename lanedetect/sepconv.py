import numpy as np
import scipy.signal

def decompose_separable_kernel(kernel):
	#y_i = np.argmax(np.sum(np.abs(kernel), axis=0))
	#x_i = np.argmax(np.sum(np.abs(kernel), axis=1))
	
	#x = kernel[x_i:x_i+1,:]
	#y = kernel[:,y_i:y_i+1]
	
	#x = x / x[0][np.argmax(np.abs(x[0]))]

	y, s, x = np.linalg.svd(kernel)
	y = y[:,0:1]*np.sqrt(s[0])
	x = x.T
	x = np.conj(x[:,0:1])*np.sqrt(s[0])
	return (y, x.T)

def separable_convolution(data, (y, x), **kwargs):
	data = scipy.signal.convolve(data, y, **kwargs)
	data = scipy.signal.convolve(data, x, **kwargs)
	return data

def test_separable_convolution():
	import pylab
	kernel = np.array([[-3, 0, 3],
			   [-10, 0, 10],
			   [-3, 0, 3]], dtype=np.float)

	data = np.zeros((100, 100))
	data[:,20:] = 1.0

	pylab.gray()
	pylab.imshow(scipy.signal.convolve(data, kernel, mode='valid'))

	pylab.figure()
	y, x = decompose_separable_kernel(kernel)
	data = scipy.signal.convolve(data, y, mode='valid')
	data = scipy.signal.convolve(data, x, mode='valid')
	pylab.imshow(data)
	pylab.show()

def test_decomposition():

	kernel = np.array([[-3, 0, 3],
			   [-10, 0, 10],
			   [-3, 0, 3]], dtype=np.float)
	#a = np.array([[-0.2, 0.6, -0.2]], dtype=np.float).T
	#b = a.T
	#kernel = np.dot(a, b)
	y, x = decompose_separable_kernel(kernel)
	print kernel
	print np.dot(y, x)

if __name__ == '__main__':
	test_separable_convolution()

