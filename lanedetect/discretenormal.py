import numpy as np
from scipy.stats import norm
import pylab

def discretenormal(limits, nbins, scale=1.0):
	bins = np.linspace(*limits, num=nbins)

	mass = np.array([norm.pdf(b, scale=scale) for b in bins])
	pylab.plot(bins, mass)
	pylab.show()

if __name__ == '__main__':
	discretenormal((-1, 1), 30, scale=0.5)
