import sys
from itertools import izip, chain
import copy

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

from __init__ import *

def simulate_static_fixations(noise_std, fix_dur, dt,
		location_dist=lambda: np.random.random(2)*10):
	mean = location_dist()
	
	rate = dt/fix_dur
	i = 0
	while True:
		new_mean = np.random.poisson(rate)
		if new_mean > 0:
			mean = location_dist()
		
		
		measurement = mean + np.random.randn(len(mean))*noise_std
		#if np.random.poisson(rate*2.0) > 0:
		#	measurement += np.random.randn(2)*10

		new_dt = yield measurement, mean
		if new_dt is not None:
			dt = new_dt

		i += 1

def plot_static_fixations():
	dt = 1/60.0
	gen = simulate_static_fixations(1.0, 1.0, dt)
	# No 2d fromiter in numpy :(
	t = np.arange(0, 60, dt)
	data = np.array([gen.next() for i in range(len(t))])

	plt.plot(t, data[:,0], '.')
	plt.plot(t, data[:,1])
	plt.show()

class IncrementalSs:
	"""
	Incremental sum of squares calculation

	>>> n_sims = 10
	>>> sample_len = 10
	>>> dim = 1
	>>> std_gen = lambda: np.abs(np.random.randn(dim)*1.0)
	>>> errors = []
	>>> for i in range(n_sims):
	...     ss = IncrementalSs()
	...     std = std_gen()
	...     data = np.random.randn(sample_len, dim)*std
	...     for d in data:
	...         dummy = ss(d)
	...     est = ss.ss
	...     truth = np.sum((data - np.mean(data))**2)
	...     errors.extend(np.abs(truth-est))
	>>> np.mean(errors) < 1e-14
	True
	"""
	def __init__(self):
		self.__call__ = self.__lazyinit
	
	def __lazyinit(self, t, x):
		dim = np.size(x)
		self.m = np.zeros(dim)
		self.ss = np.zeros(dim)
		self.n = 0.0
		self.__call__ = self.__doit
		return self(t, x)
	
	def __doit(self, t, x):
		self.n += 1.0
		delta = x - self.m
		self.m += delta/self.n
		self.ss += delta*(x - self.m)
		return self.ss
	
	def fit(self, x):
		return self.m
	

class IncrementalLinearRegression:
	"""
	Incremental (1D) linear regression.

	Can also do a "naive" multiple regression of form
	y_1 = x_1*a_1 + b_1 + e_1
	y_2 = x_2*a_2 + b_2 + e_2
	...
	yn = x_3*a_n + b_n + e_n

	Where e_1, e_2 ... e_n are uncorrelated, ie the
	covariance matrix is diagonal.
	"""
	def __call__(self, x, y):
		# Some sanity. size(x) > 1 and size(y) == 1
		# would make some sense, but not for our
		# purposes.
		assert np.size(x) == np.size(y) or np.size(x) == 1

		self.xs = IncrementalSs()
		self.ys = IncrementalSs()
		self.sxy = np.zeros(np.size(y))
		self.__call__ = self.__doit
		return self(x, y)

	def __doit(self, x, y):
		xs = self.xs(x, x)
		ys = self.ys(x, y)
		
		self.n = n = self.xs.n
		self.m = self.ys.m
		# TODO: This is wrong, the x and y deltas
		#	should be the "previous" ones
		inc = (n - 1)/(n)*(x-self.xs.m)*(y-self.ys.m)
		self.sxy += inc
		if xs == 0.0:
			return 0.0
		resid = ys - self.sxy**2/xs
		return resid
	
	def fit(self, x):
		if self.xs.ss == 0:
			slope = 0
		else:
			slope = self.sxy/self.xs.ss

		intercept = self.ys.m - slope*self.xs.m
		return x*slope + intercept


def piecewise_reg_slow(data, dt, noise_std, fix_dur, lik_limit=20,
		hypothesis_callback=lambda h, i: None):
	rate = dt/fix_dur
	dim = len(data[0])

	# TODO: The index/dt separation is a mess!

	# I have a gut feeling, that this should be corrected
	# by the number of dimensions, as
	# the likelihood is to the "dim'th power" in the
	# gaussian. Although this would be fixed if there'd
	# also be the step-size in the model.
	# TODO: Figure this out!
	new_lik = scipy.stats.poisson.logsf(1, rate)

	normer = np.log(1.0/(noise_std*np.sqrt(2*np.pi)))
	varcoeff = 1.0/(2*noise_std**2)

	class Hypothesis:
		def __init__(self, split=0, parent=None):
			self.split = split
			# Ugly!!
			self.parent = parent
			if parent is None:
				self.parent_loglik = 0
			else:
				self.parent_loglik = parent.loglikelihood
			
			# This could be shared by a "generation" (ie ones
			# with a common last split)
			# TODO: Are there even "generations" anymore?
			#self.ss = IncrementalSs()
			self.ss = IncrementalLinearRegression()
			

		def update_likelihood(self, i):
			# TODO: There's something a bit fishy going on here.
			# 	the mean doesnt correspond exactly to taking
			#	np.mean of the span
			ss = self.ss(dt*i, data[i])
			my_loglik = np.sum(self.ss.n*normer - varcoeff*ss)
			self.loglikelihood = self.parent_loglik + my_loglik + new_lik
		
		@property
		def splits(self):
			splits = []
			while self:
				splits.append(self.split)
				self = self.parent
			return splits[::-1]

		def fit(self, i):
			if i < self.split:
				return self.parent.fit(i)
			return self.ss.fit(i*dt)

		def fit_nodes(self, i):
			while self is not None:
				yield i, self.fit(i)
				yield self.split, self.fit(self.split)
				i = self.split - 1
				self = self.parent
			


	hypotheses = [Hypothesis()]
	hypotheses[0].update_likelihood(0)
	for i in range(1, len(data)):
		parent = copy.deepcopy(hypotheses[0]) # This is stupid
		new = Hypothesis(i, parent)
		hypotheses.append(new)
		for h in hypotheses: h.update_likelihood(i)
		hypotheses.sort(key=lambda h: h.loglikelihood, reverse=True)
		winner_lik = hypotheses[0].loglikelihood
		for hi, h in enumerate(hypotheses):
			if winner_lik - h.loglikelihood > lik_limit:
				break

		hypotheses = hypotheses[:hi+1]
		hypothesis_callback(hypotheses, i)
		
	return hypotheses[0].splits

def static_reg_test():
	dt = 1/60.0
	gen = simulate_static_fixations(1.0, 1.0, dt)
	# No 2d fromiter in numpy :(
	t = np.arange(0, 30.0, dt)
	data = (gen.next() for i in range(len(t)))
	data = zip(*data)
	data, thruth = map(np.array, data)

	include = np.unique(np.random.randint(0, len(t), len(t)*0.9))
	include.sort()
	include = slice(None, None)
	t = t[include]
	data = data[include]
	thruth = thruth[include]
	#for split in np.array(splits)*dt:
	#	plt.axvline(split)
	
	
	plt.ion()
	plt.figure(1)

	def mesplot():
		plt.subplot(2,1,1)
		plt.plot(t, data[:,0], '.', alpha=0.3, label="Measurements", color='black')
		plt.subplot(2,1,2)
		plt.plot(t, data[:,1], '.', alpha=0.3, color='black')
		
		"""
		plt.subplot(2,1,1)
		plt.plot(t, thruth[:,0], label="Truth", alpha=0.5, color='green')

		plt.subplot(2,1,2)
		plt.plot(t, thruth[:,1], alpha=0.5, color='green')
		"""

	

	def plot_hypotheses(hypotheses, i):
		if i%10 != 0 and i != len(data) - 1: return
		plt.subplot(2,1,1); plt.cla()
		plt.subplot(2,1,2); plt.cla()
		mesplot()
		plotted = hypotheses[:10]
		weights = [h.loglikelihood for h in plotted]
		weights /= np.sum(weights)
		for h, w in zip(plotted, weights):
			splits, means = zip(*h.fit_nodes(i))
			splits = np.array(splits)*dt
			means = np.array(means)
			plt.subplot(2,1,1)
			plt.plot(splits, means[:,0], color='red', linewidth=3, alpha=w)
			plt.subplot(2,1,2)
			plt.plot(splits, means[:,1], color='red', linewidth=3, alpha=w)
			
		plt.figure(1)
		plt.draw()

	mesplot()
	ot = t
	odata = data
	#raw_input()
	splits, valid, winner, params =\
		piecewise_linear_regression_pseudo_em(t, data, 1.0, 1.0,
		#hypothesis_callback=plot_hypotheses
		)
	
	t = t[valid]
	data = data[valid]
	segs = segmentation_to_table(splits, t, data)
	noise_std, dur = estimate_parameters(splits, t, data)
	
	"""
	stats = slope_stats(splits, t, data)
	pdf = slope_density(stats, noise_std)
	plt.ioff()
	plt.figure()
	start, end = -10, 10
	X, Y = np.mgrid[start:end:0.1, start:end:0.1]
	coords = np.vstack((X.flatten(), Y.flatten())).T
	density = pdf(coords).reshape(X.shape)
	plt.contour(X, Y, density)
	plt.show()
	"""


	#fit = regression_interpolator(splits, t, data)

	#recon = fit(t)
	
	#ind, fit = zip(*splits.fit_nodes(len(data)-1))
	#fit_t = np.array(ind)*dt
	#recon = np.array(fit)
	plt.subplot(2,1,1)
	#plt.plot(t, recon[:,0], 'r', label="Estimate")
	plt.plot([segs.t0, segs.t1], [segs.d0[:,0], segs.d1[:,0]],
		'r', linewidth=2, alpha=0.9)
	#plt.plot(t[splits], recon[:,0][splits], 'ro', alpha=0.3)
	plt.plot(ot[~valid], odata[~valid][:,0], 'rx')
	plt.subplot(2,1,2)
	#plt.plot(t, recon[:,1], 'r')
	#plt.plot(t[splits], recon[:,1][splits], 'ro', alpha=0.3)
	plt.plot([segs.t0, segs.t1], [segs.d0[:,1], segs.d1[:,1]],
		'r', linewidth=2, alpha=0.9)
	plt.plot(ot[~valid], odata[~valid][:,1], 'rx')
	plt.draw()
	#for split in splits:
	#	plt.axvline(t[split])
	
	plt.subplot(2,1,1)
	#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	#	ncol=3, mode="expand", borderaxespad=0.)
	
	plt.subplot(2,1,1)
	plt.plot(ot, thruth[:,0], label="Truth", alpha=1.0)

	plt.subplot(2,1,2)
	plt.plot(ot, thruth[:,1], alpha=1.0)
	
	plt.ioff()
	plt.show()


static_reg_test()
