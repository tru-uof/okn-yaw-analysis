import numpy as np
import scipy

# TODO: Could be optimized a lot
def piecewise_linear_regression(t, data, noise_std, fix_dur, swarm_max=None,
		outlier_prop=0.6):
	if swarm_max is None:
		swarm_max = min(2*len(data), 512)
		#swarm_max = 2*len(data)
	
	noise_std = np.array(noise_std)
	dim = len(data[0])

	# I have a gut feeling, that this should be corrected
	# by the number of dimensions, as
	# the likelihood is to the "dim'th power" in the
	# gaussian. Although this would be fixed if there'd
	# also be the step-size in the model.
	# TODO: Figure this out!
	dt = np.diff(t)
	new_lik = np.empty(len(t))
	new_lik[1:] = -scipy.stats.poisson.logsf(1, dt)
	# Don't penalize the first measurement just for
	# coming first
	new_lik[0] = 0.0

	
	normer = np.log(1.0/(noise_std*np.sqrt(2*np.pi)))
	varcoeff = 1.0/(2*noise_std**2)


	pool = np.recarray(swarm_max, dtype=[
		('id', int),
		('new_lik', float),
		('parent_lik', float),
		('reg_lik', float),
		('n', float),
		('lik', float),
		('xm', float),
		('ym', float, dim),
		('xs', float),
		('ys', float, dim),
		('xys', float, dim),
		])
	
	pool[:] = 0

	hypotheses = pool[:1]
	root = hypotheses[0]
	root.new_lik = 0
	root.ym = data[0]
	root.xm = t[0]
	root.n = 1.0
	
	splits = {}
	splits[0] = [0]
	
	outliers = {}
	outliers[0] = []

	winner = 0
	
	n_forks = 2
	# TODO: Allocate buffers for temporary results
	# TODO: Verfify that the bug that causes the split at index 1 to
	#	almost always occur is now fixed.
	for i in range(1, len(data)):
		for j in range(1, n_forks + 1):
			if len(hypotheses) + 1 <= len(pool):
				hypotheses = pool[:len(hypotheses)+1]
			else:
				del splits[hypotheses[-j].id]
				del outliers[hypotheses[-j].id]
					
				for field in hypotheses.dtype.names:
					hypotheses[-j][field] *= 0.0

		hypotheses[-2].id = i
		hypotheses[-2].new_lik = new_lik[i]
		hypotheses[-2].parent_lik = hypotheses[0].lik
		splits[i] = splits[hypotheses[0].id][:] + [i]
		outliers[i] = outliers[hypotheses[0].id][:]
		
		# Create an outlier hypothesis
		hypotheses[-1] = hypotheses[0]
		hypotheses[-1].id = -i
		hypotheses[-1].parent_lik = hypotheses[0].parent_lik
		hypotheses[-1].new_lik += new_lik[i]*outlier_prop

		splits[-i] = splits[hypotheses[0].id][:]
		outliers[-i] = outliers[hypotheses[0].id][:] + [i]

		# Hide the outlier hypothesis from the data-update
		hypotheses = hypotheses[:-1]
		
		hypotheses.n += 1.0
		
		dx = (t[i] - hypotheses.xm)

		hypotheses.xm += dx/hypotheses.n
		hypotheses.xs += dx*(t[i] - hypotheses.xm)

		dx = dx.reshape(-1, 1)
		n = hypotheses.n.reshape(-1, 1)
		
		dy = (data[i] - hypotheses.ym)

		hypotheses.ym += dy/n
		hypotheses.ys += dy*(data[i] - hypotheses.ym)
		
		hypotheses.xys += (n - 1)/(n)*dx*dy
		
		# Avoid division by zero (we have a 0/0 = 0 situation
		# for the new hypothesis)
		hypotheses[-1].xs = 1.0
	
		resid = hypotheses.ys - hypotheses.xys**2/hypotheses.xs.reshape(-1, 1)
		hypotheses[-1].xs = 0.0
		
		# TODO: Make incremental?
		hypotheses.reg_lik = -np.sum(n*normer - varcoeff*resid, axis=1)
		
		# Pull the outlier hypothesis back
		hypotheses = pool[:len(hypotheses)+1]
	
		hypotheses.lik = hypotheses.reg_lik + hypotheses.new_lik + hypotheses.parent_lik

		# TODO: Could be optimized by pruning hypotheses that
		# can't win
		#winner = np.argmax(hypotheses.lik)
		#loser = np.argmin(hypotheses.lik)
		# TODO: Full sort not needed
		hypotheses.sort(order='lik')
		


	splits = splits[hypotheses[winner].id]
	outliers = outliers[hypotheses[winner].id]
	
	splits = np.array(splits)
	pruned_splits = np.array(splits, copy=True)
	for i in outliers:
		pruned_splits[splits >= i] -= 1
	
	mask = np.ones(len(t), dtype=bool)
	mask[outliers] = False
	return pruned_splits, mask, hypotheses[winner]

def estimate_parameters(splits, ts, data):
	splits = list(splits) + [len(data)]
	var = []
	weights = []
	durations = []
	dfs = []
	for i in range(len(splits)-1):
		span = slice(splits[i], splits[i+1])
		t = ts[span]
		d = data[span]
		n = len(t)

		if n < 3: continue
		
		mt = np.mean(t)
		md = np.mean(d, axis=0)
		dd = (d - md)
		dt = (t - mt)
		sd = np.sum(dd**2, axis=0)
		st = np.sum(dt**2)

		sdt = np.sum(dd*dt.reshape(-1, 1), axis=0)
		
		resid = sd - sdt**2/st

		#TODO: What should the DFS be here?
		#	Very quick simulation shows that
		#	this formulations seems to give
		#	unbiased (or less biased) results and
		#	makes sort of sense.
		var.append(resid/(n - 2))
		dfs.append(n - 2)
	
	# TODO: Is this the non-biased MLE?
	# Take the square root of the pooled sample variance. Ignores
	# samples where n < 3, as they don't really have a meaningful
	# s^2.
	dfs = np.array(dfs).reshape(-1, 1)
	std = np.sqrt(np.sum(np.multiply(var, dfs), axis=0)/np.sum(dfs))

	# TODO: Is this the MLE?
	
	if len(splits) < 3:
		duration = np.nan
	else:
		duration = np.mean(np.diff(ts[splits[:-1]]))
	
	return std, duration

def piecewise_linear_regression_pseudo_em(t, data, noise_std, fix_dur, **kwargs):
	likelihood = -np.inf
	params = [noise_std, fix_dur]
	while True:
		splits, mask, winner =\
			piecewise_linear_regression(t,
				data, params[0], params[1],
				**kwargs)

		if winner.lik == likelihood:
			break

		if winner.lik < likelihood:
			mask = prev_mask
			splits = prev_splits
			winner = prev_winner
			params = prev_params
			break
		
		# Ignores the duration parameter now as it
		# causes weird "convergences" and can't really
		# be estimated separately from std.
		params[0] = estimate_parameters(splits, t[mask], data[mask])[0]
		
		likelihood = winner.lik
		prev_mask = mask
		prev_winner = winner
		prev_splits = splits
		prev_params = params

	
	return splits, mask, winner, params

def deoutlier(t, data, splits, outlier_limit=2):
	lens = np.diff(splits)
	outlier_spans = lens <= outlier_limit
	new_splits = np.array(splits[:])
	inliers = np.ones(len(t), dtype=np.bool)
	
	for i in np.flatnonzero(outlier_spans):
		n = splits[i+1] - splits[i]
		inliers[splits[i]:splits[i+1]] = False
		new_splits[i+1:] -= n
	
	new_splits = new_splits[~outlier_spans]
	
	return list(new_splits), inliers

def deoutliered_pseudo_em_piecewise_linear_regression(t, data, noise_std, fix_dur, **kwargs):
	inliers = np.ones(len(t), dtype=bool)
	n_outliers = 0
	orig_t = t
	orig_data = data
	
	while True:
		splits, winner, (noise_std, fix_dur) =\
			piecewise_linear_regression_pseudo_em(t, data, noise_std, fix_dur)
		splits, new_inliers = deoutlier(t, data, splits)
		new_outliers = np.sum(~new_inliers) - n_outliers
		if new_outliers == 0:
			break
		n_outliers += new_outliers
		inliers |= new_inliers

		t = orig_t[inliers]
		data = orig_data[inliers]
	
	return splits, winner, (noise_std, fix_dur), inliers

def regression_ss(x, y):
	"""
	>>> x = np.random.randn(100)
	>>> y = np.random.randn(100)
	>>> res_ss1, sx, sy, sxy, mx, my = regression_ss(x, y)
	>>> slope = sxy/sx
	>>> intercept = my - slope*mx
	>>> fit = np.polyfit(x, y, 1, full=True)
	>>> res_ss2 = np.sum(fit[1][0])
	>>> float(np.abs(res_ss1 - res_ss2)) < 1e-10
	True
	>>> float(np.abs(fit[0][0] - slope)) < 1e-10
	True
	>>> float(np.abs(fit[0][1] - intercept)) < 1e-10
	True
	
	>>> x = np.random.randn(2)
	>>> y = np.random.randn(2)
	>>> float(regression_ss(x, y)[0]) < 1e-10
	True
	"""
	x = np.reshape(x, (len(x), -1))
	y = np.reshape(y, (len(y), -1))
	
	xm = np.mean(x, axis=0)
	ym = np.mean(y, axis=0)

	dx = (x - np.mean(x, axis=0))
	dy = (y - np.mean(y, axis=0))

	sx = np.sum(dx**2, axis=0)
	sy = np.sum(dy**2, axis=0)
	
	sxy = np.sum(dx*dy, axis=0)

	resid = sy - sxy**2/sx

	return resid, sx, sy, sxy, xm, ym

def slope_stats(splits, ts, data):
	splits = list(splits) + [len(data)]
	stats = []
	dummy = np.zeros(data[0].shape)*np.nan
	zero = np.zeros(1)
	for i in range(len(splits)-1):
		span = slice(splits[i], splits[i+1])
		t = ts[span]
		d = data[span]
		n = float(len(t))
		if n < 2:
			stats.append((n, dummy, dummy, zero))
			continue

		resid, sx, sy, sxy = regression_ss(t, d)
		if n < 3:
			s_slope = dummy
		else:
			s_slope = np.sqrt(resid/sx/(n-2.0))
		
		slope = sxy/sx
		stats.append((n, slope, s_slope, sx))
	
	return stats

def segmentation_to_table(splits, ts, data):
	splits = list(splits) + [len(data)]
	
	dim = len(data[0])
	n_rows = len(splits)-1
	rows = np.recarray(n_rows, dtype=[
		('t0', float),
		('t1', float),
		('d0', float, dim),
		('d1', float, dim),
		('n', int)
		])
		
	for i in range(len(splits)-1):
		span = slice(splits[i], splits[i+1])
		t = ts[span]
		d = data[span]
		n = float(len(t))
		if n < 2:
			d0 = d[0]
			d1 = d[-1]
		else:
			fit = np.polyfit(t, d, 1)
			d0 = tuple(np.polyval(fit, t[0]))
			d1 = tuple(np.polyval(fit, t[-1]))

		rows[i] = (t[0], t[-1], d0, d1, n)
		#stats.append((n, slope, s_slope, sx))
	
	return rows

	

def slope_density(stats, std):
	#stats = slope_stats(splits, ts, data)
	n, slope, s_slope, sx = map(np.array, zip(*stats))
	valid = n >= 2
	n = n[valid]
	slope = slope[valid]
	s_slope = s_slope[valid]
	sx = sx[valid].reshape(-1, 1)

	test = np.array([[0, 0]])
	df = n - 2
	w = n/np.sum(n)
	
	n = n.reshape(-1, 1)
	est_std = np.sqrt((std**2*n/sx/(n-1.0)))
	
	def density(y):
		# TODO: Figure out some broadcasting magic for this!
		axdens = []
		for i in range(slope.shape[0]):
			#normed = (slope[i] - y)*s_slope[i] # TODO: VERIFY!!
			#density = np.prod(scipy.stats.t.pdf(normed, df[i]), axis=1)
			
			# I guess we are allowed to use normal distribution, as
			# the residual std is "known"
			normed = (slope[i] - y)*est_std[i] # TODO: VERIFY!!
			density = np.prod(scipy.stats.norm.pdf(normed), axis=1)
			density *= w[i] # Does this make sense?
			axdens.append(density)
		return np.sum(axdens, axis=0)
		#return np.mean(axdens, axis=0)
	return density

def regression_interpolator(splits, ts, data):
	splits = list(splits) + [len(data)]
	sts = []
	fits = []
	for i in range(len(splits)-1):
		span = slice(splits[i], splits[i+1])
		t = ts[span]
		d = data[span]
		
		if len(t) == 1:
			sts.append(t[0])
			fits.append(d[0])
			continue

		fit = np.polyfit(t, d, 1)
		fit_t = t[0], t[-1]
		dims = [np.polyval(f, fit_t) for f in fit.T]
		sts.extend(fit_t)
		fits.extend(np.array(dims).T)
	
	return scipy.interpolate.interp1d(np.array(sts), np.array(fits), axis=0, bounds_error=False)

if __name__ == '__main__':
	import doctest
	doctest.testmod()

