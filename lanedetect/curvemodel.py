import scipy.stats
import numpy as np

from scipy.stats import norm

class CurvePoint(object):
	def __init__(self, position, angle, curvature, weight=1.0, probability=1.0):
		self.position = position
		self.angle = angle
		self.curvature = curvature
		self.weight = weight
		self.probability = probability

	def next_probability(self, position, angle, curvature):
		curvature_p = norm(angle, loc=self.angle, scale=pi/6.0)
		exp_angle = self.angle+self.angle_delta
		angle_p = norm(angle, loc=exp_angle, scale=pi/6.0)

		#exp_position = 

	def simulate_next(self, distance):
		# TODO: Take the distance in to account
		pos, angle = point_at_curvature(self.position, self.angle, self.curvature, distance)
		std = 0.02**2
		k = np.random.normal(loc=self.curvature, scale=std)
		#p = norm.pdf(k, loc=self.curvature, scale=std)
		
		
		new = CurvePoint(pos, angle, k)
		return new

def point_at_curvature(src, tangent_angle, k, dist):
	if k == 0:
		new = src + dist*np.array((np.cos(tangent_angle), np.sin(tangent_angle)))
		return new, tangent_angle
	r = 1.0/k
	norm_angle = tangent_angle - np.sign(r)*np.pi/2.0
	displacement = np.abs(r)*np.array((np.cos(norm_angle), np.sin(norm_angle)))
	
	origin = src + displacement
	#pylab.scatter(*origin)
	#pylab.plot(*zip(src, origin), color='red')


	central_angle = -dist/r
	new_orientation = (norm_angle - np.pi) + central_angle

	#print np.degrees(norm_angle)
	
	#cir = pylab.Circle(origin, r, fill=False)
	#pylab.gca().add_patch(cir)
	
	new_point = np.abs(r)*np.array((np.cos(new_orientation), np.sin(new_orientation)))
	new_point += origin
	return new_point, tangent_angle+central_angle

def radius_of_curvature(a, ad, b):
	ax, ay = a
	bx, by = b
	s = np.sin(ad-np.pi/2.0)
	c = np.cos(ad-np.pi/2.0)
	
	t = (by**2-2*ay*by+bx**2-2*ax*bx+ay**2+ax**2)/((2*by-2*ay)*s+(2*bx-2*ax)*c)

	x = ax + c*t
	y = ay + s*t

	center = np.array((x, y))
	r = np.linalg.norm(center - a)

	return 1.0/r, center

def three_point_circle(p1, p2, p3):
	a, b = p1
	c, d = p2
	e, f = p3
	
	k = 0.5*((a**2+b**2)*(e-c) + (c**2+d**2)*(a-e) + (e**2+f**2)*(c-a)) / (b*(e-c)+d*(a-e)+f*(c-a))
	h = 0.5*((a**2+b**2)*(f-d) + (c**2+d**2)*(b-f) + (e**2+f**2)*(d-b)) / (a*(f-d)+c*(b-f)+e*(d-b))
	r = np.sqrt((a-h)**2 + (b-k)**2)
	return (h, k), r

def angle_at_point(c, p):
	dx = p[0] - c[0]
	dy = p[1] - c[1]
	inner_angle = np.arctan2(dy, dx)
	tangent_angle = inner_angle + np.pi/2.0
	return tangent_angle

import pylab
def simulate():
	p = CurvePoint([0,0], np.pi/2.0, 0)

	def simulate_one(p, n):
		for i in range(n):
			p = p.simulate_next(10)
			yield (p.position, p.weight)
	
	curves = [list(simulate_one(p, 100)) for i in range(100)]
	for curve in curves:
		curve, weights = zip(*curve)
		pylab.plot(*zip(*curve), alpha=0.1, color='black')
	pylab.show()



def three_point_test():
	import collections

	points = collections.deque(maxlen=3)
	def add_point(event):
		x, y = event.xdata, event.ydata
		points.append((x, y))
		pylab.cla()
		pylab.scatter(*zip(*points))
		pylab.xlim(-10, 10)
		pylab.ylim(-10, 10)

		pylab.draw()
		if len(points) < 3: return

		c, r = three_point_circle(*points)
		cir = pylab.Circle(c, r)
		pylab.gca().add_patch(cir)

		for p in points:
			angle = angle_at_point(c, p)
			if angle < 0:
				angle += 2*np.pi
			if angle >= np.pi:
				angle = angle - np.pi
			print np.degrees(angle)
			dx, dy = np.array((np.cos(angle), np.sin(angle)))
			pylab.text(p[0], p[1], "%.2f"%np.degrees(angle))
			pylab.arrow(p[0], p[1], dx, dy)
		pylab.show()
		
	
	#pylab.scatter(*zip(a, b, c))
	#c, r = three_point_circle(a, b, c)
	#print c, r
	pylab.xlim(-10, 10)
	pylab.ylim(-10, 10)

	pylab.connect('button_release_event', add_point)
	pylab.show()

if __name__ == '__main__':
	three_point_test()
	#simulate()


def curv_point_test():
	start = np.array([1, 1])
	r = 1.0
	
	pylab.scatter(*start, color='red')
	angle = np.radians(45)
	pylab.plot(*zip(start, start+(np.cos(angle), np.sin(angle))))
	end, d = point_at_curvature(start, angle, 0, 2*np.pi*r/4)
	pylab.plot(*zip(end, end+(np.cos(d), np.sin(d))))
	pylab.scatter(*end, color='black')
	pylab.show()

#curv_point_test()

def secant_to_circle(a, ad, b):
	ax, ay = a
	bx, by = b
	s = np.sin(ad-np.pi/2.0)
	c = np.cos(ad-np.pi/2.0)
	
	t = (by**2-2*ay*by+bx**2-2*ax*bx+ay**2+ax**2)/((2*by-2*ay)*s+(2*bx-2*ax)*c)

	x = ax + c*t
	y = ay + s*t

	center = np.array((x, y))
	r = np.linalg.norm(center - a)	

	return (x, y), r

def curvature_test():
	a = np.array([0, 0])
	b = np.array([0, 1])

	d = np.radians(10)
	
	pylab.ion()
	for d in np.radians(range(1, 360, 3)):
		pylab.cla()
		center, r = secant_to_circle(a, d, b)
		if r < 1000:
			cir = pylab.Circle((center), r)
			pylab.gca().add_patch(cir)
		print 1/r
		d_v = [np.cos(d), np.sin(d)]
		pylab.plot(*zip(a, d_v+a))

		pylab.plot(*zip(a, center))
		pylab.plot(*zip(b, center))
	
		pylab.scatter(*zip(a, b), color='red')
		pylab.xlim(-3, 3)
		pylab.ylim(-3, 3)
		pylab.draw()
	
	
	"""
	print np.arccos(np.dot(d_v, b)/(np.linalg.norm(d_v)*np.linalg.norm(b)))
	tang_angle = np.arccos(np.dot(d_v, b)/(np.linalg.norm(d_v)*np.linalg.norm(b)))
	tang_angle = np.pi/2.0 - tang_angle

	r = np.linalg.norm(a - b)/2.0/np.cos(tang_angle)
	
	tangent = d + np.radians(90)

	center = (r*np.cos(tangent), r*np.sin(tangent))
	"""
	
		

	
	pylab.show()


