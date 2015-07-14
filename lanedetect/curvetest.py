import pylab
import numpy as np
import curvemodel


angle = np.pi/4.0 #+ np.pi/3.0
position = [0.0, 0.0]

pylab.scatter(*position)
pylab.plot(*zip(position, position + np.array((np.cos(angle), np.sin(angle)))))

pylab.xlim(-100, 100)
pylab.ylim(-100, 100)

def curver(e):
	x, y = e.xdata, e.ydata
	k, center = curvemodel.radius_of_curvature(position, angle, (x, y))
	k *= np.linalg.norm(np.subtract((x,y), position))
	print k
	print center
	pylab.scatter(*center)
	pylab.draw()

pylab.gca().set_aspect('equal')
pylab.connect('motion_notify_event', curver)
pylab.show()
