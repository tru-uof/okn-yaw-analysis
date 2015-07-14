
def param_line_to_points(param, size):
	a = np.cos(param[1])
	b = np.sin(param[1])
	x0 = param[0]*a
	y0 = param[0]*b
	
	# This is very ugly
	start = (x0 + 10000*(-b), y0 + 10000*a)
	end = (x0 - 10000*(-b), y0 - 10000*a)
	return cv.ClipLine((size[1], size[0]), start, end)

def param_line_to_points_(line, width):
	line = h.polar_line_to_cartesian(line[0], line[1])
	start = (0, line[1])
	end = (width, width*line[0]+line[1])
	return (start, end)

def polar_line_to_cartesian(line):
	a = -np.cos(line[1])/np.sin(line[1])
	y0 = line[0]/(np.sin(line[1]))
	return (a, y0)

def get_vanishing_point(a, b):
	if(a[0] - b[0] == 0): return (0, 0)
	x = (b[1] - a[1])/(a[0] - b[0])
	y = a[0]*x + a[1]
	return (x, y)

