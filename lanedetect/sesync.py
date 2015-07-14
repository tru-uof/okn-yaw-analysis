#!/usr/bin/python

import gst
import gobject
import threading
import itertools
import functools
import subprocess
import logging
from PIL import Image, ImageOps
from StringIO import StringIO
import numpy as np

logger = logging.getLogger("ocr_sync")

gobject.threads_init()

class MainLoopThread(threading.Thread):
	def __enter__(self):
		self.start()

	def __exit__(self, *args):
		self.loop.quit()
		
	def run(self):
		self.loop = gobject.MainLoop()
		self.loop.run()

def log_messages(*args):
	#print args
	return True

def run_in_gobject_loop(func):
	def wrapper(*args, **kwargs):
		with MainLoopThread():
			func(*args, **kwargs)
	return wrapper

def video_frames(infile, message_listener=log_messages):
	pipeline = gst.parse_launch("""
		filesrc name=src ! decodebin name=dec sync=false !
			ffmpegcolorspace !
			pngenc snapshot=false !
			appsink sync=false name=sink
		""")
	
	pipeline.get_bus().add_signal_watch()
	pipeline.get_bus().connect("message", message_listener)
	src = pipeline.get_by_name("src")
	sink = pipeline.get_by_name("sink")
	src.set_property("location", infile)
	
	pipeline.set_state(gst.STATE_PAUSED)
	pipeline.set_state(gst.STATE_PLAYING)
	if sink.emit("pull-preroll") is None:
		return
	
	while True:
		frame = sink.emit("pull-buffer")
		if not frame: return
		
		seek = yield frame
		if seek:
			seek = seek*gst.SECOND
			seek += frame.timestamp
			pipeline.seek_simple(gst.FORMAT_TIME,
						gst.SEEK_FLAG_FLUSH,
						int(seek))

class TimestampOcr:
	def __call__(self, frame):
		cmd = "gocr -C 0-9. -"
		proc = subprocess.Popen(cmd, shell=True,
			stdin=subprocess.PIPE,
			stderr=subprocess.PIPE,
			stdout=subprocess.PIPE)

		timestamp = proc.communicate(frame)[0]
		timestamp = timestamp.strip()
		timestamp = timestamp.strip('_')
		try:
			timestamp = float(timestamp)
		except ValueError:
			return None
		return timestamp

def c(f, g):
	return lambda *args, **kwargs: f(g(*args, **kwargs))

def video_timestamps(videofile, timestamper, stepsize=100.0):
	frames = video_frames(videofile)
	
	frame = frames.next()
	while frame:
		frame_ts = timestamper(frame)
		if frame_ts is not None:
			video_ts = frame.timestamp/float(gst.SECOND)
			yield video_ts, frame_ts
		
		frame = frames.send(stepsize)

def ransab_poly1d(a, b, outlier_share, niters):
	"""
	A very very naive robustish linear fit

	TODO: Implement a real RANSAC
	"""
	a = np.array(a)
	b = np.array(b)
	n = int(len(a)*(1-outlier_share))
	best = None
	for i in range(niters):
		s = np.random.permutation(len(a))[:n]
		fit = np.polyfit(a[s], b[s], 1)
		residuals = np.polyval(fit, a[s]) - b[s]
		mse = np.mean(residuals**2)
		if not best or mse < best[0]:
			best = (mse, fit, s)
	
	return best[1], np.sort(best[2])

def get_timestamp_fit(videofile, preprocessor=lambda f: None,
			residual_handler=lambda *args: None):
	
	timestamper = video_timestamps(videofile, c(TimestampOcr(), preprocessor), 500.0)
	timestamps = list(timestamper)
	if len(timestamps) < 5:
		raise BadFitException("Not enough valid frame timestamps found in %s"%videofile)

	vid_ts, frame_ts = [np.array(a, dtype=np.float) for a in zip(*timestamps)]
	
	frame_to_vid, sample = ransab_poly1d(frame_ts, vid_ts, 0.3, 1000)
	frame_ts = frame_ts[sample]
	vid_ts = vid_ts[sample]
	#frame_to_vid = np.polyfit(frame_ts, vid_ts, 1)
	
	fitted = np.polyval(frame_to_vid, frame_ts)
	residuals = fitted - vid_ts
	
	stats = (np.max(np.abs(residuals)),
		np.mean(residuals),
		np.mean(np.abs(residuals)))
		
	logger.info("Fit residual abs max: %f, mean %f, abs mean %f"%stats)
	
	
	import pylab
	pylab.subplot(2,1,1)
	pylab.plot(frame_ts, fitted)
	pylab.scatter(frame_ts, vid_ts)
	pylab.subplot(2,1,2)
	pylab.plot(frame_ts, residuals)
	pylab.show()
	
	
	residual_handler(*stats)
	
	return frame_to_vid

class BadFitException(Exception): pass

def se_frame_preprocessor(frame):
	pos, size = (3, 24), (80, 24)

	img = Image.open(StringIO(frame))
	output = StringIO()
	img = img.crop((pos[0], pos[1], pos[0]+size[0], pos[1]+size[1]))
	img = ImageOps.invert(img)
	img.save(output, format="PPM")
	return output.getvalue()

def mean_residual_checker(err_limit):
	def checker(*stats):
		err = stats[2]
		if err < err_limit: return
		raise BadFitException("Mean abs residual %f higher than %f"%(err, err_limit))
	return checker
			
se_timestamp_fit = functools.partial(get_timestamp_fit, 
				preprocessor=se_frame_preprocessor,
				residual_handler=mean_residual_checker(0.1))

if __name__ == '__main__':
	import sys
	logging.basicConfig(level=logging.DEBUG)
	print se_timestamp_fit(sys.argv[1])
	
