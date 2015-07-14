import cv
import sys
import numpy as np

class LaneHypothesis(object):
	def __init__(self, position, weight=1.0, parent=None):
		self.position = position
		self.parent = parent

	def spawn(self, position):
		return LaneHypothesis(position, parent=self)

	#def get_prior
