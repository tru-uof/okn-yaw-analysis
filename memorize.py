import os
import cPickle
import shelve
import atexit

import logging
logger = logging.getLogger('tru.memorize')

# TODO: Needs cleaning up

import time
class TimeCounter:
	def __init__(self, name):
		self.name = name
		self.start_time = time.clock()
		self.stop_time = None
	
	def stop(self):
		if(self.stop_time is None):
			self.stop_time = time.clock()

	def __str__(self):
		self.stop()
		interval = self.stop_time - self.start_time
		return "Task %s took %.3f seconds"%(self.name, interval)

import functools
def B(func):
	def wrapper(*args, **kwargs):
		t = TimeCounter(str(func))
		result = func(*args, **kwargs)
		logger.debug(str(t))
		return result
	
	functools.update_wrapper(wrapper, func)
	return wrapper

class Memorizer(object):
	def value_hash(self, value): pass
	def merge_hashes(self, *args): pass
	def callspecHash(self, callspec): pass
	def hashIsInMemory(self, callhash): pass
	def storeHashResult(callhash, result): pass
	def getHashResult(self): pass
	
	def isInMemory(self, callspec):
		return self.hashIsInMemory(self.callspecHash(callspec))
	
	def getCachedResult(self, callspec):
		return self.getHashResult(self.callspecHash(callspec))

	def storeResult(callspec, result):
		return self.storeHashResult(self.callspecHash(callspec), result)

	def function_hash(self, func):
		try:
			code = func.func_code.co_code
		except AttributeError:
			code = ""
		return self.value_hash((func.__name__, func.__globals__['__file__'], code))

	@classmethod
	def fullCallspec(cls, (func, args, kargs)):
		funcspec = (func.__name__, func.__globals__['__file__'])
		return (funcspec, args, kargs)

	def memorize(self, func):
		def wrapper(*args, **kargs):
			return self.memorize_call(func, *args, **kargs)
		return wrapper

	__call__ = memorize

	def memorize_call(self, func, *args, **kwargs):
		callspec = (func, args, kwargs)
		callhash = self.callspecHash(callspec)
		return self.memorize_call_with_hash(callhash, func, *args, **kwargs)

	def memorize_call_with_hash(self, callhash, func, *args, **kwargs):
		if(B(self.hashIsInMemory)(callhash)):
			return B(self.getHashResult)(callhash)
		
		result = B(func)(*args, **kwargs)
		B(self.storeHashResult)(callhash, result)
		return result


class FileShelver(object):
	def __init__(self, directory):
		if not os.path.exists(directory):
			os.mkdir(directory)
		self.directory = directory
	
	def _path(self, key):
		return os.path.join(self.directory, key)
	
	def __contains__(self, key):
		return os.path.exists(self._path(key))

	def __setitem__(self, key, item):
		f = open(self._path(key), 'w')
		cPickle.dump(item, f, protocol=2)
		f.close()

	def __getitem__(self, key):
		with open(self._path(key), 'r') as f:
			return cPickle.load(f)

	def sync(self): pass

	def close(self): pass

import numpy as np
class NumpyFileShelver(FileShelver):
	def __npy_path(self, key):
		return self._path(key)+'.npy'

	def __contains_npy(self, key):
		return os.path.exists(self.__npy_path(key))

	def __contains__(self, key):
		if self.__contains_npy(key): return True
		return FileShelver.__contains__(self, key)

	def __setitem__(self, key, item):
		if not isinstance(item, np.ndarray):
			FileShelver.__setitem__(self, key, item)
			return
		
		np.save(self.__npy_path(key), item)

	def __getitem__(self, key):
		if not self.__contains_npy(key):
			return FileShelver.__getitem__(self, key)

		return np.load(self.__npy_path(key))


class StreamHasher:
	def __init__(self, hasher):
		self.hash = hasher()

	def write(self, data):
		self.hash.update(data)

import hashlib
class ShelvingMemorizer(Memorizer):
	def __init__(self, directory, indexpath, hasher=hashlib.md5):
		#self.index = shelve.open(indexpath, protocol=2)
		self.index = NumpyFileShelver(directory)
		atexit.register(self._cleanup)
		self.hasher = hasher
	
	def merge_hashes(self, *args):
		return self.hasher(reduce(lambda x,y: x+y, args)).hexdigest()

	def value_hash(self, value):
		hash = StreamHasher(self.hasher)
		cPickle.dump(value, hash, protocol=2)
		digest = hash.hash.hexdigest()
		return digest

	def callspecHash(self, callspec):
		callspec = Memorizer.fullCallspec(callspec)
		return self.value_hash(callspec)

	def hashIsInMemory(self, callhash):
		return callhash in self.index
	
	def getHashResult(self, callhash):
		return self.index[callhash]
	
	def storeHashResult(self, callhash, result):
		# TODO: Store results in separate files
		#assert callhash not in self.index, "Memorizer key collision!"
		logging.debug("Storing value with hash %s"%callhash)
		self.index[callhash] = result
		self.index.sync()

	def _cleanup(self):
		if(self.index):
			self.index.close()

memorize = ShelvingMemorizer('.py_memory', '.py_memory/index.shelve')


def pack(label, func, *args, **kargs):
	def __wrapper(key_for_identification):
		return func(*args, **kargs)
	#print('__wrapper', str(__wrapper))
	#m = tru.memorize.Memorizer('.py_memory', ".py_memory/%s.shelve" % (label,))
	
	return memorize(__wrapper)(label)


class HashedValue:
	def __init__(self, myhash, value=None):
		self.myhash = myhash
		if value is None:
			self.has_cache = False
		else:
			self.value_cache = value
			self.has_cache = True

	def __call__(self):
		if not self.has_cache:
			self.value_cache = memorize.getHashResult(self.myhash)
			self.has_cache = True
		
		return self.value_cache

def memorize_hash(func, args, kwargs):
	my_hash = memorize.function_hash(memorize_hash)
	func_hash = memorize.function_hash(func)
	
	arg_spec = __apply_to_args(lambda a: a.myhash, args, kwargs)

	arg_hash = memorize.value_hash(arg_spec)
	full_hash = memorize.merge_hashes(my_hash, func_hash, arg_hash)
	
	# Return the hash from memory
	if memorize.hashIsInMemory(full_hash):
		logger.debug("Function %s in memory with hash %s"%(func, full_hash))
		return HashedValue(memorize.getHashResult(full_hash))
	
	logger.debug("Cache miss on %s with \n\targ hash %s \n\tfull hash %s"%(
					func, arg_hash, full_hash))
	args, kwargs = __unpack_hash_args(args, kwargs)
	result_value = func(*args, **kwargs)
	result_hash = memorize.value_hash(result_value)
	logger.debug("Storing call result with hash %s"%result_hash)
	memorize.storeHashResult(result_hash, result_value)
	logger.debug("Storing call with hash %s"%full_hash)
	memorize.storeHashResult(full_hash, result_hash)
	assert memorize.hashIsInMemory(full_hash)
	
	return HashedValue(result_hash, result_value)
	

def __unpack_hash(obj):
	if not isinstance(obj, HashedValue):
		return obj
	
	return obj()

def __pack_hash(obj):
	if isinstance(obj, HashedValue):
		return obj

	obj_hash = memorize.value_hash(obj)
	if not memorize.hashIsInMemory(obj_hash):
		memorize.storeHashResult(obj_hash, obj)
		return HashedValue(obj_hash, obj)
	
	return HashedValue(obj_hash)

def __apply_to_args(func, args, kwargs):
	args = [func(a) for a in args]
	kwargs = dict((k,func(v)) for k,v in kwargs.iteritems())
	return args, kwargs

__unpack_hash_args = functools.partial(__apply_to_args, __unpack_hash)
unpack_hash_args = __unpack_hash_args
__pack_hash_args = functools.partial(__apply_to_args, __pack_hash)

def hash_mem(func):
	def wrapper(*args, **kwargs):
		return_hash = kwargs.pop('_return_hash', False)

		try:
			args, kwargs = __pack_hash_args(args, kwargs)
			hash_value = memorize_hash(func, args, kwargs)
		except cPickle.PicklingError:
			logger.debug("Failed to pickle args of %s"%(func))
			args, kwargs = __unpack_hash_args(args, kwargs)
			value = func(*args, **kwargs)
			try:
				value_hash = memorize.value_hash(value)
				hash_value = HashedValue(value_hash, value)
			except cPickle.PicklingError:
				logger.debug("Failed to pickle return value of %s"%func)
				hash_value = HashedValue(None, value)
		
		
		if return_hash:
			return hash_value
		else:
			return hash_value()
				
	wrapper._hash_aware = True

	return wrapper


