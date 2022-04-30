import sys
import uuid

def globalize(func):
	"""This decorator allows for nested functions to be 
	utilized for parallel processing using the multiprocessing package
	Example: 
	def outer_fcn():
	  @globalize
	  def inner_fcn():
	    #<do something>
	    return
	  r = pool.imap(inner_fcn, range(40))
	return
	"""
	def result(*args, **kwargs):
		return func(*args, **kwargs)
	result.__name__ = result.__qualname__ = uuid.uuid4().hex
	setattr(sys.modules[result.__module__], result.__name__, result)
	return result

def nargout():
	"""
	yields the number of arguments available for assignment 
	Example: 
	def fcn():
	  n_out = nargout()
	  print(n_out)
	  return
	a,b,c = fcn()
	: 3
	"""
	import traceback
	callInfo = traceback.extract_stack()
	callLine = str(callInfo[-3].line)
	if callLine.find('=') < 0:
		return 0
	split_equal = callLine.split('=')
	split_comma = split_equal[0].split(',')
	num = len(split_comma)
	return num

class LazyProperty(object):
	""" 
	A decorator for object methods that allows for delayed assignment of object properties
	Example: 
	call xgb_shap():
		def __init__(self, a):
			self.a = a
		
		@LazyProperty
		def ex_prop():
			return np.random.rand(5)
	"""
	def __init__(self, func):
		self.func = func
	
	def __get__(self, instance, owner):
		if instance is None:
			return self
		else:
			value = self.func(instance)
			setattr(instance, self.func.__name__, value)
			return value
