import numpy as np

DEFAULT_EPS = 1e-5

# TODO: replace with relative error margins
def discrete_metric(a, b, tolerance=DEFAULT_EPS):
  return int(max(np.absolute(a - b)) > tolerance)

def norm(x, order=None, axis=None, keepdims=False):
  return np.linalg.norm(x, order, axis, keepdims)

# Reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
def induced_metric(a, b, order=None, axis=None, keepdims=False):
  return norm(a - b, order, axis, keepdims)

def close_enough(a, b, metric):
  return metric(a, b) <= DEFAULT_EPS
