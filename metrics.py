import numpy as np

DEFAULT_EPS = 1e-5

def discrete_metric(a, b, tolerance=DEFAULT_EPS):
  return int(max(np.absolute(a - b)) > tolerance)

def norm(x, order=None, axis=None, keepdims=False):
  return np.linalg.norm(x, order, axis, keepdims)

# Reference: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
# Default: Euclidean distance (l^2)
def induced_metric(a, b, order=None, axis=None, keepdims=False):
  return norm(a - b, order, axis, keepdims)

# TODO: replace with relative error margins
def close_enough(a, b, metric):
  return metric(a, b) <= DEFAULT_EPS
