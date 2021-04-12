#Zhang, Pereira, LeDuc
import numpy as np
import scipy.special as scs
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

# Kullback-Leibler divergence, modified to be symmetric
# Input: two CDFs of identical shape, e.g. generated from the ecdf function
# Note: This is a semi-metric as it doesn't satisfy the triangle inequality
def sym_kl(cdf1 , cdf2, dx):
    # First step is to take them and turn them into PDFs
    f1 = np.gradient(cdf1)
    f1 /= np.sum(f1)
    
    f2 = np.gradient(cdf2)
    f2 /= np.sum(f2)
    
    # Scipy defines KL divergence as x*log(x/y)+x-y but it all comes out in the end.
    # Same values in each array are finite and nonzero so adding these together makes
    # the extra stuff cancel out!
    ent1 = scs.kl_div(f1, f2)
    p1 = np.sum(ent1[np.where(np.isfinite(ent1))])
    
    ent2 = scs.kl_div(f2, f1)
    p2 = np.sum(ent2[np.where(np.isfinite(ent2))])
   
    return 0.5*(p1+p2)

def ks_test(cdf1, cdf2 ):
  ## This is the common Kolmogorov-Smirnov test for the equality of two distributions.
  # Based on the scipy implementation
  # D = sup_x(|F1(x)-F2(x)|) Basically the L_inf metric but for CDFs
  # Output 1 = result of the KS test.
  diffs = np.abs(cdf1-cdf2)
  return np.max(diffs)
