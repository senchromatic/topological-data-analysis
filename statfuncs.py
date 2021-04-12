import numpy as np
import pylab as pl
import scipy.special as scs


# Empirical Cumulative Distribution Function
def ecdf(sample):
    
    # Convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)
    
    # Find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)
    
    # Take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between (0, 1]
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    
    return quantiles, cumprob
