import numpy as np

# Empirical Cumulative Distribution Function
def ecdf(sample):
    
    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)
    
    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)
    
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities in (0, 1]
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    
    return quantiles, cumprob
