import numpy as np
import pylab as pl
import scipy.special as scs
def ecdf(sample):
    #empirical CDF
    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)
    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob

def sym_kl(cdf1 , cdf2 ):
    #Note: This is a semi-metric as it doesn't satisfy the triangle inequality
    #Modified KL divergence to be symmetric
    #Input: two CDFs. First step is to take them and turn them into PDFs
    f1 = np.gradient(cdf1)
    f1 /= np.sum(f1)
    
    f2 = np.gradient(cdf2)
    f2 /= np.sum(f2)
    #Scipy defines KL divergence as x*log(x/y)+x-y but it all comes out in the end.
    #Same values in each array are finite and nonzero so adding these together makes
    #the extra stuff cancel out!
    ent1 = scs.kl_div(f1, f2)
    p1 = np.sum(ent1[np.where(np.isfinite(ent1))])
    
    ent2 = scs.kl_div(f2, f1)
    p2 = np.sum(ent2[np.where(np.isfinite(ent2))])
   
    return 0.5*(p1+p2)
