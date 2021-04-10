import numpy as np
import metrics as met

# Synthetic data, two identical arrays
dist1 = np.array([0,1/2,2/3,1,1,1,1])
dist2 = np.array([0,1/2,2/3,1,1,1,1])
# Both of these should come out as zero after kl and ks
klVal = met.sym_kl(dist1, dist2, 1)
ksVal = met.ks_test(dist1, dist2)

print("dist1 = "+str(dist1))
print("dist2 = "+str(dist2))
print("Expect both the KLD and KS to be zero")
print("KLD = "+str(klVal))
print("KS = "+str(ksVal))

# Different array of data, should have non-zero Kolmogorov-Smirnov distance
dist3 = np.array([0.01,0.1,0.5, 0.5,0.51,1,1])
ksVal3 = met.ks_test(dist1, dist3)
print("dist3 = "+str(dist3))
print("Expect Ks(dist1, dist3)="+str(np.max(np.abs(dist1-dist3))))
print("KS = "+str(ksVal3))

# Assert symmetry of the custom Kullback-Leibler divergence semi-metric 
print("Expect that KLD(dist1, dist3) = KLD(dist3, dist1)")
print("KLD(dist1, dist3) = "+str(met.sym_kl(dist1, dist3, 1))) 
print("KLD(dist3, dist1) = "+str(met.sym_kl(dist3, dist1, 1))) 
