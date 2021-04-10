## LeDuc, Pereira, Zhang
#This is a first hack at working with the project data using the KL divergence to measure the distance
#between two probability distributions of the depth of minimum sound speed.
import abstract_simplicial_complex as asc
import z2array as z2
import numpy as np
import sets as st
import scipy.interpolate as sci
import metrics as met
import pylab as pl
import datetime as dt
import pandas as pd
import statfuncs as sf

# TODO: wrap code logic into a function, make this a parameter
MIN_POINTS_IN_DISTRIBUTION = 4

def create_simplices(names, coords, metric):
    a = asc.Simplex()
    for ndx in range(len(names)):   
        a.add_point(asc.Point(names[ndx], coords[:,ndx], metric))
    return a

data1 = np.genfromtxt('data/MITprof_mar2016_argo0708.nc.csv', delimiter = ',')
data2 = np.genfromtxt('data/MITprof_mar2016_argo0910.nc.csv', delimiter = ',')

data = np.concatenate((data1, data2[:,2:]), axis = 1)
data = data[:,0:1000]#Just for testing purposes to make sure everything goes through okay
##breakpoint()
dates = pd.to_datetime( data[2,1:]-719529, unit='D')
depths = data[3:, 0]
lats = np.around(data[0,1:]/3)*3
lons = np.around(data[1,1:]/3)*3

ssps = data[3:,1:]
nProfiles = len(ssps[0,:])

## Can find PDFs of sound speed minimum or maximum
# sound speed minimum
mins = np.argmin( ssps, axis = 0 )
minDepths = depths[mins]
# mask by month first, then divide into lat/lon boxes?
##for mm in [1]:#[1,2,3,4,5,6,7,8,9,10,11,12]
##    monthMask = ( dates.month == mm )
##    thisMonthMinDepths = minDepths[monthMask]
##    thisMonthLats = lats[monthMask]
uLats = np.unique(lats)
##thisMonthLons = lons[monthMask]
uLons = np.unique( lons )

[latGrid,lonGrid] = np.meshgrid( uLats, uLons )
theseLats = np.reshape(latGrid, [1,-1])
theseLons = np.reshape(lonGrid, [1,-1])

spatialDist = np.zeros([len(depths), len( theseLats.transpose() )])
for ndx in range(0, len(theseLats.transpose())):
    latNdcs = np.where(lats == theseLats[0,ndx])
    lonNdcs = np.where(lons == theseLons[0,ndx])
    ndcs = np.intersect1d( latNdcs, lonNdcs )
    if len(np.unique(minDepths[ndcs])) >= MIN_POINTS_IN_DISTRIBUTION:
        #If there are enough points to make a distribution
        [x,y] = sf.ecdf(minDepths[ndcs])
        # Interpolate the ecdf onto the depth grid we started with, then adjust the
        # values as needed. We need these CDFs to measure the 
        f = sci.interp1d( x,y, fill_value="extrapolate")
        yi = f(depths)
        tooLow = yi<0
        yi[tooLow] = 0
    
        tooHigh = yi>1
        yi[tooHigh] = 1
        spatialDist[:,ndx] = yi
    else:
        #No appropriate dists in the box
        theseLats[0,ndx] = -1000
        theseLons[0,ndx] = -1000
[foo,mask] = np.where(theseLats>-1000)
maskedLats = theseLats[0,mask]
maskedLons = theseLons[0,mask]
maskedDists = spatialDist[:,mask]

distMat = np.zeros([len(maskedLats), len(maskedLats)])
a = 0.05
c_a = np.sqrt(-np.log(a/2)*0.5)

ksStat = c_a*np.sqrt(2/len(depths))#%Value the metric needs to exceed to reject the ks test null hypothesis at the a-lvl
for ii in range(0,len(maskedLats)):
    for jj in range(ii+1, len(maskedLats)):
        distMat[ii,jj] = met.ks_test(maskedDists[:,ii], maskedDists[:,jj])
        distMat[jj,ii] = distMat[ii,jj]
   
names = [str(maskedLats[ndx])+"deg N by "+str(maskedLons[ndx])+"deg E" for ndx in range(len(maskedLats))]
simplicialComplex = create_simplices(names,maskedDists, met.ks_test)
vrc = asc.vietoris_rips(simplicialComplex.points, 2, 0.5)
print("2-implices:")
for i in vrc.k_simplices(2):
    print(i)
print("\n\n\n1-simplices:")
for i in vrc.k_simplices(1):
    print(i)
