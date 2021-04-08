import abstract_simplicial_complex as asc
import z2array as z2
import numpy as np
import scipy.interpolate as sci

import pylab as pl
import datetime as dt
import pandas as pd
import statfuncs as sf

data1 = np.genfromtxt('data/MITprof_mar2016_argo0708.nc.csv', delimiter = ',')
data2 = np.genfromtxt('data/MITprof_mar2016_argo0910.nc.csv', delimiter = ',')

data = np.concatenate((data1, data2[:,2:]), axis = 1)
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
    if len(np.unique(minDepths[ndcs]))>=4:
        [x,y] = sf.ecdf(minDepths[ndcs])
        f = sci.interp1d( x,y, fill_value="extrapolate")
        yi = f(depths)
        tooLow = yi<0
        yi[tooLow] = 0
    
        tooHigh = yi>1
        yi[tooHigh] = 1
        spatialDist[:,ndx] = yi
    else:
        theseLats[0,ndx] = -1000
        theseLons[0,ndx] = -1000
[foo,mask] = np.where(theseLats>-1000)
maskedLats = theseLats[0,mask]
maskedLons = theseLons[0,mask]
maskedDists = spatialDist[:,mask]

distMat = np.zeros([len(maskedLats), len(maskedLats)])

for ii in range(0,len(maskedLats)):
    for jj in range(ii+1, len(maskedLats)):
        distMat[ii,jj] = sf.SymKL(maskedDists[:,ii], maskedDists[:,jj])
        distMat[jj,ii] = distMat[ii,jj]

