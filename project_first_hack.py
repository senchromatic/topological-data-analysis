## LeDuc, Pereira, Zhang
# This is a first hack at working with the project data using the KL divergence to measure the distance
# between two probability distributions of the depth of minimum sound speed.

import numpy as np
import pandas as pd
import pylab as pl # This gets used a lot I promise

from abstract_simplicial_complex import Point, Simplex, vietoris_rips
from filtration import Filtration
from metrics import ks_test
from random import sample, seed
from scipy.interpolate import interp1d
from statfuncs import ecdf


# Global constants (bad)
# TODO: move these into a config file?
USE_RANDOM_SAMPLING = True  # Read a random sample of points from the input data
MIN_POINTS_FOR_KL_DIVERGENCE = 4  # Minimum number of points needed to compute KL divergence
TEST_SAMPLE_SIZE = 2000  # Number of points to extract from dataset (non-random sampling)
MIN_SIGNIFICANCE_LEVEL = 0.05  # Used in Kolmogorov-Smirnov test
MAX_ASC_DIMENSION = 2  # Maximum dimension of the simplices considered in the output ASC

# Read a subset of data from multiple years (currently, 2 years -- hardcoded)
def read_raw_data(sample_size=None, sample_randomly=True, random_seed=0):
    seed(random_seed)
    data1 = np.genfromtxt('data/MITprof_mar2016_argo0708.nc.csv', delimiter = ',')
    data2 = np.genfromtxt('data/MITprof_mar2016_argo0910.nc.csv', delimiter = ',')
    data = np.concatenate((data1, data2[:, 2:]), axis = 1)
    depths = data[3:,0]
    N = data.shape[1]  # total number of data points
    # Choose a subset of points to return, if sample_size is specified
    if (sample_size is not None) and (sample_size < N):
        if sample_randomly:
            all_indices = range(data.shape[1])
            selected_indices = sample(all_indices, sample_size)
        else:
            selected_indices = range(sample_size)
        data = data[:, selected_indices]
    return data, depths

# Preprocess data to extract dates, ocean depths, latitudes, longitudes, and sound speed profiles
def extract_features(data):
    NUM_DAYS_TO_UNIX_EPOCH = 719529  # Days between January 1, 0000 to January 1, 1970
    dates = pd.to_datetime(data[2, 1:] - NUM_DAYS_TO_UNIX_EPOCH, unit='D')
    
    # Spatial dimensions

    latitudes = np.around(data[0, 1:] / 3) * 3  # length: N
    longitudes = np.around(data[1, 1:] / 3) * 3  # length: N
    
    # Sound speed profiles
    sound_speed_profiles = data[3:, 1:]  # dimensions: K x N
    
    return dates, latitudes, longitudes, sound_speed_profiles

# Divide the Earth's surface into geographic (latitude/longitude) boxes,
# and return the coordinates at the boundary of each box.
def locate_grid_boundaries(latitudes, longitudes):
    # Can find probability density functions of sound speed minimum or maximum
    # <unused>
    # mask by month first, then divide into lat/lon boxes?
    ## sound_speed_minimums = np.argmin(sound_speed_profiles, axis = 0)
    ## min_depths = depths[sound_speed_minimums]
    ## for mm in [1]:  # [1,2,3,4,5,6,7,8,9,10,11,12]
    ##     month_mask = ( dates.month == mm )
    ##     this_month_min_depths = min_depths[month_mask]
    ##     this_month_lats = lats[month_mask]
    ##     this_month_lons = lons[month_mask]
    ##     ...
    # </unused>
    unique_latitudes = np.unique(latitudes)
    unique_longitudes = np.unique(longitudes)
    
    # Cartesian product between latitude and longitude gridpoints
    [lat_grid, lon_grid] = np.meshgrid(unique_latitudes, unique_longitudes)
    return np.reshape(lat_grid, [1, -1])[0], np.reshape(lon_grid, [1, -1])[0]

# Compute CDFs on each geographic box
def compute_boxed_cdfs(latitudes, longitudes, grid_latitudes, grid_longitudes, sound_speed_profiles, min_points_per_box=MIN_POINTS_FOR_KL_DIVERGENCE):
    min_speeds = np.argmin(sound_speed_profiles, axis = 0)
    min_depths = depths[min_speeds]
    num_boxes = len(grid_latitudes)
    
    spatial_cdf = np.zeros([len(depths), num_boxes])
    subsample_sizes = np.zeros(num_boxes)
    
    for k in range(len(grid_latitudes)):
        lat_indices = np.where(latitudes == grid_latitudes[k])
        lon_indices = np.where(longitudes == grid_longitudes[k])
        box_indices = np.intersect1d(lat_indices, lon_indices)
        
        # Consider the number of points in this box
        subsample_sizes[k] = len(np.unique(min_depths[box_indices]))
        # and whether the subsample size is sufficient
        if subsample_sizes[k] < min_points_per_box:
            continue
        
        # Compute empirical CDF on the depths
        quantiles, probabilities = ecdf(min_depths[box_indices])

        # Interpolate the ecdf onto the depth grid we started with, then adjust the
        # values as needed. We need these CDFs to measure the distance between points. 
        interpolator = interp1d(quantiles, probabilities, fill_value="extrapolate")
        p = interpolator(depths)  # interpolated cumulative probabilities
        
        # Probabilities must be between 0 and 1
        too_low = p < 0
        p[too_low] = 0
        
        too_high = p > 1
        p[too_high] = 1
        
        spatial_cdf[:, k] = p
    
    # Exclude boxes with too few data points
    mask = np.where(subsample_sizes >= min_points_per_box)
    masked_lats = grid_latitudes[mask]
    masked_lons = grid_longitudes[mask]
    masked_cdfs = spatial_cdf[:, mask]
    return masked_lats, masked_lons, masked_cdfs

# Compute the Kolmogorov-Smirnov statistic between each pair of geographic boxes  
def compute_pairwise_ks_statistics(latitudes, longitudes, cdfs):
    # <unused>
    ## Kolmogorov-Smirnov constants
    # a = MIN_SIGNIFICANCE_LEVEL
    # c_a = np.sqrt(-np.log(a/2)*0.5)
    ## Value the metric needs to exceed to reject the ks test null hypothesis at the given significance level
    # critical_value = c_a * np.sqrt(2 / len(depths))
    # </unused>
    
    num_boxes = len(latitudes)
    assert(num_boxes == len(longitudes))
    ks_matrix = np.zeros([num_boxes, num_boxes])
    
    # Calculate the test statistic on each box
    for i in range(0, num_boxes):
        for j in range(i+1, num_boxes):
            ks_matrix[i, j] = ks_test(cdfs[:, i], cdfs[:, j])
            ks_matrix[j, i] = ks_matrix[i, j]
    
    return ks_matrix

# Human-readable format for a geographic point
def stringify_coordinates(latitude, longitude):
    return str(latitude) + "deg N by " + str(longitude) + "deg E"

def generate_geographic_names(latitudes, longitudes):
    num_masked_boxes = len(masked_latitudes)
    assert(num_masked_boxes == len(masked_longitudes))
    return [stringify_coordinates(masked_latitudes[k], masked_longitudes[k]) for k in range(num_masked_boxes)]

# Create a set of N points, where N = |names| = |coords|
# names: list of strings
# coords: list of coordinates
def create_point_cloud(names, coords, metric):
    cloud = set()
    
    N = len(names)
    assert(N == coords.shape[1])
    
    for k in range(len(names)):   
        cloud.add(Point(names[k], coords[:, k], metric))
    return cloud


if __name__ == '__main__':
    sample_data, depths = read_raw_data(sample_size=TEST_SAMPLE_SIZE, sample_randomly=USE_RANDOM_SAMPLING)
    
    dates, latitudes, longitudes, sound_speed_profiles = extract_features(sample_data)
    
    grid_latitudes, grid_longitudes = locate_grid_boundaries(latitudes, longitudes)
    
    masked_latitudes, masked_longitudes, masked_cdf = compute_boxed_cdfs(latitudes, longitudes, grid_latitudes, grid_longitudes, sound_speed_profiles)
    
    # TODO: investigate why masked_cdfs returned by compute_boxed_cdfs has 1 extra dimension compared to local variable in function
    masked_cdfs = masked_cdf[:, 0, :]
    
    # distance_matrix = compute_pairwise_ks_statistics(masked_latitudes, masked_longitudes, masked_cdfs)
    
    # Generate an ASC based on Kolmogorov-Smirnov distances fed into Vietoris-Rips algorithm
    geographic_names = generate_geographic_names(masked_latitudes, masked_longitudes)
    point_cloud = create_point_cloud(geographic_names, masked_cdfs, ks_test)
    
    # f = Filtration(point_cloud, len(point_cloud) - 1)
    f = Filtration(point_cloud, max_dimension=MAX_ASC_DIMENSION)
    f.print_metadata()
    f.generate_filtration(verbosity=1)
    # f.print_filtration()
    # for asc in f.asc_sequence:
    #     for k in range(1, MAX_ASC_DIMENSION+1):
    #         asc.compute_boundary(k)
    
    # for rr in np.arange(0.1, 1, 0.1):
    #     rips_asc = vietoris_rips(point_cloud, MAX_ASC_DIMENSION, rr)
    #     print("Radius = "+str(rr))
    #     print(rips_asc)
    #     # Print simplices
    #     for k in range(1,MAX_ASC_DIMENSION+1):
    #         rips_asc.compute_boundary(k)
