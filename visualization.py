# Functions for plotting persistence as a function of birth times (diameter cutoffs).

from collections import defaultdict
from matplotlib import pyplot
import numpy as np
# One color per homology dimension (0: blue, 1: green, 2: red, 3: yellow)
DIMENSIONAL_COLOR_CODES = "bgry"
DIMENSIONAL_COLOR_LABELS = ["connected components (k=0)", "holes (k=1)", "voids (k=2)", "3-homologies (k=3)"]

DEFAULT_OPACITY = 0.3  # alpha = 1 - transparency
DEFAULT_MARKER_SIZE = 25  # (pixels width) ^ 2

# pivots_rc: list of (row, column) pairs
# diameters: list of diameter corresponding to each index among values in pivots_rc 
# dimensions: list of simplex dimension corresponding to each index among values in pivots_rc 
# xy_fmt: format string for the x=y line
def plot_birth_death(pivots_rc, diameters, dimensions, xy_fmt="k--", output_filename="birth_death.png", verbose=True):
    # Prepare plotting
    figure, axes = pyplot.subplots()
    axes.set(xlabel="Birth (diameter)", ylabel="Death (diameter)", title="Birth-Death Plot")
    axes.plot([0, 1], xy_fmt, alpha=DEFAULT_OPACITY)  # y = x
    # For each dimension
    for k in range(min(dimensions), max(dimensions)):
        x = []  # r: row (birth)
        y = []  # c: column (death) 
        for r, c in pivots_rc:
            # Check whether the dimension matches
            if dimensions[r] != k:
                continue
            # Add a point in that dimension's list
            x.append(diameters[r])  
            y.append(diameters[c])
        # Plot homologies at dimension k
        axes.scatter(x, y, color=DIMENSIONAL_COLOR_CODES[k], label=DIMENSIONAL_COLOR_LABELS[k], alpha=DEFAULT_OPACITY, s=DEFAULT_MARKER_SIZE)
        if verbose:
            print("(Birth diameter, death diameter) for " + str(k) + "-homologies, ordered by increasing birth: ")
            print(list(zip(x,y)))
    axes.legend()
    figure.savefig(output_filename)

# Persistence is defined as death - birth diameter
# pivots_rc: list of (row, column) pairs
# diameters: list of diameter corresponding to each index among values in pivots_rc 
# dimensions: list of simplex dimension corresponding to each index among values in pivots_rc 
# y0_fmt: format string for the y=0 line
def plot_birth_persistence(pivots_rc, diameters, dimensions, y0_fmt="k--", output_filename="birth_persistence.png", verbose=True):
    figure, axes = pyplot.subplots()
    axes.set(xlabel="Birth (diameter)", ylabel="Persistence (diameter)", title="Birth-Persistence Plot")
    axes.plot([0, 1], [0, 0], y0_fmt, alpha=DEFAULT_OPACITY)  # y = 0
    # For each dimension
    for k in range(min(dimensions), max(dimensions)):
        x = []  # r: row (birth)
        y = []  # r-c: row - column (persistence)
        for r, c in pivots_rc:
            # Check whether the dimension matches
            if dimensions[r] != k:
                continue
            # Add a point in that dimension's list
            x.append(diameters[r])
            y.append(diameters[c] - diameters[r])
        # Plot homologies at dimension k
        axes.scatter(x, y, color=DIMENSIONAL_COLOR_CODES[k], label=DIMENSIONAL_COLOR_LABELS[k], alpha=DEFAULT_OPACITY, s=DEFAULT_MARKER_SIZE)
        if verbose:
            print("(Birth diameter, persistence) for " + str(k) + "-homologies with persistence > 0, orderd by decreasing persistence: ")
            # Sort by persistence, and then swap order in each tuple to print (diameter, persistence)
            positive_yx = []
            for i in range(len(y)):
                if y[i] > 0:
                    positive_yx.append((y[i], x[i]))
            positive_yx.sort(reverse=True)
            positive_xy = [(x,y) for y,x in positive_yx]
            print(positive_xy)
    axes.legend()
    figure.savefig(output_filename)

def plot_barcodes( pivots_rc, diameters, dimensions, output_filename="barcodes.png", verbose=True):
    figure = pyplot.subplots()
    n_plots = len(range(min(dimensions), max(dimensions)))
    for k in range(min(dimensions), max(dimensions)):
        sp = pyplot.subplot( n_plots, 1, 1+k)
        sp.set(xlabel="Diameter", ylabel="Homology number", title="Barcodes for H"+str(k))
        start = []
        fin = []
        y = [0]
        for r,c in pivots_rc:
            if dimensions[r]!=k:
                continue
          
            start.append(diameters[r])
            fin.append(diameters[c])
            y.append(y[-1]+1)
        inds = np.flip(np.argsort( np.array(fin)-np.array(start) ))
        nn = 1
        for ii in inds:
            
            sp.plot( [start[ii], fin[ii]], [nn,nn], color=DIMENSIONAL_COLOR_CODES[k] )
            nn+=1
    
            
