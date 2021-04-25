# Functions for plotting persistence as a function of birth times (diameter cutoffs).

from collections import defaultdict
from matplotlib import pyplot

# One color per homology dimension (0: blue, 1: green, 2: red, 3: yellow)
DIMENSIONAL_COLORS = "bgry"

# pivots_rc: list of (row, column) pairs
# diameters: list of diameter corresponding to each index among values in pivots_rc 
# dimensions: list of simplex dimension corresponding to each index among values in pivots_rc 
# data_marker: marker style for the birth-death points
# xy_fmt: format string for the x=y line
def plot_birth_death(pivots_rc, diameters, dimensions, data_marker=".", xy_fmt="r--", output_filename="birth_death.png", verbose=True):
    x = defaultdict(list)
    y = defaultdict(list)
    # For each dimension
    for k in range(min(dimensions), max(dimensions) + 1):
        # r: row (birth)
        # c: column (death)
        for r, c in pivots_rc:
            # Check whether the dimension matches
            if dimensions[r] != k:
                continue
            # Add a point in that dimension's list
            x[k].append(diameters[r])
            y[k].append(diameters[c])
        if verbose:
            print("(Birth diameter, death diameter) for " + str(k) + "-homologies, ordered by increasing birth: ")
            print(list(zip(x[k],y[k])))
    figure, axes = pyplot.subplots()
    # Plot homologies at each dimension
    axes.set(xlabel="Birth (diameter)", ylabel="Death (diameter)", title="Birth-Death Plot")
    for k in range(min(dimensions), max(dimensions) + 1):    
        axes.plot(x[k], y[k], data_marker + DIMENSIONAL_COLORS[k])
    axes.plot([0, 1], xy_fmt)
    figure.savefig(output_filename)

# Persistence is defined as death - birth diameter
# pivots_rc: list of (row, column) pairs
# diameters: list of diameter corresponding to each index among values in pivots_rc 
# dimensions: list of simplex dimension corresponding to each index among values in pivots_rc 
# data_marker: marker style for the birth-death points
# xy_fmt: format string for the x=y line
def plot_birth_persistence(pivots_rc, diameters, dimensions, data_marker=".", output_filename="birth_persistence.png", verbose=True):
    x = defaultdict(list)
    y = defaultdict(list)
    # For each dimension
    for k in range(min(dimensions), max(dimensions) + 1):
        # r: row (birth)
        # c: column (death)
        for r, c in pivots_rc:
            # Check whether the dimension matches
            if dimensions[r] != k:
                continue
            # Add a point in that dimension's list
            x[k].append(diameters[r])
            y[k].append(diameters[c] - diameters[r])
        if verbose:
            print("(Birth diameter, persistence) for " + str(k) + "-homologies with persistence > 0, orderd by decreasing persistence: ")
            positive_yx = []
            for i in range(len(y[k])):
                if y[k][i] > 0:
                    positive_yx.append((y[k][i], x[k][i]))
            positive_yx.sort(reverse=True)
            positive_xy = [(x,y) for y,x in positive_yx]
            print(positive_xy)
    figure, axes = pyplot.subplots()
    axes.set(xlabel="Birth (diameter)", ylabel="Persistence (diameter)", title="Birth-Persistence Plot")
    # Plot homologies at each dimension
    for k in range(min(dimensions), max(dimensions) + 1):
        axes.plot(x[k], y[k], data_marker + DIMENSIONAL_COLORS[k])
    figure.savefig(output_filename)
