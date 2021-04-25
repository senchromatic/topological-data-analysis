# Functions for plotting persistence as a function of birth times (diameter cutoffs).

from collections import defaultdict
from matplotlib import pyplot

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
    axes.plot([0, 1], xy_fmt)  # y = x
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
    axes.plot([0, 1], [0, 0], y0_fmt)  # y = 0
    axes.legend()
    figure.savefig(output_filename)
