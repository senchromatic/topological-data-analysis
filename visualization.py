from matplotlib import pyplot

# pivots_rc: list of (row, column) pairs
# diameters: list of diameter corresponding to each index among values in pivots_rc 
# data_fmt: format string for the birth-death points
# xy_fmt: format string for the x=y line
def plot_birth_death(pivots_rc, diameters, data_fmt="b.", xy_fmt="r--", output_filename="birth_death.png", verbose=True):
    x = [diameters[r] for r, c in pivots_rc]
    y = [diameters[c] for r, c in pivots_rc]
    if verbose:
        print("(Birth diameter, death diameter) points ordered by increasing birth: ")
        print(list(zip(x,y)))
    figure, axes = pyplot.subplots()
    axes.plot(x, y, data_fmt)
    axes.set(xlabel="Birth (diameter)", ylabel="Death (diameter)", title="Birth-Death Plot")
    axes.plot([0, 1], xy_fmt)
    figure.savefig(output_filename)

# Persistence is defined as death - birth diameter
# pivots_rc: list of (row, column) pairs
# diameters: list of diameter corresponding to each index among values in pivots_rc 
# data_fmt: format string for the birth-persistence points
# xy_fmt: format string for the x=y line
def plot_birth_persistence(pivots_rc, diameters, data_fmt="b.", output_filename="birth_persistence.png", verbose=True):
    x = [diameters[r] for r, c in pivots_rc]
    y = [yi - xi for xi,yi in zip(x, [diameters[c] for r, c in pivots_rc])]
    if verbose:
        print("(Birth diameter, persistence) points with persistence > 0, orderd by decreasing persistence: ")
        positive_yx = []
        for i in range(len(x)):
            if y[i] > 0:
                positive_yx.append((y[i], x[i]))
        positive_yx.sort(reverse=True)
        positive_xy = [(x,y) for y,x in positive_yx]
        print(positive_xy)
    figure, axes = pyplot.subplots()
    axes.plot(x, y, data_fmt)
    axes.set(xlabel="Birth (diameter)", ylabel="Persistence (diameter)", title="Birth-Persistence Plot")
    figure.savefig(output_filename)
