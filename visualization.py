from matplotlib import pyplot

# pivots_rc: list of (row, column) pairs
# diameters: list of diameter corresponding to each index among values in pivots_rc 
# data_fmt: format string for the birth-death points
# xy_fmt: format string for the x=y line
def plot_birth_death(pivots_rc, diameters, data_fmt="b.", xy_fmt="r--", output_filename="birth_death.png"):
    x = [diameters[r] for r, c in pivots_rc]
    y = [diameters[c] for r, c in pivots_rc]
    figure, axes = pyplot.subplots()
    axes.plot(x, y, data_fmt)
    axes.set(xlabel="Birth (diameter)", ylabel="Death (diameter)", title="Birth-Death Plot")
    axes.plot([0, 1], xy_fmt)
    figure.savefig(output_filename)