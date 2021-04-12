# Zhang, Pereira, LeDuc
# A library for additional set operations beyond the Python standard library.

from itertools import chain, combinations

# Source: https://stackoverflow.com/a/1482316
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

