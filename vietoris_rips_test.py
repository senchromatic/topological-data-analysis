# Unit tests for the Vietoris-Rips algorithm.
# Test 1
#  Points:
#   A(-1, 0), B(1, 0), C(0, 1)
#  Illustration:
#     .
#    . .
#
# Test 2
#  Points:
#   A(0, -2), B(0, -1), C (0, 1), D(0, 2)
#  Illustration:
#   .
#   .
#   
#   .
#   .

from abstract_simplicial_complex import Point, Simplex, ASC, vietoris_rips
from metrics import induced_metric
import numpy as np


def test_1():
    print("\nTest 1\n")
    points = set([
        Point(name='A', coordinates=np.array([-1,  0]), dist_metric=induced_metric),
        Point(name='B', coordinates=np.array([+1,  0]), dist_metric=induced_metric),
        Point(name='C', coordinates=np.array([ 0, +1]), dist_metric=induced_metric)
    ])
    
    for max_diam in [0.5, 1.5, 2.5]:
        rips_complex = vietoris_rips(points, len(points) - 1, max_diam)
        print("Distance threshold of %.1f:\n" % max_diam)
        print(rips_complex)
        print()

    print("\n")

def test_2():
    print("\nTest 2\n")
    points = set([
        Point(name='A', coordinates=np.array([ 0,  -2]), dist_metric=induced_metric),
        Point(name='B', coordinates=np.array([ 0,  -1]), dist_metric=induced_metric),
        Point(name='C', coordinates=np.array([ 0,  +1]), dist_metric=induced_metric),
        Point(name='D', coordinates=np.array([ 0,  +2]), dist_metric=induced_metric)
    ])
    
    for max_diam in range(5):
        rips_complex = vietoris_rips(points, len(points) - 1, max_diam)
        print("Distance threshold of %.1f:\n" % max_diam)
        print(rips_complex)
        print()

    print("\n")


if __name__ == '__main__':
    
    test_1()

    test_2()