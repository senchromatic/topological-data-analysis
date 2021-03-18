
# Computes non-oriented boundaries for p-chains

from abstract_simplicial_complex import Point, Simplex, Boundary, ASC
from enum import Enum

import numpy as np


# Organisms are organic entities.
organisms = Simplex()

def create_organisms():
  # Terrestrial animals:
  for name in ["Cow", "Rabbit", "Horse", "Dog"]:
    organisms.add_point(Point(name, np.array([1,1])))
  # Aquatic animals
  for name in ["Fish", "Dolphin", "Oyster"]:
    organisms.add_point(Point(name, np.array([0,1])))
  # Plants
  for name in ["Broccoli", "Fern", "Onion", "Apple"]:
    organisms.add_point(Point(name, np.array([1,0])))

def get_organism(name):
  return organisms.get_point_by_name(name)

def point_comparison_test(verbose=True):
  cow = get_organism("Cow")
  rabbit = get_organism("Rabbit")
  dolphin = get_organism("Dolphin")
  apple = get_organism("Apple")
  if verbose:
    print("Here is a land animal:", cow)
    print("Here is an aquatic animal:", dolphin)
    print("Here is a land plant:", apple)
    print("Cow same as rabbit?", cow == rabbit)
    print("Cow similar to rabbit?", cow.similar_to(rabbit))
    print("Rabbit similar to dolphin?", rabbit.similar_to(dolphin))
    print("Dolphin similar to apple?", dolphin.similar_to(apple))

# Computes the boundary for two triangles sharing a common edge (BC)
# Triangle 1 = ABC, Triangle 2 = BCD
def two_triangles_test(verbose=True):
  A = np.array([-1,  0])
  B = np.array([ 0, -1])
  C = np.array([ 0, +1])
  D = np.array([+1,  0])

  t1 = Simplex()
  t1.add_point(Point('A', A))
  t1.add_point(Point('B', B))
  t1.add_point(Point('C', C))

  t2 = Simplex()
  t2.add_point(Point('B', B))
  t2.add_point(Point('C', C))
  t2.add_point(Point('D', D))

  asc = ASC()
  asc.add_simplex(t1)
  asc.add_simplex(t2)

  if verbose:
    print(asc.compute_boundary())

# Returns a new simplicial complex (labelled A, from homework assignment),
# and prints debug statements if verbosity is enabled.
def create_asc_a(verbose=False):
  asc_a = ASC()

  if verbose:
    print("------------- Abstract Simplicial Complex A -------------")
  
  # Generate points in ASC
  for p in organisms.points:
    asc_a.add_simplex(Simplex({p}))
  if verbose:
    print("Number of 0-simplices:", len(asc_a.k_simplices(0)))
  
  # Generate line segments in ASC
  for p1 in organisms.points:
    for p2 in organisms.points - {p1}:
      if p1.similar_to(p2):
        new_simplex = Simplex({p1, p2})
        asc_a.add_simplex(new_simplex)
  if verbose:
    print("Number of 1-simplices:", len(asc_a.k_simplices(1)))
  
  # Generate triangles in ASC
  for p1 in organisms.points:
    for p2 in organisms.points - {p1}:
      for p3 in organisms.points - {p1, p2}:
        if p1.similar_to(p2) and p1.similar_to(p3) and p2.similar_to(p3):
          new_simplex = Simplex({p1, p2, p3})
          asc_a.add_simplex(new_simplex)
  
  if verbose:
    print("Number of 2-simplices:", len(asc_a.k_simplices(2)))
    print("\n")

    print(asc_a)
    print("\n\n")

  return asc_a

# Returns a new simplicial complex (labelled B, from homework assignment),
# and prints debug statements if verbosity is enabled.
def create_asc_b(verbose=False):
  asc_b = ASC()

  if verbose:
    print("------------- Abstract Simplicial Complex B -------------")
  
  # Generate points in ASC
  for p in organisms.points:
    asc_b.add_simplex(Simplex({p}))
  if verbose:
    print("Number of 0-simplices:", len(asc_b.k_simplices(0)))
  
  # Add all line segments to ASC
  for p1,p2 in {("Cow", "Rabbit"), ("Cow", "Fish"), ("Cow", "Oyster"), ("Cow", "Oyster"), ("Cow", "Broccoli"), ("Cow", "Onion"), ("Cow", "Apple"), ("Rabbit", "Fish"), ("Rabbit", "Oyster"), ("Rabbit", "Broccoli"), ("Rabbit", "Onion"), ("Rabbit", "Apple"), ("Fish", "Oyster"), ("Fish", "Broccoli"), ("Fish", "Onion"), ("Fish", "Apple"), ("Oyster", "Broccoli"), ("Oyster", "Onion"), ("Oyster", "Apple"), ("Broccoli", "Onion"), ("Broccoli", "Apple"), ("Onion", "Apple"), ("Horse", "Dog"), ("Horse", "Dolphin"), ("Horse", "Fern"), ("Dog", "Dolphin"), ("Dog", "Fern"), ("Dolphin", "Fern")}:
    asc_b.add_simplex(Simplex({get_organism(p1), get_organism(p2)}))
  if verbose:
    print("Number of 1-simplices:", len(asc_b.k_simplices(1)))
    
  # Add all triangles to ASC
  for p1,p2,p3 in {("Cow", "Broccoli", "Apple"), ("Cow", "Onion", "Apple"), ("Rabbit", "Broccoli", "Apple"), ("Rabbit", "Onion", "Apple"), ("Fish", "Broccoli", "Apple"), ("Fish", "Onion", "Apple"), ("Oyster", "Broccoli", "Apple"), ("Oyster", "Onion", "Apple")}:
    asc_b.add_simplex(Simplex({get_organism(p1), get_organism(p2), get_organism(p3)}))
  if verbose:
    print("Number of 2-simplices:", len(asc_b.k_simplices(2)))
  
  if verbose:
    print("\n")
    print(asc_b)
    print("\n\n")

  return asc_b


if __name__ == '__main__':

  # Point cloud
  create_organisms()

  # Unit tests
  # print("All organisms:", organisms)
  # point_comparison_test()
  # two_triangles_test()
  
  asc_a = create_asc_a(verbose=False)
  asc_b = create_asc_b(verbose=False)

  print("------------- Abstract Simplicial Complex A -------------")
  for k in range(2):
    print("\nBoundary over " + str(k+1) + "-simplices in entire abstract simplicial complex A:")
    print(asc_a.compute_boundary(k=k+1))
    print("\nBoundary for each " + str(k+1) + "-simplex, computed separately:")
    # asc_a.display_simplex_boundaries(k=k+1)
    asc_a.compute_boundary(k=k+1, matrix=True, verbose=True)
  print("\n")

  print("------------- Abstract Simplicial Complex B -------------")
  for k in range(2):
    print("\nBoundary over " + str(k+1) + "-simplices in entire abstract simplicial complex B:")
    print(asc_b.compute_boundary(k=k+1))
    print("\nBoundary for each " + str(k+1) + "-simplex, computed separately:")
    # asc_b.display_simplex_boundaries(k=k+1)
    asc_b.compute_boundary(k=k+1, matrix=True, verbose=True)
  print("\n")
