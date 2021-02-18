# This example program creates two ASC (abstract simplicial complex) labeled A and B,
# and then outputs their simplices in order of increasing dimension.
# 
# ASC A is represented as a biological taxonomy on living things with two key characteristics in this sample -
# Coordinate 1 (biosphere): 0 for hydosphere (water), 1 biosphere (land)
# Coordinate 2 (kingdom): 0 for plantae (plants), 1 for animalia (animals)
# Based on proximity in terms of coordinate 1 and 2, simplices of dimension up to k=2 are automatically generated.
# The other dimensions used to distinguish each "species" are encoded as a string (sequence of characters), its name.
# 
# ASC B is created by directly constructing the collection of simplices.

from abstract_simplicial_complex import Point, Simplex, ASC
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

def point_comparison_test():
  cow = get_organism("Cow")
  rabbit = get_organism("Rabbit")
  dolphin = get_organism("Dolphin")
  apple = get_organism("Apple")
  print("Here is a land animal:", cow)
  print("Here is an aquatic animal:", dolphin)
  print("Here is a land plant:", apple)
  print("Is the cow the same as a rabbit?", cow == rabbit)
  print("Is the cow similar to a rabbit?", cow.similar_to(rabbit))
  print("Is the rabbit similar to a dolphin?", rabbit.similar_to(dolphin))
  print("Is the dolphin similar to an apple?", dolphin.similar_to(apple))

# Create all k-simplices for a given metric and distance threshold
def cech_complex(points, k):
  pass  # TODO

# Create all k-simplices for a given metric and distance threshold
def vietoris_rips_complex(points, k):
  pass  # TODO

if __name__ == '__main__':
  # Point cloud
  create_organisms()
  # print("All organisms:", organisms)
  # point_comparison_test()
  
  
  asc_a = ASC()
  print("------------- Abstract Simplicial Complex A -------------")
  
  # Generate points in ASC
  for p in organisms.points:
    asc_a.add_simplex(Simplex({p}))
  print("Number of 0-simplices:", len(asc_a.k_simplices(0)))
  
  # Generate line segments in ASC
  for p1 in organisms.points:
    for p2 in organisms.points - {p1}:
      if p1.similar_to(p2):
        new_simplex = Simplex({p1, p2})
        asc_a.add_simplex(new_simplex)
  print("Number of 1-simplices:", len(asc_a.k_simplices(1)))
  
  # Generate triangles in ASC
  for p1 in organisms.points:
    for p2 in organisms.points - {p1}:
      for p3 in organisms.points - {p1, p2}:
        if p1.similar_to(p2) and p1.similar_to(p3) and p2.similar_to(p3):
          new_simplex = Simplex({p1, p2, p3})
          asc_a.add_simplex(new_simplex)
  print("Number of 2-simplices:", len(asc_a.k_simplices(2)))
  print("\n")
  
  print(asc_a)
  print("\n\n")
  
  asc_b = ASC()
  print("------------- Abstract Simplicial Complex B -------------")
  
  # Generate points in ASC
  for p in organisms.points:
    asc_b.add_simplex(Simplex({p}))
  print("Number of 0-simplices:", len(asc_b.k_simplices(0)))
  
  # Add all line segments to ASC
  for p1,p2 in {("Cow", "Rabbit"), ("Cow", "Fish"), ("Cow", "Oyster"), ("Cow", "Oyster"), ("Cow", "Broccoli"), ("Cow", "Onion"), ("Cow", "Apple"), ("Rabbit", "Fish"), ("Rabbit", "Oyster"), ("Rabbit", "Broccoli"), ("Rabbit", "Onion"), ("Rabbit", "Apple"), ("Fish", "Oyster"), ("Fish", "Broccoli"), ("Fish", "Onion"), ("Fish", "Apple"), ("Oyster", "Broccoli"), ("Oyster", "Onion"), ("Oyster", "Apple"), ("Broccoli", "Onion"), ("Broccoli", "Apple"), ("Onion", "Apple"), ("Horse", "Dog"), ("Horse", "Dolphin"), ("Horse", "Fern"), ("Dog", "Dolphin"), ("Dog", "Fern"), ("Dolphin", "Fern")}:
    asc_b.add_simplex(Simplex({get_organism(p1), get_organism(p2)}))
  print("Number of 1-simplices:", len(asc_b.k_simplices(1)))
    
  # Add all triangles to ASC
  for p1,p2,p3 in {("Cow", "Broccoli", "Apple"), ("Cow", "Onion", "Apple"), ("Rabbit", "Broccoli", "Apple"), ("Rabbit", "Onion", "Apple"), ("Fish", "Broccoli", "Apple"), ("Fish", "Onion", "Apple"), ("Oyster", "Broccoli", "Apple"), ("Oyster", "Onion", "Apple")}:
    asc_b.add_simplex(Simplex({get_organism(p1), get_organism(p2), get_organism(p3)}))
  print("Number of 2-simplices:", len(asc_b.k_simplices(2)))
  
  print("\n")
  print(asc_b)
  print("\n\n")

