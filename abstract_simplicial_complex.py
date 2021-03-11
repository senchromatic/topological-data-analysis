# Defines a Point, Simplex, and ASC (abstract simplicial complex).

import metrics

from copy import deepcopy
from functools import total_ordering
from sets import powerset


# A point must have a unique identifier or a (partial) coordinate vector as representation.
# If no metric is provided, then the discrete metric will be used by default.
@total_ordering
class Point:
  
  def __init__(self, name=None, coordinates=None, dist_metric=None, show_coords=False):
    if name is None and coordinates is None:
      raise ValueError
    self.name = name
    self.coordinates = coordinates
    self.dist_metric = dist_metric if dist_metric else metrics.discrete_metric
    self.show_coords = show_coords
  
  # Formats:
  #   [name]([coordinates])  -- if both exist
  #   [name]                 -- only name
  #   (coordinates)          -- only coordinates
  def __str__(self):
    s = ""
    if self.name:
      s += self.name
    if self.show_coords and self.coordinates is not None:
      s += '(' + ' '.join([str(x) for x in self.coordinates]) + ')'
    return s
  
  def distance_to(self, other_point):
    return self.dist_metric(self.coordinates, other_point.coordinates)
  
  # Compare (approximately) by coordinates.
  def similar_to(self, other_point):
    return self.distance_to(other_point) <= metrics.DEFAULT_EPS
  
  # If both name and coordinates are provided, and the metrics are equal, these two points are indistinguishable.
  # Otherwise, compare by the arguments provided.
  def __eq__(self, other):
    have_names = not(self.name is None or other.name is None)
    metric_comparable = not(self.coordinates is None or other.coordinates is None) and self.dist_metric == other.dist_metric
    same_names = self.name == other.name
    close_enough = self.similar_to(other)
    if have_names and metric_comparable:
      return same_names and close_enough
    if have_names:
      return same_names
    if metric_comparable:
      return close_enough
    raise ValueError("Unable to compare", self, "and", other)
  
  def __lt__(self, other):
    if not(self.name is None or other.name is None):
      return self.name < other.name
    return metrics.norm(self.coordinates) < metrics.norm(other.coordinates)
  
  def __hash__(self):
    return hash(self.name)

# A k-simplex consists of (d+1) points.
@total_ordering
class Simplex:
  def __init__(self, points=None):
    self.points = points if points else set()

  def dimension(self):
    return len(self.points) - 1
  
  def __str__(self, sep=', '):
    return '{' + sep.join(sorted(map(str, self.points))) + '}'
  
  def add_point(self, new_point):
    self.points.add(new_point)
  
  def add_points(self, other_simplex):
    self.points = self.points | other_simplex.points

  def remove_point(self, existing_point):
    self.points.remove(existing_point)
  
  def get_point_by_name(self, name):
    for p in self.points:
      if p.name == name:
        return p
    return None
  
  def __lt__(self, other):
    return sorted(list(self.points)) < sorted(list(other.points))
  
  def __eq__(self, other):
    return self.points == other.points
  
  # Compute boundary on a simplex.
  def compute_boundary(self, verbose=False):
    bdy = Boundary() 
    if verbose:
      print("simplex: ", self)
    for p in self.points:
      face = deepcopy(self)
      face.remove_point(p)
      if verbose:
        print("face: ", face)
      bdy.xor(face)
    if verbose:
      print()
    return bdy
  
  def __hash__(self):
    return hash('|'.join(map(str, sorted(self.points))))

# A boundary is a p-chain with coefficients over Z_2, from a free abelian group over simplices.
class Boundary():

  def __init__(self):
    self.simplices = set()

  def __str__(self, sep=' + '):
    return sep.join(sorted(map(str, self.simplices)))

  def xor(self, sim):
    if sim in self.simplices:
      self.simplices.remove(sim)
    else:
      self.simplices.add(sim)

# An abstract simplicial complex is a collection of simplices closed under the subset operation.
class ASC:
  def __init__(self, simplices=None):
    self.simplices = simplices if simplices else set()
  
  # We define the dimension of an ASC as the maximum dimension among its simplices, if -1 if it's empty.
  def highest_dimension(self):
    return max([sim.dimension() for sim in self.simplices]) if self.simplices else -1
  
  def all_dimensions(self):
    return set([sim.dimension() for sim in self.simplices])
  
  # All simplices of dimesion k, for a given k
  def k_simplices(self, k):
    return set(filter(lambda sim : sim.dimension() == k, self.simplices))
  
  # String representation of simplices sorted by (non-decreasing) dimension
  def __str__(self, sep="\n\n"):
    return '{' + sep.join(['; '.join(map(str, sorted(self.k_simplices(k)))) for k in self.all_dimensions()]) + '}'
  
  # Check closure by taking all subsets.
  def is_valid(self):
    for subset in powerset(self.simplices):
      sim = Simplex(subset)
      if sim not in self.simplices:
        return False
    return True
  
  def add_simplex(self, sim):
    self.simplices.add(sim)

  # Computes the boundary for all simplices of dimension k.
  # If k is unspecified, the ASC's dimension will be used by default.
  # Algorithm: For each simplex, take one point at a time
  #            and delete it to form a face, then add this face to the boundary.
  # Let S be the collection of simplices.
  # Time complexity: O(k|S|)
  # Memory usage: O(k|S|)
  def compute_boundary(self, k=None, verbose=False):
    if k is None:
      k = self.highest_dimension()
    
    sims = self.k_simplices(k)
    bdy = Boundary()
    for sim in sims:
      if verbose:
        print("simplex: ", sim)
      for p in sim.points:
        face = deepcopy(sim)
        face.remove_point(p)
        if verbose:
          print("face: ", face)
        bdy.xor(face)
      if verbose:
        print()
    return bdy
  
  # Computes and prints the boundary for each simplex separately
  def display_simplex_boundaries(self, k=None, verbose=False):
    for sim in self.k_simplices(k):
      print("Boundary of", sim, ":", sim.compute_boundary(verbose=verbose))