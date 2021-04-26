# This module has two classes.
# Each Z2SparseSquareMatrix is a collection of Z2SparseColumns.

from collections import defaultdict

# Constant for undefined lowest positive entry number
UNDEFINED_LOW_FOR_EMPTY_COLUMN = -1

# A column of coefficients in Z_2
# entries: set of positive indices 
class Z2SparseColumn:
    def __init__(self):
        self.entries = set()
    
    def __str__(self, sep=", "):
        return sep.join(map(str, sorted(self.entries))) if self.entries else ""
    
    # r: row number
    def flip_bit(self, r):
        if r in self.entries:
            self.entries.remove(r)
        else:
            self.entries.add(r)
    
    # Returns the row number of the lowest positive entry, else -1 if undefined (for empty column)
    def get_low(self):
        return max(self.entries) if self.entries else UNDEFINED_LOW_FOR_EMPTY_COLUMN
    
    def add(self, other_column):
        # Equivalent to xor (set_difference is equally efficient)
        # self.entries = self.entries ^ other_column.entries
        for r in other_column.entries:
            self.flip_bit(r)

# A sparse, square matrix (MxM) with coefficients in Z_2
# columns: A dictionary from column number to the Z2SparseColumn
# M: number of rows and columns
# is_reduced: whether column_reduction has been called
class Z2SparseSquareMatrix:
    def __init__(self, M):
        self.M = M
        self.columns = [Z2SparseColumn() for _ in range(M)]
        self.is_reduced = False
    
    # Print an ordered list of entries in each column
    def __str__(self, sep='\n'):
        s = ""
        for c,col in enumerate(self.columns):
            s += "Column " + str(c) + ": " + str(col) + sep
        return s
    
    # c: column number
    # r: row number
    def flip_entry(self, c, r):
        self.columns[c].flip_bit(r)
    
    # See pseudocode described in page 182 of Edelsbrunner, Harer - "Computational Topology"
    def column_reduction(self, verbose=False):
        pivot_cache = defaultdict(list)  # Dictionary from get_low to matching columns with index < cj
        for cj in range(self.M):
            if verbose:
                print("Reducing column", cj, "of", self.M)
            # Skip empty column
            if self.columns[cj].get_low() == UNDEFINED_LOW_FOR_EMPTY_COLUMN:
                continue
            added_columns = False  # True if we find an entry in cj can be zeroed out by a pivot (in ci) to the left
            first_search = True
            while first_search or added_columns:
                first_search = False
                added_columns = False
                cj_low = self.columns[cj].get_low()  # Store the get_low value for column cj for reuse
                # This loop will execute at most once, since pivots clear out the entire row to the left
                for ci in pivot_cache[cj_low]:
                    if cj_low == UNDEFINED_LOW_FOR_EMPTY_COLUMN:
                        continue
                    self.columns[cj].add(self.columns[ci])
                    added_columns = True
                    cj_low = self.columns[cj].get_low()
            pivot_cache[self.columns[cj].get_low()].append(cj)
        self.is_reduced = True
    
    # Returns an ordered list of (column, row) indices for pivots positions
    # (should only be called on reduced matrix)
    def find_pivots_cr(self):
        assert(self.is_reduced)
        pivots = []
        for c,col in enumerate(self.columns):
            # Ignore empty columns
            if not col.entries:
                continue
            pivots.append((c, self.columns[c].get_low()))
        return pivots
    
    # Returns an ordered list of (row, column) indices for pivots positions
    # (should only be called on reduced matrix)
    def find_pivots_rc(self):
        return sorted([(r, c) for c, r in self.find_pivots_cr()])