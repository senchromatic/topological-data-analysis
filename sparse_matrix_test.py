# Example is from Figure VII.4 on page 184 of Edelsbrunner, Harer - "Computational Topology"

from sparse_matrix import Z2SparseSquareMatrix

if __name__ == '__main__':
    triangle_matrix = Z2SparseSquareMatrix(8)
    for c,r in [(1, 0), (2, 0), (3, 0), (4, 1), (4, 2), (5, 2), (5, 3), (6, 1), (6, 3), (7, 4), (7, 5), (7, 6)]:
        triangle_matrix.flip_entry(c, r)
    triangle_matrix.column_reduction()
    
    print(triangle_matrix)
    print("Pivots:", triangle_matrix.find_pivots_cr())
    
    cycle_indices = triangle_matrix.find_all_cycle_indices()
    print("Cycle indices:", cycle_indices)
    print("Cycles:")
    for ci in cycle_indices:
        print(triangle_matrix.get_cycle_at_index(ci))