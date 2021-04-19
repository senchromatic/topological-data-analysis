# Zhang, Pereira, LeDuc
# Code for doing matrix algebra in Z/2
import numpy as np


class Z2array(np.ndarray):
    """
    Subclasses NumPy array class to
    be able to do operations with matrices
    in Z2^n (n-dimensional matrices whose entries
    are in the field Z2), respecting proper
    Z2 arithmetic.

    Broadcastable operations (NumPy ufuncs) are
    overloaded. So, matrix arithmetic will work
    as expected. One caveat: Assignment operators
    such as += are *not* overloaded yet, so
    for example:

    arr1 = arr1 + arr2 will give correct results,
    but
    arr1 += arr2 will give incorrect results.

    Functions from np.linalg or SciPy will not respect
    the overloading. We must make our own versions
    (such as the z2rank function below).
    """

    def __new__(cls, input_array, info=None):
        """
        Python constructor. Runs before __init__
        method. See Python documentation.

        Will convert matrix to matrix with entries
        in Z2.
        """
        # Input array is an already formed ndarray instance.
        # We convert it to a Z2^n array here.
        obj = np.asarray(input_array.astype(np.int) % 2).view(cls)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Overloads NumPy ufunc handling for arrays
        of Z2array type. Ensures arithmetic operations
        are performed respecting the field Z2.

        See https://numpy.org/doc/stable/user/basics.
        subclassing.html#array-ufunc-for-ufuncs

        and

        https://numpy.org/doc/stable/reference/ufuncs.html

        for an explanation of NumPy ufuncs and how to
        overload them.
        """
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Z2array):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Z2array):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(Z2array, self).__array_ufunc__(ufunc, method,
                                                       *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        # This line below does all the actual work to ensure Z2^n arithmetic.
        # Everything else is just handling. See documentation for subclassing
        # NumPy ndarray class.
        results = tuple(((np.asarray(result).view(np.int) % 2).view(Z2array)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results


def z2array(*args, **kwargs):
    """
    Use this function to construct
    Z2array objects. Do not instantiate
    directly as Z2array(foo).

    This function wraps the NumPy
    constructor and is used in the same
    way, e.g. arr = z2array([[1, 0], [0,1]]).
    """
    return Z2array(np.array(*args, **kwargs))


def z2array_zeros(shape):
    return z2array(np.zeros(shape, dtype=int))


def z2array_ones(shape):
    return z2array(np.ones(shape, dtype=int))


def z2row_echelon(z2mat, copy_mat=True):
    """
    Return row echelon form of matrix,
    respecting Z2 arithmetic.

    Based on the recursive implementation
    here, https://math.stackexchange.com/
    questions/3073083/how-to-reduce
    -matrix-into-row-echelon-form-
    in-python/3073117,
    but modified to work in Z2 and to not
    use functional recursion, which gives
    a stack recursion limit for large matrices
    like those we encounter when calculation
    boundaries of our ASCs with real data.

    (The arithmetic below will work correctly
    because of the overloaded array class.
    The original will not work because +=/-=
    with array indexing is not yet overloaded
    in Z2array.)
    """
    if not (isinstance(z2mat, Z2array)
            and len(z2mat.shape) == 2):
        raise ValueError("Not supported!")

    if copy_mat:
        A = z2mat.copy()
    else:
        A = z2mat

    # A_ is a pointer to a subarray of A.
    # We can do the recursion here without
    # recursive function calls.
    A_ = A  # A reference, not a copy.
    r, c = A_.shape  # We'll recalculate these as we go.
    while (r > 0 and c > 0):

        # Find pivot
        col_all_zeros = True  # We use this bool flag for a check below.
        for i in range(len(A_)):
            if A_[i, 0] != 0:
                # We found a nonzero element, so column is not all zeros.
                # And i is the pivot row.
                col_all_zeros = False
                break
        # If the column is all zeros, move on to the next column.
        if col_all_zeros:
            A_ = A_[:, 1:]
            r, c = A_.shape
            continue

        # Do row exchange if needed.
        if i > 0:
            ith_row = A_[i].copy()
            A_[i] = A_[0]
            A_[0] = ith_row

        """
        Zero out column entries below pivot.

        This arithmetic is in Z2 if the input
        numpy array is of our Z2array type.
        Hence the type-checking at the top.
        """
        A_[1:] = A_[1:] - (A_[0] * A_[1:, 0:1])

        # Move on to next column.
        A_ = A_[1:, 1:]
        r, c = A_.shape

    return A


def z2rank(z2mat, nullspace=True):
    """
    Count number of nonzero rows in the row echelon
    form of the input Z2 matrix.

    If the matrix represents a linear transformation,
    this is the dimension of the image.

    If nullspace=True, also returns the dimension
    of the null space of the transformation
    (True by default).
    """
    rrz2mat = z2row_echelon(z2mat)
    # Cast as integer to sum across rows quickly.
    row_sums_as_int = np.array(rrz2mat, dtype=np.int).sum(axis=1)
    # Zero rows will be the rows with 0 sums in the integer version.
    rank = len(row_sums_as_int[row_sums_as_int > 0])
    if nullspace:
        # Dimension of domain is the number of columns.
        c = z2mat.shape[1]
        # Rank-nullity theorem.
        null_rank = c - rank
        return rank, null_rank
    return rank


def pivotcol_idxs_row_echelon(rez2mat):
    """
    Returns the indices of the pivot columns
    of the input matrix. Expects input matrix
    to already be in row echelon form.

    The column indices returned are 0-indexed.

    Relies on the fact that our GE function
    z2row_echelon only does row exchanges
    (no column exchanges).

    The corresponding columns in the original
    matrix (not row echelon) give a basis
    for the image of the transformation.
    This fact is used in z2_image_basis,
    which relies on this function.
    """
    r = rez2mat.shape[0]  # Number of rows.
    # List of pivot indices we will return.
    pivot_idxs = []
    # Row in which last pivot was found.
    # Set to -1 to avoid special handling for NoneType
    # when checking for the first pivot, while handling 0-indexing.
    #
    # Using i,j convention for row and column indices.
    last_pivot_i = -1
    # Iterate through columns.
    for j, col in enumerate(rez2mat.T):
        # Iterate through entries in column.
        lowest_nonzero_i = None
        for i, entry in enumerate(col):
            # Zero entries are at the bottom.
            if entry != 0:
                # Index of lowest nonzero entry in
                # this column.
                lowest_nonzero_i = i
        # If lowest nonzero entry in this column is
        # in a row with a higher index (below) that of the previous pivot
        # (or has an index higher than the initial dummy value of -1 for
        # the first pivot), then this is a pivot column.
        if lowest_nonzero_i is not None and (lowest_nonzero_i > last_pivot_i):
            # This is a pivot, so add column index to list.
            pivot_idxs.append(j)
            # Update our check.
            last_pivot_i = lowest_nonzero_i
        # If we have found the last possible pivot,
        # we can stop searching for more.
        if last_pivot_i == r - 1:
            break
    return pivot_idxs


def z2_image_basis(z2mat, idxs=False):
    """
    Returns a basis for the image space of the linear
    transformation represented by the input Z2 matrix
    (original matrix not in row echelon form).

    The return is a Z2 matrix whose columns are
    a basis for the image of the transformation.

    Optionally, also returns indices of the columns
    that give the basis if idxs=True (False by default).
    """
    rez2mat = z2row_echelon(z2mat, copy_mat=True)
    pivot_idxs = pivotcol_idxs_row_echelon(rez2mat)
    pivotcols = z2mat[:, pivot_idxs]
    if idxs:
        return pivotcols, pivot_idxs
    return pivotcols


def z2_null_basis(z2mat):
    """
    Returns a basis for the null space of the linear
    transformation represented by the input Z2 matrix
    (original matrix not in row echelon form).

    The return is a Z2 matrix whose columns are
    a basis for the null space of the transformation.

    If the rank of the null space is 0, returns the
    zero vector of the domain.
    """
    r, c = z2mat.shape
    # Do GE on the original matrix.
    rez2mat = z2row_echelon(z2mat, copy_mat=True)
    # Get indices of pivot columns.
    pivot_idxs = pivotcol_idxs_row_echelon(rez2mat)
    # If matrix has full column rank, the kernel is just
    # the zero vector of the domain.
    if len(pivot_idxs) == c:
        return z2array_zeros((c, 1))

    # Otherwise, we continue below. (Not optimized)

    # Take the transpose of the reduced matrix.
    trez2mat = rez2mat.T
    # Identity matrix.
    id = z2array(np.eye(c, dtype=np.int32))
    # Expand the transpose of the reduced matrix.
    exptrez2mat = z2array(np.concatenate([trez2mat, id], axis=1))
    # Do GE again on this expanded matrix.
    reexptrez2mat = z2row_echelon(exptrez2mat, copy_mat=False)
    # Get indices of the zero rows (nonexpanded part).
    lhs = reexptrez2mat[:, 0:r]
    # Cast as integer to sum across rows quickly.
    lhs_row_sums = np.array(lhs, dtype=np.int).sum(axis=1)
    # Zero rows will be the rows with 0 sums in the integer version.
    zero_row_idxs = np.where(lhs_row_sums == 0)[0].tolist()
    # Transpose of the expanded part of the zero rows is a basis
    # for the null space of the original matrix.
    rhs_zero_rows = reexptrez2mat[zero_row_idxs, r:]
    nullspace_basis = z2array(rhs_zero_rows.T)

    return nullspace_basis


if __name__ == "__main__":

    """
    Some quick tests. Will add better
    unit tests later.
    """
    print("\n\nChecking correct casting from plain old integers:")
    A = z2array([[1, 2, 3], [1, 2, 3]])
    A_ = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.float64)
    b_pass_A = ((A.astype(np.float64) - A_).sum() == 0
                and isinstance(A, Z2array))
    print(f"\nA:\n{A}")
    print(f"Passes? {b_pass_A}")

    B = z2array([[1, 0, 1], [-1, 1, -7]])
    B_ = np.array([[1, 0, 1], [1, 1, 1]], dtype=np.float64)
    b_pass_B = ((B.astype(np.float64) - B_).sum() == 0
                and isinstance(B, Z2array))
    print(f"\nB:\n{B}")
    print(f"Passes? {b_pass_B}")

    C = 3*A - 7*B
    C_ = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float64)
    b_pass_C = ((C.astype(np.float64) - C_).sum() == 0
                and isinstance(C, Z2array))
    print(f"\n3A - 7B:\n{C}")
    print(f"Passes? {b_pass_C}")

    print("\n\nChecking matrix multiplication:")
    A = z2array([[1, 1, 0, 0],
                 [1, 0, 1, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 1],
                 [0, 0, 0, 1]])
    b = z2array([[1, 0, 0, 1]]).T
    x = A @ b
    x_ = z2array([[1, 1, 0, 1, 1]]).T
    b_pass_x = ((x.astype(np.float64) - x_).sum() == 0
                and isinstance(x, Z2array))
    print(f"Matrix A:\n{A}")
    print(f"Vector b:\n{b}")
    print(f"\nA @ b = x:\n{x}")
    print(f"Passes? {b_pass_x}")

    print("\n\nChecking rank calculation:")
    """
    Note: Regular NumPy matrix_rank function
    will not work, because it is not overloaded
    (and not trivial to do so). Use the z2_rank
    function above.
    """
    rank_A = z2rank(A)
    rank_A_ = (3, 1)
    b_pass_rankA = (rank_A == rank_A_)
    print(f"\nMatrix A:\n{A}")
    print(f"\nRank of matrix A: {rank_A} (should be {rank_A_})")
    print(f"Passes? {b_pass_rankA}")
    B = z2array([[1, 0, 0, 0],
                 [1, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 1, 0, 1],
                 [0, 0, 0, 1]])
    rank_B = z2rank(B)
    rank_B_ = (4, 0)
    b_pass_rankB = (rank_B == rank_B_)
    print(f"\nMatrix B:\n{B}")
    print(f"Rank of matrix B: {rank_B} (should be {rank_B_})")
    print(f"Passes? {b_pass_rankB}")

    print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Visual check for row echelon forms:")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
    np.random.seed(42)
    M1 = Z2array(np.abs(np.random.rand(10, 10)).round().astype(np.int))
    M2 = Z2array(np.abs(np.random.rand(7, 7)).round().astype(np.int))
    M3 = Z2array(np.abs(np.random.rand(4, 3)).round().astype(np.int))
    M4 = Z2array(np.abs(np.random.rand(9, 2)).round().astype(np.int))
    M5 = Z2array(np.abs(np.random.rand(5, 8)).round().astype(np.int))
    rrM1 = z2row_echelon(M1)
    rrM2 = z2row_echelon(M2)
    rrM3 = z2row_echelon(M3)
    rrM4 = z2row_echelon(M4)
    rrM5 = z2row_echelon(M5)
    rankM1 = z2rank(M1)
    rankM2 = z2rank(M2)
    rankM3 = z2rank(M3)
    rankM4 = z2rank(M4)
    rankM5 = z2rank(M5)
    print(f"\n\nArray 1:\n{M1}")
    print(f"\nRow echelon form:\n{rrM1}")
    print(f"\nRank: {rankM1}")
    print(f"\n\nArray 2:\n{M2}")
    print(f"\nRow echelon form:\n{rrM2}")
    print(f"\nRank: {rankM2}")
    print(f"\n\nArray 3:\n{M3}")
    print(f"\nRow echelon form:\n{rrM3}")
    print(f"\nRank: {rankM3}")
    print(f"\n\nArray 4:\n{M4}")
    print(f"\nRow echelon form:\n{rrM4}")
    print(f"\nRank: {rankM4}")
    print(f"\n\nArray 5:\n{M5}")
    print(f"\nRow echelon form:\n{rrM5}")
    print(f"\nRank: {rankM5}")

    print("\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Visual check for bases:")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(f"\nTesting bases with Array 1:\n{M1}")
    print(f"\nRow echelon form:\n{rrM1}")
    print(f"\nRank: {rankM1}")
    basis_M1, pivotcol_idxs_M1 = z2_image_basis(M1, idxs=True)
    print(f"\nPivot columns of Array 1: {pivotcol_idxs_M1}")
    print(
        f"\nBasis for image of linear transformation represented by Array 1:\n{basis_M1}")
    null_basisM1 = z2_null_basis(M1)
    print(f"\nBasis for null space:\n{null_basisM1}")
    print("\n Multiplying vectors in the null space of Array 1 by Array 1:")
    print(M1 @ null_basisM1)

    print(f"\nTesting bases with Array 2:\n{M2}")
    print(f"\nRow echelon form:\n{rrM2}")
    print(f"\nRank: {rankM2}")
    basis_M2, pivotcol_idxs_M2 = z2_image_basis(M2, idxs=True)
    print(f"\nPivot columns of Array 2: {pivotcol_idxs_M2}")
    print(
        f"\nBasis for image of linear transformation represented by Array 2:\n{basis_M2}")
    null_basisM2 = z2_null_basis(M2)
    print(f"\nBasis for null space:\n{null_basisM2}")
    print("\n Multiplying vectors in the null space of Array 2 by Array 2:")
    print(M2 @ null_basisM2)

    print(f"\nTesting bases with Array 3:\n{M3}")
    print(f"\nRow echelon form:\n{rrM3}")
    print(f"\nRank: {rankM3}")
    basis_M3, pivotcol_idxs_M3 = z2_image_basis(M3, idxs=True)
    print(f"\nPivot columns of Array 3: {pivotcol_idxs_M3}")
    print(f"\nBasis for image of linear transformation represented by Array 3:\n{basis_M3}")
    null_basisM3 = z2_null_basis(M3)
    print(f"\nBasis for null space:\n{null_basisM3}")
    print("\n Multiplying vectors in the null space of Array 3 by Array 3:")
    print(M3 @ null_basisM3)

    print(f"\nTesting bases with Array 4:\n{M4}")
    print(f"\nRow echelon form:\n{rrM4}")
    print(f"\nRank: {rankM4}")
    basis_M4, pivotcol_idxs_M4 = z2_image_basis(M4, idxs=True)
    print(f"\nPivot columns of Array 4: {pivotcol_idxs_M4}")
    print(
        f"\nBasis for image of linear transformation represented by Array 4:\n{basis_M4}")
    null_basisM4 = z2_null_basis(M4)
    print(f"\nBasis for null space:\n{null_basisM4}")
    print("\n Multiplying vectors in the null space of Array 4 by Array 4:")
    print(M4 @ null_basisM4)

    print(f"\nTesting bases with Array 5:\n{M5}")
    print(f"\nRow echelon form:\n{rrM5}")
    print(f"\nRank: {rankM5}")
    basis_M5, pivotcol_idxs_M5 = z2_image_basis(M5, idxs=True)
    print(f"\nPivot columns of Array 5: {pivotcol_idxs_M5}")
    print(
        f"\nBasis for image of linear transformation represented by Array 5:\n{basis_M5}")
    null_basisM5 = z2_null_basis(M5)
    print(f"\nBasis for null space:\n{null_basisM5}")
    print("\n Multiplying vectors in the null space of Array 5 by Array 5:")
    print(M5 @ null_basisM5)

    # Takes a while to run because the matrix is big, but passes.
    print("\n\nRecursion test")
    big_matrix = Z2array(np.abs(np.random.rand(5000, 2000)).round().astype(np.int))
    z2row_echelon(big_matrix)
    # If we don't get a recursion error, we pass.
    print("Passed. (No recursion limit error).")
