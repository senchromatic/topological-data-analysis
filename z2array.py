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

    Copied from the recursive implementation
    here, https://math.stackexchange.com/
    questions/3073083/how-to-reduce
    -matrix-into-row-echelon-form-
    in-python/3073117,
    but slightly modified for Z2.

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

    # Check if we're done.
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # Find pivot
    for i in range(len(A)):
        if A[i, 0] != 0:
            break
    else:
        B = z2row_echelon(A[:, 1:],
                          copy_mat=False)
        return np.hstack([A[:, :1], B])

    # Do row exchange if needed.
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    """
    Zero out column entries below pivot.

    This arithmetic is in Z2 if the input
    numpy array is of our Z2array type.
    Hence the type-checking at the top.
    """
    A[1:] = A[1:] - (A[0] * A[1:, 0:1])

    # Move on to next column.
    B = z2row_echelon(A[1:, 1:],
                      copy_mat=False)

    return np.vstack([A[:1], np.hstack([A[1:, :1], B])])


def z2rank(z2mat):
    """
    Count number of nonzero rows
    in row echelon form.
    """
    rrz2mat = z2row_echelon(z2mat)
    # Cast as integer to sum across rows quickly.
    row_sums_as_int = np.array(rrz2mat, dtype=np.int).sum(axis=1)
    # 0 rows will be the rows with 0 sums in the integer version.
    rank = len(row_sums_as_int[row_sums_as_int > 0])
    return rank


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
    rank_A_ = 3
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
    rank_B_ = 4
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
