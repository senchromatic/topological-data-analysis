import numpy as np
import scipy.linalg as linalg


class Z2array(np.ndarray):

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array.astype(np.int) % 2).view(cls)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
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

        results = tuple(((np.asarray(result).view(np.int) % 2).view(Z2array)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results


def z2array(*args, **kwargs):
    return Z2array(np.array(*args, **kwargs))


def z2rank(z2mat):
    """
    Want to use optimized NumPy/SciPy C/Fortran
    routines with Python bindings rather than
    write a slow pure-Python Gaussian elimination
    function. That's the reason for this function:
    making this computation *fast* in wall-clock
    time, even for big matrices.

    How?
    Exploit theorem that for real m x n matrix A,
    PA = LU (P an m x m elementary matrix,
    L n x n lower triangular, U m x n upper triangular)
    and rank(A) = rank(U) = # nonzero rows in U.

    Wrinkle: The SciPy LU algorithm used is for real
    matrices and our matrices are Z2.

    Solution:
    We convert A from Z2 to reals to exploit the SciPy
    algorithm for the factorization, then convert U to Z2.
    The rank of U *as a Z2 matrix* will be the rank of A
    *as a Z2 matrix*.

    Why?
    Because P just changes the order of rows of A, and for
    real matrices X, Y, Z and modulo 2,
    X = YZ => (X mod 2) = (Y mod2) (Z mod2) mod2,
    because a+b mod2 = (a mod2) + (b mod2) mod2 and
    ab mod2 = (a mod2)(b mod2) = (a mod2)(b mod2) mod2.

    So, in our case, (PA mod2) = P(A mod2) = PA (since
    A was a Z2 matrix when we started),
    so PA = (L mod2) (U mod2) mod2. Multiplying by P does
    not change the rank of A. Hence, rank PA = rank A.

    [Finish checking if this is always true and fill gap.]
    And rank A = rank (U mod 2) = # nonzero rows in (U mod2)
    follows from back-subsitution step in the LU algorithm for
    solving the system:

    Ax = b with PA = LU, so LUx = Pb. Setting y = Ux,

    1. Solve Ly = Pb for y.
    2. Solve Ux = y for x.

    Back to implementation:
    After U is in Z2, count number of zero rows in U to get
    rank of A. For this, we convert the data type temporarily
    back to regular integers. Then we just sum over each row
    (since our previous conversion to Z2Array made all elements
    nonnegative) and check how many rows of U are nonzero.

    We do it this way to play nicely with our overloading
    of some NumPy methods in the Z2Array class.
    """
    # LU factorization with pivoting.
    P, L, U = linalg.lu(z2mat)
    # Convert back to our Z2 NumPy type.
    U_z2 = Z2array(U)
    L_z2 = Z2array(L)
    P_z2 = Z2array(P)
    print("Original")
    print(z2mat)
    print("PA")
    print(P_z2 @ Z2array(z2mat))
    print("(L mod2)(U mod2)")
    print((L.astype(np.int) % 2) @ (U.astype(np.int) % 2))
    # Count nonzero rows.
    row_sums_as_int = np.array(U_z2, dtype=np.int).sum(axis=1)
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
    #z2rank(np.random.rand(5,5) * 27)
