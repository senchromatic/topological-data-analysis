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
    Came up with a possible counterexample.
    Will probably just have to write a
    Gaussian elimination function for Z2array,
    which won't be fast, but it will have to do.
    """
    # LU factorization with pivoting.
    P, L, U = linalg.lu(z2mat)
    # Convert back to our Z2 NumPy type.
    U_z2 = Z2array(U)
    L_z2 = Z2array(L)
    P_z2 = Z2array(P)
    print("A")
    print(z2mat)
    print("L")
    print(L)
    print("U")
    print(U)
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
    # Counter example
    z2rank(np.abs(np.random.rand(10,10)).round().astype(np.int))
