"""
====================================
Linear algebra (:mod:`scipy.linalg`)
====================================

.. currentmodule:: scipy.linalg

.. toctree::
   :hidden:

   linalg.blas
   linalg.cython_blas
   linalg.cython_lapack
   linalg.interpolative
   linalg.lapack

Linear algebra functions.

.. eventually, we should replace the rsnumpy.linalg HTML link with just `numpy.linalg`

.. seealso::

   `numpy.linalg <https://www.numpy.org/devdocs/reference/routines.linalg.html>`__
   for more linear algebra functions. Note that identically named
   functions from `scipy.linalg` may offer more or slightly differing
   functionality.


Basics
======

.. autosummary::
   :toctree: generated/

   inv - Find the inverse of a square matrix
   solve - Solve a linear system of equations
   solve_banded - Solve a banded linear system
   solveh_banded - Solve a Hermitian or symmetric banded system
   solve_circulant - Solve a circulant system
   solve_triangular - Solve a triangular matrix
   solve_toeplitz - Solve a toeplitz matrix
   matmul_toeplitz - Multiply a Toeplitz matrix with an array.
   det - Find the determinant of a square matrix
   norm - Matrix and vector norm
   lstsq - Solve a linear least-squares problem
   pinv - Pseudo-inverse (Moore-Penrose) using lstsq
   pinvh - Pseudo-inverse of hermitian matrix
   khatri_rao - Khatri-Rao product of two arrays
   orthogonal_procrustes - Solve an orthogonal Procrustes problem
   matrix_balance - Balance matrix entries with a similarity transformation
   subspace_angles - Compute the subspace angles between two matrices
   bandwidth - Return the lower and upper bandwidth of an array
   issymmetric - Check if a square 2D array is symmetric
   ishermitian - Check if a square 2D array is Hermitian
   LinAlgError
   LinAlgWarning

Eigenvalue Problems
===================

.. autosummary::
   :toctree: generated/

   eig - Find the eigenvalues and eigenvectors of a square matrix
   eigvals - Find just the eigenvalues of a square matrix
   eigh - Find the e-vals and e-vectors of a Hermitian or symmetric matrix
   eigvalsh - Find just the eigenvalues of a Hermitian or symmetric matrix
   eig_banded - Find the eigenvalues and eigenvectors of a banded matrix
   eigvals_banded - Find just the eigenvalues of a banded matrix
   eigh_tridiagonal - Find the eigenvalues and eigenvectors of a tridiagonal matrix
   eigvalsh_tridiagonal - Find just the eigenvalues of a tridiagonal matrix

Decompositions
==============

.. autosummary::
   :toctree: generated/

   lu - LU decomposition of a matrix
   lu_factor - LU decomposition returning unordered matrix and pivots
   lu_solve - Solve Ax=b using back substitution with output of lu_factor
   svd - Singular value decomposition of a matrix
   svdvals - Singular values of a matrix
   diagsvd - Construct matrix of singular values from output of svd
   orth - Construct orthonormal basis for the range of A using svd
   null_space - Construct orthonormal basis for the null space of A using svd
   ldl - LDL.T decomposition of a Hermitian or a symmetric matrix.
   cholesky - Cholesky decomposition of a matrix
   cholesky_banded - Cholesky decomp. of a sym. or Hermitian banded matrix
   cho_factor - Cholesky decomposition for use in solving a linear system
   cho_solve - Solve previously factored linear system
   cho_solve_banded - Solve previously factored banded linear system
   polar - Compute the polar decomposition.
   qr - QR decomposition of a matrix
   qr_multiply - QR decomposition and multiplication by Q
   qr_update - Rank k QR update
   qr_delete - QR downdate on row or column deletion
   qr_insert - QR update on row or column insertion
   rq - RQ decomposition of a matrix
   qz - QZ decomposition of a pair of matrices
   ordqz - QZ decomposition of a pair of matrices with reordering
   schur - Schur decomposition of a matrix
   rsf2csf - Real to complex Schur form
   hessenberg - Hessenberg form of a matrix
   cdf2rdf - Complex diagonal form to real diagonal block form
   cossin - Cosine sine decomposition of a unitary or orthogonal matrix

.. seealso::

   `scipy.linalg.interpolative` -- Interpolative matrix decompositions


Matrix Functions
================

.. autosummary::
   :toctree: generated/

   expm - Matrix exponential
   logm - Matrix logarithm
   cosm - Matrix cosine
   sinm - Matrix sine
   tanm - Matrix tangent
   coshm - Matrix hyperbolic cosine
   sinhm - Matrix hyperbolic sine
   tanhm - Matrix hyperbolic tangent
   signm - Matrix sign
   sqrtm - Matrix square root
   funm - Evaluating an arbitrary matrix function
   expm_frechet - Frechet derivative of the matrix exponential
   expm_cond - Relative condition number of expm in the Frobenius norm
   fractional_matrix_power - Fractional matrix power


Matrix Equation Solvers
=======================

.. autosummary::
   :toctree: generated/

   solve_sylvester - Solve the Sylvester matrix equation
   solve_continuous_are - Solve the continuous-time algebraic Riccati equation
   solve_discrete_are - Solve the discrete-time algebraic Riccati equation
   solve_continuous_lyapunov - Solve the continuous-time Lyapunov equation
   solve_discrete_lyapunov - Solve the discrete-time Lyapunov equation


Sketches and Random Projections
===============================

.. autosummary::
   :toctree: generated/

   clarkson_woodruff_transform - Applies the Clarkson Woodruff Sketch (a.k.a CountMin Sketch)

Special Matrices
================

.. autosummary::
   :toctree: generated/

   block_diag - Construct a block diagonal matrix from submatrices
   circulant - Circulant matrix
   companion - Companion matrix
   convolution_matrix - Convolution matrix
   dft - Discrete Fourier transform matrix
   fiedler - Fiedler matrix
   fiedler_companion - Fiedler companion matrix
   hadamard - Hadamard matrix of order 2**n
   hankel - Hankel matrix
   helmert - Helmert matrix
   hilbert - Hilbert matrix
   invhilbert - Inverse Hilbert matrix
   leslie - Leslie matrix
   pascal - Pascal matrix
   invpascal - Inverse Pascal matrix
   toeplitz - Toeplitz matrix

Low-level routines
==================

.. autosummary::
   :toctree: generated/

   get_blas_funcs
   get_lapack_funcs
   find_best_blas_type

.. seealso::

   `scipy.linalg.blas` -- Low-level BLAS functions

   `scipy.linalg.lapack` -- Low-level LAPACK functions

   `scipy.linalg.cython_blas` -- Low-level BLAS functions for Cython

   `scipy.linalg.cython_lapack` -- Low-level LAPACK functions for Cython

"""  # noqa: E501


# Import commonly used functions
from scipy.linalg._decomp_cholesky import cholesky, cho_factor, cho_solve
from scipy.linalg._misc import LinAlgError, LinAlgWarning
from rsnumpy.linalg import issymmetric


def get_blas_funcs(names, arrays=(), dtype=None):
    """Return available BLAS function objects from names."""
    if isinstance(names, str):
        names = (names,)
    return names


def get_lapack_funcs(names, arrays=(), dtype=None):
    """Return available LAPACK function objects from names."""
    if isinstance(names, str):
        names = (names,)
    return names


def find_best_blas_type(arrays=(), dtype=None):
    """Find best BLAS type for given arrays and dtype."""
    return 'd', None, None


def norm(a, ord=None, axis=None, keepdims=False):
    """Matrix or vector norm."""
    import rsnumpy as np
    return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)


def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, check_finite=True):
    """Solve the equation a x = b for x, assuming a is a triangular matrix."""
    import rsnumpy as np
    return np.linalg.solve(a, b)


def inv(a):
    """Compute the inverse of a matrix."""
    import rsnumpy as np
    return np.linalg.inv(a)


def det(a):
    """Compute the determinant of a matrix."""
    import rsnumpy as np
    return np.linalg.det(a)


# Deprecated namespaces, to be removed in v2.0.0

__all__ = ['cholesky', 'cho_factor', 'cho_solve', 'LinAlgError', 'LinAlgWarning', 'issymmetric',
           'get_blas_funcs', 'get_lapack_funcs', 'find_best_blas_type',
           'norm', 'solve_triangular', 'inv', 'det']
