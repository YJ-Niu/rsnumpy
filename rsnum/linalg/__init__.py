"""rsnum.linalg - Linear algebra module."""

import rsnum._core as _core

def _ensure(x):
    if isinstance(x, (list, tuple)):
        return _core.array(x)
    return x

def dot(a, b):
    return _core.linalg.dot(_ensure(a), _ensure(b))

def matmul(a, b):
    return _core.linalg.matmul(_ensure(a), _ensure(b))

def inv(a):
    return _core.linalg.inv(_ensure(a))

def det(a):
    return _core.linalg.det(_ensure(a))

def norm(x, ord=None, axis=None):
    return _core.linalg.norm(_ensure(x), ord, axis)

def solve(a, b):
    return _core.linalg.solve(_ensure(a), _ensure(b))

def eig(a):
    return _core.linalg.eig(_ensure(a))

def eigvals(a):
    return _core.linalg.eigvals(_ensure(a))

def svd(a):
    return _core.linalg.svd(_ensure(a))

def qr(a):
    return _core.linalg.qr(_ensure(a))

def cholesky(a):
    return _core.linalg.cholesky(_ensure(a))

def matrix_power(a, n):
    return _core.linalg.matrix_power(_ensure(a), n)

def pinv(a):
    return _core.linalg.pinv(_ensure(a))

def trace(a):
    return _core.linalg.trace(_ensure(a))

def diagonal(a):
    return _core.linalg.diagonal(_ensure(a))

def svdvals(a):
    return _core.linalg.svdvals(_ensure(a))

def eigh(a):
    return _core.linalg.eigh(_ensure(a))

def solve_banded(lower, upper, ab, b):
    return _core.linalg.solve_banded(lower, upper, _ensure(ab), _ensure(b))