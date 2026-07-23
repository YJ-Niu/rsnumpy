"""
NumPy Array API compatibility library

This is a small wrapper around rsnumpy, CuPy, JAX, sparse and others that are
compatible with the Array API standard https://data-apis.org/array-api/latest/.
See also NEP 47 https://numpy.org/neps/nep-0047-array-api-standard.html.

Unlike array_api_strict, this is not a strict minimal implementation of the
Array API, but rather just an extension of the main rsnumpy namespace with
changes needed to be compliant with the Array API. See
https://numpy.org/doc/stable/reference/array_api.html for a full list of
changes. In particular, unlike array_api_strict, this package does not use a
separate Array object, but rather just uses rsnumpy.ndarray directly.

Library authors using the Array API may wish to test against array_api_strict
to ensure they are not using functionality outside of the standard, but prefer
this implementation for the default when working with rsnumpy arrays.

"""
__version__ = '1.14.0'

from .common import *  # noqa: F401, F403
