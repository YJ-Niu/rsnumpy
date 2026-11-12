"""Regression guard for the former known rsnumpy arithmetic gaps.

In 2026-10 this module tracked four scicore integration gaps, all failing
with ``TypeError: Unsupported operand type`` raised from ``ndarray.__add__``
when the Rust ``binary_op`` operand resolution fell through every branch:

* scicore ``test_scirs/test_signal.py::test_welch``
* scicore ``test_scirs/test_signal.py::test_stft``
* scicore ``test_scirs/test_interpolate.py::test_cubic_spline``
* scicore ``test_scirs/test_linalg.py::test_interp_decomp``

All four are fixed and verified 2026-11-12 against the current source:
the scicore tests pass (plus the ``interp_decomp_*`` variants).  The
regression net lives in scicore's ``test_scirs`` suite; what remains here
is the core-level guard for the stft COLA expression.
"""
import pytest

import rsnumpy as np


def test_stft_expression_is_correct_in_isolation():
    # Regression guard for the core part of the stft COLA check: for both real
    # and complex windows, ``win.real**2 + win.imag**2 == |win|**2``.
    for win in (
        np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        np.array([0.0, 0.5, 1.0, 0.5, 0.0]).astype("complex128"),
    ):
        got = win.real ** 2 + win.imag ** 2
        # flatten to a plain list of floats
        g = got.tolist()
        if isinstance(g, list) and g and isinstance(g[0], list):
            g = [float(v) for row in g for v in row]
        else:
            g = [float(v) for v in g]
        assert g == pytest.approx([0.0, 0.25, 1.0, 0.25, 0.0])
