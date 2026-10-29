"""Characterization of the known rsnumpy arithmetic gaps.

These are the four scicore integration tests that currently fail with
``TypeError: Unsupported operand type`` raised from ``ndarray.__add__``:

* scicore ``test_scirs/test_signal.py::test_welch``
* scicore ``test_scirs/test_signal.py::test_stft``
* scicore ``test_scirs/test_interpolate.py::test_cubic_spline``
* scicore ``test_scirs/test_linalg.py::test_interp_decomp_linear_operator``

Root-cause investigation (this repo, 2026-10): the defect is in the operand
resolution of ``binary_op`` (src/lib.rs), which resolves the right operand in
this order:

  1. ``b.extract::<f64>()``   — succeeds for *any* float-coercible object.
  2. ``b.cast::<NdArray>()``  — only a raw Rust ``NdArray`` instance.
  3. ``b.extract::<NdArray>()`` — buffer / array-interface coercible.
  4. otherwise ``TypeError: Unsupported operand type``.

Two distinct failure modes flow from this ordering:

* **Wrong value (scalar misread).** A *wrapped* rsnumpy ndarray coerces to
  ``float`` via its ``__float__`` (its first element), so ``raw + wrapper``
  takes branch 1 and computes ``raw + wrapper[0]`` instead of element-wise.
  This one reproduces in isolation and is locked in by the ``xfail`` test
  ``test_raw_arithmetic_with_wrapper`` in test_arithmetic.py.

* **Exception (unrecognised operand).** The four scicore paths below hand
  ``__add__`` an intermediate operand that none of the three branches accepts,
  so it falls to branch 4 and raises.  The triggering operand state (a shape /
  dtype / view combination produced by an earlier step) only appears inside the
  full scicore sequence — it does not reproduce from core ops in isolation — so
  it is tracked here and asserted in scicore's ``test_scirs`` suite rather than
  unit-tested at the rsnumpy core level.

The core element-wise arithmetic itself (scalar / raw / 2-D broadcasting, and
the ``win.real**2 + win.imag**2`` stft COLA expression) is sound in isolation.

This module is a living registry: when a gap is fixed, delete its entry and
promote the corresponding scicore test out of ``xfail``.
"""
import pytest

import rsnumpy as np

KNOWN_GAPS = [
    ("stft", "win.real**2 + win.imag**2 falls through __add__'s raw branch"),
    ("welch", "power-spectral averaging hits the same __add__ branch"),
    ("cubic_spline", "spline coefficient accumulation hits the __add__ branch"),
    ("interp_decomp", "interpolative-decomp LinearOperator path hits __add__"),
]


@pytest.mark.parametrize(
    ("name", "desc"), KNOWN_GAPS, ids=[g[0] for g in KNOWN_GAPS]
)
def test_known_gap_documented(name, desc):
    """Placeholder guard: documents that `name` is a tracked integration gap.

    Skipped (not failed) — the gap does not reproduce from rsnumpy core ops in
    isolation, so there is no core-level assertion to make.  The real coverage
    lives in scicore's ``test_scirs`` suite, where the four tests above are the
    regression net.
    """
    pytest.skip(f"tracked integration gap ({name}): {desc}")


def test_stft_expression_is_correct_in_isolation():
    # Regression guard for the core part of the stft COLA check: for both real
    # and complex windows, ``win.real**2 + win.imag**2 == |win|**2``.  This
    # passes today; it documents that the arithmetic core is sound and isolates
    # the failure to the surrounding scicore sequence.
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
