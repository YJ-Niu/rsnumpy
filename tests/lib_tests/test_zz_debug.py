from functools import partial

import rsnumpy as np
from rsnumpy._core.numeric import normalize_axis_tuple  # noqa: F401
from rsnumpy.exceptions import AxisError, ComplexWarning  # noqa: F401
from rsnumpy.lib._function_base_impl import _QuantileMethods  # noqa: F401
from rsnumpy.lib._nanfunctions_impl import (  # noqa: F401
    _divide_by_count, _nan_mask, _replace_nan)
from rsnumpy.testing import assert_equal  # noqa: F401

_ndat = np.array([[0.6244, np.nan, 0.2692, 0.0116, np.nan, 0.1170],
                  [0.5351, -0.9403, np.nan, 0.2100, 0.4759, 0.2833],
                  [np.nan, np.nan, np.nan, 0.1042, np.nan, -0.5954],
                  [0.1610, np.nan, np.nan, 0.1859, 0.3146, np.nan]])
_rdat = [np.array([0.6244, 0.2692, 0.0116, 0.1170]),
         np.array([0.5351, -0.9403, 0.2100, 0.4759, 0.2833]),
         np.array([0.1042, -0.5954]),
         np.array([0.1610, 0.1859, 0.3146])]


def test_debug():
    percentile = partial(np.percentile, method="inverted_cdf")
    nanpercentile = partial(np.nanpercentile, method="inverted_cdf")

    def gen_weights(d):
        return np.ones_like(d)

    tgt = [percentile(d, 28, weights=gen_weights(d)) for d in _rdat]
    print('TGT:', [float(x) for x in tgt])
    print('NDAT after tgt:', _ndat.tolist())
    res = nanpercentile(_ndat, 28, axis=1, weights=gen_weights(_ndat), out=None)
    print('RES:', res.tolist())
    assert True
