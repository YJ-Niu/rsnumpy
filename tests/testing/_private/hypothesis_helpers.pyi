from typing import Final

import hypothesis as hypothesis  # noqa: F401
from hypothesis import given as given, settings as settings, strategies as strategies  # noqa: F401
from hypothesis.extra import rsnumpy as _hynp
from hypothesis.extra.rsnumpy import arrays as arrays  # noqa: F401
from hypothesis.strategies import sampled_from as sampled_from  # noqa: F401

hynp = _hynp  # ensure that `hynp` is considered as importable by type-checkers
st = strategies

# True when the optional ``hypothesis`` dependency is installed.
HAS_HYPOTHESIS: Final[bool] = ...
