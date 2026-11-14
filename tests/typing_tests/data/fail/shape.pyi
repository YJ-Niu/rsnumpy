from typing import Any

import rsnumpy as np

# test bounds of _ShapeT_co

np.ndarray[tuple[str, str], Any]  # type: ignore[type-var]
