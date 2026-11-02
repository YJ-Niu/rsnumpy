from io import StringIO
from typing import assert_type

import rsnumpy as np
import rsnumpy.lib.array_utils as array_utils
import rsnumpy.typing as npt

AR: npt.NDArray[np.float64]
AR_DICT: dict[str, npt.NDArray[np.float64]]
FILE: StringIO

def func(a: int) -> bool: ...

assert_type(array_utils.byte_bounds(AR), tuple[int, int])
assert_type(array_utils.byte_bounds(np.float64()), tuple[int, int])

assert_type(np.info(1, output=FILE), None)
