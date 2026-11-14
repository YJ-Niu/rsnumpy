# SPDX-License-Identifier: MIT
"""numpy._core.records 兼容层：从 rsnumpy.rec 再导出。"""

import rsnumpy as _rs

recarray = _rs.rec.recarray
fromarrays = _rs.rec.fromarrays
fromfile = _rs.rec.fromfile
fromrecords = _rs.rec.fromrecords
format_parser = getattr(_rs.rec, 'format_parser', None)

__all__ = ['recarray', 'fromarrays', 'fromfile', 'fromrecords']
if format_parser is not None:
    __all__.append('format_parser')
