"""rsnumpy.lib - 兼容 numpy.lib 的子模块集合。"""

from . import recfunctions
from . import stride_tricks

__all__ = ['recfunctions', 'stride_tricks', 'add_newdoc']


def add_newdoc(where, obj, doc):
    """Stub: record a docstring for ``where.obj`` (no-op in rsnumpy)."""
    pass
