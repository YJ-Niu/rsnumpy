"""rsnumpy.lib - 兼容 numpy.lib 的子模块集合。"""

from . import recfunctions
from . import stride_tricks

__all__ = ['recfunctions', 'stride_tricks', 'add_newdoc',
           '_index_tricks_impl']


def add_newdoc(where, obj, doc):
    """Stub: record a docstring for ``where.obj`` (no-op in rsnumpy)."""
    pass


def __getattr__(name):
    if name == '_index_tricks_impl':
        from . import _index_tricks_impl
        return _index_tricks_impl
    raise AttributeError(f"module 'rsnumpy.lib' has no attribute {name!r}")
