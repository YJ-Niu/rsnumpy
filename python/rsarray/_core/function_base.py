# SPDX-License-Identifier: MIT
"""numpy._core.function_base 兼容实现（文档工具）。

``add_newdoc`` 与 NumPy 2.x 行为一致：文档无法写入时静默忽略，
但被文档对象不存在时会抛错。
"""

import inspect
import types
import warnings

__all__ = ['add_newdoc', 'add_docstring', '_add_docstring']


def _needs_add_docstring(obj):
    """C 层对象需要 add_newdoc 补文档；纯 Python 对象不需要。"""
    if isinstance(obj, (type, types.FunctionType, types.MethodType)):
        return False
    return True


def add_docstring(func, doc, appender=None):
    """设置 docstring（失败静默；numpy ``add_docstring`` 简化版）。"""
    if not doc:
        return
    if appender:
        doc = appender(doc)
    try:
        func.__doc__ = doc
    except Exception:
        pass


def _add_docstring(obj, doc, warn_on_python):
    doc = inspect.cleandoc(doc)

    if warn_on_python and not _needs_add_docstring(obj):
        warnings.warn(
            f"add_newdoc was used on a pure-python object {obj}. "
            "Prefer to attach it directly to the source.",
            UserWarning, stacklevel=3)

    # 对类型尝试直接赋值 __doc__（heap type 可行；静态类型静默失败）。
    if isinstance(obj, type):
        head, sep, body = doc.partition("\n--\n\n")
        try:
            obj.__doc__ = body if sep else doc
        except Exception:
            pass
        else:
            if not sep:
                return
            doc = head + sep

    try:
        add_docstring(obj, doc)
    except Exception:
        pass


def add_newdoc(place, obj, doc, warn_on_python=True):
    """为既有对象附加文档（numpy ``add_newdoc`` 兼容）。

    Parameters
    ----------
    place : str
        模块的绝对点分名（如 "numpy._core"）。
    obj : str
        对象名（通常是类或函数名）。
    doc : str | tuple | list
        字符串直接作为文档；元组 ``(attr, docstring)`` 作用于对象的
        属性；列表则逐项应用。
    warn_on_python : bool
        对纯 Python 对象使用时是否告警。
    """
    new = getattr(__import__(place, globals(), {}, [obj]), obj)
    if isinstance(doc, str):
        _add_docstring(new, doc, warn_on_python)
    elif isinstance(doc, tuple):
        attr, docstring = doc
        _add_docstring(getattr(new, attr), docstring, warn_on_python)
    elif isinstance(doc, list):
        for attr, docstring in doc:
            _add_docstring(getattr(new, attr), docstring, warn_on_python)
