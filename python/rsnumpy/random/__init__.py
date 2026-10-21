"""rsnumpy.random - 随机数生成模块"""

import rsnumpy.num_core as _core
import random as _py_random


# _core.random.Generator may be Python's built-in typing.Generator, not a
# real RNG.  Provide a working fallback.
if not hasattr(_core.random.Generator, 'integers'):
    class _PyGenerator:
        """Python-based random number generator fallback."""

        def __init__(self, seed=None):
            self._rng = _py_random.Random(seed)

        def integers(self, low, high=None, size=None, dtype=None, endpoint=False):
            import rsnumpy as np
            if high is None:
                low, high = 0, low
            if size is None:
                return self._rng.randint(low, high)
            return np.asarray([self._rng.randint(low, high) for _ in range(size)])

        def uniform(self, low=0.0, high=1.0, size=None):
            import rsnumpy as np
            if size is None:
                return self._rng.uniform(low, high)
            return np.asarray([self._rng.uniform(low, high) for _ in range(size)])

        def normal(self, loc=0.0, scale=1.0, size=None):
            import rsnumpy as np
            if size is None:
                return self._rng.gauss(loc, scale)
            return np.asarray([self._rng.gauss(loc, scale) for _ in range(size)])

        def random(self, size=None):
            import rsnumpy as np
            if size is None:
                return self._rng.random()
            return np.asarray([self._rng.random() for _ in range(size)])

        def standard_normal(self, size=None):
            return self.normal(0.0, 1.0, size)

        def multinomial(self, n, pvals, size=None):
            import rsnumpy as np
            # Simple categorical sampling
            pvals = np.asarray(pvals)
            pvals = pvals / pvals.sum()
            cdf = np.cumsum(pvals)
            if size is None:
                size = ()
            result = []
            for _ in range(np.prod(size) if isinstance(size, tuple) else size):
                u = self._rng.random()
                idx = int(cdf.searchsorted(u))
                result.append(idx)
            return np.asarray(result).reshape(size)

    _core.random.Generator = _PyGenerator


class _WrappedGenerator:
    """Generator 的薄包装，把核心返回的数组提升为 rsnumpy 数组。

    核心生成器（`_core.random.Generator`）返回的是 Rust 侧的 `ndarray`，它和
    `rsnumpy.ndarray` 不是同一个类：后者才有那些补齐了 NumPy 语义的 Python 层
    方法。差别最明显的是 `astype` —— 核心版只接受字符串，而 NumPy 语义要求也能
    吃 `dtype` 实例，因此诸如

        a = rng.random(shape).astype(np.dtype('float64'))

    的写法（numpy 约定 scipy 测试里到处在用）在裸核心对象上会直接报错。

    这里只做一件事：转发所有属性，并把方法返回值里的核心数组换成 `rsnumpy.ndarray`。
    """

    __slots__ = ('_inner',)

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                import rsnumpy as np
                # Wrap builtins.ndarray results into rsnumpy ndarray
                if type(result).__module__ == 'builtins' and type(result).__name__ == 'ndarray':
                    return np.asarray(result)
                return result
            return wrapper
        return attr

    def _convert(self, value):
        if type(value).__module__ == 'builtins' and \
                type(value).__name__ == 'ndarray':
            # `_core.ndarray`：Rust 侧类型，包一层即可获得 Python 层 API。
            import rsnumpy as np
            return np.asarray(value)
        return value

    def random(self, *args, **kwargs):
        return self._convert(self._inner.random(*args, **kwargs))

    def standard_normal(self, *args, **kwargs):
        return self._convert(self._inner.standard_normal(*args, **kwargs))

    def normal(self, *args, **kwargs):
        return self._convert(self._inner.normal(*args, **kwargs))

    def uniform(self, *args, **kwargs):
        return self._convert(self._inner.uniform(*args, **kwargs))

    def integers(self, *args, **kwargs):
        return self._convert(self._inner.integers(*args, **kwargs))

    def multinomial(self, n, pvals, size=None):
        import rsnumpy as np
        pvals = np.asarray(pvals, dtype=float)
        pvals = pvals / pvals.sum()
        cdf = np.cumsum(pvals)
        if size is None:
            size = ()
        n_samples = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        # Count occurrences of each category
        result = np.zeros(len(pvals), dtype=int)
        for _ in range(n_samples):
            u = float(np.asarray(self._inner.random()).tolist())
            if isinstance(u, list):
                u = u[0]
            idx = int(np.asarray(cdf).searchsorted(u) if hasattr(cdf, 'searchsorted') else 0)
            idx = min(idx, len(pvals) - 1)
            result[idx] += 1
        return np.asarray(result)

    def permuted(self, x, axis=0, *args, **kwargs):
        import rsnumpy as np
        arr = np.asarray(x)
        idx = self._inner.permutation(arr.shape[axis])
        return np.take(arr, idx, axis=axis)


class mtrand_module:
    """mtrand 模块，提供全局随机数生成器"""
    def __init__(self):
        self._rand = _core.random.RandomState(None)


class random_module:
    """随机数生成模块"""

    def __init__(self):
        self._random = _core.random
        self.mtrand = mtrand_module()

    @property
    def Generator(self):
        return _core.random.Generator

    @property
    def RandomState(self):
        return _core.random.RandomState

    def default_rng(self, seed=None):
        """创建默认的随机数生成器（新 API）。

        参数:
            seed: 随机种子。

        返回:
            Generator: 随机数生成器。
        """
        # If seed is already a wrapped generator, return it directly
        if isinstance(seed, _WrappedGenerator):
            return seed
        return _WrappedGenerator(_core.random.Generator(seed))

    def seed(self, s):
        """设置随机种子。"""
        return self._random.seed(s)
    
    def rand(self, *args):
        """生成均匀分布的随机数。"""
        from ..__init__ import ndarray
        if len(args) == 0:
            return ndarray(self._random.rand(1))
        return ndarray(self._random.rand(*args))
    
    def randn(self, *args):
        """生成标准正态分布的随机数。"""
        from ..__init__ import ndarray
        if len(args) == 0:
            return ndarray(self._random.randn(1))
        return ndarray(self._random.randn(*args))
    
    def randint(self, low, high=None, size=None):
        """生成整数随机数。"""
        from ..__init__ import ndarray
        return ndarray._wrap(self._random.randint(low, high, size), "int64")
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """生成均匀分布的随机数。"""
        from ..__init__ import ndarray
        return ndarray(self._random.uniform(low, high, size))
    
    def random(self, size=None):
        """生成 [0.0, 1.0) 区间均匀分布的随机数。"""
        from ..__init__ import ndarray
        return ndarray(self._random.uniform(0.0, 1.0, size))
    
    def normal(self, loc=0.0, scale=1.0, size=None):
        """生成正态分布的随机数。"""
        from ..__init__ import ndarray
        if size is None:
            size = 1
        return ndarray(self._random.normal(loc, scale, size))
    
    def randn_like(self, a):
        """生成与输入数组形状相同的标准正态分布随机数。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        return ndarray(self._random.randn(*arr.shape))
    
    def rand_like(self, a):
        """生成与输入数组形状相同的均匀分布随机数。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        return ndarray(self._random.rand(*arr.shape))
