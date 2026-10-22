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

        def vonmises(self, kappa, size=None):
            import rsnumpy as np
            import math
            if size is None:
                size = ()
            n_samples = int(np.prod(size)) if isinstance(size, tuple) else int(size)
            result = []
            for _ in range(n_samples):
                while True:
                    u1 = self._rng.random()
                    u2 = self._rng.random()
                    z = math.cos(math.pi * u1)
                    if u2 < math.exp(kappa * (z - 1)):
                        break
                angle = math.acos(max(-1.0, min(1.0, z)))
                if self._rng.random() < 0.5:
                    angle = -angle
                result.append(angle)
            return np.asarray(result).reshape(size)

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


class _MT19937:
    """Minimal stand-in for NumPy's MT19937 bit generator.

    Only the attributes that downstream code (e.g. rsscipy's ``_util``)
    inspects are provided: ``_seed_seq`` and ``state``.
    """

    def __init__(self, seed=None):
        self._seed_seq = None
        self._seed = seed

    @property
    def state(self):
        return {'bit_generator': 'MT19937', 'state': {'key': [0]}, 'pos': 0}

    def spawn(self, n_children):
        return [type(self)(None) for _ in range(n_children)]


class _RandomStateWrapper:
    """Python wrapper around the core RandomState.

    Adds the legacy attributes SciPy expects (``_bit_generator``) and
    distributions the Rust core does not implement (``multivariate_normal``).
    """

    def __init__(self, seed=None):
        self._seed = seed
        self._inner = _core.random.RandomState(seed)

    def __deepcopy__(self, memo):
        # The Rust core object is not picklable.  Re-seed from the original
        # seed so the copy is independent of (and initially identical to)
        # this wrapper, matching numpy's deepcopy semantics for unused
        # generators.
        return _RandomStateWrapper(self._seed)

    def __copy__(self, memo=None):
        return _RandomStateWrapper(self._seed)

    def _next_u(self):
        return float(self._inner.random(1).tolist()[0])

    def _next_normal(self):
        return float(self._inner.randn(1).tolist()[0])

    def __getattr__(self, name):
        if name == 'multivariate_normal':
            def mvn(mean, cov, size=None):
                return _mvn_sample(self._next_normal, mean, cov, size)
            return mvn
        if name == 'choice':
            def choice(a, size=None, replace=True, p=None):
                import rsnumpy as np
                if isinstance(a, int):
                    pop = a
                else:
                    arr0 = np.asarray(a)
                    pop = arr0.shape[0]
                if size is None:
                    out = [self._draw_index(pop, p)]
                    sz_shape = ()
                elif isinstance(size, int):
                    out = [self._draw_index(pop, p) for _ in range(size)]
                    sz_shape = (size,)
                else:
                    sz_shape = tuple(size)
                    out = [self._draw_index(pop, p)
                           for _ in range(int(np.prod(sz_shape)))]
                if not replace:
                    seen = set()
                    fixed = []
                    for v in out:
                        while v in seen:
                            v = self._draw_index(pop, p)
                        seen.add(v)
                        fixed.append(v)
                    out = fixed
                if isinstance(a, int):
                    res = np.asarray(out)
                else:
                    res = np.take(np.asarray(a), np.asarray(out))
                return res.reshape(sz_shape) if size is not None else res[()]
            return choice
        if name == 'permutation':
            def permutation(x):
                import rsnumpy as np
                n = x if isinstance(x, int) else np.asarray(x).shape[0]
                idx = list(range(n))
                for i in range(n - 1, 0, -1):
                    j = int(self._next_u() * (i + 1))
                    idx[i], idx[j] = idx[j], idx[i]
                if isinstance(x, int):
                    return np.asarray(idx)
                return np.take(np.asarray(x), np.asarray(idx))
            return permutation
        if name == 'shuffle':
            def shuffle(x):
                n = x.shape[0]
                for i in range(n - 1, 0, -1):
                    j = int(self._next_u() * (i + 1))
                    x[i], x[j] = x[j], x[i]
            return shuffle
        if name == 'get_state':
            def get_state():
                # Snapshot tuple mirroring NumPy's legacy layout.  The Rust
                # core does not expose the MT19937 buffer, so report the
                # seed plus an empty internal buffer.
                return ('MT19937', [self._seed], 0, 0, 0.0)
            return get_state
        if name == 'set_state':
            def set_state(state):
                # Re-seed from the recorded seed when possible; the core
                # cannot restore an arbitrary MT19937 buffer.
                try:
                    seed = state[1][0]
                except (IndexError, TypeError):
                    seed = None
                self._inner = _core.random.RandomState(seed)
            return set_state
        if name == 'standard_normal':
            def standard_normal(size=None):
                return self._inner.randn(*( (size,) if isinstance(size, int)
                                            else tuple(size) if size else (1,)))
            return standard_normal
        if name == 'randint':
            def randint(low, high=None, size=None, dtype=None, endpoint=False):
                import rsnumpy as np
                if high is None:
                    low, high = 0, low
                if size is None:
                    vals = [low + int(self._next_u() * (high - low))]
                    shape = ()
                elif isinstance(size, int):
                    vals = [low + int(self._next_u() * (high - low))
                            for _ in range(size)]
                    shape = (size,)
                else:
                    shape = tuple(size)
                    n = int(np.prod(shape))
                    vals = [low + int(self._next_u() * (high - low))
                            for _ in range(n)]
                res = np.asarray(vals, dtype=dtype or 'int64')
                return res.reshape(shape) if shape else res.reshape(())[()]
            return randint
        return getattr(self._inner, name)

    def _draw_index(self, pop, p):
        if p is None:
            return int(self._next_u() * pop)
        import rsnumpy as np
        cdf = np.cumsum(np.asarray(p, dtype='float64'))
        u = self._next_u()
        return int(np.searchsorted(cdf, u))

    @property
    def _bit_generator(self):
        return _MT19937(None)


def _mvn_sample(next_standard_normal, mean, cov, size):
    """Draw multivariate normal samples via the SVD factor of ``cov``."""
    import rsnumpy as np
    mean = np.asarray(mean, dtype='float64').reshape(-1).tolist()
    cov = np.asarray(cov, dtype='float64')
    d = len(mean)
    _, s, vt = np.linalg.svd(cov, full_matrices=False)
    sl = [max(float(x), 0.0) ** 0.5 for x in s.tolist()]
    vtl = [[float(x) for x in row] for row in vt.tolist()]
    if size is None:
        shape = ()
    elif isinstance(size, int):
        shape = (size,)
    else:
        shape = tuple(size)
    n = int(np.prod(shape)) if shape else 1
    out = []
    for _ in range(n):
        z = [next_standard_normal() for _ in range(d)]
        row = []
        for i in range(d):
            acc = mean[i]
            for j in range(d):
                acc += vtl[j][i] * sl[j] * z[j]
            row.append(acc)
        out.append(row)
    return np.asarray(out).reshape(shape + (d,))


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

    __slots__ = ('_inner', '_seed')

    def __init__(self, inner=None):
        # Accept NumPy-style construction: Generator(seed) or Generator()
        self._seed = inner
        if isinstance(inner, _WrappedGenerator):
            self._seed = inner._seed
            inner = inner._inner
        elif not hasattr(inner, 'random'):
            try:
                inner = _core.random.Generator(inner)
            except OverflowError:
                # NumPy accepts arbitrarily large seeds (SeedSequence);
                # fold them into the range the core understands.
                inner = _core.random.Generator(int(inner) % (2**63))
        self._inner = inner

    def __deepcopy__(self, memo):
        return _WrappedGenerator(self._seed)

    def __copy__(self):
        return _WrappedGenerator(self._seed)

    def _u(self):
        """One uniform float in [0, 1) from the inner generator."""
        v = self._inner.random()
        if hasattr(v, 'item'):
            v = v.item()
        return float(v)

    def _make_choice(self):
        """NumPy-compatible ``choice`` (the Rust core only takes array `a`
        and no ``p``; int populations and weights are handled here)."""
        def choice(a, size=None, replace=True, p=None):
            import rsnumpy as np
            if hasattr(a, 'shape') or isinstance(a, (list, tuple)):
                values = np.asarray(a)
                pop = values.shape[0] if values.ndim > 0 else values.size
            else:
                values = None
                pop = int(a)
            if size is None:
                sz_shape = ()
                n = 1
            elif isinstance(size, (int,)):
                sz_shape = (int(size),)
                n = int(size)
            else:
                sz_shape = tuple(int(x) for x in size)
                n = 1
                for x in sz_shape:
                    n *= x
            if not replace and pop < n:
                raise ValueError(
                    "Cannot take a larger sample than population when "
                    "'replace=false'")
            if p is not None:
                pw = np.asarray(p, dtype='float64').reshape(-1).tolist()
                total = float(sum(pw))
                cum = []
                acc = 0.0
                for x in pw:
                    acc += float(x) / total
                    cum.append(acc)
                cum[-1] = 1.0

                def draw():
                    u = self._u()
                    for i, c in enumerate(cum):
                        if u <= c:
                            return i
                    return len(cum) - 1
            else:
                def draw():
                    return int(self._u() * pop)
            idx = [draw() for _ in range(n)]
            if not replace:
                seen = set()
                fixed = []
                for v in idx:
                    while v in seen:
                        v = draw()
                    seen.add(v)
                    fixed.append(v)
                idx = fixed
            if values is None:
                res = np.asarray(idx)
            else:
                res = np.take(values, np.asarray(idx))
            if size is None:
                return res.reshape(sz_shape)[()]
            return res.reshape(sz_shape)
        return choice

    def _make_shuffle(self):
        """NumPy-compatible in-place ``shuffle`` (the Rust core only accepts
        its own ndarray type, so shuffle at the Python layer)."""
        def shuffle(x, axis=0):
            import rsnumpy as np
            arr = x if axis == 0 else np.moveaxis(x, axis, 0)
            n = arr.shape[0]
            for i in range(n - 1, 0, -1):
                j = int(self.integers(i + 1))
                if j == i:
                    continue
                tmp = arr[i].copy()
                arr[i] = arr[j]
                arr[j] = tmp
        return shuffle

    def __getattr__(self, name):
        if name == 'choice':
            return self._make_choice()
        if name == 'shuffle':
            return self._make_shuffle()
        attr = getattr(self._inner, name, None)
        if attr is None:
            # Fallback for methods not available on the inner generator
            if name == 'vonmises':
                def vonmises_wrapper(kappa, size=None, mu=0.0, dtype=None):
                    import rsnumpy as np
                    import math
                    # Broadcast mu/kappa to a common shape, then optionally to
                    # ``size``.  This lets callers pass array parameters (as
                    # rsscipy's distribution layer does) as well as scalars.
                    mu_b, kappa_b = np.broadcast_arrays(
                        np.asarray(mu, dtype='float64'),
                        np.asarray(kappa, dtype='float64'))
                    if size is None:
                        shape = tuple(mu_b.shape)
                    else:
                        shape = (tuple(size) if isinstance(size, (tuple, list))
                                 else (int(size),))
                        mu_b = np.broadcast_to(mu_b, shape)
                        kappa_b = np.broadcast_to(kappa_b, shape)
                    flat_mu = mu_b.reshape(-1).tolist()
                    flat_k = kappa_b.reshape(-1).tolist()
                    result = []
                    for m, k in zip(flat_mu, flat_k):
                        m = float(m)
                        k = float(k)
                        while True:
                            u1 = self._inner.random().item()
                            u2 = self._inner.random().item()
                            z = math.cos(math.pi * u1)
                            if u2 < math.exp(k * (z - 1)):
                                break
                        angle = math.acos(max(-1.0, min(1.0, z)))
                        if self._inner.random().item() < 0.5:
                            angle = -angle
                        result.append(m + angle)
                    result_arr = np.asarray(result).reshape(shape)
                    if dtype is not None and hasattr(result_arr, 'astype'):
                        result_arr = result_arr.astype(dtype)
                    return result_arr
                return vonmises_wrapper
            if name == 'multivariate_normal':
                def mvn_wrapper(mean, cov, size=None):
                    def _std_normal():
                        v = self._inner.standard_normal()
                        if hasattr(v, 'item'):
                            v = v.item()
                        return float(v)
                    return _mvn_sample(_std_normal, mean, cov, size)
                return mvn_wrapper
            if name == 'permutation':
                def permutation_wrapper(x, axis=0):
                    import rsnumpy as np
                    arr = np.asarray(x)
                    idx = self._inner.permutation(arr.shape[axis])
                    return np.take(arr, idx, axis=axis)
                return permutation_wrapper
            if name == 'permutated':
                def permutated_wrapper(x, axis=0, *args, **kwargs):
                    import rsnumpy as np
                    arr = np.asarray(x)
                    idx = self._inner.permutation(arr.shape[axis])
                    return np.take(arr, idx, axis=axis)
                return permutated_wrapper
            raise AttributeError(f"'{type(self._inner).__name__}' object has no attribute '{name}'")
        if callable(attr):
            def wrapper(*args, **kwargs):
                # Strip dtype from kwargs - rsnumpy core doesn't support it
                dtype = kwargs.pop('dtype', None)
                result = attr(*args, **kwargs)
                import rsnumpy as np
                # Wrap builtins.ndarray results into rsnumpy ndarray
                if type(result).__module__ == 'builtins' and type(result).__name__ == 'ndarray':
                    result = np.asarray(result)
                # Convert dtype if specified
                if dtype is not None and hasattr(result, 'astype'):
                    result = result.astype(dtype)
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
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.random(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def standard_normal(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.standard_normal(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def normal(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.normal(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def uniform(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.uniform(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

    def integers(self, *args, **kwargs):
        dtype = kwargs.pop('dtype', None)
        result = self._convert(self._inner.integers(*args, **kwargs))
        if dtype is not None and hasattr(result, 'astype'):
            result = result.astype(dtype)
        return result

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
        self._rand = _RandomStateWrapper(None)


class _SeedSequence:
    """Minimal stand-in for ``np.random.SeedSequence``.

    Only the attributes downstream code inspects (``entropy``) are kept.
    """

    def __init__(self, entropy=None):
        self.entropy = entropy


class random_module:
    """随机数生成模块"""

    def __init__(self):
        self._random = _core.random
        self.mtrand = mtrand_module()

    @property
    def Generator(self):
        # Expose the wrapper class so ``isinstance(x, np.random.Generator)``
        # works for generators produced by ``default_rng``.
        return _WrappedGenerator

    @property
    def RandomState(self):
        return _RandomStateWrapper

    @property
    def SeedSequence(self):
        return _SeedSequence

    @property
    def BitGenerator(self):
        return _MT19937

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
        return _WrappedGenerator(seed)

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
