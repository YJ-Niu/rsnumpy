"""rsnum.random - 随机数生成模块"""

import rsnum._core as _core


class random_module:
    """随机数生成模块"""

    def __init__(self):
        self._random = _core.random

    def default_rng(self, seed=None):
        """创建默认的随机数生成器（新 API）。

        参数:
            seed: 随机种子。

        返回:
            Generator: 随机数生成器。
        """
        return _core.random.Generator(seed)

    def seed(self, s):
        """设置随机种子。"""
        return self._random.seed(s)
    
    def rand(self, *args):
        """生成均匀分布的随机数。"""
        if len(args) == 0:
            return self._random.rand(1)
        return self._random.rand(*args)
    
    def randn(self, *args):
        """生成标准正态分布的随机数。"""
        if len(args) == 0:
            return self._random.randn(1)
        return self._random.randn(*args)
    
    def randint(self, low, high=None, size=None):
        """生成整数随机数。"""
        return self._random.randint(low, high, size)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        """生成均匀分布的随机数。"""
        return self._random.uniform(low, high, size)
    
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