"""rsnum.linalg - Linear algebra module."""

import rsnum._core as _core


def _ensure(x):
    from ..__init__ import ndarray
    if isinstance(x, (list, tuple)):
        return _core.ndarray(x)
    elif isinstance(x, ndarray):
        return x._array
    return x


class linalg_module:
    """线性代数模块"""
    
    @staticmethod
    def dot(a, b):
        """计算两个数组的点积。
        
        参数:
            a: 第一个数组。
            b: 第二个数组。
        
        返回:
            标量或 ndarray: 点积结果。
        """
        from ..__init__ import ndarray
        a_arr = ndarray(a)
        b_arr = ndarray(b)
        a_data = a_arr.tolist()
        b_data = b_arr.tolist()
        
        a_is_1d = not isinstance(a_data[0], list)
        b_is_1d = not isinstance(b_data[0], list)
        
        if a_is_1d and b_is_1d:
            return sum(x * y for x, y in zip(a_data, b_data))
        
        if a_is_1d:
            rows_b = len(b_data)
            cols_b = len(b_data[0]) if isinstance(b_data[0], list) else 1
            if cols_b == 1:
                return sum(a_data[i] * b_data[i][0] for i in range(len(a_data)))
            result = [sum(a_data[k] * b_data[k][j] for k in range(len(a_data))) for j in range(cols_b)]
            return ndarray(result)
        
        if b_is_1d:
            rows_a = len(a_data)
            return [sum(a_data[i][k] * b_data[k] for k in range(len(b_data))) for i in range(rows_a)]
        
        rows_a = len(a_data)
        cols_a = len(a_data[0])
        rows_b = len(b_data)
        cols_b = len(b_data[0])
        
        if cols_a != rows_b:
            raise ValueError(f"Shape mismatch: {cols_a} != {rows_b}")
        
        result = [[sum(a_data[i][k] * b_data[k][j] for k in range(cols_a)) for j in range(cols_b)] for i in range(rows_a)]
        return ndarray(result)
    
    @staticmethod
    def matmul(a, b):
        """计算两个数组的矩阵乘积。"""
        return linalg_module.dot(a, b)
    
    @staticmethod
    def inv(a):
        """计算矩阵的逆。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        data = arr.tolist()
        n = len(data)
        
        aug = []
        for i in range(n):
            aug.append(data[i] + [0.0] * n)
            aug[i][n + i] = 1.0
        
        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            pivot = aug[col][col]
            if abs(pivot) < 1e-10:
                raise ValueError("Matrix is singular")
            
            for j in range(n * 2):
                aug[col][j] /= pivot
            
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    for j in range(n * 2):
                        aug[row][j] -= factor * aug[col][j]
        
        result = [[aug[i][j + n] for j in range(n)] for i in range(n)]
        return ndarray(result)
    
    @staticmethod
    def det(a):
        """计算矩阵的行列式。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        data = arr.tolist()
        n = len(data)
        
        sign = 1
        det = 1.0
        
        aug = [row[:] for row in data]
        
        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]
                sign *= -1
            
            pivot = aug[col][col]
            if abs(pivot) < 1e-10:
                return 0.0
            
            det *= pivot
            
            for row in range(col + 1, n):
                factor = aug[row][col] / pivot
                for j in range(col, n):
                    aug[row][j] -= factor * aug[col][j]
        
        return det * sign
    
    @staticmethod
    def norm(x, ord=None, axis=None):
        """计算矩阵或向量的范数。"""
        from ..__init__ import ndarray
        import math
        arr = ndarray(x)
        data = arr.tolist()
        
        def flatten(d):
            if isinstance(d, list):
                result = []
                for item in d:
                    result.extend(flatten(item))
                return result
            return [d]
        
        flat = flatten(data)
        
        if ord is None or ord == 2:
            return math.sqrt(sum(v * v for v in flat))
        elif ord == 1:
            return sum(abs(v) for v in flat)
        elif ord == float('inf'):
            return max(abs(v) for v in flat)
        else:
            return sum(abs(v) ** ord for v in flat) ** (1.0 / ord)
    
    @staticmethod
    def solve(a, b):
        """求解线性方程组。"""
        from ..__init__ import ndarray
        a_arr = ndarray(a)
        b_arr = ndarray(b)
        a_data = a_arr.tolist()
        b_data = b_arr.tolist()
        
        n = len(a_data)
        
        aug = []
        for i in range(n):
            aug.append(a_data[i][:] + ([b_data[i]] if isinstance(b_data[i], (int, float)) else b_data[i]))
        
        for col in range(n):
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > abs(aug[max_row][col]):
                    max_row = row
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            pivot = aug[col][col]
            if abs(pivot) < 1e-10:
                raise ValueError("Matrix is singular")
            
            for row in range(col + 1, n):
                factor = aug[row][col] / pivot
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]
        
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            x[i] /= aug[i][i]
        
        return ndarray(x)
    
    @staticmethod
    def eig(a):
        """计算矩阵的特征值和特征向量。"""
        return _core.linalg.eig(_ensure(a))
    
    @staticmethod
    def eigvals(a):
        """计算矩阵的特征值。"""
        return _core.linalg.eigvals(_ensure(a))
    
    @staticmethod
    def svd(a):
        """计算奇异值分解。"""
        return _core.linalg.svd(_ensure(a))
    
    @staticmethod
    def qr(a):
        """计算 QR 分解。"""
        return _core.linalg.qr(_ensure(a))
    
    @staticmethod
    def cholesky(a):
        """计算 Cholesky 分解。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        data = arr.tolist()
        n = len(data)
        
        L = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    s = sum(L[i][k] ** 2 for k in range(j))
                    L[i][j] = (data[i][j] - s) ** 0.5
                else:
                    s = sum(L[i][k] * L[j][k] for k in range(j))
                    L[i][j] = (data[i][j] - s) / L[j][j]
        
        return ndarray(L)
    
    @staticmethod
    def matrix_power(a, n):
        """计算矩阵的幂。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        data = arr.tolist()
        
        if n == 0:
            return ndarray([[1.0 if i == j else 0.0 for j in range(len(data))] for i in range(len(data))])
        
        result = [row[:] for row in data]
        
        for _ in range(n - 1):
            result = linalg_module.dot(result, data)
        
        return ndarray(result) if isinstance(result, list) else result
    
    @staticmethod
    def pinv(a):
        """计算矩阵的伪逆。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        data = arr.tolist()
        
        rows = len(data)
        cols = len(data[0])
        
        if rows <= cols:
            ata = linalg_module.dot(linalg_module.transpose(data), data)
            ata_inv = linalg_module.inv(ata)
            return linalg_module.dot(ata_inv, linalg_module.transpose(data))
        else:
            aat = linalg_module.dot(data, linalg_module.transpose(data))
            aat_inv = linalg_module.inv(aat)
            return linalg_module.dot(linalg_module.transpose(data), aat_inv)
    
    @staticmethod
    def transpose(a):
        """转置矩阵（内部辅助函数）。"""
        if not isinstance(a, list):
            return a
        rows = len(a)
        cols = len(a[0])
        return [[a[i][j] for i in range(rows)] for j in range(cols)]
    
    @staticmethod
    def trace(a):
        """计算矩阵的迹。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        data = arr.tolist()
        return sum(data[i][i] for i in range(len(data)))
    
    @staticmethod
    def diagonal(a):
        """返回矩阵的对角线元素。"""
        from ..__init__ import ndarray
        arr = ndarray(a)
        data = arr.tolist()
        return ndarray([data[i][i] for i in range(len(data))])
    
    @staticmethod
    def svdvals(a):
        """计算奇异值。"""
        return _core.linalg.svdvals(_ensure(a))
    
    @staticmethod
    def eigh(a):
        """计算 Hermitian 或对称矩阵的特征值和特征向量。"""
        return _core.linalg.eigh(_ensure(a))
    
    @staticmethod
    def solve_banded(lower, upper, ab, b):
        """求解带状线性方程组。"""
        from ..__init__ import ndarray, _wrap_result
        return _wrap_result(_core.linalg.solve_banded(lower, upper, _ensure(ab), _ensure(b)))
