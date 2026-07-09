import rsnumpy as np
import time
import math

start_time = time.time()

def mark_print():
    print("\n++++++++++++++++++++++++\n")


# 使用标量类型
dt = np.dtype([('age', np.int8)])
a = np.array([(10,), (20,), (30,)], dtype=dt)
print(a['age'])

student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)

a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(a)

a = np.arange(15).reshape(3, 5)
print(a)
# print(b)
# print(np.array_equal(a, b))  # True
# print(a.shape == b.shape)    # True
# print(a.dtype == b.dtype)    # True
print(a.dtype)
# print(b.dtype)

 
a = np.arange(24)
print(a.ndim)             # a 现只有一个维度
# 现在调整其大小
b = a.reshape(2, 4, 3)  # b 现在拥有三个维度
print(b.ndim)
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)

a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape = (3, 2)
print(a)

a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.reshape(3, 2)
print(b)

# 数组的 dtype 为 int8（一个字节）
x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
print(x.itemsize)
 
# 数组的 dtype 现在为 float64（八个字节）
y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
print(y.itemsize)

x = np.array([1, 2, 3, 4, 5])
print(x.flags)

x3 = np.empty([2, 3], dtype=int)
print(x3)
print()

x2 = np.zeros([2, 3], dtype=int)
print(x2)
print()

x3 = np.zeros(shape=(2, 3), dtype=float, order='C')
print(x3)

print("+++++++++++++++++\n")
# 默认为浮点数
x = np.zeros(5)
print(x)
 
# 设置类型为整数
y = np.zeros((5,), dtype=int)
print(y)

print("+++++++++++++++++\n")
# 自定义类型
z = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
print(z)

x = np.ones(5)
print(x)
 
# 自定义类型
x = np.ones([2, 2], dtype=int)
print(x)

print("3. +++++++++++++++++\n")
# 创建一个 3x3 的二维数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 
# 创建一个与 arr 形状相同的，所有元素都为 0 的数组
zeros_arr = np.zeros_like(arr)
print(zeros_arr)

print("4. +++++++++++++++++\n")
# 创建一个 3x3 的二维数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
 
# 创建一个与 arr 形状相同的，所有元素都为 1 的数组
ones_arr = np.ones_like(arr)

print(ones_arr)

print("5. +++++++++++++++++\n")
x = [1, 2, 3]
a = np.asarray(x)
print(a, "\n==>\n", "[1  2  3]")

x = (1, 2, 3)
a = np.asarray(x)
print(a, "\n==>\n", "[1  2  3]")

x = [(1, 2, 3), (4, 5)]
a = np.asarray(x)
print(a, "\n==>\n", "[[1  2  3]\n[4  5]]")

x = [1, 2, 3]
a = np.asarray(x, dtype=float)
print(a, "\n==>\n", "[1.  2.  3.]")

s = b'Hello World'
a = np.frombuffer(s, dtype='S1')
print(a)

# 使用 range 函数创建列表对象
list = list(range(5))
it = iter(list)
 
# 使用迭代器创建 ndarray
x = np.fromiter(it, dtype=float)
print(x)

x = np.arange(5)
print(x)
# 设置了 dtype
x = np.arange(5, dtype=float)
print(x)


x = np.arange(10, 20, 2)
print(x)

a = np.linspace(1, 10, 10)
print(a)

a = np.linspace(1, 1, 10)
print(a)

a = np.linspace(10, 20, 5, endpoint=False)
print(a)

a = np.linspace(1, 10, 10, retstep=True)
print(11, a)

# 拓展例子
b = np.linspace(1, 10, 10).reshape([10, 1])
print(b)

a = np.logspace(1.0, 2.0, num=10)
print(a)

a = np.logspace(0, 9, 10, base=2)
print(a)

a = np.arange(10)
s = slice(2, 7, 2)   # 从索引 2 开始到索引 7 停止，间隔为2
print(a[s])

a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
b = a[5]
print(b)

a = np.arange(10)
print(a[2:])

a = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
print(a[2:5])

a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a)
# 从某个索引处开始切割
print('从数组索引 a[1:] 处开始切割')
print(a[1:])

print("+++++++++++++++++\n")
a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print(a[..., 1])   # 第2列元素
print(a[1, ...])   # 第2行元素
print(a[..., 1:])  # 第2列及剩下的所有元素
print("+++++++++++++++++\n")
x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0, 1, 2], [0, 1, 0]]
print(y)

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print('我们的数组是：')
print(x)
print("\n")
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols]
print('这个数组的四个角元素是：')
print(y)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = a[1:3, 1:3]
print("b:", b)

c = a[1:3, [1, 2]]
print("c:", c)

d = a[..., 1:]
print("d:", d)

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print('我们的数组是：')
print(x)
print("\n")
# 现在我们会打印出大于 5 的元素
print("大于 5 的元素是：")
print(x[x > 5])

a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5])
print(a[~np.isnan(a)])

a = np.array([1, 2+6j, 5, 3.5+5j])
print(a[np.iscomplex(a)])

x = np.arange(9)
print(x)
# 一维数组读取指定下标对应的元素
print("-------读取下标对应的元素-------")
x2 = x[[0, 6]]  # 使用花式索引
print(x2)

print(x2[0])
print(x2[1])

x = np.arange(32).reshape((8, 4))
print(x)
# 二维数组读取指定下标对应的行
print("-------读取下标对应的行-------")
print(x[[4, 2, 1, 7]])

x = np.arange(32).reshape((8, 4))
print(x[[-4, -2, -1, -7]])

x = np.arange(32).reshape((8, 4))
print(x[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a * b
print(c)

a = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
b = np.array([0, 1, 2])
print(a + b)

a = np.array([[0, 0, 0],
              [10, 10, 10],
              [20, 20, 20],
              [30, 30, 30]])
b = np.array([1, 2, 3])
bb = np.tile(b, (4, 1))  # 重复 b 的各个维度
print(a + bb)

a = np.arange(6).reshape(2, 3)
print('原始数组是：')
print(a)
print('\n')
print('迭代输出元素：')
for x in np.nditer(a):
    print(x, end=", ")
print('\n')


c = np.array([1.3, 2.4, 3.5])
for x in np.nditer(c):
    print(x, end=", ")
print('\n')
print('\n')
print('\n')

# a = np.arange(6).reshape(2, 3)
# for x in np.nditer(a.T):
#     print(x, end=", ")
# print('\n')
 
# for x in np.nditer(a.T.copy(order='C')):
#     print(x, end=", ")
# print('\n')

 
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
print('原始数组的转置是：')
b = a.T
print(b)
print('\n')
print('以 C 风格顺序排序：')
c = b.copy(order='C')
print(c)
for x in np.nditer(c):
    print(x, end=", ")
print('\n')
print('以 F 风格顺序排序：')
c = b.copy(order='F')
print(c)
for x in np.nditer(c, order='F'):
    print(x, end=", ")
print()
a = np.arange(6).reshape(2, 3)
print('原始数组是：')
print(a)
print('\n')
print(a.T)
print('迭代输出元素：')
for x in np.nditer(a.T):
    print(x, end=", ")
print('\n')
 
for x in np.nditer(a.T.copy(order='F')):
    print(x, end=", ")
print('\n')

print("+++++++++++++++++\n")
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
print('以 C 风格顺序排序：')
for x in np.nditer(a, order='C'):
    print(x, end=", ")
print('\n')
print('以 F 风格顺序排序：')
for x in np.nditer(a, order='F'):
    print(x, end=", ")
print('\n')
print("+++++++++++++++++\n")

a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
for x in np.nditer(a, op_flags=['readwrite']):
    x[...] = 2*x
print('修改后的数组是：')
print(a)
print('\n')

print("+++++++++++++++++\n")
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('原始数组是：')
print(a)
print('\n')
print('修改后的数组是：')
for x in np.nditer(a, flags=['multi_index'], order='F'):
    print(x, end=", ")

print('')

print("+++++++++++++++++\n")
a = np.arange(0, 60, 5)
a = a.reshape(3, 4)
print('第一个数组为：')
print(a)
print('\n')
print('第二个数组为：')
b = np.array([1, 2, 3, 4], dtype=int)
print(b)
print('\n')
print('修改后的数组为：')
for x, y in np.nditer([a, b]):
    print("%d:%d" % (x, y), end=", ")
print('')
print("+++++++++++++++++\n")
a = np.arange(8)
print('原始数组：')
print(a)
print('\n')
 
b = a.reshape(4, 2)
print('修改后的数组：')
print(b)

a = np.arange(9).reshape(3, 3)
print('原始数组：')
for row in a:
    print(row)
 
# 对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
print('迭代后的数组：')
for element in a.flat:
    print(element)

print('')
print("+++++++++++++++++\n")
a = np.arange(8).reshape(2, 4)
 
print('原数组：')
print(a)
print('\n')
# 默认按行
 
print('展开的数组：')
print(a.flatten())
print('\n')
 
print('以 F 风格顺序展开的数组：')
print(a.flatten(order='F'))
print('\n')

print('')
print("+++++++++++++++++\n")
a = np.arange(8).reshape(2, 4)
 
print('原数组：')
print(a)
print('\n')
 
print('调用 ravel 函数之后：')
print(a.ravel())
print('\n')
 
print('以 F 风格顺序调用 ravel 函数之后：')
print(a.ravel(order='F'))
print('\n')

a = np.arange(12).reshape(3, 4)
 
print('原数组：')
print(a)
print('\n')
 
print('对换数组：')
print(a.T)


print("+++++++++++++++++\n")

# 创建了三维的 ndarray
a = np.arange(8).reshape(2, 2, 2)
 
print('原数组：')
print(a)
print('获取数组中一个值：')
print(np.where(a == 6))
print(a[1, 1, 0])  # 为 6
print('\n')
 
 
# 将轴 2 滚动到轴 0（宽度到深度）
 
print('调用 rollaxis 函数：')
b = np.rollaxis(a, 2, 0)
print(b)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [0, 1, 1]
# 最后一个 0 移动到最前面
print(np.where(b == 6))
print('\n')
 
# 将轴 2 滚动到轴 1：（宽度到高度）
 
print('调用 rollaxis 函数：')
c = np.rollaxis(a, 2, 1)
print(c)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [1, 0, 1]
# 最后的 0 和 它前面的 1 对换位置
print(np.where(c == 6))
print('\n')

a = np.arange(8)
print('原始数组：')
print(a)
print('\n')
 
b = a.reshape(4, 2)
print('修改后的数组：')
print(b)

a = np.arange(9).reshape(3, 3)
print('原始数组：')
for row in a:
    print(row)
 
# 对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
print('迭代后的数组：')
for element in a.flat:
    print(element)
a = np.arange(8).reshape(2, 4)
 
print('原数组：')
print(a)
print('\n')
# 默认按行
 
print('展开的数组：')
print(a.flatten())
print('\n')
 
print('以 F 风格顺序展开的数组：')
print(a.flatten(order='F'))

a = np.arange(8).reshape(2, 4)
 
print('原数组：')
print(a)
print('\n')
 
print('调用 ravel 函数之后：')
print(a.ravel())
print('\n')
 
print('以 F 风格顺序调用 ravel 函数之后：')
print(a.ravel(order='F'))

a = np.arange(12).reshape(3, 4)
 
print('原数组：')
print(a)
print('\n')
 
print('对换数组：')
print(np.transpose(a))
a = np.arange(12).reshape(3, 4)
 
print('原数组：')
print(a)
print('\n')
 
print('转置数组：')
print(a.T)

# 创建了三维的 ndarray
a = np.arange(8).reshape(2, 2, 2)
 
print('原数组：')
print(a)
print('获取数组中一个值：')
print(np.where(a == 6))
print(a[1, 1, 0])  # 为 6
print('\n')
 
 
# 将轴 2 滚动到轴 0（宽度到深度）
 
print('调用 rollaxis 函数：')
b = np.rollaxis(a, 2, 0)
print(b)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [0, 1, 1]
# 最后一个 0 移动到最前面
print(np.where(b == 6))
print('\n')
 
# 将轴 2 滚动到轴 1：（宽度到高度）
 
print('调用 rollaxis 函数：')
c = np.rollaxis(a, 2, 1)
print(c)
# 查看元素 a[1,1,0]，即 6 的坐标，变成 [1, 0, 1]
# 最后的 0 和 它前面的 1 对换位置
print(np.where(c == 6))
print('\n')

# 创建了三维的 ndarray
a = np.arange(8).reshape(2, 2, 2)
 
print('原数组：')
print(a)
print('\n')
# 现在交换轴 0（深度方向）到轴 2（宽度方向）
 
print('调用 swapaxes 函数后的数组：')
print(np.swapaxes(a, 2, 0))

print('')
print("++++++++++++++++++++++++++++++++++++++++++++\n")

x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])
 
# 对 y 广播 x
b = np.broadcast(x, y)
# 它拥有 iterator 属性，基于自身组件的迭代器元组
 
print('对 y 广播 x：')
r, c = b.iters
 
# Python3.x 为 next(context) ，Python2.x 为 context.next()
print(next(r), next(c))
print(next(r), next(c))
print('\n')
# shape 属性返回广播对象的形状
 
print('广播对象的形状：')
print(b.shape)
print('\n')
# 手动使用 broadcast 将 x 与 y 相加
b = np.broadcast(x, y)
c = np.empty(b.shape)
 
print('手动使用 broadcast 将 x 与 y 相加：')
print(c.shape)
print('\n')
c.flat = [u + v for (u, v) in b]
 
print('调用 flat 函数：')
print(c)
print('\n')
# 获得了和 NumPy 内建的广播支持相同的结果
 
print('x 与 y 的和：')
print(x + y)


print('')
print("+++++++++++++++++++++++++++++++++\n")
a = np.arange(4).reshape(1, 4)
 
print('原数组：')
print(a)
print('\n')
 
print('调用 broadcast_to 函数之后：')
print(np.broadcast_to(a, (4, 4)))

print('\n')
print("+++++++++++++++++++++++++++++++++\n")
x = np.array(([1, 2], [3, 4]))
 
print('数组 x：')
print(x)
print('\n')
y = np.expand_dims(x, axis=0)
 
print('数组 y：')
print(y)
print('\n')
 
print('数组 x 和 y 的形状：')
print(x.shape, y.shape)
print('\n')
# 在位置 1 插入轴
y = np.expand_dims(x, axis=1)
 
print('在位置 1 插入轴之后的数组 y：')
print(y)
print('\n')
 
print('x.ndim 和 y.ndim：')
print(x.ndim, y.ndim)
print('\n')
 
print('x.shape 和 y.shape：')
print(x.shape, y.shape)
mark_print()
x = np.arange(9).reshape(1, 3, 3)
 
print('数组 x：')
print(x)
print('\n')
y = np.squeeze(x)
 
print('数组 y：')
print(y)
print('\n')
 
print('数组 x 和 y 的形状：')
print(x.shape, y.shape)

mark_print()
a = np.array([[1, 2], [3, 4]])
 
print('第一个数组：')
print(a)
print('\n')
b = np.array([[5, 6], [7, 8]])
 
print('第二个数组：')
print(b)
print('\n')
# 两个数组的维度相同
 
print('沿轴 0 连接两个数组：')
print(np.concatenate((a, b)))
print('\n')
 
print('沿轴 1 连接两个数组：')
print(np.concatenate((a, b), axis=1))
mark_print()
a = np.array([[1, 2], [3, 4]])
 
print('第一个数组：')
print(a)
print('\n')
b = np.array([[5, 6], [7, 8]])
 
print('第二个数组：')
print(b)
print('\n')
 
print('沿轴 0 堆叠两个数组：')
print(np.stack((a, b), axis=0))
print('\n')
 
print('沿轴 1 堆叠两个数组：')
print(np.stack((a, b), axis=1))

mark_print()
a = np.array([[1, 2], [3, 4]])
 
print('第一个数组：')
print(a)
print('\n')
b = np.array([[5, 6], [7, 8]])
 
print('第二个数组：')
print(b)
print('\n')
 
print('水平堆叠：')
c = np.hstack((a, b))
print(c)
print('\n')

mark_print()
a = np.array([[1, 2], [3, 4]])
 
print('第一个数组：')
print(a)
print('\n')
b = np.array([[5, 6], [7, 8]])
 
print('第二个数组：')
print(b)
print('\n')
 
print('竖直堆叠：')
c = np.vstack((a, b))
print(c)

mark_print()
a = np.arange(9)
 
print('第一个数组：')
print(a)
print('\n')
 
print('将数组分为三个大小相等的子数组：')
b = np.split(a, 3)
print(b)
print('\n')
 
print('将数组在一维数组中表明的位置分割：')
b = np.split(a, [4, 7])
print(b)

mark_print()
a = np.arange(16).reshape(4, 4)
print('第一个数组：')
print(a)
print('\n')
print('默认分割（0轴）：')
b = np.split(a, 2)
print(b)
print('\n')

print('沿水平方向分割：')
c = np.split(a, 2, axis=1)
print(c)
print('\n')

print('沿水平方向分割：')
d = np.hsplit(a, 2)
print(d)

mark_print()
harr = np.floor(10 * np.random.random((2, 6)))
print('原array：')
print(harr)
 
print('拆分后：')
print(np.hsplit(harr, 3))

mark_print()
a = np.arange(16).reshape(4, 4)
 
print('第一个数组：')
print(a)
print('\n')
 
print('竖直分割：')
b = np.vsplit(a, 2)
print(b)

mark_print()
a = np.array([[1, 2, 3], [4, 5, 6]])
 
print('第一个数组：')
print(a)
print('\n')
 
print('第一个数组的形状：')
print(a.shape)
print('\n')
b = np.resize(a, (3, 2))
 
print('第二个数组：')
print(b)
print('\n')
 
print('第二个数组的形状：')
print(b.shape)
print('\n')
# 要注意 a 的第一行在 b 中重复出现，因为尺寸变大了
 
print('修改第二个数组的大小：')
b = np.resize(a, (3, 3))
print(b)

mark_print()
a = np.array([[1, 2, 3], [4, 5, 6]])
 
print('第一个数组：')
print(a)
print('\n')
 
print('向数组添加元素：')
print(np.append(a, [7, 8, 9]))
print('\n')
 
print('沿轴 0 添加元素：')
print(np.append(a, [[7, 8, 9]], axis=0))
print('\n')
 
print('沿轴 1 添加元素：')
print(np.append(a, [[5, 5, 5], [7, 8, 9]], axis=1))

mark_print()
a = np.array([[1, 2], [3, 4], [5, 6]])
 
print('第一个数组：')
print(a)
print('\n')
 
print('未传递 Axis 参数。 在删除之前输入数组会被展开。')
print(np.insert(a, 3, [11, 12]))
print('\n')
print('传递了 Axis 参数。 会广播值数组来配输入数组。')
 
print('沿轴 0 广播：')
print(np.insert(a, 1, [11], axis=0))
print('\n')
 
print('沿轴 1 广播：')
print(np.insert(a, 1, 11, axis=1))

mark_print()
a = np.arange(12).reshape(3, 4)
 
print('第一个数组：')
print(a)
print('\n')
 
print('未传递 Axis 参数。 在插入之前输入数组会被展开。')
print(np.delete(a, 5))
print('\n')
 
print('删除第二列：')
print(np.delete(a, 1, axis=1))
print('\n')
 
print('包含从数组中删除的替代值的切片：')
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(np.delete(a, np.s_[::2]))

mark_print()
# 取第0~3行，列方向每隔一列
idx = np.s_[0:3, ::2]
print(idx)
# 输出： (slice(0, 3, None), slice(None, None, 2))

arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print(arr_2d[idx])

mark_print()
a = np.array([5, 2, 6, 2, 7, 5, 6, 8, 2, 9])
 
print('第一个数组：')
print(a)
print('\n')
 
print('第一个数组的去重值：')
u = np.unique(a)
print(u)
print('\n')
 
print('去重数组的索引数组：')
u, indices = np.unique(a, return_index=True)
print(indices)
print('\n')
 
print('我们可以看到每个和原数组下标对应的数值：')
print(a)
print('\n')
 
print('去重数组的下标：')
u, indices = np.unique(a, return_inverse=True)
print(u)
print('\n')
 
print('下标为：')
print(indices)
print('\n')
 
print('使用下标重构原数组：')
print(u[indices])
print('\n')
 
print('返回去重元素的重复数量：')
u, counts = np.unique(a, return_counts=True)
print(u)
print(counts)

mark_print()
arr1 = np.array([True, False, True], dtype=bool)
arr2 = np.array([False, True, False], dtype=bool)

result_and = np.bitwise_and(arr1, arr2)
result_or = np.bitwise_or(arr1, arr2)
result_xor = np.bitwise_xor(arr1, arr2)
result_not = np.bitwise_not(arr1)

print("AND:", result_and)  # [False, False, False]
print("OR:", result_or)    # [True, True, True]
print("XOR:", result_xor)  # [True, True, True]
print("NOT:", result_not)  # [False, True, False]

# 按位取反
arr_invert = np.invert(np.array([1, 2], dtype=np.int8))
print("Invert:", arr_invert)  # [-2, -3]

# 左移位运算
arr_left_shift = np.left_shift(5, 2)
print("Left Shift:", arr_left_shift)  # 20

# 右移位运算
arr_right_shift = np.right_shift(10, 1)
print("Right Shift:", arr_right_shift)  # 5

mark_print()
print('13 和 17 的二进制形式：')
a, b = 13, 17
print(bin(a), bin(b))
print('\n')
 
print('13 和 17 的位与：')
print(np.bitwise_and(13, 17))

mark_print()
a, b = 13, 17
print('13 和 17 的二进制形式：')
print(bin(a), bin(b))
 
print('13 和 17 的位或：')
print(np.bitwise_or(13, 17))

mark_print()
print('13 的位反转，其中 ndarray 的 dtype 是 uint8：')
print(np.invert(np.array([13], dtype=np.uint8)))
print('\n')
# 比较 13 和 242 的二进制表示，我们发现了位的反转
 
print('13 的二进制表示：')
print(np.binary_repr(13, width=8))
print('\n')
 
print('242 的二进制表示：')
print(np.binary_repr(242, width=8))


mark_print()
print('将 10 左移两位：')
print(np.left_shift(10, 2))
print('\n')
 
print('10 的二进制表示：')
print(np.binary_repr(10, width=8))
print('\n')
 
print('40 的二进制表示：')
print(np.binary_repr(40, width=8))
#  '00001010' 中的两位移动到了左边，并在右边添加了两个 0。
print('将 40 右移两位：')
print(np.right_shift(40, 2))
print('\n')
 
print('40 的二进制表示：')
print(np.binary_repr(40, width=8))
print('\n')
 
print('10 的二进制表示：')
print(np.binary_repr(10, width=8))
#  '00001010' 中的两位移动到了右边，并在左边添加了两个 0。

mark_print()
print('连接两个字符串：')
print(np.char.add(['hello'], [' xyz']))
print('\n')
 
print('连接示例：')
print(np.char.add(['hello', 'hi'], [' abc', ' xyz']))
print(np.char.multiply('Runoob ', 3))

print(np.char.center('Runoob', 20, fillchar='*'))
print(np.char.capitalize('runoob'))
print(np.char.title('i like runoob'))
# 操作数组
print(np.char.lower(['RUNOOB', 'GOOGLE']))
 
# 操作字符串
print(np.char.lower('RUNOOB'))
# 操作数组
print(np.char.upper(['runoob', 'google']))
 
# 操作字符串
print(np.char.upper('runoob'))

# 分隔符默认为空格
print(np.char.split('i like runoob?'))
# 分隔符为 .
print(np.char.split('www.runoob.com', sep='.'))
# 换行符 \n
print(np.char.splitlines('i\nlike runoob?'))
print(np.char.splitlines('i\rlike runoob?'))
# 移除字符串头尾的 a 字符
print(np.char.strip('ashok arunooba', 'a'))
 
# 移除数组元素头尾的 a 字符
print(np.char.strip(['arunooba', 'admin', 'java'], 'a'))
# 操作字符串
print(np.char.join(':', 'runoob'))
 
# 指定多个分隔符操作数组元素
print(np.char.join([':', '-'], ['runoob', 'google']))

print(np.char.replace('i like runoob', 'oo', 'cc'))
a = np.char.encode('runoob', 'cp500')
print(a)

a = np.char.encode('runoob', 'cp500')
print(a)
print(np.char.decode(a, 'cp500'))

a = np.array([0, 30, 45, 60, 90])
print('不同角度的正弦值：')
# 通过乘 pi/180 转化为弧度
print(np.sin(a*np.pi/180))
print('\n')
print('数组中角度的余弦值：')
print(np.cos(a*np.pi/180))
print('\n')
print('数组中角度的正切值：')
print(np.tan(a*np.pi/180))

mark_print()
a = np.array([0, 30, 45, 60, 90])
print('含有正弦值的数组：')
sin = np.sin(a*np.pi/180)
print(sin)
print('\n')
print('计算角度的反正弦，返回值以弧度为单位：')
inv = np.arcsin(sin)
print(inv)
print('\n')
print('通过转化为角度制来检查结果：')
print(np.degrees(inv))
print('\n')
print('arccos 和 arctan 函数行为类似：')
cos = np.cos(a*np.pi/180)
print(cos)
print('\n')
print('反余弦：')
inv = np.arccos(cos)
print(inv)
print('\n')
print('角度制单位：')
print(np.degrees(inv))
print('\n')
print('tan 函数：')
tan = np.tan(a*np.pi/180)
print(tan)
print('\n')
print('反正切：')
inv = np.arctan(tan)
print(inv)
print('\n')
print('角度制单位：')
print(np.degrees(inv))

mark_print()
a = np.array([1.0, 5.55, 123, 0.567, 25.532])
print('原数组：')
print(a)
print('\n')
print('舍入后：')
print(np.around(a))
print(np.around(a, decimals=1))
print(np.around(a, decimals=-1))

mark_print()
a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
print('提供的数组：')
print(a)
print('\n')
print('修改后的数组：')
print(np.floor(a))

mark_print()
a = np.array([-1.7, 1.5, -0.2, 0.6, 10])
print('提供的数组：')
print(a)
print('\n')
print('修改后的数组：')
print(np.ceil(a))
mark_print()
a = np.arange(9, dtype=np.float64).reshape(3, 3)
print('第一个数组：')
print(a)
print('\n')
print('第二个数组：')
b = np.array([10, 10, 10])
print(b)
print('\n')
print('两个数组相加：')
print(np.add(a, b))
print('\n')
print('两个数组相减：')
print(np.subtract(a, b))
print('\n')
print('两个数组相乘：')
print(np.multiply(a, b))
print('\n')
print('两个数组相除：')
print(np.divide(a, b))

mark_print()
a = np.array([0.25, 1.33, 1, 100])
print('我们的数组是：')
print(a)
print('\n')
print('调用 reciprocal 函数：')
print(np.reciprocal(a))
mark_print()
a = np.array([10, 100, 1000])
print('我们的数组是；')
print(a)
print('\n')
print('调用 power 函数：')
print(np.power(a, 2))
print('\n')
print('第二个数组：')
b = np.array([1, 2, 3])
print(b)
print('\n')
print('再次调用 power 函数：')
print(np.power(a, b))
mark_print()
a = np.array([10, 20, 30])
b = np.array([3, 5, 7])
print('第一个数组：')
print(a)
print('\n')
print('第二个数组：')
print(b)
print('\n')
print('调用 mod() 函数：')
print(np.mod(a, b))
print('\n')
print('调用 remainder() 函数：')
print(np.remainder(a, b))

mark_print()
a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 amin() 函数：')
print(np.amin(a, 1))
print('\n')
print('再次调用 amin() 函数：')
print(np.amin(a, 0))
print('\n')
print('调用 amax() 函数：')
print(np.amax(a))
print('\n')
print('再次调用 amax() 函数：')
print(np.amax(a, axis=0))

mark_print()
a = np.array([[3, 7, 5], [8, 4, 3], [2, 4, 9]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 ptp() 函数：')
print(np.ptp(a))
print('\n')
print('沿轴 1 调用 ptp() 函数：')
print(np.ptp(a, axis=1))
print('\n')
print('沿轴 0 调用 ptp() 函数：')
print(np.ptp(a, axis=0))

mark_print()
a = np.array([[10, 7, 4], [3, 2, 1]])
print('我们的数组是：')
print(a)
 
print('调用 percentile() 函数：')
# 50% 的分位数，就是 a 里排序之后的中位数
print(np.percentile(a, 50))
 
# axis 为 0，在纵列上求
print(np.percentile(a, 50, axis=0))
 
# axis 为 1，在横行上求
print(np.percentile(a, 50, axis=1))
 
# 保持维度不变
print(np.percentile(a, 50, axis=1, keepdims=True))

mark_print()
a = np.array([[30, 65, 70], [80, 95, 10], [50, 90, 60]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 median() 函数：')
print(np.median(a))
print('\n')
print('沿轴 0 调用 median() 函数：')
print(np.median(a, axis=0))
print('\n')
print('沿轴 1 调用 median() 函数：')
print(np.median(a, axis=1))

mark_print()
a = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 mean() 函数：')
print(np.mean(a))
print('\n')
print('沿轴 0 调用 mean() 函数：')
print(np.mean(a, axis=0))
print('\n')
print('沿轴 1 调用 mean() 函数：')
print(np.mean(a, axis=1))

mark_print()
a = np.array([1, 2, 3, 4])
print('我们的数组是：')
print(a)
print('\n')
print('调用 average() 函数：')
print(np.average(a))
print('\n')
# 不指定权重时相当于 mean 函数
wts = np.array([4, 3, 2, 1])
print('再次调用 average() 函数：')
print(np.average(a, weights=wts))
print('\n')
# 如果 returned 参数设为 true，则返回权重的和
print('权重的和：')
print(np.average([1, 2, 3, 4], weights=wts, returned=True))
print()

mark_print()
a = np.arange(6).reshape(3, 2)
print('我们的数组是：')
print(a)
print('\n')
print('修改后的数组：')
wt = np.array([3, 5])
print(np.average(a, axis=1, weights=wt))
print('\n')
print('修改后的数组：')
print(np.average(a, axis=1, weights=wt, returned=True))
print(np.std(a, axis=1))
mark_print()
print(np.var(a, axis=1))
mark_print()
a = np.array([[3, 7], [9, 1]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 sort() 函数：')
print(np.sort(a))
print('\n')
print('按列排序：')
print(np.sort(a, axis=0))
print('\n')
# 在 sort 函数中排序字段
dt = np.dtype([('name', 'S10'), ('age', int)])
a = np.array([("raju", 21), ("anil", 25), ("ravi", 17), ("amar", 27)], dtype=dt)
print('我们的数组是：')
print(a)
print('\n')
print('按 name 排序：')
print(np.sort(a, order='name'))

mark_print()
x = np.array([3, 1, 2])
print('我们的数组是：')
print(x)
print('\n')
print('对 x 调用 argsort() 函数：')
y = np.argsort(x)
print(y)
print('\n')
print('以排序后的顺序重构原数组：')
print(x[y])
print('\n')
print('使用循环重构原数组：')
for i in y:
    print(x[i], end=" ")

mark_print()
nm = ['raju', 'anil', 'ravi', 'amar']
dv = ['f.y.1', 's.y.2', 's.y.3', 'f.y.4']
ind = np.lexsort((dv, nm))
print('调用 lexsort() 函数：')
print(ind)
print('\n')
print('使用这个索引来获取排序后的数据：')
print([nm[i] + ", " + dv[i] for i in ind])

mark_print()
# 复数排序：
a = np.sort_complex([5, 3, 6, 2, 1])
print(a)
b = np.sort_complex([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
print(b)
# partition() 分区排序：
c = np.array([3, 4, 2, 1])
print(np.partition(c, 3))
d = np.partition(c, (1, 3))
print(d)
# 找到数组的第 3 小（index=2）的值和第 2 大（index=-2）的值
arr = np.array([46, 57, 23, 39, 1, 10, 0, 120])
print(arr[np.argpartition(arr, 2)[2]])
e = arr[np.argpartition(arr, -2)[-2]]
print(e)
# 同时找到第 3 和第 4 小的值。注意这里，用 [2,3] 同时将第 3 和第 4 小的排序好，然后可以分别通过下标 [2] 和 [3] 取得。
f = arr[np.argpartition(arr, [2, 3])[2]]
print(f)
g = arr[np.argpartition(arr, [2, 3])[3]]
print(g)

mark_print()
a = np.array([[30, 40, 70], [80, 20, 10], [50, 90, 60]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 argmax() 函数：')
print(np.argmax(a))
print('\n')
print('展开数组：')
print(a.flatten())
print('\n')
print('沿轴 0 的最大值索引：')
maxindex = np.argmax(a, axis=0)
print(maxindex)
print('\n')
print('沿轴 1 的最大值索引：')
maxindex = np.argmax(a, axis=1)
print(maxindex)
print('\n')
print('调用 argmin() 函数：')
minindex = np.argmin(a)
print(minindex)
print('\n')
print('展开数组中的最小值：')
print(a.flatten()[minindex])
print('\n')
print('沿轴 0 的最小值索引：')
minindex = np.argmin(a, axis=0)
print(minindex)
print('\n')
print('沿轴 1 的最小值索引：')
minindex = np.argmin(a, axis=1)
print(minindex)

mark_print()
a = np.array([[30, 40, 0], [0, 20, 10], [50, 0, 60]])
print('我们的数组是：')
print(a)
print('\n')
print('调用 nonzero() 函数：')
print(np.nonzero(a))
mark_print()
x = np.arange(9.).reshape(3, 3)
print('我们的数组是：')
print(x)
print('大于 3 的元素的索引：')
y = np.where(x > 3)
print(y)
print('使用这些索引来获取满足条件的元素：')
print(x[y])

mark_print()
x = np.arange(9.).reshape(3, 3)
print('我们的数组是：')
print(x)
# 定义条件, 选择偶数元素
condition = np.mod(x, 2) == 0
print('按元素的条件值：')
print(condition)
print('使用条件提取元素：')
print(np.extract(condition, x))

mark_print()
a = np.array([1, 256, 8755], dtype=np.int16)
print('我们的数组是：')
print(a)
print('以十六进制表示内存中的数据：')
print(map(hex, a))
print(map(hex, b))

# byteswap() 函数通过传入 true 来原地交换
print('调用 byteswap() 函数：')
print(a.byteswap(True))
print(b.byteswap(True))

print('十六进制形式：')
print(map(hex, a))
print(map(hex, b))
# 我们可以看到字节已经交换了
mark_print()
a = np.arange(6)
print('我们的数组是：')
print(a)
print('调用 id() 函数：')
print(id(a))
print('a 赋值给 b：')
b = a
print(b)
print('b 拥有相同 id()：')
print(id(b))
print('修改 b 的形状：')
b.shape = 3, 2
print(b)
print('a 的形状也修改了：')
print(a)

mark_print()
arr = np.arange(12)
print('我们的数组：')
print(arr)
print('创建切片：')
a = arr[3:]
b = arr[3:]
a[1] = 123
b[2] = 234
print(arr)
print(id(a), id(b), id(arr[3:]))
print(a)
print(b)
print(arr)
arr[1] = 123
arr[2] = 234
print(arr)

mark_print()
# 最开始 a 是个 3X2 的数组
a = np.arange(6).reshape(3, 2)
print('数组 a：')
print(a)
print('创建 a 的视图：')
b = a.view()
print(b)
print('两个数组的 id() 不同：')
print('a 的 id()：')
print(id(a))
print('b 的 id()：')
print(id(b))
# 修改 b 的形状，并不会修改 a
b.shape = 2, 3
print('b 的形状：')
print(b)
print('a 的形状：')
print(a)
mark_print()
a = np.array([[10, 10], [2, 3], [4, 5]])
print('数组 a：')
print(a)
print('创建 a 的深层副本：')
b = a.copy()
print('数组 b：')
print(b)
# b 与 a 不共享任何内容
print('我们能够写入 b 来写入 a 吗？')
print(b is a)
print('修改 b 的内容：')
b[0, 0] = 100
print('修改后的数组 b：')
print(b)
print('a 保持不变：')
print(a)

mark_print()
a = np.arange(4)
b = a
# 改变a中第一个元素的值
b[0] = 9
print(a)
print(b)

mark_print()
a = np.arange(12).reshape(3, 4)
 
print('原数组：')
print(a)
print('\n')
 
print('转置数组：')
print(a.T)

mark_print()

# import rsnumpy.matlib
print(np.matlib.empty((2, 2)))
print()
print(np.matlib.zeros((2, 2)))
print()
print(np.matlib.ones((2, 2)), "\n")
print(np.matlib.eye(n=3, M=4, k=0, dtype=float), "\n")
print(np.matlib.identity(5, dtype=float), "\n")
print(np.matlib.rand(3, 3), "\n")
i = np.matrix('1,2;3,4', dtype=int)
print(i, "\n")
j = np.asarray(i)
print(j, "\n")
k = np.asmatrix(j)
print(k)

mark_print()
a = np.array([[1, 2], [3, 4]])
b = np.array([[11, 12], [13, 14]])
print("aaaa", np.dot(a, b))
print()
a = np.array([[1, 2], [3, 4]])
b = np.array([[11, 12], [13, 14]])
 
# vdot 将数组展开计算内积
print(np.vdot(a, b), "\n")
print(np.inner(np.array([1, 2, 3]), np.array([0, 1, 0])), "\n")

a = np.array([[1, 2], [3, 4]])
 
print('数组 a：')
print(a)
b = np.array([[11, 12], [13, 14]])
 
print('数组 b：')
print(b)
 
print('内积：')
print(np.inner(a, b))
print()

a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
print(np.matmul(a, b))

print()
a = [[1, 0], [0, 1]]
b = [1, 2]
print(np.matmul(a, b))
print(np.matmul(b, a))
print()
a = np.arange(8).reshape(2, 2, 2)
b = np.arange(4).reshape(2, 2)
print(np.matmul(a, b))

a = np.array([[1, 2], [3, 4]])
 
print(np.linalg.det(a), "\n")
b = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
print(b)
print(np.linalg.det(b))
print(6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - -2*2))
print()

x = np.array([[1, 2], [3, 4]])
y = np.linalg.inv(x)
print(x)
print(y)
print()
print(np.dot(x, y))

mark_print()
a = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
 
print('数组 a：')
print(a)
ainv = np.linalg.inv(a)
 
print('a 的逆：')
print(ainv)
 
print('矩阵 b：')
b = np.array([[6], [-4], [27]])
print(b)
 
print('计算：A^(-1)B：')
x = np.linalg.solve(a, b)
print(x)

# 这就是线性方向 x = 5, y = 3, z = -2 的解

mark_print()
a = np.array([1, 2, 3, 4, 5])
 
# 保存到 outfile.npy 文件上
np.save('./test/outfile.npy', a)
 
# 保存到 outfile2.npy 文件上，如果文件路径末尾没有扩展名 .npy，该扩展名会被自动加上
np.save('./test/outfile2', a)
b = np.load('./test/outfile.npy')
print(b)
mark_print()
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.arange(0, 1.0, 0.1)
c = np.sin(b)
# c 使用了关键字参数 sin_array
np.savez('./test/runoob.npz', a, b, sin_array=c)
r = np.load('./test/runoob.npz')
print(r.files)  # 查看各个数组名称
print(r["arr_0"])  # 数组 a
print(r["arr_1"])  # 数组 b
print(r["sin_array"])  # 数组 c

a = np.array([1, 2, 3, 4, 5])
np.savetxt('./test/out.txt', a)
b = np.loadtxt('./test/out.txt')
 
print(b)
mark_print()
a = np.arange(0, 10, 0.5).reshape(4, -1)
print(a)
print()
np.savetxt('./test/out.txt', a, fmt="%d", delimiter=",")  # 改为保存为整数，以逗号分隔
b = np.loadtxt('./test/out.txt', delimiter=",")  # load 时也要指定为逗号分隔
print(b)

print(np.__version__)
print('\n')

def run_test():
    print("\n------------------------")


run_test()
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)
a = np.array([1, 2, 3, 4, 5, 6])
a[0] = 10
print("a: ", a)
print("a[:3]: ", a[:3])
b = a[3:]
print("b: ", b)
b[0] = 40
print("a: ", a)
print("b: ", b)
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("a.ndim: ", a.ndim)
print("a[1, 3]: ", a[1, 3])
print("a.shape: ", a.shape)
print("len(a.shape) == a.ndim: ", len(a.shape) == a.ndim)
print("a.size: ", a.size)
print("a.size == math.prod(a.shape): ", a.size == math.prod(a.shape))
print("a.dtype: ", a.dtype)
print("np.zeros(2): ", np.zeros(2))
print("np.ones(2): ", np.ones(2))
print("np.empty(2): ", np.empty(2))
print("np.arange(4): ", np.arange(4))
print("np.arange(2, 9, 2): ", np.arange(2, 9, 2))
print("np.linspace(0, 10, num=5): ", np.linspace(0, 10, num=5))
x = np.ones(2, dtype=np.int64)
print("x: ", x)
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
ss = np.sort(arr)
print("np.sort(arr): ", ss)
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
np.concatenate((a, b))
print("np.concatenate((a, b)): ", np.concatenate((a, b)))
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
print("np.concatenate((x, y), axis=0): \n", np.concatenate((x, y), axis=0))
array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]]])
print("array_example.ndim: ", array_example.ndim)
print("array_example.shape: ", array_example.shape)
print("array_example.size: ", array_example.size)
print("array_example.dtype: ", array_example.dtype)
print("array_example: ", array_example)
a = np.arange(6)
print("a: ", a)
b = a.reshape(3, 2)
print("b: ", b)
print("np.reshape(a, shape=(1, 6), order='C'): ", np.reshape(a, shape=(1, 6), order='C'))
a = np.array([1, 2, 3, 4, 5, 6])
print("a.shape: ", a.shape)
a2 = a[np.newaxis, :]
print("a2.shape: ", a2.shape)
row_vector = a[np.newaxis, :]
print("row_vector.shape: ", row_vector.shape)
col_vector = a[:, np.newaxis]
print("col_vector.shape: ", col_vector.shape)
a = np.array([1, 2, 3, 4, 5, 6])
print("a.shape: ", a.shape)
b = np.expand_dims(a, axis=1)
print("b.shape: ", b.shape)
c = np.expand_dims(a, axis=0)
print("c.shape: ", c.shape)
data = np.array([1, 2, 3])
print("data[1]: ", data[1])
print("data[0:2]: ", data[0:2])
print("data[1:]: ", data[1:])
print("data[-2:]: ", data[-2:])
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("a[a < 5]: ", a[a < 5])
five_up = (a >= 5)
print("a[five_up]: ", a[five_up])
end_time = time.time()

print("\n时间：", end_time - start_time)
