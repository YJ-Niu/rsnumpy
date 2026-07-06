"""Micro-benchmarks for rsnumpy hotspots (Python-side computation)."""
import time
import rsnumpy as np


def bench(name, fn, number=200):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(number):
        fn()
    dt = (time.perf_counter() - t0) / number
    print(f"{name:32s} {dt*1e6:10.2f} us/call")


def main():
    n = 2000
    data_list = list(range(n))
    nested = [[float(i * 7 % 13) for i in range(50)] for _ in range(50)]

    a = np.array([float((i * 131) % 997) for i in range(n)])
    arr_small = np.array([46, 57, 23, 39, 1, 10, 0, 120])

    bench("array(list[float])", lambda: np.array([float(x) for x in data_list]))
    bench("array(nested 50x50)", lambda: np.array(nested))
    bench("partition(n=8,k=3)", lambda: np.partition(arr_small, 3))
    bench("argpartition(n=8,k=2)", lambda: np.argpartition(arr_small, 2))
    bench("partition(n=2000,k=1000)", lambda: np.partition(a, 1000), number=50)
    bench("sort_complex(n=2000)", lambda: np.sort_complex(a.tolist()), number=50)
    bench("argmax(n=2000)", lambda: np.argmax(a))
    bench("argmin(n=2000)", lambda: np.argmin(a))
    bench("argsort(n=2000)", lambda: np.argsort(a), number=100)

    nm = ['raju', 'anil', 'ravi', 'amar'] * 100
    dv = ['f.y.1', 's.y.2', 's.y.3', 'f.y.4'] * 100
    bench("lexsort(n=400)", lambda: np.lexsort((dv, nm)), number=100)

    bs = np.array([1, 256, 8755, 42, 100, 7] * 50, dtype=np.int16)
    bench("byteswap(n=300)", lambda: bs.byteswap(False), number=200)

    m = np.arange(0, 60, 5).reshape(3, 4)

    def do_nditer():
        s = 0.0
        for x in np.nditer(m):
            s += x
        return s
    bench("nditer(3x4)", do_nditer, number=500)

    x = np.array([[1], [2], [3]])
    y = np.array([4, 5, 6])

    def do_broadcast():
        b = np.broadcast(x, y)
        return [u + v for (u, v) in b]
    bench("broadcast iterate(3x3)", do_broadcast, number=500)


if __name__ == "__main__":
    main()
