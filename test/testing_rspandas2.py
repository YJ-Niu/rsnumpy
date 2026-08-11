import time
start_time = time.time()

import rsnumpy as np  # noqa: E402
from collections import namedtuple  # noqa: E402
from dataclasses import make_dataclass  # noqa: E402
import rspandas as pd  # noqa: E402


def print_series(num, s):
    print("++++++++++++++++++++", num)
    print(s)
    print()


s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print_series(1, s)
print_series(2, s.index)
print_series(3, pd.Series(np.random.randn(5)))
d = {"b": 1, "a": 0, "c": 2}
print_series(4, pd.Series(d))
d = {"a": 0.0, "b": 1.0, "c": 2.0}
print_series(5, pd.Series(d))
print_series(6, pd.Series(d, index=["b", "c", "d", "a"]))
print_series(7, pd.Series(5.0, index=["a", "b", "c", "d", "e"]))
print_series(8, s.iloc[0])
print_series(9, s.iloc[:3])
print_series(10, s[s > s.median()])
print_series(11, s.iloc[[4, 3, 1]])
print_series(12, np.exp(s))
print_series(13, s.dtype)
print_series(14, s.array)
print_series(15, s.to_numpy())
print_series(16, s["a"])
s["e"] = 12.0
print_series(17, s)
print_series(18, "e" in s)
print_series(19, "f" in s)
print_series(20, s.get("f"))
print_series(21, s.get("f", np.nan))
print_series(22, s + s)
print_series(23, np.exp(s))
print_series(24, s.iloc[1:] + s.iloc[:-1])
s = pd.Series(np.random.randn(5), name="something")
print_series(25, s)
print_series(26, s.name)
s2 = s.rename("different")
print_series(27, s2.name)
d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}
df = pd.DataFrame(d)
print_series(28, df)
print_series(29, pd.DataFrame(d, index=["d", "b", "a"]))
print_series(30, pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"]))
print_series(31, df.index)
print_series(32, df.columns)
d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}
print_series(33, pd.DataFrame(d))
print_series(34, pd.DataFrame(d, index=["a", "b", "c", "d"]))
data = np.zeros((2,), dtype=[("A", "i4"), ("B", "f4"), ("C", "S10")])
data[:] = [(1, 2.0, "Hello"), (2, 3.0, "World")]
print_series(35, pd.DataFrame(data))
print_series(36, pd.DataFrame(data, index=["first", "second"]))
print_series(37, pd.DataFrame(data, columns=["C", "A", "B"]))
data2 = [{"a": 1, "b": 2}, {"a": 5, "b": 10, "c": 20}]
print_series(38, pd.DataFrame(data2))
print_series(39, pd.DataFrame(data2, index=["first", "second"]))
print_series(40, pd.DataFrame(data2, columns=["a", "b"]))
print_series(41, pd.DataFrame(
    {
        ("a", "b"): {("A", "B"): 1, ("A", "C"): 2},
        ("a", "a"): {("A", "C"): 3, ("A", "B"): 4},
        ("a", "c"): {("A", "B"): 5, ("A", "C"): 6},
        ("b", "a"): {("A", "C"): 7, ("A", "B"): 8},
        ("b", "b"): {("A", "D"): 9, ("A", "B"): 10},
    }
))
ser = pd.Series(range(3), index=list("abc"), name="ser")
print_series(42, pd.DataFrame(ser))
Point = namedtuple("Point", "x y")
print_series(43, pd.DataFrame([Point(0, 0), Point(0, 3), (2, 3)]))
Point3D = namedtuple("Point3D", "x y z")
print_series(44, pd.DataFrame([Point3D(0, 0, 0), Point3D(0, 3, 5), Point(2, 3)]))
Point = make_dataclass("Point", [("x", int), ("y", int)])
print_series(45, pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)]))
print_series(46, pd.DataFrame.from_dict(dict([("A", [1, 2, 3]), ("B", [4, 5, 6])])))
print_series(47, pd.DataFrame.from_dict(
    dict([("A", [1, 2, 3]), ("B", [4, 5, 6])]),
    orient="index",
    columns=["one", "two", "three"],
))
print_series(48, data)
print_series(49, pd.DataFrame.from_records(data, index="C"))
print_series(50, df["one"])
df["three"] = df["one"] * df["two"]
print_series(51, df)
del df["two"]
three = df.pop("three")
print_series(52, df)
df["foo"] = "bar"
print_series(53, df)
df["one_trunc"] = df["one"][:2]
print_series(54, df)
df.insert(1, "bar", df["one"])
print_series(55, df)
iris = pd.read_csv("./testing/data/iris.data")
print_series(56, iris.head())
print_series(57, iris.assign(sepal_ratio=iris["SepalWidth"] / iris["SepalLength"]).head())
print_series(58, iris.assign(sepal_ratio=lambda x: (x["SepalWidth"] / x["SepalLength"])).head())
print_series(59, iris.assign(sepal_ratio=pd.col("SepalWidth") / pd.col("SepalLength")).head())
(
    iris.query("SepalLength > 5")
    .assign(
        SepalRatio=lambda x: x.SepalWidth / x.SepalLength,
        PetalRatio=lambda x: x.PetalWidth / x.PetalLength,
    )
    .plot(kind="scatter", x="SepalRatio", y="PetalRatio")
)
dfa = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print_series(60, dfa.assign(C=lambda x: x["A"] + x["B"], D=lambda x: x["A"] + x["C"]))
print_series(61, df.loc["b"])
print_series(62, df.iloc[2])
df = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=["A", "B", "C"])
print_series(63, df + df2)
print_series(64, df - df.iloc[0])
print_series(65, df * 5 + 2)
print_series(66, 1 / df)
print_series(67, df ** 4)
df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print_series(68, df1 & df2)
print_series(69, df1 | df2)
print_series(70, df1 ^ df2)
print_series(71, -df1)
print_series(72, df[:5].T)
print_series(73, np.exp(df))
print_series(74, np.asarray(df))
ser = pd.Series([1, 2, 3, 4])
print_series(75, np.exp(ser))
ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])
print_series(76, ser1)
print_series(77, ser2)
print_series(78, np.remainder(ser1, ser2))
ser3 = pd.Series([2, 4, 6], index=["b", "c", "d"])
print_series(79, ser3)
print_series(80, np.remainder(ser1, ser3))
ser = pd.Series([1, 2, 3])
print_series(81, ser)
idx = pd.Index([4, 5, 6])
print_series(82, np.maximum(ser, idx))
baseball = pd.read_csv("./testing/data/baseball.csv")
print_series(83, baseball)
print_series(84, baseball.info())
print_series(85, baseball.iloc[-20:, :12].to_string())
print_series(86, pd.DataFrame(np.random.randn(3, 12)))
pd.set_option("display.width", 40)
print_series(87, pd.DataFrame(np.random.randn(3, 12)))
datafile = {
    "filename": ["filename_01", "filename_02"],
    "path": [
        "./testing/rspandas/filename_01",
        "./testing/rspandas/filename_02",
    ],
}
pd.set_option("display.max_colwidth", 30)
print_series(88, pd.DataFrame(datafile))
pd.set_option("display.max_colwidth", 100)
print_series(89, pd.DataFrame(datafile))
df = pd.DataFrame({"foo1": np.random.randn(5), "foo2": np.random.randn(5)})
print_series(90, df)
print_series(91, df.foo1)
index = pd.date_range("1/1/2000", periods=8)
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=["A", "B", "C"])
long_series = pd.Series(np.random.randn(1000))
print_series(92, long_series.head())
print_series(93, long_series.tail(3))
print_series(94, df[:2])
df.columns = [x.lower() for x in df.columns]
print_series(95, df)
print_series(96, s.array)
print_series(97, s.index.array)
print_series(98, s.to_numpy())
print_series(99, np.asarray(s))
end_time = time.time()
print("end_time - start_time:", end_time - start_time)
