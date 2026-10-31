"""numpy: reductions over axes (NumPy's pairwise float sums), sorting, cumulative sums."""
import numpy as np

a = np.arange(20, dtype="float64") * 0.1
print(repr(a.sum()), repr(a.mean()), repr(a.std()), repr(a.var()), repr(a.std(ddof=1)))
r = a.reshape(4, 5)
print(repr(r.sum(axis=0)), repr(r.sum(axis=1)), repr(r.sum(axis=-1)), repr(r.mean(axis=0)), repr(r.max(axis=1)))
big = np.arange(1000) * 0.1
print(repr(big.sum()), repr(big.mean()), repr(np.sum(big * big)), repr(big.reshape(10, 100).sum(axis=1)))
t = np.linspace(0, 1, 301)
print(repr(t.sum()), repr(t.std()), repr(t.reshape(7, 43).sum(axis=0)[:3]))
i = np.array([[3, 1, 2], [9, 7, 8]])
print(repr(i.sum()), repr(i.prod()), repr(i.max()), repr(i.min(axis=0)), repr(i.argmax()), repr(i.argmin(axis=1)), repr(i.mean()))
print(repr(i.cumsum()), repr(i.cumsum(axis=1)), repr(i.cumprod(axis=0)), repr(np.array([True, True, False]).sum()))
print(repr(np.array([1.0, np.nan, 3.0]).max()), repr(np.array([1.0, np.nan]).argmax()), repr(np.array([[1, 0], [1, 1]]).all(axis=1)), repr(np.array([0, 0]).any()))
s = np.array([3.5, -1.0, 2.0, np.nan, 0.0])
print(repr(np.sort(s)), repr(np.argsort(np.array([3, 1, 2]))), repr(np.unique(np.array([3, 1, 3, 2, 1]))), repr(np.sort(i, axis=0)))
s2 = np.array([5, 3, 4])
s2.sort()
print(s2, np.count_nonzero(np.array([0, 1, 2, 0])), np.allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0000001])))
print(repr(np.mean(i, axis=0)), repr(np.std(i)), repr(np.var(i, axis=1)), repr(np.max(i)), repr(np.argmax(i, axis=0)))
try:
    print(np.array([], dtype="float64").max())
except ValueError as e:
    print("ValueError:", e)
