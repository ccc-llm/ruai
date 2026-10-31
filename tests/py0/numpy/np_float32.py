"""numpy: float32 arrays: promotion, float32 arithmetic and sums, printing."""
import numpy as np

a = np.array([0.1, 0.2, 1 / 3], dtype="float32")
print(a, repr(a), a.dtype, a.sum(), repr(a.sum()), repr(a.mean()), repr(a.std()), repr(a.max()), repr(a[2]))
print(repr(a * 2.5), repr(a + np.array([1, 2, 3])), repr(a + 1), repr(np.sqrt(a)), repr(a / 3), repr(a + a.astype("float64")))
print(repr(np.array([1e-5, 1.0], dtype="float32")), repr(np.arange(30, dtype="float32") / 7))
print(repr(np.arange(12, dtype="float32").reshape(3, 4).sum(axis=0)), repr((np.arange(300, dtype="float32") * 0.1).sum()), repr(np.cumsum(a)))
b = np.zeros(3, dtype="float32")
b[0] = 0.1
b[1:] = np.array([2.5, 1e10])
print(repr(b), b, float(b[0]), repr(b @ b), repr(np.array([[1.5, 2.0]], dtype="float32") @ np.ones((2, 2), dtype="float32")))
print(repr(a > 0.15), repr(-a), repr(np.abs(-a)), repr(np.round(a, 2)), repr(a.astype("int64")), repr(np.exp(a)))
print(str(a[0]), repr(np.array([1.0, 2.0], dtype="float32").mean()), repr(np.full(2, 7.0, dtype="float32")))
