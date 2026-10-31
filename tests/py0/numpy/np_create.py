"""numpy: creating, reshaping and combining arrays; dtypes."""
import numpy as np

print(repr(np.arange(5)), repr(np.arange(2, 10, 3)), repr(np.arange(0.0, 1.0, 0.1)), repr(np.arange(1, 2, 0.25)), repr(np.arange(5, 0, -2)))
print(repr(np.linspace(0, 1, 5)), repr(np.linspace(2, 3, 4, endpoint=False)), repr(np.linspace(0, 10, 1)))
print(repr(np.zeros(3)), repr(np.ones((2, 2), dtype="int64")), repr(np.full((2, 3), 7)), repr(np.full(2, 0.5)), repr(np.eye(3)), repr(np.eye(2, 3, 1)))
print(repr(np.identity(2)), repr(np.zeros_like(np.array([1, 2]))), repr(np.ones_like(np.array([1.5]))), repr(np.full_like(np.array([1, 2]), 9)))
a = np.array([[1, 2, 3], [4, 5, 6]])
print(repr(a.reshape(3, 2)), repr(a.reshape(-1)), repr(a.reshape((3, -1))), repr(a.flatten()), repr(a.T), repr(np.transpose(a)))
print(repr(a.astype("float64")), repr(np.array([1.7, -2.7]).astype("int64")), repr(np.array([0, 2]).astype("bool")), a.dtype, a.astype("float64").dtype)
print(repr(np.array([1, 2.5])), repr(np.array([[1.0], [2.0]])), repr(np.array([[[1, 2]], [[3, 4]]])), repr(np.array([1, 2], dtype="float64")))
x = np.array([1, 2])
y = np.array([3, 4])
print(repr(np.concatenate([x, y])), repr(np.concatenate([a, a], axis=1)), repr(np.stack([x, y])), repr(np.stack([x, y], axis=1)), repr(np.vstack([x, y])), repr(np.hstack([x, y])))
z = np.asarray(x)
z[0] = 100
print(x, repr(np.array(a)), a.ndim, a.size, len(a), a.shape[1], len(a.shape), a.shape)
c = a.copy()
c[0, 0] = -1
print(a[0, 0], c[0, 0], repr(a.ravel()), a.astype("float64").flatten().tolist())
for bad in [0, 1, 2]:
    try:
        if bad == 0:
            print(a.reshape(4, 2))
        elif bad == 1:
            print(np.array([[1, 2], [3]]))
        else:
            print(np.concatenate([x, a]))
    except ValueError as e:
        print("ValueError:", e)
