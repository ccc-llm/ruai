"""numpy: matrix products and np.linalg (inv, solve, det, norm)."""
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
v = np.array([1, 1])
print(repr(a @ b), repr(a @ v), repr(v @ a), repr(v @ v), repr(np.dot(a, b)), repr(a.dot(v)), repr(np.matmul(a, b)))
print(repr(np.outer(v, np.array([1, 2, 3]))), repr(np.trace(a)))
f = np.array([[2.0, 1.0], [1.0, 3.0]])
print(repr(np.linalg.inv(f)), repr(np.linalg.solve(f, np.array([3.0, 5.0]))), repr(np.linalg.det(np.array([[4.0, 0.0], [0.0, 0.5]]))))
print(repr(np.linalg.norm(np.array([3.0, 4.0]))), repr(np.linalg.norm(np.array([[1.0, 2.0], [2.0, 4.0]]))))
x = np.arange(6).reshape(2, 3)
print(repr(x @ x.T), repr(x.T @ x), repr(np.ones((2, 2, 3)) @ np.ones((3, 2))))
for bad in [0, 1]:
    try:
        if bad == 0:
            print(x @ x)
        else:
            print(np.linalg.inv(np.array([[1.0, 2.0], [2.0, 4.0]])))
    except ValueError as e:
        print("ValueError:", e)
