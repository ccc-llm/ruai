"""numpy: indexing: ints, slices (views that share memory), masks, index arrays, assignment."""
import numpy as np

a = np.arange(12).reshape(3, 4)
print(a[1], a[-1, -1], a[:, 1], a[1:, ::2], a[::-1, 1:3], a[..., 0], a[None, 0].shape, a[:, None].shape)
v = a[0]
v[1] = 100
print(a[0], v)
w = a[:, 2]
w[:] = -1
print(a)
c = np.arange(10)
print(c[c % 3 == 0], c[[1, 3, 3]], c[np.array([0, -1])], c[[True, False] * 5])
c[c > 6] = 0
c[[0, 1]] = np.array([9, 8])
c[2:4] = 7.9
print(c)
print(a[a > 5], a[[0, 2]], a[[0, 2], 1], a[1, [0, 3]], a[[0, 1], [1, 2]], a[a[:, 0] > 0])
m = np.zeros((3, 3))
m[1] = 1.0
m[:, 2] = np.array([1.0, 2.0, 3.0])
m[0, 0] = 5
m[m == 0] = 0.5
print(m)
f = np.arange(6.0)
f[f > 2] += 0
print(f[1:4].sum(), f[::2], f[5], repr(f[5]), f[1:1])
for bad in [0, 1, 2, 3, 4]:
    try:
        if bad == 0:
            print(c[10])
        elif bad == 1:
            print(a[0, 0, 0])
        elif bad == 2:
            print(c[[True, False]])
        elif bad == 3:
            c[0:2] = np.array([1, 2, 3])
        else:
            print(a[[0, 5]])
    except IndexError as e:
        print("IndexError:", e)
    except ValueError as e:
        print("ValueError:", e)
