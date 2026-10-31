"""numpy: arithmetic with broadcasting and NumPy 2 type promotion; comparisons; errors."""
import numpy as np

i = np.array([1, 2, 3])
f = np.array([0.5, -1.5, 2.0])
b = np.array([True, False, True])
m = np.arange(6).reshape(2, 3)
for x in [i + 1, i + 1.5, i * f, i / 2, i // 2, -i // 2, i % 2, -7 % i, i ** 2, f ** 2, 2 ** i, i - True]:
    print(repr(x))
print(repr(b + b), repr(b * b), repr(b + 1), repr(b * 2.5), repr(b & b), repr(b | ~b), repr(b ^ b), repr(i & 2))
print(repr(m + i), repr(m * np.array([[10], [20]])), repr(m - m.T.T))
print(repr(i < 2), repr(f >= 0), repr(i == np.array([1, 0, 3])), repr(i != 2), repr(-f), repr(abs(f)), repr(+i))
print(repr(f // 0.5), repr(f % 0.75), repr(np.array([-7.5, 7.5]) // 2), repr(np.array([-7.5, 7.5]) % -2))
print(repr(1 / np.array([1.0, 0.0, -1.0])), repr(np.array([0.0]) / 0.0), repr(np.array([5, 0]) // np.array([0, 0])))
print(repr(np.maximum(f, 0)), repr(np.minimum(i, f)), repr(np.clip(f, -1, 1)), repr(np.power(i, 2)))
print(repr(np.sqrt(i)), repr(np.exp(f)), repr(np.log(np.array([1.0, 10.0]))), repr(np.sin(f)), repr(np.tanh(f)))
print(repr(np.floor(f)), repr(np.ceil(f)), repr(np.round(np.array([0.5, 1.5, 2.675, -0.5]), 2)), repr(np.round(np.array([25, 35]), -1)))
print(repr(np.abs(np.array([-3, 4]))), repr(np.sign(f)), repr(np.square(i)), repr(np.isnan(np.array([1.0, np.nan]))))
print(repr(np.where(i > 1, i, 0)), repr(np.where(b, 1.5, i)))
s = np.array([2.0, 4.0]).sum()
print(repr(s * 2), repr(s + i), float(s) / 3, int(s), bool(np.array([1.0]) > 0))
for bad in ["shape", "neg", "bool-sub", "float-and", "pow", "truth"]:
    try:
        if bad == "shape":
            print(i + np.array([1, 2]))
        elif bad == "neg":
            print(-b)
        elif bad == "bool-sub":
            print(b - b)
        elif bad == "float-and":
            print(f & f)
        elif bad == "pow":
            print(i ** -1)
        else:
            print(bool(i > 0))
    except ValueError as e:
        print("ValueError:", e)
    except TypeError as e:
        print("TypeError:", e)
