# Automatic differentiation with dual numbers and operator overloading:
# fit y = w x + b by gradient descent, derivatives computed exactly.
from dataclasses import dataclass


@dataclass
class Dual:
    val: float
    eps: float

    def __add__(self, o: "Dual") -> "Dual":
        return Dual(self.val + o.val, self.eps + o.eps)

    def __sub__(self, o: "Dual") -> "Dual":
        return Dual(self.val - o.val, self.eps - o.eps)

    def __mul__(self, o: "Dual") -> "Dual":
        return Dual(self.val * o.val, self.val * o.eps + self.eps * o.val)


def c(x: float) -> Dual:
    return Dual(x, 0.0)


def loss(w: Dual, b: Dual, xs: list[float], ys: list[float]) -> Dual:
    total = c(0.0)
    for x, y in zip(xs, ys):
        d = w * c(x) + b - c(y)
        total = total + d * d
    return total * c(1.0 / len(xs))


xs = [0.0, 1.0, 2.0, 3.0, 4.0]
ys = [1.1, 2.9, 5.2, 7.1, 8.8]
w = 0.0
b = 0.0
for step in range(301):
    dw = loss(Dual(w, 1.0), c(b), xs, ys).eps
    db = loss(c(w), Dual(b, 1.0), xs, ys).eps
    w -= 0.05 * dw
    b -= 0.05 * db
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss(c(w), c(b), xs, ys).val:.4f}  w={w:.3f} b={b:.3f}")
print(f"fitted line: y = {w:.2f} x + {b:.2f}")
