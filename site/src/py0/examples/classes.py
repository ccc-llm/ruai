# Dataclasses, inheritance and dunder methods.
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def norm(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


class Shape:
    def area(self) -> float:
        return 0.0

    def __str__(self) -> str:
        return "Shape"


class Rect(Shape):
    def __init__(self, w: float, h: float) -> None:
        self.w = w
        self.h = h

    def area(self) -> float:
        return self.w * self.h

    def __str__(self) -> str:
        return f"Rect({self.w} x {self.h})"


p = Point(3.0, 4.0)
print(p, p.norm())
shapes: list[Shape] = [Rect(2.0, 3.5), Rect(1.0, 1.0)]
for s in shapes:
    print(s, f"area={s.area():.2f}")
