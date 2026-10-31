"""numpy: str / repr of arrays as NumPy's arrayprint (widths, exponents, wrapping, summarizing)."""
import numpy as np


def show(a: np.ndarray) -> None:
    print(a)
    print(repr(a))


show(np.array([1.0, 2.5, -3.25]))
show(np.array([1e-5, 1.0]))
show(np.array([1e8, 1.0]))
show(np.array([0.1 + 0.2]))
show(np.array([[1, -20], [300, 4]]))
show(np.array([True, False]))
show(np.array([1/3, 2/3]))
show(np.linspace(0, 1, 30))
show(np.arange(2000.0))
show(np.arange(1500).reshape(30, 50))
show(np.array([-0.0, 0.0]))
show(np.array([123456789.0]))
show(np.array([1.5e-10, 3.0]))
show(np.array([[0.5, 1e-9], [2.0, 3.0]]))
show(np.array([1.0, np.nan, np.inf, -np.inf]))
show(np.arange(24).reshape(2, 3, 4))
show(np.arange(24.0).reshape(2, 3, 4) / 7)
show(np.zeros((2, 0)))
show(np.array([], dtype="int64"))
show(np.array([1e300, -1e-300]))
show(np.array([12345.678, 0.001]))
show(np.array([100.0, 0.5]))
show(np.arange(40) * 1000)
print(np.array(5.0), repr(np.array(5.0)), np.array([1.5]).sum(), repr(np.array([1.5]).sum()))
print(repr(np.array([1, 2]).sum()), repr(np.array([True]).all()), repr(np.array([3.0])[0]))
print(np.array([[1.0, 2.0]]).shape, np.array([1]).shape, np.array(2.0).shape, repr(np.zeros(3).dtype))
