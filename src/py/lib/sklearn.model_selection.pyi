# crate: pyrt/sklearn
# rust: ruai::pyrt::lib::sklearn
"""py0 package `sklearn.model_selection` (a subset of scikit-learn, in Rust).
`train_test_split` shuffles with NumPy's legacy RandomState, as scikit-learn."""

from numpy import ndarray
from rust import Result


def train_test_split(X: ndarray, y: ndarray, test_size: float = 0.25,
                     random_state: int | None = None) -> Result[tuple[ndarray, ndarray, ndarray, ndarray], ValueError]: ...
