# crate: pyrt/sklearn
# rust: ruai::pyrt::lib::sklearn
"""py0 package `sklearn.datasets` (a subset of scikit-learn, in Rust)."""

from numpy import ndarray


@rust("ruai::pyrt::lib::sklearn::Bunch")
class Bunch:
    @property
    def data(self) -> ndarray: ...
    @property
    def target(self) -> ndarray: ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def target_names(self) -> list[str]: ...


def load_iris() -> Bunch: ...
