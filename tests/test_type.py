import pytest

from ccrestoration import BaseModelInterface


def test_base_class() -> None:
    with pytest.raises(TypeError):
        BaseModelInterface()  # type: ignore
