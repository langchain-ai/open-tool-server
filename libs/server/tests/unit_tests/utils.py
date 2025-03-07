from typing import Any


class AnyStr:
    """A type that matches any string."""

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, str)
