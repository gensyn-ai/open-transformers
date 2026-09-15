"""Generic name→class registry. Three small registries are the entire
swap surface for the codebase (see plan/02_repo_layout.md §3).

Adding a new attention variant is: write the class, add the decorator,
flip the config string. No conditionals elsewhere.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Tiny registry keyed by string. Errors loudly on duplicate or
    unknown names because silent failures here cause silent ML bugs.
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(
                    f"{self._label} '{name}' already registered "
                    f"to {self._items[name].__name__}"
                )
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError as e:
            raise KeyError(
                f"{self._label} '{name}' not found. "
                f"Registered: {sorted(self._items)}"
            ) from e

    def keys(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
