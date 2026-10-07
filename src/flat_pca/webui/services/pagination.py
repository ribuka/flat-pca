"""Splitting a list into numbered pages."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

# Page numbers shown on each side of the current page, besides the first
# and the last page.
NEIGHBOR_PAGES = 2


@dataclass(frozen=True)
class Page[T]:
    """One page of a list.

    Attributes
    ----------
    items : list[T]
        Items on the page.
    number : int
        1-based page number.
    count : int
        Number of pages; at least 1, even for an empty list.
    total : int
        Number of items over all pages.
    start : int
        1-based position of the first item on the page, or 0 if it is empty.
    end : int
        1-based position of the last item on the page, or 0 if it is empty.
    """

    items: list[T]
    number: int
    count: int
    total: int
    start: int
    end: int

    @property
    def links(self) -> list[int | None]:
        """Return the page numbers to link, with ``None`` for a gap.

        Returns
        -------
        list[int | None]
            The first and the last page and the pages near the current one,
            in ascending order; ``None`` stands for omitted pages.
        """
        shown = sorted(
            {1, self.count}
            | set(
                range(
                    max(1, self.number - NEIGHBOR_PAGES),
                    min(self.count, self.number + NEIGHBOR_PAGES) + 1,
                )
            )
        )
        links: list[int | None] = []
        previous = 0
        for number in shown:
            if number - previous > 1:
                links.append(None)
            links.append(number)
            previous = number
        return links


def paginate[T](items: list[T], number: int, size: int) -> Page[T]:
    """Return one page of ``items``.

    Parameters
    ----------
    items : list[T]
        All items in display order.
    number : int
        Requested 1-based page number; a number past the last page shows the
        last page.
    size : int
        Number of items per page; must be positive.

    Returns
    -------
    Page[T]
        The page.

    Raises
    ------
    ValueError
        If ``number`` or ``size`` is not positive.
    """
    if number < 1 or size < 1:
        raise ValueError(f"page number and size must be positive: {number}, {size}")
    count = max(1, ceil(len(items) / size))
    number = min(number, count)
    begin = (number - 1) * size
    shown = items[begin : begin + size]
    return Page(
        items=shown,
        number=number,
        count=count,
        total=len(items),
        start=begin + 1 if shown else 0,
        end=begin + len(shown),
    )
