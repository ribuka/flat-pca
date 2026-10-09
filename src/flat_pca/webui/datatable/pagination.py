"""Splitting the rows of a table into numbered pages."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

# Page numbers shown on each side of the current page, besides the first
# and the last page.
NEIGHBOR_PAGES = 2


@dataclass(frozen=True)
class Page:
    """The position of one page among the rows of a table.

    Attributes
    ----------
    number : int
        1-based page number.
    count : int
        Number of pages; at least 1, even without rows.
    total : int
        Number of rows over all pages.
    start : int
        1-based position of the first row on the page, or 0 if it is empty.
    end : int
        1-based position of the last row on the page, or 0 if it is empty.
    """

    number: int
    count: int
    total: int
    start: int
    end: int

    @property
    def offset(self) -> int:
        """Return the 0-based position of the page's first row.

        Returns
        -------
        int
            Number of rows before the page.
        """
        return max(self.start - 1, 0)

    @property
    def length(self) -> int:
        """Return the number of rows on the page.

        Returns
        -------
        int
            ``end - start + 1``, or 0 for an empty page.
        """
        return self.end - self.start + 1 if self.start else 0

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


def paginate(total: int, number: int, size: int) -> Page:
    """Return the position of one page among ``total`` rows.

    Parameters
    ----------
    total : int
        Number of rows over all pages; must not be negative.
    number : int
        Requested 1-based page number; a number past the last page shows the
        last page.
    size : int
        Number of rows per page; must be positive.

    Returns
    -------
    Page
        The page.

    Raises
    ------
    ValueError
        If ``total`` is negative, or ``number`` or ``size`` is not positive.
    """
    if total < 0 or number < 1 or size < 1:
        raise ValueError(
            f"total must not be negative and page number and size must be positive: "
            f"{total}, {number}, {size}"
        )
    count = max(1, ceil(total / size))
    number = min(number, count)
    begin = (number - 1) * size
    length = min(size, total - begin)
    return Page(
        number=number,
        count=count,
        total=total,
        start=begin + 1 if length else 0,
        end=begin + length,
    )
