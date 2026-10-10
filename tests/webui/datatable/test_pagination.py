"""Tests for splitting the rows of a table into numbered pages."""

from __future__ import annotations

import pytest

from flat_pca.webui.datatable.pagination import Page, paginate


def test_paginate_returns_the_requested_page() -> None:
    """A page holds the 1-based positions of its rows."""
    page = paginate(25, 2, 10)

    assert page == Page(number=2, count=3, total=25, start=11, end=20)
    assert (page.offset, page.length) == (10, 10)


def test_paginate_clamps_past_the_last_page() -> None:
    """A number past the last page shows the last page."""
    page = paginate(25, 9, 10)

    assert (page.number, page.start, page.end) == (3, 21, 25)
    assert (page.offset, page.length) == (20, 5)


def test_paginate_without_rows_has_one_empty_page() -> None:
    """A table without rows has one page without rows."""
    page = paginate(0, 1, 10)

    assert (page.number, page.count, page.start, page.end) == (1, 1, 0, 0)
    assert (page.offset, page.length) == (0, 0)


@pytest.mark.parametrize(("total", "number", "size"), [(1, 0, 10), (1, 1, 0), (-1, 1, 10)])
def test_paginate_rejects_invalid_arguments(total: int, number: int, size: int) -> None:
    """The page number and size must be positive and the total not negative."""
    with pytest.raises(ValueError, match="must be positive"):
        paginate(total, number, size)
