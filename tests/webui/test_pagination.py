"""Tests for splitting a list into numbered pages."""

from __future__ import annotations

import pytest

from flat_pca.webui.services.pagination import Page, paginate


def test_paginate_returns_the_requested_page() -> None:
    """A page holds its items and their 1-based positions."""
    page = paginate(list(range(25)), 2, 10)

    assert page == Page(
        items=list(range(10, 20)), number=2, count=3, total=25, start=11, end=20
    )


def test_paginate_clamps_past_the_last_page() -> None:
    """A number past the last page shows the last page."""
    page = paginate(list(range(25)), 9, 10)

    assert page.items == list(range(20, 25))
    assert (page.number, page.start, page.end) == (3, 21, 25)


def test_paginate_empty_list_has_one_empty_page() -> None:
    """An empty list has one page without items."""
    page = paginate([], 1, 10)

    assert page.items == []
    assert (page.number, page.count, page.start, page.end) == (1, 1, 0, 0)


@pytest.mark.parametrize(("number", "size"), [(0, 10), (1, 0)])
def test_paginate_rejects_non_positive_arguments(number: int, size: int) -> None:
    """The page number and size must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        paginate([1], number, size)


@pytest.mark.parametrize(
    ("number", "count", "links"),
    [
        (1, 1, [1]),
        (1, 4, [1, 2, 3, 4]),
        (1, 10, [1, 2, 3, None, 10]),
        (5, 10, [1, None, 3, 4, 5, 6, 7, None, 10]),
        (4, 10, [1, 2, 3, 4, 5, 6, None, 10]),
        (10, 10, [1, None, 8, 9, 10]),
    ],
)
def test_page_links_skip_far_pages(
    number: int, count: int, links: list[int | None]
) -> None:
    """Links show the first, the last, and the nearby pages with gaps."""
    assert paginate(list(range(count)), number, 1).links == links
