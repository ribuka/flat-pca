"""Settings of a data table given by the application that shows it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FilterKind = Literal["text", "choice", "number", "datetime"]


@dataclass(frozen=True)
class ColumnConfig:
    """One shown column of a data table.

    Attributes
    ----------
    name : str
        Column name in the frame.
    label : str | None, default None
        Header text; the column name if ``None``.
    filter : FilterKind | None, default None
        Filter in the column menu: ``"text"`` (a search for words in any
        order and case), ``"choice"`` (any of the values checked in a list),
        ``"number"`` or ``"datetime"`` (inclusive lower and upper bounds), or
        ``None`` for no filter. A column with a filter also has a null
        filter (``is_null`` / ``is_not_null``).
    filter_label : str | None, default None
        Accessible name of a ``"text"`` or ``"choice"`` filter; ``"Filter by
        <label>"`` if ``None``.
    sortable : bool, default True
        Whether the column menu sorts by the column.
    frame_order : bool, default False
        Whether the frame's row order is the column's ascending order, as
        for a key column given in natural order. Sorting by the column then
        keeps (or reverses) the frame's order instead of comparing values.
    title_column : str | None, default None
        Frame column whose value is shown as the cell's tooltip.
    """

    name: str
    label: str | None = None
    filter: FilterKind | None = None
    filter_label: str | None = None
    sortable: bool = True
    frame_order: bool = False
    title_column: str | None = None

    @property
    def header(self) -> str:
        """Return the header text.

        Returns
        -------
        str
            ``label``, or the column name without one.
        """
        return self.name if self.label is None else self.label

    @property
    def filter_name(self) -> str:
        """Return the accessible name of a text or choice filter.

        Returns
        -------
        str
            ``filter_label``, or ``"Filter by <header>"`` without one.
        """
        return f"Filter by {self.header}" if self.filter_label is None else self.filter_label


@dataclass(frozen=True)
class TableConfig:
    """Settings of one data table.

    Attributes
    ----------
    table_id : str
        Id of the table's container element. It also names the table's query
        parameters (``<table_id>.<name>``) and element ids, so several tables
        can share a page.
    key : str
        Frame column identifying a row; selections hold its values as text.
    columns : tuple[ColumnConfig, ...]
        Shown columns, in display order.
    url : str
        URL that returns the table fragment for the table's query parameters.
    page_size : int, default 1000
        Rows per page.
    selectable : bool, default False
        Whether rows have checkboxes and the header has one for every row
        matching the filters.
    selection_name : str, default "selection"
        ``name`` of the hidden input holding the selection as a JSON array.
    selection_form : str | None, default None
        ``form`` attribute of that input: the id of the form it belongs to.
    select_all_label : str, default "Select all filtered rows"
        Accessible name of the header checkbox.
    default_sort : str | None, default None
        Column sorted by (ascending) when the query names none; ``None``
        keeps the frame's order.
    max_selected : int | None, default None
        Most rows that can be selected at once in a selectable table;
        ``None`` for no limit. At the limit the unselected rows cannot be
        checked and the table says so.
    search : bool, default False
        Whether a search box above the table looks for words (in any order
        and case) in every shown ``pl.String`` column.
    filter_chips : bool, default False
        Whether the filters in use are listed above the table as chips,
        each with a button that removes it.
    column_chooser : bool, default False
        Whether a "Columns" menu above the table shows or hides columns
        (all but the first). The browser remembers the choice per
        ``table_id``.
    histograms : bool, default False
        Whether each header shows the distribution of its column over every
        row of the frame (``column_histograms``): a histogram of a number,
        date, or datetime column, or the most frequent values of a
        categorical, enum, or boolean column. Turn it off when every row is
        not a meaningful base, such as a table of a few chosen rows.
    export_url : str | None, default None
        URL that answers a POST with the table's rows as a CSV or Parquet
        file (``parse_export``, ``export_file``); ``None`` for no "Export"
        menu above the table.
    export_name : str, default "table"
        Start of the exported file names: ``<export_name>_<YYYYmmdd-HHMMSS>``
        with ``.csv`` or ``.parquet``.
    """

    table_id: str
    key: str
    columns: tuple[ColumnConfig, ...]
    url: str
    page_size: int = 1000
    selectable: bool = False
    selection_name: str = "selection"
    selection_form: str | None = None
    select_all_label: str = "Select all filtered rows"
    default_sort: str | None = None
    max_selected: int | None = None
    search: bool = False
    filter_chips: bool = False
    column_chooser: bool = False
    histograms: bool = False
    export_url: str | None = None
    export_name: str = "table"

    def __post_init__(self) -> None:
        """Check the settings.

        Raises
        ------
        ValueError
            If the page size is not positive, a column name repeats, the
            default sort column is not a sortable column, or the selection
            limit is not a positive integer or is set on a table without
            selection.
        """
        if self.page_size < 1:
            raise ValueError(f"page_size must be positive: {self.page_size}")
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"column names must be unique: {names}")
        if self.default_sort is not None and self.default_sort not in self.sortable_columns:
            raise ValueError(f"default_sort is not a sortable column: {self.default_sort!r}")
        if self.max_selected is not None:
            if isinstance(self.max_selected, bool) or not isinstance(self.max_selected, int):
                raise ValueError(f"max_selected must be an integer: {self.max_selected!r}")
            if self.max_selected < 1:
                raise ValueError(f"max_selected must be positive: {self.max_selected}")
            if not self.selectable:
                raise ValueError("max_selected needs a selectable table")

    @property
    def prefix(self) -> str:
        """Return the prefix of the table's query parameter names.

        Returns
        -------
        str
            ``"<table_id>."``.
        """
        return f"{self.table_id}."

    @property
    def sortable_columns(self) -> dict[str, ColumnConfig]:
        """Return the sortable columns by name.

        Returns
        -------
        dict[str, ColumnConfig]
            Columns with ``sortable`` set.
        """
        return {column.name: column for column in self.columns if column.sortable}

    @property
    def filtered_columns(self) -> dict[str, ColumnConfig]:
        """Return the columns with a filter by name.

        Returns
        -------
        dict[str, ColumnConfig]
            Columns whose ``filter`` is not ``None``.
        """
        return {column.name: column for column in self.columns if column.filter is not None}
