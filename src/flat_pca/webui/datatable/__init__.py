"""A polars table filtered, sorted, and paged on the server and shown with htmx.

The package depends only on polars and Jinja2 (and htmx in the browser), so
it can be reused outside flat-pca; see ``README.md`` for its use.
"""

from .chips import FilterChip, filter_chips
from .config import ColumnConfig, FilterKind, TableConfig
from .counts import ColumnCounts, count_values
from .environment import STATIC_DIR, TEMPLATES_DIR, configure_environment
from .formatting import dtype_label, format_value
from .histograms import (
    ColumnHistogram,
    Histogram,
    HistogramBin,
    TopValue,
    TopValues,
    column_histograms,
    float_bins,
    integer_bins,
)
from .pagination import Page, paginate
from .query import (
    TableView,
    apply_state,
    choice_options,
    filter_expression,
    key_text,
    search_columns,
    search_terms,
    sort_frame,
)
from .state import NullFilter, TableState, parameter_name, parse_state

__all__ = [
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "ColumnConfig",
    "ColumnCounts",
    "ColumnHistogram",
    "FilterChip",
    "FilterKind",
    "Histogram",
    "HistogramBin",
    "NullFilter",
    "Page",
    "TableConfig",
    "TableState",
    "TableView",
    "TopValue",
    "TopValues",
    "apply_state",
    "choice_options",
    "column_histograms",
    "configure_environment",
    "count_values",
    "dtype_label",
    "filter_chips",
    "filter_expression",
    "float_bins",
    "format_value",
    "integer_bins",
    "key_text",
    "paginate",
    "parameter_name",
    "parse_state",
    "search_columns",
    "search_terms",
    "sort_frame",
]
