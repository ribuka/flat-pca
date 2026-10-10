"""A polars table filtered, sorted, and paged on the server and shown with htmx.

The package depends only on polars and Jinja2 (and htmx in the browser), so
it can be reused outside flat-pca; see ``README.md`` for its use.
"""

from .chips import FilterChip, filter_chips
from .config import ColumnConfig, FilterKind, TableConfig
from .counts import ColumnCounts, count_values
from .environment import STATIC_DIR, TEMPLATES_DIR, configure_environment
from .export import (
    ExportFile,
    ExportFormat,
    ExportRequest,
    ExportRows,
    export_file,
    export_filename,
    export_rows,
    file_content,
    parse_export,
    parse_selection,
)
from .formatting import dtype_label, format_value
from .histograms import (
    ColumnHistogram,
    Histogram,
    HistogramBin,
    TopValue,
    TopValues,
    column_histograms,
    count_histograms,
    float_bins,
    integer_bins,
    mark_histograms,
)
from .pagination import Page, paginate
from .query import (
    TableView,
    apply_state,
    filter_expression,
    key_text,
    matching_rows,
    search_columns,
    search_terms,
    sort_frame,
)
from .state import NullFilter, TableState, parameter_name, parse_state
from .summary import TableSummary, choice_options, summarize_table

__all__ = [
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "ColumnConfig",
    "ColumnCounts",
    "ColumnHistogram",
    "ExportFile",
    "ExportFormat",
    "ExportRequest",
    "ExportRows",
    "FilterChip",
    "FilterKind",
    "Histogram",
    "HistogramBin",
    "NullFilter",
    "Page",
    "TableConfig",
    "TableState",
    "TableSummary",
    "TableView",
    "TopValue",
    "TopValues",
    "apply_state",
    "choice_options",
    "column_histograms",
    "configure_environment",
    "count_histograms",
    "count_values",
    "dtype_label",
    "export_file",
    "export_filename",
    "export_rows",
    "file_content",
    "filter_chips",
    "filter_expression",
    "float_bins",
    "format_value",
    "integer_bins",
    "key_text",
    "mark_histograms",
    "matching_rows",
    "paginate",
    "parameter_name",
    "parse_export",
    "parse_selection",
    "parse_state",
    "search_columns",
    "search_terms",
    "sort_frame",
    "summarize_table",
]
