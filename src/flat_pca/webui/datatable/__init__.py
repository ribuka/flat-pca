"""A polars table filtered, sorted, and paged on the server and shown with htmx.

The package depends only on polars and Jinja2 (and htmx in the browser), so
it can be reused outside flat-pca; see ``README.md`` for its use.
"""

from .config import ColumnConfig, FilterKind, TableConfig
from .environment import STATIC_DIR, TEMPLATES_DIR, configure_environment
from .formatting import format_value
from .pagination import Page, paginate
from .query import TableView, apply_state, choice_options, filter_expression, sort_frame
from .state import TableState, parameter_name, parse_state

__all__ = [
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "ColumnConfig",
    "FilterKind",
    "Page",
    "TableConfig",
    "TableState",
    "TableView",
    "apply_state",
    "choice_options",
    "configure_environment",
    "filter_expression",
    "format_value",
    "paginate",
    "parameter_name",
    "parse_state",
    "sort_frame",
]
