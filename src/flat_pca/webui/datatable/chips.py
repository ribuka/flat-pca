"""The filters in use of a data table, as chips shown above it."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TableConfig
from .formatting import format_value
from .state import (
    EQUALS_PREFIX,
    MAX_PREFIX,
    MIN_PREFIX,
    NULL_PREFIX,
    TEXT_PREFIX,
    Bound,
    TableState,
    parameter_name,
)


@dataclass(frozen=True)
class FilterChip:
    """One filter in use, shown as a chip with a button that removes it.

    Attributes
    ----------
    label : str
        Text of the chip, such as ``lot ∈ {A, B}``, ``temp ≥ 20``, or
        ``lot is_not_null``.
    parameters : tuple[str, ...]
        Query parameter names (``<table_id>.<name>``) whose controls the
        button clears.
    """

    label: str
    parameters: tuple[str, ...]


def _range_label(header: str, lower: Bound, upper: Bound) -> str:
    """Return the chip text of inclusive bounds; ``None`` leaves a side open."""
    if lower is not None and upper is not None:
        return f"{format_value(lower)} ≤ {header} ≤ {format_value(upper)}"
    if lower is not None:
        return f"{header} ≥ {format_value(lower)}"
    return f"{header} ≤ {format_value(upper)}"


def filter_chips(state: TableState, config: TableConfig) -> list[FilterChip]:
    """Return the chips of the column filters in use.

    The search of the whole table has its own box and no chip.

    Parameters
    ----------
    state : TableState
        State whose column names have been validated, for example by
        ``parse_state``.
    config : TableConfig
        Table settings; chips name columns by their header.

    Returns
    -------
    list[FilterChip]
        In the order of the columns, per column the search (``name ~
        "words"``), the values (``lot ∈ {A, B}``), the bounds (``temp ≥ 20``,
        ``temp ≤ 30``, or ``20 ≤ temp ≤ 30`` as one chip), and the null
        filter (``lot is_null`` / ``lot is_not_null``).
    """
    chips: list[FilterChip] = []
    for column in config.columns:
        name, header = column.name, column.header
        if name in state.text:
            chips.append(
                FilterChip(
                    f'{header} ~ "{state.text[name]}"',
                    (parameter_name(config, TEXT_PREFIX + name),),
                )
            )
        if name in state.equals:
            chips.append(
                FilterChip(
                    f"{header} ∈ {{{', '.join(state.equals[name])}}}",
                    (parameter_name(config, EQUALS_PREFIX + name),),
                )
            )
        if name in state.ranges:
            lower, upper = state.ranges[name]
            chips.append(
                FilterChip(
                    _range_label(header, lower, upper),
                    (
                        parameter_name(config, MIN_PREFIX + name),
                        parameter_name(config, MAX_PREFIX + name),
                    ),
                )
            )
        if name in state.nulls:
            chips.append(
                FilterChip(
                    f"{header} {state.nulls[name]}",
                    (parameter_name(config, NULL_PREFIX + name),),
                )
            )
    return chips
