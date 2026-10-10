# datatable

A table of a polars `DataFrame` / `LazyFrame` that is filtered, sorted, and
paged on the server and shown with htmx.

- The package imports only the standard library, polars, and Jinja2
  (`tests/webui/datatable/test_table_imports.py` checks this). It does not
  depend on a web framework or on the rest of flat-pca, so it can be copied to
  another project.
- In the browser it needs htmx (any 2.x) and `datatable.js` / `datatable.css`.
- flat-pca's data selection and transform screens use it for the file table
  (`services/file_table.py`, `routes/catalog.py`), and the sidebar's
  shown-file dialog for the transform targets of the shown run
  (`services/view_file_table.py`, `routes/view_selection.py`).

Update this document whenever the component gains a feature.

## Layout

| Path | Contents |
| --- | --- |
| `config.py` | `TableConfig` and `ColumnConfig`: what the application decides. |
| `state.py` | `TableState` and `parse_state`: query parameters → table state. |
| `query.py` | `apply_state`: table state + frame → `TableView` (rows of one page, the page, matching keys, choices, counts, column types, filter chips, histograms); `matching_rows` (the filtered and sorted rows that the view and the export share), `filter_expression`, `search_columns`, and `search_terms`. A `LazyFrame` is collected only for the page's rows, for the matching keys (or their count), and for the counts. |
| `export.py` | `parse_export`: form fields → `ExportRequest`; `export_file`: the filtered or selected rows as an `ExportFile` (CSV or Parquet, its name and download headers); `export_rows`, `file_content`, `export_filename`, `parse_selection`. |
| `bounds.py` | `bound_condition`: a column compared with a lower or upper bound, exactly for integers. |
| `histograms.py` | `column_histograms`: the distribution of each column under its name (`Histogram` of bins, or `TopValues`); `integer_bins` and `float_bins`. |
| `chips.py` | `FilterChip` and `filter_chips`: the column filters in use as chips above the table. |
| `counts.py` | `ColumnCounts` and `count_values`: rows of each choice value and null values per filtered column, for the column menus. |
| `pagination.py` | `Page` and `paginate`: the page among the rows. |
| `formatting.py` | `format_value`: the cell text (`dt_cell` filter); `dtype_label`: the column type under each column name. |
| `environment.py` | `configure_environment`: registers the templates and the filter with a Jinja2 environment; `TEMPLATES_DIR`, `STATIC_DIR`. |
| `templates/datatable/macros.html` | The `container` and `fragment` macros. |
| `static/datatable.js`, `static/datatable.css` | Browser behavior and look. |

## Use

1. Configure the Jinja2 environment once and serve `STATIC_DIR`:

   ```python
   from flat_pca.webui.datatable import STATIC_DIR, configure_environment

   configure_environment(templates.env)  # adds datatable/*.html and dt_cell
   app.mount("/datatable", StaticFiles(directory=STATIC_DIR), name="datatable")
   ```

   and load `datatable.css` and `datatable.js` (after htmx) on the page.

2. Describe the table:

   ```python
   config = TableConfig(
       table_id="files",              # container id, parameter prefix "files."
       key="stem",                    # row key; selections hold it as text
       columns=(
           ColumnConfig("stem", label="file", filter="text", frame_order=True,
                        title_column="path"),
           ColumnConfig("lot", filter="choice"),
           ColumnConfig("yield_pct", filter="number"),
           ColumnConfig("date", filter="datetime"),
           ColumnConfig("n_rows", label="rows"),   # sortable, no filter
       ),
       url="/catalog/files",          # returns the fragment
       page_size=1000,
       selectable=True,
       selection_name="stems",        # name of the hidden selection input
       selection_form=None,           # its form attribute, if any
       select_all_label="Select all filtered files",
       default_sort="stem",
       max_selected=None,             # most selected rows; None for no limit
       search=True,                   # search box of the whole table
       filter_chips=True,             # chips of the filters in use
       column_chooser=True,           # "Columns" menu showing or hiding columns
       histograms=True,               # distribution of each column in its header
       export_url="/catalog/files/export",  # "Export" menu; None for none
       export_name="catalog",         # exported files: catalog_<time>.csv
   )
   ```

3. Put the container in the page; it loads the fragment from `url`:

   ```jinja
   {% from "datatable/macros.html" import container %}
   {{ container(config, selected, locked=false, triggers="catalog-updated from:body") }}
   ```

   `triggers` adds htmx triggers that reload the table. `locked` shows the
   selection with every checkbox disabled (the filters and pages still work).

4. Answer the fragment request with a thin route:

   ```python
   parameters = {key: request.query_params.getlist(key) for key in request.query_params}
   try:
       state = parse_state(parameters, config)   # ValueError -> 400
   except ValueError as error:
       raise HTTPException(status_code=400, detail=str(error)) from error
   view = apply_state(frame, state, config)      # frame: DataFrame or LazyFrame
   ```

   ```jinja
   {% from "datatable/macros.html" import fragment %}
   {{ fragment(config, view, selected) }}
   ```

   The frame is every row in its default order. `apply_state` takes the
   choices of `"choice"` filters as its fourth argument; without it they are
   the sorted distinct values of the frame. The counts beside the choices,
   the null counts, and the histograms are always taken over every row of
   the frame (not only the filtered ones); a given choice missing from the
   frame counts 0.

5. With `export_url`, answer the export (a form POST) with another thin route:

   ```python
   form = await request.form()
   parameters = {key: form.getlist(key) for key in form}
   try:
       export = parse_export(parameters, config)  # ValueError -> 400
   except ValueError as error:
       raise HTTPException(status_code=400, detail=str(error)) from error
   file = export_file(frame, export, config, datetime.now().astimezone())
   return Response(file.content, media_type=file.media_type, headers=file.headers)
   ```

   The fields are the table's query parameters (as for the fragment; `page`
   is ignored), `<table_id>.export_format` (`csv` or `parquet`),
   `<table_id>.export_rows` (`filtered` or `selected`), and the selection
   input (`selection_name`, one JSON array). `datatable.js` sends them.

## Columns

| `ColumnConfig` field | Meaning |
| --- | --- |
| `name` | Frame column. |
| `label` | Header text (default: `name`). |
| `filter` | `"text"` (a search: every whitespace-separated word as plain text, in any order and case), `"choice"` (any of the values checked in a list with their counts), `"number"` / `"datetime"` (inclusive lower and upper bounds), or `None`. A column with a filter also has a null filter. Filters combine with AND. |
| `filter_label` | Accessible name of a text filter or of the value list of a choice filter (default: `Filter by <label>`). Bounds are named `<label> lower` / `<label> upper`. |
| `sortable` | Whether the column menu sorts (default: `True`). Sorting puts nulls last and keeps the frame's order among ties. |
| `frame_order` | The frame's order is the column's ascending order (for example a key in natural order); sorting by it keeps or reverses the frame. |
| `title_column` | Frame column shown as the cell's tooltip. |

## Query parameters

Every parameter of a table is named `<table_id>.<name>`; other parameters are
ignored, so several tables can share a query string. `eq__<column>` is
repeated once per value; any other repeated parameter uses its last value.
Blank values are ignored. An unknown `<table_id>.*` parameter, a filter that
does not fit its column, or an unparsable value is a `ValueError`.

| Name | Value |
| --- | --- |
| `sort` | A sortable column (default: `default_sort`, or the frame's order). An empty value clears the sort back to the default. |
| `order` | `asc` (default) or `desc`. |
| `page` | 1-based page number; a page past the last shows the last. |
| `search` | Search of the whole table (only with `search=True`): every whitespace-separated word, in any order, ignoring case, as plain text, must be in some shown `pl.String` column of the row (`search_columns`; each word in any of them). Other types, `title_column`, and columns of the frame that are not shown are not searched; nulls hold no word. |
| `q__<column>` | Search of a `"text"` column: the value must contain every whitespace-separated word, in any order, ignoring case, as plain text (not a regular expression). `b01 lotA` matches `LotA_B01_run3`. |
| `eq__<column>` | A value of a `"choice"` column, repeated for several (`?t.eq__lot=A&t.eq__lot=B`); a row matches any of them (`is_in`). |
| `min__<column>`, `max__<column>` | Bounds of a `"number"` (an integer stays an `int` and is compared exactly with an integer column when it fits 64 bits; otherwise it is compared as a float, `±inf` past the float range) or `"datetime"` (ISO 8601, as `datetime-local` sends) column. |
| `null__<column>` | `is_null` (only the null values) or `is_not_null` (no null values) for any column with a filter. |

Null values never match a search, a value, or a bound, so setting one of
those already drops the nulls of the column: with `is_null` too, no row
matches, and `is_not_null` adds nothing. `is_null` alone lists the rows
without a value; `is_not_null` alone drops them.

The fragment carries these as inputs marked `data-dt-query` inside the
container, which sends them with `hx-include="this"`. The table view, the
header checkbox's matching keys, the export, and anything else built on
`apply_state` filter and sort through the one `matching_rows`
(`filter_expression`, the search of the whole table included, and
`sort_frame`).

## Browser behavior and events

- Above the table, by the settings:
  - `search`: a search box (`Search…`, named `Search`). Typing reloads the
    first page after 300 ms; the box keeps its text, focus, and caret across
    reloads (`hx-preserve`).
  - `column_chooser`: a `Columns` button opening a menu (a popover like the
    column menus) with a checkbox per column. Unchecking one hides its
    header and cells; the first column, which names the rows, is always
    shown. The hidden columns are kept in `localStorage` under
    `datatable:hidden-columns:<table_id>` (per browser and table id) and
    hidden again after every reload, before the rows are drawn, by a style
    sheet in the document head. Hidden columns still sort, filter, and are
    searched; the footer counts the shown columns. Without storage (a
    private window, for example) the choice lasts until the page is left.
  - `filter_chips`: a chip per column filter in use, in column order:
    `name ~ "words"`, `lot ∈ {A, B}`, `temp ≥ 20` / `temp ≤ 30` /
    `20 ≤ temp ≤ 30` (both bounds are one chip), `lot is_null` /
    `lot is_not_null`. The column is named by its header. Its × button
    (named `Remove filter <chip>`) empties the search or bounds, unchecks the
    values, or sets the null filter back to Any, and reloads the first page.
    The search of the whole table has its box and no chip.
  - `export_url`: an `Export` button opening a menu (a popover like the
    column menus) with a `CSV` and a `Parquet` button for `Filtered rows`
    (every row matching the filters and the search, on all pages) and, in a
    selectable table, for `Selected rows` (every selected row, also those the
    filters hide), each with its row count; the buttons of rows without any
    are disabled, and the selected count follows the selection. A button
    posts the table's query controls, the format, the rows, and the
    selection input's JSON array (one field, so the size of the selection is
    not bound by the limit on form fields) to `export_url` with `fetch`,
    and saves the answer under the name its `Content-Disposition` gives,
    then closes the menu. A failure shows `Export failed: <status>
    <reason>` (the JSON `detail` of the answer) in the menu, which stays
    open. The file holds the shown columns (`TableConfig.columns`, in display
    order, under their frame names; also those hidden by `Columns`, but not
    `title_column` or other frame columns) of those rows, in the table's sort
    order: a CSV in UTF-8 with a byte order mark (which Excel needs to read
    it as UTF-8), dates and datetimes in ISO 8601, and null as an empty
    field; or a Parquet file keeping the column types. It is named
    `<export_name>_<YYYYmmdd-HHMMSS>.csv` (or `.parquet`). Leave
    `export_url` unset where the rows need no export, such as a dialog
    choosing a few rows.
- Each header shows the column name, the sort mark (▲ / ▼), a funnel mark
  while a filter (values, bounds, search, or null filter) uses the column,
  and the column type under the name (`dtype_label`: polars' short names such
  as `str`, `cat`, `i64`, `f64`, `datetime[μs]`).
- `histograms`: under the type, the distribution of the column over every
  row of the frame (`column_histograms`), counted on the server and drawn
  without a script, so its shape stays the same while the filters change
  and reads as a base. Turn it off when every row of the frame is not a
  meaningful base (the shown-file dialog of flat-pca, whose rows are the
  chosen files, has none).
  - Integer (up to 64 bits), float, date, and datetime columns: an inline
    SVG histogram with the smallest and largest value under it. Integers,
    dates, and datetimes are binned by their integer value (a datetime in its
    time unit): one bin per value when the range has at most 20 values,
    otherwise at most 20 bins of `ceil(values / 20)` values. Floats get 20
    bins of equal width, half open but the last (one bin when every value is
    the same); NaN and infinite values are not counted. Bins that do not
    overlap the lower and upper bounds in use are faded (`dt-out`). Each bin's
    tooltip gives its first and last value and rows (`2 – 3: 4 rows`).
  - Categorical, enum, and boolean columns: the five most frequent values
    (ties by value) with bars as long as their rows relative to the first,
    and `Others` with the rows of the other values. Values not checked in
    the column's filter are faded.
  - Nulls are not counted (the null filter shows them); a column without a
    non-null value, or of another type (text, for example), has none.
- Clicking the column name (or Enter / Space on it) opens the column menu,
  a popover (`popover`, `popovertarget`) holding the column's controls:
  sort (Asc, Desc, Clear sort), the filter (search, values with their
  counts, or lower and upper bounds), the null filter (Any, Is null, Is not
  null, with the null count), and "Copy column name" (the frame's column
  name). A click outside or Escape closes it and returns the focus to the
  column name; Tab moves from the name into the menu. The popover is in the
  top layer, so neither the scroll box nor a `<dialog>` clips it;
  `datatable.js` places it under the column name (above it, or shifted,
  when there is no room) and keeps it there while the page scrolls.
- Typing a search or a bound reloads the first page after 300 ms; checking
  a value or a null filter reloads it at once. The menu stays open across
  the reload, with the focus (and caret) where it was. Sorting from the menu
  or copying the name closes it. Sorting or filtering returns to the first
  page. Reloads trigger `dt-reload` on the container.
- The checkbox column and the first column stay at the left while the table
  scrolls sideways (`position: sticky`, `.dt-pinned`); the header stays on
  top of them. Which columns are pinned is not configurable: the first
  column is the one naming the rows (such as the file name).
- Under the table, the footer shows "N rows, M columns" (the rows matching
  the filters and the shown columns), for a selectable table the selection
  count ("k selected", or "k / max selected" with `max_selected`) and Clear
  (which clears the whole selection, on every page and filter), "a–b of N",
  and with several pages the page controls: first, previous, next, and last
  page buttons around a page number input. Enter or leaving the input goes
  to the typed page (kept between 1 and the last page); a blank or invalid
  value puts the current page back. The number of rows per page is
  `page_size`; it cannot be changed from the page.
- The selection of a selectable table spans pages and filters. It is a JSON
  array in the hidden input `#<table_id>-selection` (named `selection_name`),
  outside the reloaded fragment, so a form or `hx-include` can send it as one
  field. Keys are the key column cast to polars `String` (`key_text`) for the
  row checkboxes, the header checkbox, and the selection alike, so keys of any
  type match; give the initial selection in the same form. A table without
  selection does not cast its keys (`data-dt-key` is `str()` of the value).
  The header checkbox selects or clears every row matching the filters and
  shows all / some / none.
- A click anywhere on a row toggles its checkbox, once (a click on the
  checkbox itself is not counted twice). Clicks on a link or another control
  in the row (`a`, `button`, `input`, `select`, `textarea`, `label`,
  `summary`, `[contenteditable]`, `[tabindex]`) and the end of a drag that
  selects text change nothing, and neither does a click on a row whose
  checkbox is disabled (every row of a locked table, and the unselected
  rows at the limit). A row that a click changes shows a pointer cursor; a
  selected row has its own background.
- Shift + click on a row or on its checkbox selects (or clears, following
  the clicked row's new state) every row shown from the row clicked before
  to this one. The range counts the rows of the current page in their shown
  order; when the row clicked before is not on it (after paging or
  filtering), only the clicked row changes. Shift + click does not select
  the text between the clicks.
- `max_selected` (a positive integer, on a selectable table only) limits the
  selection. When it is reached, the checkboxes of the unselected rows are
  disabled (selected rows can still be cleared) and the notice
  `[data-dt-limit]` above the table says so; both hold after paging,
  filtering, and sorting. The header checkbox is disabled while selecting
  every matching row would pass the limit; when they all fit, or when they
  are all selected (to clear them), it works as without a limit. A Shift +
  click range that would pass the limit changes nothing, and the row clicked
  before stays the start of the next range. An initial selection past the
  limit is kept and can only be reduced. One function in `datatable.js`
  (`canChange`) decides whether a change of the selection is allowed (row,
  range, header checkbox, and Clear), for a locked table too.
- Each change of the selection dispatches `datatable:selection-change` on the
  container. It bubbles; `event.detail.keys` holds the selected keys and
  `event.detail.max` the limit (`null` without one). The footer already
  shows the count, so a page needs the event only to act on the selection.

## Elements and classes

Every class starts with `dt-`; ids and data attributes are scoped by the
container, so several tables can share a page.

| Selector | Element |
| --- | --- |
| `.dt-root[data-datatable]` | Container (`id` = `table_id`); `data-dt-locked` when locked; `data-dt-max-selected` with the limit; `data-dt-export-url` with `export_url`. |
| `.dt-scroll` / `.dt-table` | Scroll box with a fixed header / the table. |
| `.dt-head`, `.dt-column[data-dt-column]`, `.dt-label` | Header line and the column name button opening the menu. |
| `.dt-sort-mark`, `.dt-menu-mark`, `.dt-filter-mark[data-dt-filtered]` | ▲ / ▼, the menu's ⋯, and the funnel of a filtered column. |
| `.dt-type` | Column type under the name. |
| `.dt-histogram[data-dt-histogram]`, `.dt-bins`, `.dt-bin`, `.dt-bin-bar`, `.dt-bin-hit`, `.dt-ticks` | A column's histogram: the SVG (`role="img"`, named `Histogram of <label>, <min> to <max>`), a bin (`<g>` with its `<title>`), its bar and its full-height hover area, and the smallest and largest value. |
| `.dt-top-values[data-dt-top-values]`, `.dt-top-value`, `.dt-top-bar`, `.dt-top-label`, `.dt-top-others` | Most frequent values (a list named `Most frequent values of <label>`), a value with its bar, text, and `.dt-count`, and the `Others` line. |
| `.dt-out` | A bin outside the bounds in use, or a value not checked, shown faded. |
| `.dt-toolbar` | Line above the table holding the search box and the `Columns` and `Export` buttons. |
| `.dt-search` | Search box of the whole table (`id` `<table_id>-search`). |
| `.dt-columns-button`, `.dt-menu[data-dt-columns-menu]`, `[data-dt-show-column]` | `Columns` button, its menu (`id` `<table_id>-columns-menu`, named `Columns`), and the checkbox of each column. |
| `.dt-export-button`, `.dt-menu[data-dt-export-menu]`, `.dt-export-rows`, `.dt-export-label`, `[data-dt-export][data-dt-export-rows]`, `[data-dt-export-selected-count]`, `.dt-export-error[data-dt-export-error]` | `Export` button, its menu (`id` `<table_id>-export-menu`, named `Export`), a group of rows (named `Filtered rows` / `Selected rows`) with its label and count, a download button (`csv` or `parquet`, `filtered` or `selected`, named `Export <rows> as <format>`), the selected count, and the failure notice (`role="alert"`). |
| `.dt-chips`, `.dt-chip`, `.dt-chip-label`, `[data-dt-remove-filter]` | List of the filters in use (named `Filters in use`), a chip, its text, and its × button (a JSON array of the parameter names it clears). |
| `.dt-menu[popover][data-dt-menu]` | A menu: a column's (`id` `<table_id>-menu-<n>`, `role="dialog"`, named `<label> menu`, `data-dt-menu` = the column name) or the `Columns` or `Export` menu (empty `data-dt-menu`); `data-dt-placed` once placed. |
| `.dt-menu-section`, `.dt-sorts`, `.dt-filter`, `.dt-choices`, `.dt-choice`, `.dt-nulls`, `.dt-count` | Menu parts: sort buttons (`[data-dt-sort][data-dt-order]`, `aria-pressed`), the filter, the value list, the null filter, and counts. |
| `[data-dt-copy]` | "Copy column name" button. |
| `.dt-check`, `[data-dt-check-all]`, `[data-dt-row-check]` | Selection checkboxes. |
| `.dt-pinned` | Cells of the checkbox column and the first column, pinned at the left. |
| `.dt-limit[data-dt-limit]` | Notice that the selection limit is reached (`hidden` below it). |
| `tr[data-dt-key]` | Row with its key. |
| `.dt-footer` | Footer under the table. |
| `.dt-shape[data-dt-shape]`, `[data-dt-column-count]` | "N rows, M columns", and M (the shown columns). |
| `.dt-selection-state`, `.dt-selected-count[data-dt-selected-count]`, `[data-dt-clear]` | Selection count and the Clear button (selectable tables). |
| `.dt-pager`, `[data-dt-page-range]`, `[data-dt-page]`, `.dt-page-jump`, `.dt-page-input[data-dt-page-input][data-dt-page-current]` | "a–b of N", the first / previous / next / last page buttons (named `First page` and so on), and the page number input. |

## CSS variables

Set them on `.dt-root` (or an ancestor) to change the look. The defaults
follow `prefers-color-scheme`; `data-dt-theme="light"` or `"dark"` on the
container or an ancestor fixes the scheme.

| Variable | Use |
| --- | --- |
| `--dt-fg`, `--dt-muted` | Text, and headers / the footer texts / locked rows / the limit notice. |
| `--dt-border` | Cell and box borders. |
| `--dt-bg` | Background of the pinned cells, which cover the cells scrolled under them. |
| `--dt-header-bg`, `--dt-row-hover-bg`, `--dt-row-selected-bg` | Header, hovered row, and selected row backgrounds. |
| `--dt-accent` | Sort and filter marks, and the chosen sort in a menu. |
| `--dt-radius` | Box corner radius. |
| `--dt-font`, `--dt-font-size` | Font family and table font size. |
| `--dt-cell-padding` | Cell padding. |
| `--dt-check-width` | Width of the checkbox column, where the pinned first column starts. |
| `--dt-max-height` | Height of the scroll box (about the header and 12 rows). |
| `--dt-filter-max-width` | Width limit of the filter inputs. |
| `--dt-mono-font` | Font of the column types. |
| `--dt-menu-bg`, `--dt-menu-shadow`, `--dt-menu-max-width` | Menu background, shadow, and width limit. |
| `--dt-search-width` | Width of the search box. |
| `--dt-chip-bg` | Background of the filter chips. |
| `--dt-histogram-width`, `--dt-histogram-height` | Width of a histogram or of the top values, and height of a histogram's bars. |
| `--dt-histogram-fg`, `--dt-histogram-out-opacity` | Color of the bars, and opacity of the faded ones. |

Buttons and inputs otherwise take the page's own styles.
