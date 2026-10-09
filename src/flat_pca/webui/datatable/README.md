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
| `query.py` | `apply_state`: table state + frame → `TableView` (rows of one page, the page, matching keys, choices). A `LazyFrame` is collected only for the page's rows and for the matching keys (or their count). |
| `pagination.py` | `Page` and `paginate`: the page among the rows. |
| `formatting.py` | `format_value`: the cell text (`dt_cell` filter). |
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
   the sorted distinct values of the frame.

## Columns

| `ColumnConfig` field | Meaning |
| --- | --- |
| `name` | Frame column. |
| `label` | Header text (default: `name`). |
| `filter` | `"text"` (case-insensitive substring), `"choice"` (one value from a drop-down), `"number"` / `"datetime"` (inclusive lower and upper bounds), or `None`. Null values match no filter. Filters combine with AND. |
| `filter_label` | Accessible name of a text or choice filter (default: `Filter by <label>`). Bounds are named `<label> lower` / `<label> upper`. |
| `sortable` | Whether the header sorts (default: `True`). Sorting puts nulls last and keeps the frame's order among ties. |
| `frame_order` | The frame's order is the column's ascending order (for example a key in natural order); sorting by it keeps or reverses the frame. |
| `title_column` | Frame column shown as the cell's tooltip. |

## Query parameters

Every parameter of a table is named `<table_id>.<name>`; other parameters are
ignored, so several tables can share a query string. A repeated parameter uses
its last value; blank values are ignored. An unknown `<table_id>.*` parameter,
a filter that does not fit its column, or an unparsable value is a
`ValueError`.

| Name | Value |
| --- | --- |
| `sort` | A sortable column (default: `default_sort`, or the frame's order). |
| `order` | `asc` (default) or `desc`. |
| `page` | 1-based page number; a page past the last shows the last. |
| `q__<column>` | Substring of a `"text"` column. |
| `eq__<column>` | Value of a `"choice"` column. |
| `min__<column>`, `max__<column>` | Bounds of a `"number"` (an integer stays an `int` and is compared exactly with an integer column when it fits 64 bits; otherwise it is compared as a float, `±inf` past the float range) or `"datetime"` (ISO 8601, as `datetime-local` sends) column. |

The fragment carries these as inputs marked `data-dt-query` inside the
container, which sends them with `hx-include="this"`.

## Browser behavior and events

- Typing in a filter reloads the first page after 300 ms; choosing a value
  reloads it at once. Clicking a header sorts ascending, then toggles. Sorting
  or filtering returns to the first page. Reloads trigger `dt-reload` on the
  container.
- The selection of a selectable table spans pages and filters. It is a JSON
  array in the hidden input `#<table_id>-selection` (named `selection_name`),
  outside the reloaded fragment, so a form or `hx-include` can send it as one
  field. Keys are the key column cast to polars `String` (`key_text`) for the
  row checkboxes, the header checkbox, and the selection alike, so keys of any
  type match; give the initial selection in the same form. A table without
  selection does not cast its keys (`data-dt-key` is `str()` of the value). The header checkbox selects or clears every row matching the filters
  and shows all / some / none.
- `max_selected` (a positive integer, on a selectable table only) limits the
  selection. When it is reached, the checkboxes of the unselected rows are
  disabled (selected rows can still be cleared) and the notice
  `[data-dt-limit]` above the table says so; both hold after paging,
  filtering, and sorting. The header checkbox is disabled while selecting
  every matching row would pass the limit; when they all fit, or when they
  are all selected (to clear them), it works as without a limit. An initial
  selection past the limit is kept and can only be reduced. One function in
  `datatable.js` (`canChange`) decides whether a change of the selection is
  allowed, for a locked table too.
- Each change of the selection dispatches `datatable:selection-change` on the
  container. It bubbles; `event.detail.keys` holds the selected keys and
  `event.detail.max` the limit (`null` without one), so the page can show
  `n / max`. The table itself does not show the count.

## Elements and classes

Every class starts with `dt-`; ids and data attributes are scoped by the
container, so several tables can share a page.

| Selector | Element |
| --- | --- |
| `.dt-root[data-datatable]` | Container (`id` = `table_id`); `data-dt-locked` when locked; `data-dt-max-selected` with the limit. |
| `.dt-scroll` / `.dt-table` | Scroll box with a fixed header / the table. |
| `.dt-sort[data-dt-sort]`, `.dt-sort-mark` | Header sort button and its ▲ / ▼. |
| `.dt-filter` | Filters under a header. |
| `.dt-check`, `[data-dt-check-all]`, `[data-dt-row-check]` | Selection checkboxes. |
| `.dt-limit[data-dt-limit]` | Notice that the selection limit is reached (`hidden` below it). |
| `tr[data-dt-key]` | Row with its key. |
| `.dt-pager`, `[data-dt-page]`, `.dt-page-current`, `[data-dt-page-range]` | Page links and "a–b of N". |

## CSS variables

Set them on `.dt-root` (or an ancestor) to change the look. The defaults
follow `prefers-color-scheme`; `data-dt-theme="light"` or `"dark"` on the
container or an ancestor fixes the scheme.

| Variable | Use |
| --- | --- |
| `--dt-fg`, `--dt-muted` | Text, and headers / "a–b of N" / locked rows / the limit notice. |
| `--dt-border` | Cell and box borders. |
| `--dt-header-bg`, `--dt-row-hover-bg` | Header and hovered row backgrounds. |
| `--dt-accent`, `--dt-accent-fg` | Sort marks and the current page, and its text. |
| `--dt-radius` | Box corner radius. |
| `--dt-font`, `--dt-font-size` | Font family and table font size. |
| `--dt-cell-padding` | Cell padding. |
| `--dt-max-height` | Height of the scroll box (about the header and 12 rows). |
| `--dt-filter-max-width` | Width limit of the filter inputs. |

Buttons and inputs otherwise take the page's own styles.
