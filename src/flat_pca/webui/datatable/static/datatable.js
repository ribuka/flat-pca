// Data tables: bridges between a table's controls and htmx. Filtering,
// sorting, and paging stay on the server. Each table is a `[data-datatable]`
// container (the `container` macro); several can share a page. See README.md.
(() => {
  const QUERY_DELAY_MS = 300;
  const RELOAD_EVENT = "dt-reload";
  const SELECTION_EVENT = "datatable:selection-change";

  // The selection of a selectable table over all pages and filters lives as
  // a JSON array in its hidden input (`data-dt-selection` names its id),
  // outside the reloaded fragment.
  function selectionInput(root) {
    return document.getElementById(root.dataset.dtSelection);
  }

  function selectedKeys(root) {
    return new Set(JSON.parse(selectionInput(root).value));
  }

  // Returns the most rows that can be selected (data-dt-max-selected), or
  // null for no limit.
  function maxSelected(root) {
    const max = root.dataset.dtMaxSelected;
    return max === undefined ? null : Number(max);
  }

  // Keeps the selection and tells the page: `datatable:selection-change`
  // bubbles from the container with the selected keys in `detail.keys` and
  // the limit in `detail.max` (null for none).
  function setSelectedKeys(root, keys) {
    selectionInput(root).value = JSON.stringify([...keys]);
    root.dispatchEvent(
      new CustomEvent(SELECTION_EVENT, { bubbles: true, detail: { keys: [...keys], max: maxSelected(root) } }),
    );
  }

  // Decides whether `keys` can be added to (`select`) or removed from the
  // selection `selected`: nothing changes in a locked table, and adding
  // must keep the selection within the limit. Every control that changes
  // the selection asks this.
  function canChange(root, selected, keys, select) {
    if (root.hasAttribute("data-dt-locked")) {
      return false;
    }
    const max = maxSelected(root);
    if (!select || max === null) {
      return true;
    }
    const added = keys.filter((key) => !selected.has(key));
    return new Set(added).size + selected.size <= max;
  }

  // Returns the keys of every row matching the table's filters, on all of
  // its pages.
  function matchingKeys(root) {
    return JSON.parse(root.querySelector("[data-dt-matching]").textContent);
  }

  // Shows the selection in the row checkboxes, and in the header checkbox
  // whether all, some, or none of the rows matching the filters are
  // selected. A checkbox is disabled when its click could not change the
  // selection (canChange): every one in a locked table (data-dt-locked),
  // whose filters and pages still work, and at the limit the unselected
  // rows and a header that would select past it. The limit notice shows
  // while the selection is at the limit.
  function syncChecks(root) {
    const keys = selectedKeys(root);
    for (const box of root.querySelectorAll("[data-dt-row-check]")) {
      box.checked = keys.has(box.value);
      box.disabled = !canChange(root, keys, [box.value], !box.checked);
    }
    const matching = matchingKeys(root);
    const checked = matching.filter((key) => keys.has(key)).length;
    const header = root.querySelector("[data-dt-check-all]");
    header.checked = matching.length > 0 && checked === matching.length;
    header.indeterminate = checked > 0 && checked < matching.length;
    header.disabled = matching.length === 0 || !canChange(root, keys, matching, !header.checked);
    const notice = root.querySelector("[data-dt-limit]");
    if (notice) {
      notice.hidden = keys.size < maxSelected(root);
    }
    syncSelectionState(root, keys);
  }

  // Shows the selection count under the table ("k selected", or
  // "k / max selected" with a limit) and enables Clear when it can clear.
  function syncSelectionState(root, keys) {
    const count = root.querySelector("[data-dt-selected-count]");
    if (count) {
      const max = maxSelected(root);
      count.textContent = max === null ? `${keys.size} selected` : `${keys.size} / ${max} selected`;
    }
    const clear = root.querySelector("[data-dt-clear]");
    if (clear) {
      clear.disabled = keys.size === 0 || !canChange(root, keys, [...keys], false);
    }
    const exported = root.querySelector("[data-dt-export-selected-count]");
    if (exported) {
      exported.textContent = String(keys.size);
    }
    for (const button of root.querySelectorAll("[data-dt-export][data-dt-export-rows='selected']")) {
      button.disabled = keys.size === 0;
    }
  }

  function tableOf(element) {
    return element.closest("[data-datatable]");
  }

  // Adds `targets` to the selection (`select`) or removes them, when
  // canChange allows it, then shows the selection. Returns whether it
  // changed.
  function changeSelection(root, targets, select) {
    const keys = selectedKeys(root);
    const allowed = canChange(root, keys, targets, select);
    if (allowed) {
      for (const key of targets) {
        if (select) {
          keys.add(key);
        } else {
          keys.delete(key);
        }
      }
      setSelectedKeys(root, keys);
    }
    syncChecks(root);
    return allowed;
  }

  // The header checkbox selects or clears every row matching the filters,
  // shown on this page or not.
  document.addEventListener("change", (event) => {
    const root = tableOf(event.target);
    if (root?.dataset.dtSelection && event.target.matches("[data-dt-check-all]")) {
      changeSelection(root, matchingKeys(root), event.target.checked);
    }
  });

  // The row last clicked (its key) in each table: the start of a range
  // that Shift + click selects.
  const rangeAnchors = new WeakMap();

  // Returns the keys of the rows shown from the anchor row to `row`, both
  // included, or only `row`'s key when the anchor is not shown on the page.
  function rangeKeys(root, row) {
    const rows = [...root.querySelectorAll("tbody tr[data-dt-key]")];
    const anchor = rangeAnchors.get(root);
    const from = rows.findIndex((shown) => shown.dataset.dtKey === anchor);
    const to = rows.indexOf(row);
    if (from < 0) {
      return [row.dataset.dtKey];
    }
    return rows.slice(Math.min(from, to), Math.max(from, to) + 1).map((shown) => shown.dataset.dtKey);
  }

  // Selects (`select`) or clears the clicked row, or with Shift the rows
  // from the anchor to it. A range that would pass the limit changes
  // nothing and keeps the anchor; otherwise the row becomes the anchor.
  function clickRow(root, row, select, range) {
    const targets = range ? rangeKeys(root, row) : [row.dataset.dtKey];
    if (changeSelection(root, targets, select)) {
      rangeAnchors.set(root, row.dataset.dtKey);
    }
  }

  // Elements of a row that handle their own clicks.
  const INTERACTIVE = "a, button, input, select, textarea, label, summary, [contenteditable], [tabindex]";

  function selectableRow(target) {
    const row = target.closest?.("tbody tr[data-dt-key]");
    const root = row && tableOf(row);
    return root?.dataset.dtSelection ? { root, row, box: row.querySelector("[data-dt-row-check]") } : null;
  }

  // A row checkbox, or a click anywhere else on a row, selects or clears
  // the row (with Shift, the range from the row clicked before). A click on
  // a link or another control in the row, the end of a drag selecting
  // text, or a row whose checkbox is disabled (a locked table, or an
  // unselected row at the limit) changes nothing. The checkbox has already
  // toggled itself when its click arrives.
  document.addEventListener("click", (event) => {
    const found = selectableRow(event.target);
    if (!found?.box) {
      return;
    }
    const { root, row, box } = found;
    if (event.target === box) {
      clickRow(root, row, box.checked, event.shiftKey);
      return;
    }
    if (box.disabled || event.target.closest(INTERACTIVE)) {
      return;
    }
    if (!event.shiftKey && !document.getSelection().isCollapsed) {
      return;
    }
    clickRow(root, row, !box.checked, event.shiftKey);
  });

  // Shift + click would otherwise select the text between the two clicks.
  document.addEventListener("mousedown", (event) => {
    if (event.shiftKey && selectableRow(event.target)?.box) {
      event.preventDefault();
    }
  });

  // Clear empties the selection over every page and filter.
  document.addEventListener("click", (event) => {
    const clear = event.target.closest?.("[data-dt-clear]");
    const root = clear && tableOf(clear);
    if (root?.dataset.dtSelection) {
      changeSelection(root, [...selectedKeys(root)], false);
    }
  });

  function isChosen(control) {
    return control.type === "checkbox" || control.type === "radio";
  }

  // Returns the values of query controls as they would be sent: unchecked
  // boxes and radios are left out.
  function fieldsOf(controls) {
    const fields = new URLSearchParams();
    for (const control of controls) {
      if (!isChosen(control) || control.checked) {
        fields.append(control.name, control.value);
      }
    }
    return fields;
  }

  function queryControls(element) {
    return element.querySelectorAll("[data-dt-query][name]");
  }

  // Returns the table's query (sort, order, page, search, and filters) as
  // applied: an unapplied draft of a column menu is left out
  // (withAppliedFilters).
  function appliedQueryFields(root) {
    const fields = fieldsOf(queryControls(root));
    withAppliedFilters(root, fields);
    return fields;
  }

  // Returns the file name of a download from its Content-Disposition
  // header: `filename*` (percent-encoded UTF-8) or else `filename`.
  function downloadName(disposition, fallback) {
    const encoded = /filename\*=UTF-8''([^;]+)/i.exec(disposition ?? "");
    if (encoded) {
      try {
        return decodeURIComponent(encoded[1]);
      } catch {
        // Fall back to the plain name.
      }
    }
    return /filename="([^"]*)"/i.exec(disposition ?? "")?.[1] ?? fallback;
  }

  // Returns the reason of a failed response: the `detail` of a JSON body
  // (as FastAPI answers), or the start of the text.
  function errorDetail(text) {
    try {
      const detail = JSON.parse(text).detail;
      if (typeof detail === "string") {
        return detail;
      }
    } catch {
      // Not JSON.
    }
    return text.slice(0, 200);
  }

  function saveFile(blob, name) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
  }

  // An export button posts the table's applied query, the format and rows
  // asked for, and the selection (the hidden input's JSON array, as one
  // field) to `data-dt-export-url`, and saves the file it answers. The
  // server filters and sorts the rows as for the table. A failure is shown
  // in the menu, which stays open; a success closes it.
  async function exportRows(root, button) {
    const menu = button.closest("[data-dt-menu]");
    const error = menu.querySelector("[data-dt-export-error]");
    const buttons = [...menu.querySelectorAll("[data-dt-export]")];
    const fields = appliedQueryFields(root);
    fields.append(`${root.id}.export_format`, button.dataset.dtExport);
    fields.append(`${root.id}.export_rows`, button.dataset.dtExportRows);
    if (root.dataset.dtSelection) {
      const input = selectionInput(root);
      fields.append(input.name, input.value);
    }
    const disabled = buttons.map((other) => other.disabled);
    buttons.forEach((other) => { other.disabled = true; });
    error.hidden = true;
    try {
      const response = await fetch(root.dataset.dtExportUrl, { method: "POST", body: fields });
      if (!response.ok) {
        throw new Error(`${response.status} ${errorDetail(await response.text())}`);
      }
      const fallback = `export.${button.dataset.dtExport}`;
      saveFile(await response.blob(), downloadName(response.headers.get("Content-Disposition"), fallback));
      closeMenu(button);
    } catch (failure) {
      error.textContent = `Export failed: ${failure.message}`;
      error.hidden = false;
    } finally {
      buttons.forEach((other, index) => { other.disabled = disabled[index]; });
      if (root.dataset.dtSelection) {
        syncSelectionState(root, selectedKeys(root));
      }
    }
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest?.("[data-dt-export]");
    const root = button && tableOf(button);
    if (root?.dataset.dtExportUrl) {
      exportRows(root, button);
    }
  });

  const queryTimers = new WeakMap();

  // Sets query values of a table and reloads it from the server. `values`
  // are keyed by control name without the table's prefix.
  function reload(root, values) {
    clearTimeout(queryTimers.get(root));
    const prefix = `${root.id}.`;
    for (const [name, value] of Object.entries(values)) {
      root.querySelector(`[data-dt-query][name="${CSS.escape(prefix + name)}"]`).value = value;
    }
    htmx.trigger(root, RELOAD_EVENT);
  }

  // Typing in the search box of the whole table reloads the first page
  // after a pause. The filters of the column menus wait for Apply (below).
  const TYPED_QUERY = "input[data-dt-query]:not([type='checkbox'], [type='radio'])";

  document.addEventListener("input", (event) => {
    const root = tableOf(event.target);
    if (!root || !event.target.matches(TYPED_QUERY) || event.target.closest("[data-dt-menu]")) {
      return;
    }
    clearTimeout(queryTimers.get(root));
    queryTimers.set(
      root,
      setTimeout(() => reload(root, { page: "1" }), QUERY_DELAY_MS),
    );
  });

  // The page number input under the table moves to the page typed, kept
  // between the first and the last, on Enter or when it loses the focus.
  // Anything else puts the current page back. An input notes the page it
  // asked for (`askedPages`), so that the `change` it fires after Enter,
  // also when the reload removes it, does not ask again; Enter always asks,
  // and a failed reload forgets the note, so the same page can be retried.
  const askedPages = new WeakMap();

  function goToTypedPage(input, retry) {
    const current = Number(input.dataset.dtPageCurrent);
    const typed = Math.round(Number(input.value));
    if (input.value.trim() === "" || !Number.isFinite(typed)) {
      input.value = askedPages.get(input) ?? current;
      return;
    }
    const number = Math.min(Math.max(typed, 1), Number(input.max));
    input.value = number;
    if (number === current || (!retry && askedPages.get(input) === number)) {
      return;
    }
    askedPages.set(input, number);
    reload(tableOf(input), { page: String(number) });
  }

  document.addEventListener("change", (event) => {
    if (event.target.matches("[data-dt-page-input]") && event.target.isConnected && tableOf(event.target)) {
      goToTypedPage(event.target, false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches?.("[data-dt-page-input]") && tableOf(event.target)) {
      event.preventDefault();
      goToTypedPage(event.target, true);
    }
  });

  // Only the note of the page whose request failed is forgotten: a request
  // replaced by a newer one (hx-sync) also ends unsuccessfully, and must not
  // drop the note of the page the newer one asks for.
  document.addEventListener("htmx:afterRequest", (event) => {
    const root = event.detail.elt;
    if (event.detail.successful || !root.matches?.("[data-datatable]")) {
      return;
    }
    const path = event.detail.pathInfo?.finalRequestPath ?? "";
    const failedPage = new URL(path, window.location.href).searchParams.get(`${root.id}.page`);
    for (const input of root.querySelectorAll("[data-dt-page-input]")) {
      if (String(askedPages.get(input)) === failedPage) {
        askedPages.delete(input);
      }
    }
  });

  // Clears a filter control: a typed filter is emptied, its values
  // unchecked, and a null filter set back to "Any" (the empty value).
  function clearControl(control) {
    if (control.type === "checkbox") {
      control.checked = false;
    } else if (control.type === "radio") {
      control.checked = control.value === "";
    } else {
      control.value = "";
    }
  }

  // A chip of a filter in use clears the controls of its parameters
  // (`data-dt-remove-filter`, a JSON array of names) and reloads the first
  // page.
  document.addEventListener("click", (event) => {
    const remove = event.target.closest?.("[data-dt-remove-filter]");
    const root = remove && tableOf(remove);
    if (!root) {
      return;
    }
    for (const name of JSON.parse(remove.dataset.dtRemoveFilter)) {
      root.querySelectorAll(`[data-dt-query][name="${CSS.escape(name)}"]`).forEach(clearControl);
    }
    reload(root, { page: "1" });
  });

  // Choices of the look of a table that the browser remembers per table id
  // (`name` is the kind of choice): kept in localStorage under
  // `datatable:<name>:<table_id>` as JSON, and in memory (`preferences`),
  // so a choice holds across reloads until the page is left even where the
  // storage cannot be read or written.
  const preferences = new Map();

  function preferenceKey(root, name) {
    return `datatable:${name}:${root.id}`;
  }

  // Returns the remembered choice, or `fallback` when there is none or
  // `valid` rejects the stored value.
  function preference(root, name, fallback, valid) {
    const key = preferenceKey(root, name);
    if (preferences.has(key)) {
      return preferences.get(key);
    }
    let value = fallback;
    try {
      const stored = localStorage.getItem(key);
      if (stored !== null) {
        const parsed = JSON.parse(stored);
        value = valid(parsed) ? parsed : fallback;
      }
    } catch {
      // Unreadable storage: the fallback.
    }
    preferences.set(key, value);
    return value;
  }

  function storePreference(root, name, value) {
    const key = preferenceKey(root, name);
    preferences.set(key, value);
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // The choice in memory lasts until the page is left.
    }
  }

  // The "Columns" menu (`[data-dt-columns-menu]`) shows or hides columns.
  // The hidden ones are remembered (`hidden-columns`) and hidden by a style
  // sheet in the document head, which outlives the reloads, so a reloaded
  // table shows no hidden column even for a moment.
  function storedHiddenColumns(root) {
    return new Set(preference(root, "hidden-columns", [], Array.isArray));
  }

  function storeHiddenColumns(root, names) {
    storePreference(root, "hidden-columns", [...names]);
  }

  // Hides the columns unchecked in the "Columns" menu, checks its boxes from
  // the remembered choice when `fromStorage`, and counts the shown columns
  // under the table. The first column is always shown.
  function syncColumns(root, fromStorage) {
    const menu = root.querySelector("[data-dt-columns-menu]");
    const sheetId = `${root.id}-dt-hidden-columns`;
    let sheet = document.getElementById(sheetId);
    if (!menu) {
      sheet?.remove();
      return;
    }
    const boxes = [...menu.querySelectorAll("[data-dt-show-column]")];
    if (fromStorage) {
      const hidden = storedHiddenColumns(root);
      for (const box of boxes) {
        box.checked = box.disabled || !hidden.has(box.dataset.dtShowColumn);
      }
    }
    const hidden = new Set(boxes.filter((box) => !box.checked).map((box) => box.dataset.dtShowColumn));
    const cells = [...root.querySelectorAll(".dt-table thead tr > th")];
    const rules = cells
      .map((cell, index) => [cell.querySelector("[data-dt-column]")?.dataset.dtColumn, index + 1])
      .filter(([name]) => hidden.has(name))
      .map(([, child]) => `#${CSS.escape(root.id)} .dt-table tr > :nth-child(${child}) { display: none; }`);
    if (!sheet) {
      sheet = document.createElement("style");
      sheet.id = sheetId;
      document.head.append(sheet);
    }
    sheet.textContent = rules.join("\n");
    const count = root.querySelector("[data-dt-column-count]");
    if (count) {
      count.textContent = String(boxes.length - hidden.size);
    }
  }

  document.addEventListener("change", (event) => {
    const root = tableOf(event.target);
    if (root && event.target.matches("[data-dt-show-column]")) {
      syncColumns(root, false);
      storeHiddenColumns(
        root,
        [...root.querySelectorAll("[data-dt-show-column]")].filter((box) => !box.checked).map((box) => box.dataset.dtShowColumn),
      );
    }
  });

  // The "Pin columns" toggle (`[data-dt-pin-toggle]`) pins the cells marked
  // `.dt-pinned` (the checkbox column and the first column) at the left. The
  // choice is remembered (`pinned-columns`, off by default) and shown on the
  // container (`data-dt-pinned`), which the reloads keep, so a reloaded table
  // does not jump; the toggle shows it with `aria-pressed`. A table without
  // the toggle pins nothing.
  function syncPinned(root) {
    const toggle = root.querySelector("[data-dt-pin-toggle]");
    const pinned = Boolean(toggle) && preference(root, "pinned-columns", false, (value) => typeof value === "boolean");
    root.toggleAttribute("data-dt-pinned", pinned);
    toggle?.setAttribute("aria-pressed", String(pinned));
  }

  document.addEventListener("click", (event) => {
    const toggle = event.target.closest?.("[data-dt-pin-toggle]");
    const root = toggle && tableOf(toggle);
    if (root) {
      storePreference(root, "pinned-columns", !root.hasAttribute("data-dt-pinned"));
      syncPinned(root);
    }
  });

  // Column widths. Every column but the checkbox column starts at
  // `--dt-col-width`; the handle on the right edge of its header
  // (`[data-dt-resize]`, the column name) resizes it by a drag, fits it to
  // its header and shown values by a double-click, and moves it by
  // `WIDTH_STEP_PX` with ←/→ while focused. A column is never narrower than
  // `minimumWidth`. The widths set are remembered (`column-widths`, pixels
  // by column name) and applied, like the hidden columns, by a style sheet in
  // the document head that outlives the reloads; "Reset column widths" in
  // the "Columns" menu forgets them.
  const WIDTH_STEP_PX = 10;

  function storedWidths(root) {
    return preference(
      root,
      "column-widths",
      {},
      (value) => value !== null && typeof value === "object" && !Array.isArray(value)
        && Object.values(value).every((width) => Number.isFinite(width) && width > 0),
    );
  }

  // Keeps a column's width in memory, and in the storage when `persist`
  // (at the end of a drag, so that a drag does not write at every move).
  function setWidth(root, name, width, persist) {
    const widths = { ...storedWidths(root), [name]: Math.round(width) };
    if (persist) {
      storePreference(root, "column-widths", widths);
    } else {
      preferences.set(preferenceKey(root, "column-widths"), widths);
    }
    syncWidths(root);
  }

  // Sizes the resized columns, and enables "Reset column widths" while
  // there are any. A remembered width is kept within the current minimum
  // (`minimumWidth`), should the page have raised it, or the padding, since;
  // the rule keeps `--dt-col-min-width` for changes after this.
  function syncWidths(root) {
    const widths = storedWidths(root);
    const sheetId = `${root.id}-dt-column-widths`;
    let sheet = document.getElementById(sheetId);
    if (!sheet) {
      sheet = document.createElement("style");
      sheet.id = sheetId;
      document.head.append(sheet);
    }
    const cells = [...root.querySelectorAll(".dt-table thead tr > th")];
    sheet.textContent = cells
      .map((cell, index) => [cell, cell.querySelector("[data-dt-resize]")?.dataset.dtResize, index + 1])
      .filter(([, name]) => name !== undefined && Object.hasOwn(widths, name))
      .map(([cell, name, child]) => `#${CSS.escape(root.id)} .dt-table thead tr > th:nth-child(${child})`
        + ` { width: max(var(--dt-col-min-width), ${Math.max(minimumWidth(cell), widths[name])}px); }`)
      .join("\n");
    for (const handle of root.querySelectorAll("[data-dt-resize]")) {
      const width = Math.round(handle.parentElement.getBoundingClientRect().width);
      handle.setAttribute("aria-valuenow", String(width));
      handle.setAttribute("aria-valuetext", `${width} px`);
    }
    const reset = root.querySelector("[data-dt-reset-widths]");
    if (reset) {
      reset.disabled = Object.keys(widths).length === 0;
    }
  }

  // Returns the width of an element's contents in `cell` (an element made
  // at once and removed) when laid out as `css` asks.
  function probeWidth(cell, css, contents = []) {
    const probe = document.createElement("div");
    probe.className = "dt-measure";
    probe.style.cssText = css;
    probe.append(...contents);
    cell.append(probe);
    const width = probe.getBoundingClientRect().width;
    probe.remove();
    return width;
  }

  // The narrowest a column gets: `--dt-col-min-width`, and never less than
  // the histogram with the cell's left and right padding, so that the
  // histogram and the column name's button stay in view.
  function minimumWidth(cell) {
    const style = getComputedStyle(cell);
    const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    return probeWidth(cell, `width: max(var(--dt-col-min-width), calc(var(--dt-histogram-width) + ${padding}px))`);
  }

  // Returns the width a cell needs to show its contents whole: its contents
  // copied at their natural width (but the column menu and the handle),
  // with its padding and borders.
  function neededWidth(cell) {
    const contents = [...cell.childNodes]
      .filter((node) => node.nodeType !== Node.ELEMENT_NODE || !node.matches("[popover], [data-dt-resize]"))
      .map((node) => node.cloneNode(true));
    const style = getComputedStyle(cell);
    const edges = ["paddingLeft", "paddingRight", "borderLeftWidth", "borderRightWidth"]
      .reduce((sum, edge) => sum + parseFloat(style[edge]), 0);
    return probeWidth(cell, "", contents) + edges;
  }

  // Fits a column to its header and the values of the rows shown on the
  // page, within the minimum.
  function fitWidth(root, handle) {
    const header = handle.parentElement;
    const child = [...header.parentElement.children].indexOf(header) + 1;
    const cells = [header, ...root.querySelectorAll(`.dt-table tbody tr > :nth-child(${child})`)];
    const needed = Math.ceil(Math.max(...cells.map(neededWidth)));
    setWidth(root, handle.dataset.dtResize, Math.max(minimumWidth(header), needed), true);
  }

  // A drag follows the pointer (captured by the handle) from the width the
  // column had, live, and remembers the width where it ends. Taking the
  // pointer down cancels the text selection and the clicks under it.
  document.addEventListener("pointerdown", (event) => {
    const handle = event.target.closest?.("[data-dt-resize]");
    const root = handle && tableOf(handle);
    if (!root || event.button !== 0) {
      return;
    }
    event.preventDefault();
    handle.focus({ preventScroll: true });
    const header = handle.parentElement;
    const name = handle.dataset.dtResize;
    const startX = event.clientX;
    const startWidth = header.getBoundingClientRect().width;
    const minimum = minimumWidth(header);
    handle.setPointerCapture(event.pointerId);
    handle.dataset.dtResizing = "";
    root.dataset.dtResizing = "";
    const move = (moved) => {
      setWidth(root, name, Math.max(minimum, startWidth + moved.clientX - startX), false);
    };
    const end = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("lostpointercapture", end);
      delete handle.dataset.dtResizing;
      delete root.dataset.dtResizing;
      if (Object.hasOwn(storedWidths(root), name)) {
        storePreference(root, "column-widths", storedWidths(root));
      }
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("lostpointercapture", end);
  });

  document.addEventListener("dblclick", (event) => {
    const handle = event.target.closest?.("[data-dt-resize]");
    const root = handle && tableOf(handle);
    if (root) {
      event.preventDefault();
      fitWidth(root, handle);
    }
  });

  document.addEventListener("keydown", (event) => {
    const handle = event.target.closest?.("[data-dt-resize]");
    const root = handle && tableOf(handle);
    const step = { ArrowLeft: -WIDTH_STEP_PX, ArrowRight: WIDTH_STEP_PX }[event.key];
    if (!root || step === undefined) {
      return;
    }
    event.preventDefault();
    const header = handle.parentElement;
    const width = header.getBoundingClientRect().width + step;
    setWidth(root, handle.dataset.dtResize, Math.max(minimumWidth(header), width), true);
  });

  document.addEventListener("click", (event) => {
    const reset = event.target.closest?.("[data-dt-reset-widths]");
    const root = reset && tableOf(reset);
    if (root) {
      storePreference(root, "column-widths", {});
      syncWidths(root);
    }
  });

  // A value (or a column name) cut short by its column shows whole as its
  // tooltip while the pointer is on it. A cell keeps the tooltip the
  // application gave it (`title_column`).
  document.addEventListener("mouseover", (event) => {
    const cell = event.target.closest?.(".dt-table td, .dt-table .dt-label");
    if (!cell || !tableOf(cell) || (cell.hasAttribute("title") && !("dtAutoTitle" in cell.dataset))) {
      return;
    }
    if (cell.scrollWidth > cell.clientWidth) {
      cell.title = cell.textContent.trim();
      cell.dataset.dtAutoTitle = "";
    } else if ("dtAutoTitle" in cell.dataset) {
      cell.removeAttribute("title");
      delete cell.dataset.dtAutoTitle;
    }
  });

  // Menus are popovers (`[data-dt-menu]`): a column's, opened by its name,
  // and the "Columns" menu. The browser opens and closes them (a click
  // outside or Esc closes one) and shows them in the top layer, so neither
  // the table's scroll box nor a <dialog> clips them; this code places them
  // under the button opening them and keeps the one that was open open
  // across reloads.
  const MENU_GAP_PX = 4;
  const MENU_MARGIN_PX = 8;

  function menuButton(menu) {
    return document.querySelector(`[popovertarget="${CSS.escape(menu.id)}"]`);
  }

  // Puts a menu under its button, or above it when there is no room
  // below, and always inside the window. It stays invisible until placed
  // (`data-dt-placed`), since its size is known only once it shows.
  function placeMenu(menu) {
    const anchor = menuButton(menu).getBoundingClientRect();
    const width = document.documentElement.clientWidth;
    const height = document.documentElement.clientHeight;
    const box = menu.getBoundingClientRect();
    const left = Math.max(MENU_MARGIN_PX, Math.min(anchor.left, width - box.width - MENU_MARGIN_PX));
    let top = anchor.bottom + MENU_GAP_PX;
    if (top + box.height > height - MENU_MARGIN_PX) {
      const above = anchor.top - MENU_GAP_PX - box.height;
      top = above >= MENU_MARGIN_PX ? above : height - box.height - MENU_MARGIN_PX;
    }
    menu.style.left = `${left}px`;
    menu.style.top = `${Math.max(MENU_MARGIN_PX, top)}px`;
    menu.dataset.dtPlaced = "";
  }

  function openMenus() {
    return document.querySelectorAll("[data-dt-menu]:popover-open");
  }

  // Toggle events do not bubble, so listen while capturing.
  document.addEventListener(
    "toggle",
    (event) => {
      if (!event.target.matches?.("[data-dt-menu]")) {
        return;
      }
      if (event.newState === "open") {
        placeMenu(event.target);
      } else {
        delete event.target.dataset.dtPlaced;
      }
    },
    true,
  );

  // An open menu follows its column name when the page or the table scrolls.
  for (const type of ["scroll", "resize"]) {
    window.addEventListener(
      type,
      () => {
        for (const menu of openMenus()) {
          placeMenu(menu);
        }
      },
      { capture: true, passive: true },
    );
  }

  function closeMenu(element) {
    const menu = element.closest("[data-dt-menu]");
    if (menu?.matches(":popover-open")) {
      menu.hidePopover();
    }
  }

  // Copies text to the clipboard, also where navigator.clipboard is missing
  // (a page served over plain HTTP from another host). The fallback selects
  // the text in a textarea put in `container` (the open menu), which stays
  // usable inside a modal <dialog>, where the rest of the page is inert.
  function copyText(text, container) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).catch(() => {});
      return;
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    container.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }

  // Escape in an open menu closes it and returns the focus to its column
  // name, also for a menu reopened after a reload (which has no invoker to
  // return to). Cancelling the key keeps a surrounding <dialog> open.
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || event.defaultPrevented) {
      return;
    }
    const menu = document.activeElement?.closest?.("[data-dt-menu]");
    if (!menu?.matches(":popover-open")) {
      return;
    }
    event.preventDefault();
    menu.hidePopover();
    menuButton(menu).focus();
  });

  // A sort button of a column menu sorts by its column (or clears the sort
  // with an empty `data-dt-sort`) and closes the menu; "Copy column name"
  // copies it and closes the menu; the page buttons move between pages.
  document.addEventListener("click", (event) => {
    const root = tableOf(event.target);
    if (!root) {
      return;
    }
    const sort = event.target.closest("[data-dt-sort]");
    if (sort) {
      closeMenu(sort);
      reload(root, { sort: sort.dataset.dtSort, order: sort.dataset.dtOrder, page: "1" });
      return;
    }
    const copy = event.target.closest("[data-dt-copy]");
    if (copy) {
      copyText(copy.dataset.dtCopy, copy.closest("[data-dt-menu]"));
      closeMenu(copy);
      return;
    }
    const page = event.target.closest("[data-dt-page]");
    if (page) {
      reload(root, { page: page.dataset.dtPage });
    }
  });

  // The filters of a column menu (one with `[data-dt-apply]`) are a draft
  // while it is open: changing them reloads nothing. Opening the menu notes
  // its filters as applied (`drafts`, per table: the menu's id and its
  // fields). Apply reloads the first page with the draft and closes the
  // menu; Cancel, Escape, a click outside, or any other closing puts the
  // applied filters back. Clear empties the column's filters, reloads the
  // first page, and closes the menu. A reload while a draft is open (the
  // search box or a page button reached with Tab, or a trigger of the page)
  // and an export send the applied filters, and the menu reopened after the
  // reload gets its draft back.
  const drafts = new WeakMap();

  function hasFilters(menu) {
    return Boolean(menu.matches?.("[data-dt-menu]") && menu.querySelector("[data-dt-apply]"));
  }

  // Returns the open draft of `menu`, or undefined when it has none.
  function draftOf(menu) {
    const draft = drafts.get(tableOf(menu));
    return draft?.menu === menu.id ? draft : undefined;
  }

  // Sets filter controls to `fields` (as fieldsOf returns them).
  function setControls(controls, fields) {
    for (const control of controls) {
      if (isChosen(control)) {
        control.checked = fields.getAll(control.name).includes(control.value);
      } else {
        control.value = fields.get(control.name) ?? "";
      }
    }
  }

  // Shows whether a menu's draft differs from the applied filters: then
  // Apply is enabled and stands out (`data-dt-dirty` on the menu) and the
  // notice of unapplied changes shows. An open menu is placed again, since
  // the notice changes its height.
  function syncDraft(menu) {
    const draft = draftOf(menu);
    const dirty = draft !== undefined && fieldsOf(queryControls(menu)).toString() !== draft.applied.toString();
    menu.toggleAttribute("data-dt-dirty", dirty);
    menu.querySelector("[data-dt-apply]").disabled = !dirty;
    const status = menu.querySelector("[data-dt-draft-status]");
    if (status.hidden !== !dirty) {
      status.hidden = !dirty;
      if (menu.matches(":popover-open")) {
        placeMenu(menu);
      }
    }
  }

  // Puts the applied filters of the table's open draft, if any, in place of
  // the draft's in `fields` (URLSearchParams or FormData).
  function withAppliedFilters(root, fields) {
    const draft = drafts.get(root);
    const menu = draft && document.getElementById(draft.menu);
    if (!menu) {
      return;
    }
    for (const name of new Set([...queryControls(menu)].map((control) => control.name))) {
      fields.delete(name);
    }
    for (const [name, value] of draft.applied) {
      fields.append(name, value);
    }
  }

  function applyDraft(menu) {
    const root = tableOf(menu);
    drafts.delete(root);
    closeMenu(menu);
    reload(root, { page: "1" });
  }

  function clearFilters(menu) {
    const root = tableOf(menu);
    drafts.delete(root);
    queryControls(menu).forEach(clearControl);
    closeMenu(menu);
    reload(root, { page: "1" });
  }

  // beforetoggle comes before the menu closes, also when a click outside
  // closes it, so the applied filters are back before that click lands.
  // Toggle events do not bubble, so listen while capturing. A menu taken
  // out of the page by a reload closes without the event and keeps its
  // draft for the reopening.
  document.addEventListener(
    "beforetoggle",
    (event) => {
      const menu = event.target;
      if (!hasFilters(menu)) {
        return;
      }
      const draft = draftOf(menu);
      if (event.newState === "open") {
        if (!draft) {
          drafts.set(tableOf(menu), { menu: menu.id, applied: fieldsOf(queryControls(menu)) });
        }
      } else if (draft) {
        setControls(queryControls(menu), draft.applied);
        drafts.delete(tableOf(menu));
      }
      syncDraft(menu);
    },
    true,
  );

  for (const type of ["input", "change"]) {
    document.addEventListener(type, (event) => {
      const menu = event.target.closest?.("[data-dt-menu]");
      if (menu && event.target.matches("[data-dt-query]") && hasFilters(menu) && tableOf(menu)) {
        syncDraft(menu);
      }
    });
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest?.("[data-dt-apply], [data-dt-cancel], [data-dt-clear-filter]");
    const menu = button?.closest("[data-dt-menu]");
    if (!menu || !tableOf(menu)) {
      return;
    }
    if (button.matches("[data-dt-apply]")) {
      applyDraft(menu);
    } else if (button.matches("[data-dt-clear-filter]")) {
      clearFilters(menu);
    } else {
      closeMenu(menu);
      menuButton(menu)?.focus();
    }
  });

  // Enter in a search or a bound of a column menu is Apply (nothing while
  // the draft equals the applied filters). Enter that ends an IME
  // composition only confirms the text.
  document.addEventListener("keydown", (event) => {
    const menu = event.target.closest?.("[data-dt-menu]");
    if (event.key !== "Enter" || event.isComposing || !menu || !event.target.matches(TYPED_QUERY) || !hasFilters(menu)) {
      return;
    }
    event.preventDefault();
    if (!menu.querySelector("[data-dt-apply]").disabled) {
      applyDraft(menu);
    }
  });

  document.addEventListener("htmx:configRequest", (event) => {
    const root = event.detail.elt;
    if (root?.matches?.("[data-datatable]")) {
      withAppliedFilters(root, event.detail.formData);
    }
  });

  // A reload replaces the menus. The menu open before it, the control
  // focused in it, and its draft are noted here and restored after the
  // swap, so that a reload while a menu is open keeps it open as it was.
  const reopenedMenus = new WeakMap();

  document.addEventListener("htmx:beforeSwap", (event) => {
    const root = event.detail.target ?? event.target;
    if (!root?.matches?.("[data-datatable]")) {
      return;
    }
    const menu = root.querySelector("[data-dt-menu]:popover-open");
    if (!menu) {
      reopenedMenus.delete(root);
      drafts.delete(root);
      return;
    }
    const focused = menu.contains(document.activeElement) ? document.activeElement : null;
    reopenedMenus.set(root, {
      menu: menu.id,
      id: focused?.id || null,
      name: focused?.name || null,
      value: focused?.value ?? null,
      selection: focused && typeof focused.selectionStart === "number"
        ? [focused.selectionStart, focused.selectionEnd]
        : null,
      draft: hasFilters(menu) && draftOf(menu) ? fieldsOf(queryControls(menu)) : null,
    });
  });

  function reopenMenu(root) {
    const noted = reopenedMenus.get(root);
    reopenedMenus.delete(root);
    const menu = noted && document.getElementById(noted.menu);
    if (!menu) {
      drafts.delete(root);
      return;
    }
    try {
      menu.showPopover({ source: menuButton(menu) });
    } catch {
      // Another popover took its place or the table left the page.
      drafts.delete(root);
      return;
    }
    if (noted.draft && hasFilters(menu)) {
      setControls(queryControls(menu), noted.draft);
      syncDraft(menu);
    }
    // Place it at once: a control in a menu not yet placed cannot be focused.
    placeMenu(menu);
    const focused = noted.id
      ? menu.querySelector(`#${CSS.escape(noted.id)}`)
      : noted.name && [...menu.querySelectorAll(`[name="${CSS.escape(noted.name)}"]`)]
        .find((control) => control.value === noted.value);
    if (focused) {
      focused.focus({ preventScroll: true });
      if (noted.selection) {
        try {
          focused.setSelectionRange(...noted.selection);
        } catch {
          // Number and date inputs have no text selection.
        }
      }
    }
  }

  // Reopen as soon as the new menus are in the page, so that no Escape or
  // click outside falls between the swap and the reopening, and hide the
  // hidden columns and show the pinned state and the column widths before
  // the new rows are drawn.
  document.addEventListener("htmx:afterSwap", (event) => {
    const root = event.detail.target ?? event.target;
    if (root?.matches?.("[data-datatable]")) {
      syncColumns(root, true);
      syncPinned(root);
      syncWidths(root);
      reopenMenu(root);
    }
  });

  // Settling resets the attributes of elements kept by id to those of the
  // response, which drops the placement of a reopened menu; place it again
  // in the same task, before it is drawn. A menu closed meanwhile stays
  // closed.
  document.addEventListener("htmx:afterSettle", (event) => {
    const root = event.detail.elt;
    if (!root.matches?.("[data-datatable]")) {
      return;
    }
    for (const menu of root.querySelectorAll("[data-dt-menu]:popover-open")) {
      placeMenu(menu);
    }
    if (root.dataset.dtSelection) {
      syncChecks(root);
    }
  });
})();
