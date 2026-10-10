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

  // Returns the table's query controls (sort, order, page, search, and
  // filters) as they would be sent to reload it: unchecked boxes and radios
  // are left out.
  function queryFields(root) {
    const fields = new URLSearchParams();
    for (const control of root.querySelectorAll("[data-dt-query][name]")) {
      if ((control.type === "checkbox" || control.type === "radio") && !control.checked) {
        continue;
      }
      fields.append(control.name, control.value);
    }
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

  // An export button posts the table's query controls, the format and rows
  // asked for, and the selection (the hidden input's JSON array, as one
  // field) to `data-dt-export-url`, and saves the file it answers. The
  // server filters and sorts the rows as for the table. A failure is shown
  // in the menu, which stays open; a success closes it.
  async function exportRows(root, button) {
    const menu = button.closest("[data-dt-menu]");
    const error = menu.querySelector("[data-dt-export-error]");
    const buttons = [...menu.querySelectorAll("[data-dt-export]")];
    const fields = queryFields(root);
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

  // Typing in a filter reloads the first page after a pause; checking a
  // value or a null filter reloads it at once.
  const TYPED_QUERY = "input[data-dt-query]:not([type='checkbox'], [type='radio'])";
  const CHOSEN_QUERY = "input[data-dt-query]:is([type='checkbox'], [type='radio'])";

  document.addEventListener("input", (event) => {
    const root = tableOf(event.target);
    if (!root || !event.target.matches(TYPED_QUERY)) {
      return;
    }
    clearTimeout(queryTimers.get(root));
    queryTimers.set(
      root,
      setTimeout(() => reload(root, { page: "1" }), QUERY_DELAY_MS),
    );
  });

  document.addEventListener("change", (event) => {
    const root = tableOf(event.target);
    if (root && event.target.matches(CHOSEN_QUERY)) {
      reload(root, { page: "1" });
    }
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

  // A chip of a filter in use clears the controls of its parameters
  // (`data-dt-remove-filter`, a JSON array of names) and reloads the first
  // page: a typed filter is emptied, its values unchecked, and a null filter
  // set back to "Any" (the empty value).
  document.addEventListener("click", (event) => {
    const remove = event.target.closest?.("[data-dt-remove-filter]");
    const root = remove && tableOf(remove);
    if (!root) {
      return;
    }
    for (const name of JSON.parse(remove.dataset.dtRemoveFilter)) {
      for (const control of root.querySelectorAll(`[data-dt-query][name="${CSS.escape(name)}"]`)) {
        if (control.type === "checkbox") {
          control.checked = false;
        } else if (control.type === "radio") {
          control.checked = control.value === "";
        } else {
          control.value = "";
        }
      }
    }
    reload(root, { page: "1" });
  });

  // The "Columns" menu (`[data-dt-columns-menu]`) shows or hides columns.
  // The hidden ones are remembered per table id in localStorage and hidden
  // by a style sheet in the document head, which outlives the reloads, so
  // a reloaded table shows no hidden column even for a moment. The latest
  // choice of each table id is also kept in memory (`hiddenColumns`), so it
  // holds across reloads until the page is left even where the storage
  // cannot be read or written.
  const hiddenColumns = new Map();

  function hiddenColumnsKey(root) {
    return `datatable:hidden-columns:${root.id}`;
  }

  function storedHiddenColumns(root) {
    if (hiddenColumns.has(root.id)) {
      return hiddenColumns.get(root.id);
    }
    let hidden = new Set();
    try {
      const stored = JSON.parse(localStorage.getItem(hiddenColumnsKey(root)) ?? "[]");
      hidden = new Set(Array.isArray(stored) ? stored : []);
    } catch {
      // Unreadable storage: every column is shown.
    }
    hiddenColumns.set(root.id, hidden);
    return hidden;
  }

  function storeHiddenColumns(root, names) {
    hiddenColumns.set(root.id, new Set(names));
    try {
      localStorage.setItem(hiddenColumnsKey(root), JSON.stringify([...names]));
    } catch {
      // The choice in memory lasts until the page is left.
    }
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

  // A reload replaces the menus. The menu open before it, and the control
  // focused in it, are noted here and restored after the swap, so that
  // checking values or typing a search keeps the menu open.
  const reopenedMenus = new WeakMap();

  document.addEventListener("htmx:beforeSwap", (event) => {
    const root = event.detail.target ?? event.target;
    if (!root?.matches?.("[data-datatable]")) {
      return;
    }
    const menu = root.querySelector("[data-dt-menu]:popover-open");
    if (!menu) {
      reopenedMenus.delete(root);
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
    });
  });

  function reopenMenu(root) {
    const noted = reopenedMenus.get(root);
    reopenedMenus.delete(root);
    const menu = noted && document.getElementById(noted.menu);
    if (!menu) {
      return;
    }
    try {
      menu.showPopover({ source: menuButton(menu) });
    } catch {
      // Another popover took its place or the table left the page.
      return;
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
  // hidden columns before the new rows are drawn.
  document.addEventListener("htmx:afterSwap", (event) => {
    const root = event.detail.target ?? event.target;
    if (root?.matches?.("[data-datatable]")) {
      syncColumns(root, true);
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
