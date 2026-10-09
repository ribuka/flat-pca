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
  }

  function tableOf(element) {
    return element.closest("[data-datatable]");
  }

  // A row checkbox selects its row; the header checkbox selects or clears
  // every row matching the filters, shown on this page or not.
  document.addEventListener("change", (event) => {
    const root = tableOf(event.target);
    if (!root || !root.dataset.dtSelection || !event.target.matches("[data-dt-check-all], [data-dt-row-check]")) {
      return;
    }
    const keys = selectedKeys(root);
    const targets = event.target.matches("[data-dt-check-all]") ? matchingKeys(root) : [event.target.value];
    if (canChange(root, keys, targets, event.target.checked)) {
      for (const key of targets) {
        if (event.target.checked) {
          keys.add(key);
        } else {
          keys.delete(key);
        }
      }
      setSelectedKeys(root, keys);
    }
    syncChecks(root);
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

  function queryValue(root, name) {
    return root.querySelector(`[data-dt-query][name="${CSS.escape(`${root.id}.${name}`)}"]`).value;
  }

  // Typing in a filter reloads the first page after a pause; choosing a
  // value reloads it at once.
  document.addEventListener("input", (event) => {
    const root = tableOf(event.target);
    if (!root || !event.target.matches("input[data-dt-query]")) {
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
    if (root && event.target.matches("select[data-dt-query]")) {
      reload(root, { page: "1" });
    }
  });

  // A column name sorts by that column, ascending first and then toggling;
  // the page links move between pages.
  document.addEventListener("click", (event) => {
    const root = tableOf(event.target);
    if (!root) {
      return;
    }
    const sort = event.target.closest("[data-dt-sort]");
    if (sort) {
      const descending = queryValue(root, "sort") === sort.dataset.dtSort && queryValue(root, "order") === "asc";
      reload(root, { sort: sort.dataset.dtSort, order: descending ? "desc" : "asc", page: "1" });
      return;
    }
    const page = event.target.closest("[data-dt-page]");
    if (page) {
      reload(root, { page: page.dataset.dtPage });
    }
  });

  document.addEventListener("htmx:afterSettle", (event) => {
    const root = event.detail.elt;
    if (root.matches?.("[data-datatable]") && root.dataset.dtSelection) {
      syncChecks(root);
    }
  });
})();
