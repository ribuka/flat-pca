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

  // Column menus are popovers (`[data-dt-menu]`) opened by the column
  // names. The browser opens and closes them (a click outside or Esc closes
  // one) and shows them in the top layer, so neither the table's scroll box
  // nor a <dialog> clips them; this code places them under their column
  // name and keeps the one that was open open across reloads.
  const MENU_GAP_PX = 4;
  const MENU_MARGIN_PX = 8;

  function menuButton(menu) {
    return menu.closest("th").querySelector("[data-dt-column]");
  }

  // Puts a menu under its column name, or above it when there is no room
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
  // (a page served over plain HTTP from another host).
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).catch(() => {});
      return;
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }

  // A sort button of a column menu sorts by its column (or clears the sort
  // with an empty `data-dt-sort`) and closes the menu; "Copy column name"
  // copies it and closes the menu; the page links move between pages.
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
      copyText(copy.dataset.dtCopy);
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
    const root = event.detail.target;
    if (!root.matches?.("[data-datatable]")) {
      return;
    }
    const menu = root.querySelector("[data-dt-menu]:popover-open");
    if (!menu) {
      reopenedMenus.delete(root);
      return;
    }
    const focused = menu.contains(document.activeElement) ? document.activeElement : null;
    reopenedMenus.set(root, {
      column: menu.dataset.dtMenu,
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
    const menu = noted && root.querySelector(`[data-dt-menu="${CSS.escape(noted.column)}"]`);
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

  document.addEventListener("htmx:afterSettle", (event) => {
    const root = event.detail.elt;
    if (!root.matches?.("[data-datatable]")) {
      return;
    }
    reopenMenu(root);
    if (root.dataset.dtSelection) {
      syncChecks(root);
    }
  });
})();
