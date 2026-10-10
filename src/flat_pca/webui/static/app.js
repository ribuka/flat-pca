// Bridges between page controls and htmx. Data processing stays on the server.
// The data tables have their own script (datatable/static/datatable.js).

const BUSY_SHOW_DELAY_MS = 300;
const BUSY_TICK_MS = 250;

// The page-wide overlay shown while a request or a job is pending: it greys
// out the page, blocks input, and shows a spinner, a message, the elapsed
// seconds, optional details, and a cancel button. `start` takes the action of
// the cancel button and these options:
// - `owner`: a key telling who started the overlay (see `owner()`).
// - `message`: the text above the elapsed seconds.
// - `elapsedS`: the seconds already elapsed when the overlay starts.
// - `keepOnCancel`: whether cancelling keeps the overlay until its owner
//   stops it, as for a job that ends some time after the cancel request.
const busyOverlay = (() => {
  const overlay = document.getElementById("busy-overlay");
  const message = document.getElementById("busy-message");
  const elapsed = document.getElementById("busy-elapsed");
  const detail = document.getElementById("busy-detail");
  const cancel = document.getElementById("busy-cancel");
  const layout = document.querySelector(".layout");
  const defaultMessage = message.textContent;
  let showTimer = null;
  let tickTimer = null;
  let cancelAction = null;
  let currentOwner = null;
  let keepOpen = false;
  let startedAt = 0;

  function showElapsed() {
    elapsed.textContent = `${Math.max(0, Math.floor((performance.now() - startedAt) / 1000))}`;
  }

  function stop() {
    clearTimeout(showTimer);
    clearInterval(tickTimer);
    cancelAction = null;
    currentOwner = null;
    keepOpen = false;
    overlay.hidden = true;
    overlay.classList.remove("busy-visible");
    layout.inert = false;
    message.textContent = defaultMessage;
    detail.replaceChildren();
    detail.hidden = true;
    cancel.disabled = false;
  }

  // Counts the elapsed seconds from `elapsedS` now, and shows `source`'s
  // children (copied) below them, or nothing without a source.
  function update(elapsedS, source = null) {
    startedAt = performance.now() - elapsedS * 1000;
    showElapsed();
    detail.replaceChildren(...(source ? source.cloneNode(true).childNodes : []));
    detail.hidden = !source;
  }

  function start(onCancel, options = {}) {
    stop();
    cancelAction = onCancel;
    currentOwner = options.owner ?? null;
    keepOpen = options.keepOnCancel ?? false;
    message.textContent = options.message ?? defaultMessage;
    update(options.elapsedS ?? 0);
    // Input is blocked at once; the overlay is drawn only after the delay.
    layout.inert = true;
    overlay.hidden = false;
    showTimer = setTimeout(() => {
      overlay.classList.add("busy-visible");
      cancel.focus();
    }, BUSY_SHOW_DELAY_MS);
    tickTimer = setInterval(showElapsed, BUSY_TICK_MS);
  }

  // With `keepOnCancel`, the action returns a promise of whether the cancel
  // request was accepted; a refused or failed request lets the user retry.
  cancel.addEventListener("click", async () => {
    const action = cancelAction;
    if (!keepOpen) {
      stop();
      if (action) {
        action();
      }
      return;
    }
    const cancelledOwner = currentOwner;
    cancel.disabled = true;
    message.textContent = "Cancelling…";
    let accepted = false;
    try {
      accepted = action ? await action() : true;
    } catch {
      // A failed request is reported below like a refused one.
    }
    if (!accepted && currentOwner === cancelledOwner) {
      cancel.disabled = false;
      message.textContent = "Could not cancel. Try again.";
    }
  });

  // Returns the `owner` of the shown overlay, or null.
  function owner() {
    return currentOwner;
  }

  return { start, update, stop, owner };
})();

// The job status panel (of a fit or transform run): while its run is queued
// or running, the overlay covers the page with the panel's message, the run's
// elapsed time, and its progress and remaining time, and its cancel button
// cancels the run. A poll of the panel that finds the run finished closes the
// overlay and opens the panel's group to show the result.
function syncJobOverlay() {
  const panel = document.querySelector("[data-job-status]");
  const owner = busyOverlay.owner();
  const jobOwned = owner !== null && owner.startsWith("job:");
  if (!panel || panel.dataset.jobActive === undefined) {
    if (jobOwned) {
      busyOverlay.stop();
      const group = panel?.closest("details");
      if (group) {
        group.open = true;
      }
    }
    return;
  }
  const key = `job:${panel.dataset.runId}`;
  if (owner !== key) {
    const cancelUrl = panel.dataset.cancelUrl;
    // A run that is no longer active (409) is closed by the next poll.
    const cancelRun = async () => {
      const response = await fetch(cancelUrl, { method: "POST" });
      return response.ok || response.status === 409;
    };
    busyOverlay.start(cancelRun, {
      owner: key,
      message: panel.dataset.busyMessage,
      keepOnCancel: true,
    });
  }
  busyOverlay.update(Number(panel.dataset.elapsedS), panel.querySelector("[data-job-progress]"));
}

document.addEventListener("htmx:afterSettle", syncJobOverlay);

// A control that fails the browser's validation inside a collapsed group
// opens the group, so the browser can focus it and show the message.
document.addEventListener(
  "invalid",
  (event) => {
    for (let group = event.target.closest("details"); group; group = group.parentElement.closest("details")) {
      group.open = true;
    }
  },
  true,
);

// Submits a GET form whenever one of its controls changes, including the
// controls joined to it from other cards with the form attribute.
document.addEventListener("change", (event) => {
  const form = event.target.form;
  if (form?.matches("[data-auto-submit]")) {
    form.requestSubmit();
  }
});

// While the next page loads, the overlay covers the page. Cancelling stops
// the load and resets the form to the shown conditions, which are its
// defaults. The server still finishes the computation.
document.addEventListener("submit", (event) => {
  const form = event.target.closest("form[data-auto-submit]");
  if (form) {
    busyOverlay.start(() => {
      window.stop();
      form.reset();
    });
  }
});

// The controls of the main part that are not drawn by the server: the
// conditions of a page restored by "back" (from the bfcache or with restored
// form state) that were never shown, and the overlay of an active fit or
// transform run.
function syncMainControls() {
  for (const form of document.querySelectorAll("form[data-auto-submit]")) {
    form.reset();
  }
  syncJobOverlay();
}

// Functions that stop the work of the shown main part, such as pending trend
// requests, before it is replaced.
let mainDisposers = [];

// Sets up the main part of a loaded page or of a replaced one: the heatmaps
// with their sliders and trends, the point tables, the figures, and the
// controls of syncMainControls. `pointStems` maps the id of a point table to
// the stems it shows at first.
function initMain(main, pointStems = {}) {
  for (const root of main.querySelectorAll("[data-explore]")) {
    mainDisposers.push(initExplore(root));
  }
  pointTables.clear();
  for (const root of main.querySelectorAll(".point-table")) {
    pointTables.set(root.id, initPointTable(root, pointStems[root.id] ?? []));
  }
  for (const target of main.querySelectorAll("[data-plot]")) {
    initPlot(target);
  }
  syncMainControls();
}

// Stops the work of the main part and frees its figures, whose responsive
// resizing listens on the window.
function disposeMain(main) {
  for (const dispose of mainDisposers) {
    dispose();
  }
  mainDisposers = [];
  for (const plot of main.querySelectorAll(".js-plotly-plot")) {
    Plotly.purge(plot);
  }
}

// A loaded page sets up its main part. A page restored from the bfcache
// keeps its figures and shows the sidebar choices the server holds now.
window.addEventListener("pageshow", (event) => {
  busyOverlay.stop();
  if (!event.persisted) {
    initMain(document.querySelector("main.content"));
    return;
  }
  syncMainControls();
  htmx.ajax("GET", "/sidebar/selection", { target: "#view-selection", swap: "innerHTML" });
});

// The display screens' choices: the shown transform run and the model (chosen
// on the transform page) and the shown files (in the sidebar's dialog or by a
// click on a score point). A change refreshes the sidebar's choices and, when the main
// element names the changed choice in data-view-swap, the main part from the
// current URL; the page is not reloaded. The overlay covers the page meanwhile; cancelling stops the
// requests and shows the choices the server holds.
const VIEW_OWNER = "view-selection";
// The pending refresh, aborted by a newer one or by cancelling.
let viewRefresh = null;
// The stems shown by each point table (by id) after the next replacement of
// the main part.
let keptPointStems = {};

function stopViewOverlay() {
  if (busyOverlay.owner() === VIEW_OWNER) {
    busyOverlay.stop();
  }
}

function cancelViewChange() {
  viewRefresh?.abort();
  viewRefresh = null;
  htmx.ajax("GET", "/sidebar/selection", { target: "#view-selection", swap: "innerHTML" });
}

// Returns the HTML of a part of the page; throws if the request fails.
async function fetchPart(url, signal, headers = {}) {
  const response = await fetch(url, { signal, headers, cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${url}`);
  }
  return response.text();
}

// Refreshes the page after the choice `changed` ("run", "files", or "model") changed.
// Both parts are replaced together once both have arrived; if either fails,
// nothing is replaced and the overlay closes. `pointStems` is passed to
// initMain.
async function refreshView(changed, pointStems = {}) {
  viewRefresh?.abort();
  const controller = new AbortController();
  viewRefresh = controller;
  const main = document.querySelector("main.content");
  const swapMain = main.dataset.viewSwap.split(" ").includes(changed);
  try {
    const [sidebar, content] = await Promise.all([
      fetchPart("/sidebar/selection", controller.signal),
      swapMain ? fetchPart(window.location.href, controller.signal, { "HX-Request": "true" }) : null,
    ]);
    if (viewRefresh !== controller) {
      return;
    }
    htmx.swap("#view-selection", sidebar, { swapStyle: "innerHTML" });
    if (content !== null) {
      disposeMain(main);
      keptPointStems = pointStems;
      htmx.swap(main, content, { swapStyle: "innerHTML" });
    }
  } catch {
    // A failed request closes the overlay below; an aborted one is handled
    // by whoever aborted it.
  }
  if (viewRefresh === controller) {
    viewRefresh = null;
    stopViewOverlay();
  }
}

document.addEventListener("htmx:afterSettle", (event) => {
  if (event.detail.elt.matches("main.content")) {
    const pointStems = keptPointStems;
    keptPointStems = {};
    initMain(event.detail.elt, pointStems);
  }
});

// htmx triggers a response's events (HX-Trigger) and htmx:afterRequest on the
// element that sent the request, and they reach the listeners above it only
// while it is in the page. A choice replaced while its request is pending,
// such as a "Show" form of the transform run list reloaded meanwhile, passes
// them on itself, so that the page is still refreshed and the overlay closed.
const relayedChoices = new WeakSet();

function relayFromReplacedChoice(elt) {
  if (relayedChoices.has(elt)) {
    return;
  }
  relayedChoices.add(elt);
  elt.addEventListener("view-selection-changed", (event) => {
    if (!elt.isConnected) {
      document.body.dispatchEvent(new CustomEvent(event.type, { bubbles: true, detail: event.detail }));
    }
  });
  elt.addEventListener("htmx:afterRequest", (event) => {
    if (!elt.isConnected && !event.detail.successful) {
      stopViewOverlay();
    }
  });
}

document.addEventListener("htmx:beforeRequest", (event) => {
  const form = event.detail.elt.closest("[data-view-choice]");
  if (!form) {
    return;
  }
  relayFromReplacedChoice(event.detail.elt);
  // The modal dialog would cover the overlay, so a choice made in it closes
  // it. Its contents stay until the request ends, since htmx triggers the
  // response's events (HX-Trigger) on the form.
  const dialog = form.closest("dialog");
  if (dialog) {
    dialog.dataset.submitting = "true";
    dialog.close();
  }
  busyOverlay.start(
    () => {
      htmx.trigger(form, "htmx:abort");
      cancelViewChange();
    },
    { owner: VIEW_OWNER },
  );
});

document.addEventListener("htmx:afterRequest", (event) => {
  if (event.detail.elt.closest("[data-view-choice]") && !event.detail.successful) {
    stopViewOverlay();
  }
});

// The server announces an accepted choice with this event (HX-Trigger).
document.addEventListener("view-selection-changed", (event) => {
  refreshView(event.detail.changed);
});

// Once settled, the sidebar's choices are processed by htmx and marked ready.
document.addEventListener("htmx:afterSettle", (event) => {
  if (event.detail.elt.id === "view-selection") {
    event.detail.elt.dataset.ready = "true";
  }
});

// The dialog choosing the shown files. Its contents are fetched each time it
// opens, so it starts from the chosen files, not from a cancelled selection.
// "Select" posts the selection (a [data-view-choice] form, handled above);
// Cancel, the close button, and Esc close it without saving.
function openViewFilesDialog() {
  const dialog = document.getElementById("view-files-dialog");
  const content = dialog.querySelector("[data-dialog-content]");
  delete dialog.dataset.submitting;
  content.innerHTML = '<p class="muted">Loading…</p>';
  dialog.showModal();
  htmx.ajax("GET", "/sidebar/selection/files/dialog", {
    source: content,
    target: content,
    swap: "innerHTML",
  });
}

// A closed dialog drops its contents, so that its table and inputs do not
// stay in the page; one closed by "Select" drops them when its request ends.
function dropClosedDialogContents(dialog) {
  if (!dialog.open && !dialog.dataset.submitting) {
    dialog.querySelector("[data-dialog-content]").replaceChildren();
  }
}

document.getElementById("view-files-dialog").addEventListener("close", (event) => {
  dropClosedDialogContents(event.target);
});

// Also drops what a request of a closed dialog brought in after it closed.
document.addEventListener("htmx:afterRequest", (event) => {
  const dialog = event.detail.elt.closest("dialog");
  if (!dialog) {
    return;
  }
  if (event.detail.elt.closest("[data-view-choice]")) {
    delete dialog.dataset.submitting;
  }
  dropClosedDialogContents(dialog);
});

document.addEventListener("click", (event) => {
  if (event.target.closest("[data-view-files-open]")) {
    openViewFilesDialog();
  } else if (event.target.closest("[data-dialog-close]")) {
    event.target.closest("dialog").close();
  }
});

// Adds a clicked file to the sidebar's shown files, then refreshes the page
// with `shownStems` in the point table of the figure. The request names the
// run of the figure, so a figure drawn before the sidebar's run changed adds
// nothing. At the limit of shown files, or for such a stale figure, a warning
// is shown instead.
async function selectClickedFile(target, stem, shownStems) {
  const warning = document.getElementById("plot-select-warning");
  const controller = new AbortController();
  // The overlay blocks further clicks until the page is refreshed.
  busyOverlay.start(
    () => {
      controller.abort();
      cancelViewChange();
    },
    { owner: VIEW_OWNER },
  );
  let result = {};
  try {
    const response = await fetch(target.dataset.selectUrl, {
      method: "POST",
      body: new URLSearchParams({ run: target.dataset.runId, stem }),
      signal: controller.signal,
    });
    result = response.ok ? await response.json() : {};
  } catch {
    // A failed request or response is reported below like a refusal.
  }
  if (controller.signal.aborted) {
    return;
  }
  if (result.added) {
    refreshView("files", { [target.dataset.pointTable]: shownStems });
    return;
  }
  stopViewOverlay();
  const message = result.message ?? `Could not add ${stem} to the shown files.`;
  if (warning) {
    warning.textContent = message;
    warning.hidden = false;
  }
}

// The settings of the fit form (Preprocess / PCA screen) survive a switch to
// another screen and back, in this tab. The page of a reopened fit run
// (?run=) shows that run's settings, which become the kept ones; any other
// visit shows the kept settings in place of the defaults.
const FIT_FORM_KEY = "flat-pca:fit-form";
// The confirmation of a large memory estimate is given anew for each run.
const FIT_FORM_UNKEPT = new Set(["confirm_memory"]);

// Returns the values of the fit form by name, each a list (as in FormData).
function fitFormValues(form) {
  const values = {};
  for (const [name, value] of new FormData(form)) {
    if (!FIT_FORM_UNKEPT.has(name) && typeof value === "string") {
      (values[name] ??= []).push(value);
    }
  }
  return values;
}

function keepFitForm(form) {
  try {
    sessionStorage.setItem(FIT_FORM_KEY, JSON.stringify(fitFormValues(form)));
  } catch {
    // Without storage the settings last only for this page.
  }
}

// Returns the kept values of the fit form, or null without any.
function keptFitFormValues() {
  try {
    const values = JSON.parse(sessionStorage.getItem(FIT_FORM_KEY) ?? "null");
    return values !== null && typeof values === "object" && !Array.isArray(values) ? values : null;
  } catch {
    return null;
  }
}

// Returns whether `name` is a bound of a range ({name}_lower or _upper) that
// is disabled in `values`. Such a bound keeps what the server drew: the
// catalog range of the current fit target, which may differ from the kept
// one and bound the field (min and max).
function isDisabledFitRangeBound(form, name, values) {
  const match = /^(.+)_(lower|upper)$/.exec(name);
  const enabled = match && `${match[1]}_enabled`;
  return Boolean(enabled && form.elements[enabled] && !Array.isArray(values[enabled]));
}

// Shows `values` (see fitFormValues) in the fit form. A checkbox is checked
// only when its value is kept under its name, as FormData leaves out the
// unchecked ones. A field takes its kept value, a select only one of its
// options; a field without a kept value, or the bound of a disabled range,
// keeps what the server drew.
function showFitFormValues(form, values) {
  for (const control of form.elements) {
    if (
      !control.name ||
      FIT_FORM_UNKEPT.has(control.name) ||
      isDisabledFitRangeBound(form, control.name, values)
    ) {
      continue;
    }
    const kept = Array.isArray(values[control.name]) ? values[control.name].map(String) : [];
    if (control.type === "checkbox") {
      control.checked = kept.includes(control.value);
    } else if (kept.length === 0 || control.type === "submit" || control.type === "button") {
      continue;
    } else if (control.tagName === "SELECT") {
      if ([...control.options].some((option) => option.value === kept[0])) {
        control.value = kept[0];
      }
    } else {
      control.value = kept[0];
    }
  }
}

// Shows the kept settings on a loaded fit page, or keeps those of a reopened
// run, and refreshes the memory estimate for the shown settings.
function initFitForm() {
  const form = document.getElementById("fit-form");
  if (!form) {
    return;
  }
  if (new URLSearchParams(window.location.search).has("run")) {
    keepFitForm(form);
    return;
  }
  const values = keptFitFormValues();
  if (values === null) {
    return;
  }
  showFitFormValues(form, values);
  htmx.trigger(form, "change");
}

window.addEventListener("pageshow", (event) => {
  if (!event.persisted) {
    initFitForm();
  }
});

for (const type of ["input", "change"]) {
  document.addEventListener(type, (event) => {
    const form = event.target.form;
    if (form?.id === "fit-form") {
      keepFitForm(form);
    }
  });
}

const INACTIVE_OPACITY = 0.1;
const INACTIVE_ZORDER = -1;

// Returns whether a trace of a figure is made inactive from its legend.
function isInactiveTrace(trace) {
  return trace?.opacity === INACTIVE_OPACITY;
}

// Returns the indices of the traces toggled together with trace `index`:
// those of its legend group, or the trace alone.
function legendGroupIndices(gd, index) {
  const group = gd.data[index].legendgroup;
  if (!group) {
    return [index];
  }
  return gd.data.flatMap((trace, i) => (trace.legendgroup === group ? [i] : []));
}

// Makes each of the traces `indices` active or inactive by `active`: an
// inactive trace stays drawn, faint and behind the active ones.
function setTracesActive(gd, indices, active) {
  return Plotly.restyle(
    gd,
    {
      opacity: active.map((on) => (on ? 1 : INACTIVE_OPACITY)),
      zorder: active.map((on) => (on ? 0 : INACTIVE_ZORDER)),
    },
    indices,
  );
}

// Makes the legend of a figure fade traces instead of hiding them. A click
// on a legend item toggles its legend group between active and inactive; a
// right click makes only that group active, or every trace again when only
// that group already is. A double click does nothing, so that Plotly's
// showing of one trace does not mix with the fading. The state lies in the
// opacity of the traces, so a figure redrawn by `Plotly.react` with new
// traces starts with every trace active. Call it once per figure, after it
// is first drawn: the legend items are redrawn with the figure, so the
// listeners sit on the figure itself.
function attachLegendOpacity(gd) {
  gd.on("plotly_legendclick", (event) => {
    // Plotly reports a right click as a click too; the context menu handles it.
    if (event.event?.button === 2) {
      return false;
    }
    const indices = legendGroupIndices(gd, event.curveNumber);
    const active = indices.some((index) => isInactiveTrace(gd.data[index]));
    setTracesActive(gd, indices, indices.map(() => active));
    return false;
  });
  gd.on("plotly_legenddoubleclick", () => false);
  gd.addEventListener("contextmenu", (event) => {
    const toggle = event.target.closest?.(".legendtoggle");
    if (!toggle) {
      return;
    }
    event.preventDefault();
    // A legend item names the first trace of its group by Plotly's data of
    // the item; the order of the items does not follow the traces when a
    // group has several traces or a trace has no item.
    const index = toggle.parentNode.__data__?.[0]?.trace?.index;
    if (index === undefined) {
      return;
    }
    const group = new Set(legendGroupIndices(gd, index));
    const isolated = gd.data.every(
      (trace, i) => isInactiveTrace(trace) !== group.has(i),
    );
    const indices = gd.data.map((_, i) => i);
    setTracesActive(gd, indices, indices.map((i) => isolated || group.has(i)));
  });
}

// Returns the layout changes that make room for a legend of `legendHeight`
// pixels above the plot area of a figure with the layout `base`, which has
// none: the height and the top margin grow by it, so the plot area keeps its
// height however many rows the legend wraps into. The width keeps following
// the container, which a new height alone would stop.
function legendRoomLayout(base, legendHeight) {
  const room = Math.ceil(legendHeight);
  return { height: base.height + room, "margin.t": base.margin.t + room, autosize: true };
}

// Gives a drawn figure, laid out by `base` with its legend above the plot
// area, the room for its legend unless it already has it.
async function fitLegendRoom(gd, base) {
  const room = legendRoomLayout(base, gd._fullLayout.legend?._height ?? 0);
  if (room.height !== gd.layout.height || room["margin.t"] !== gd.layout.margin.t) {
    await Plotly.relayout(gd, room);
  }
}

const TREND_DEBOUNCE_MS = 150;

// Returns the index of the axis value closest to `value`.
function nearestIndex(axis, value) {
  let best = 0;
  for (let index = 1; index < axis.length; index += 1) {
    if (Math.abs(axis[index] - value) < Math.abs(axis[best] - value)) {
      best = index;
    }
  }
  return best;
}

// Returns the heatmap shapes of a crosshair through (x, y).
function crosshair(x, y) {
  const line = { color: "#ffffff", width: 1, dash: "dot" };
  return [
    { type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line },
    { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: y, y1: y, line },
  ];
}

// Returns the heatmap annotations of the triangle markers that point at
// (x, y) from above and from the left of the plot area.
function markers(x, y) {
  const marker = { showarrow: false, font: { size: 14, color: "#1a73e8" } };
  return [
    { ...marker, text: "▼", xref: "x", yref: "paper", x, y: 1, yanchor: "bottom" },
    { ...marker, text: "▶", xref: "paper", yref: "y", x: 0, y, xanchor: "right" },
  ];
}

// Returns the index of the axis value that a heatmap slider selects.
function sliderIndex(slider, values) {
  return nearestIndex(values, Number(slider.value));
}

// Sets up a heatmap slider whose value is an axis value (a wavelength or a
// StepTime), so that the thumb lies on the row or column of the selected
// value even when the values are unevenly spaced. The slider starts at the
// middle value; dragging snaps it to the nearest value and the keys move it
// by one value. Calls `onChange` after each move.
function initAxisSlider(slider, values, onChange) {
  slider.step = "any";
  slider.min = values[0];
  slider.max = values[values.length - 1];
  slider.value = values[Math.floor((values.length - 1) / 2)];
  slider.addEventListener("input", () => {
    slider.value = values[sliderIndex(slider, values)];
    onChange();
  });
  slider.addEventListener("keydown", (event) => {
    const low = nearestIndex(values, Number(slider.min));
    const high = nearestIndex(values, Number(slider.max));
    const index = sliderIndex(slider, values);
    const target = {
      ArrowUp: index + 1,
      ArrowRight: index + 1,
      PageUp: index + 1,
      ArrowDown: index - 1,
      ArrowLeft: index - 1,
      PageDown: index - 1,
      Home: low,
      End: high,
    }[event.key];
    if (target === undefined) {
      return;
    }
    event.preventDefault();
    slider.value = values[Math.min(high, Math.max(low, target))];
    onChange();
  });
}

// Places a heatmap slider along one axis of the plot area: the vertical
// StepTime slider beside the heatmap ("y") or the wavelength slider below it
// ("x"). The slider spans the values inside the shown axis range (all of them
// unless zoomed), and its thumb lies on the row or column of its value. Reads
// the plot area from Plotly's computed layout. Returns whether the slider
// value changed because it fell outside the shown range.
function alignSlider(heatmap, slider, values, axis) {
  const size = heatmap._fullLayout._size;
  const range = heatmap._fullLayout[`${axis}axis`].range;
  const low = Math.min(...range);
  const high = Math.max(...range);
  const shown = values
    .map((value, index) => ({ value, index }))
    .filter(({ value }) => value >= low && value <= high);
  if (shown.length === 0) {
    slider.hidden = true;
    return false;
  }
  slider.hidden = false;
  const before = slider.value;
  const first = shown[0].value;
  const last = shown[shown.length - 1].value;
  slider.min = first;
  slider.max = last;
  const thumb = parseFloat(getComputedStyle(slider).getPropertyValue("--thumb-size"));
  if (axis === "y") {
    const toPixel = (value) => size.t + (size.h * (high - value)) / (high - low);
    slider.style.top = `${toPixel(last) - thumb / 2}px`;
    slider.style.height = `${toPixel(first) - toPixel(last) + thumb}px`;
  } else {
    const toPixel = (value) => size.l + (size.w * (value - low)) / (high - low);
    slider.style.left = `${toPixel(first) - thumb / 2}px`;
    slider.style.width = `${toPixel(last) - toPixel(first) + thumb}px`;
  }
  return slider.value !== before;
}

// Aligns both heatmap sliders; returns whether either slider value changed.
function alignSliders(heatmap, wavelengthSlider, stepTimeSlider, axes) {
  const wavelengthMoved = alignSlider(heatmap, wavelengthSlider, axes.wavelengths, "x");
  const stepTimeMoved = alignSlider(heatmap, stepTimeSlider, axes.step_times, "y");
  return wavelengthMoved || stepTimeMoved;
}

// Spectral exploration and the model screen's component heatmap: the heatmap
// click and the two sliders choose one point; the trends at that point are
// fetched from the server. Returns a function that stops the pending and later
// trend requests from drawing, for when the main part is replaced.
function initExplore(root) {
  const heatmap = root.querySelector("#explore-heatmap");
  const figure = JSON.parse(root.querySelector("#explore-heatmap-figure").textContent);
  const axes = JSON.parse(root.querySelector("#explore-axes").textContent);
  const wavelengthSlider = root.querySelector("#explore-wavelength");
  const stepTimeSlider = root.querySelector("#explore-step-time");
  const byStepTime = root.querySelector("#explore-trend-step-time");
  const byWavelength = root.querySelector("#explore-trend-wavelength");
  let timer = null;
  let latest = 0;
  let disposed = false;
  // A copy of the server layout of each trend figure, without room for the
  // legend: Plotly keeps and changes the drawn layout object itself. A figure
  // in it has its listeners, so a redraw adds none again.
  const baseLayouts = new Map();

  async function drawTrend(gd, trend) {
    const first = !baseLayouts.has(gd);
    baseLayouts.set(gd, structuredClone(trend.layout));
    await Plotly.react(gd, trend.data, trend.layout, { responsive: true });
    if (first) {
      attachLegendOpacity(gd);
      // A resize may wrap the legend into another number of rows.
      gd.on("plotly_afterplot", () => fitLegendRoom(gd, baseLayouts.get(gd)));
    }
    await fitLegendRoom(gd, baseLayouts.get(gd));
  }

  async function fetchTrends(wavelength, stepTime) {
    const request = ++latest;
    const url = `${root.dataset.trendUrl}&wavelength=${wavelength}&step_time=${stepTime}`;
    // A newer selection invalidates this request at every await.
    const response = await fetch(url);
    if (!response.ok || request !== latest) {
      return;
    }
    const trends = await response.json();
    if (request !== latest) {
      return;
    }
    await drawTrend(byStepTime, trends.by_step_time);
    if (request !== latest) {
      return;
    }
    await drawTrend(byWavelength, trends.by_wavelength);
    if (request !== latest) {
      return;
    }
    root.dataset.trendWavelength = trends.wavelength;
    root.dataset.trendStepTime = trends.step_time;
  }

  function update() {
    if (disposed) {
      return;
    }
    const wavelength = axes.wavelengths[sliderIndex(wavelengthSlider, axes.wavelengths)];
    const stepTime = axes.step_times[sliderIndex(stepTimeSlider, axes.step_times)];
    root.querySelector("#explore-wavelength-value").textContent = `${wavelength}`;
    Plotly.relayout(heatmap, {
      shapes: crosshair(wavelength, stepTime),
      annotations: markers(wavelength, stepTime),
    });
    latest += 1;
    clearTimeout(timer);
    timer = setTimeout(() => fetchTrends(wavelength, stepTime), TREND_DEBOUNCE_MS);
  }

  initAxisSlider(wavelengthSlider, axes.wavelengths, update);
  initAxisSlider(stepTimeSlider, axes.step_times, update);
  Plotly.newPlot(heatmap, figure.data, figure.layout, { responsive: true }).then(() => {
    if (disposed) {
      return;
    }
    // Every redraw, including a resize or a zoom, may move the rows and
    // columns. A zoom that hides the selected point moves it into the shown
    // range.
    alignSliders(heatmap, wavelengthSlider, stepTimeSlider, axes);
    heatmap.on("plotly_afterplot", () => {
      if (alignSliders(heatmap, wavelengthSlider, stepTimeSlider, axes)) {
        update();
      }
    });
    heatmap.on("plotly_click", (event) => {
      const point = event.points[0];
      wavelengthSlider.value = axes.wavelengths[nearestIndex(axes.wavelengths, point.x)];
      stepTimeSlider.value = axes.step_times[nearestIndex(axes.step_times, point.y)];
      update();
    });
    update();
  });

  return () => {
    disposed = true;
    latest += 1;
    clearTimeout(timer);
  };
}

// The score and T²/Q screens' table of the files of the chosen points. The
// page embeds the rows of every file, first cell the stem; `show` replaces
// the shown rows with those of the given stems, in the embedded order. The
// table starts with the rows of `stems`, such as the clicked file after the
// main part is replaced.
function initPointTable(root, stems) {
  const rows = JSON.parse(root.querySelector("[data-point-rows]").textContent);
  const positions = new Map(rows.map((row, position) => [row[0], position]));
  const body = root.querySelector("tbody");

  function show(stems) {
    const shown = [...new Set(stems)]
      .filter((stem) => positions.has(stem))
      .map((stem) => positions.get(stem))
      .sort((a, b) => a - b);
    body.replaceChildren(
      ...shown.map((position) => {
        const row = document.createElement("tr");
        for (const value of rows[position]) {
          const cell = document.createElement("td");
          cell.textContent = value;
          row.append(cell);
        }
        return row;
      }),
    );
    root.querySelector(".table-scroll").hidden = shown.length === 0;
    root.querySelector("[data-point-table-empty]").hidden = shown.length > 0;
    root.dataset.shownCount = `${shown.length}`;
  }

  show(stems);
  return { show };
}

// The point tables of the shown main part by id (see initMain).
const pointTables = new Map();

// Returns the stems named by the points of a Plotly event.
function eventStems(event) {
  return (event?.points ?? [])
    .map((point) => point.customdata)
    .filter((stem) => stem !== undefined);
}

// Model, score, and T²/Q screens: draws each embedded figure. Clicking a
// point that names a file, or choosing points by a box or a lasso, shows the
// files in the figure's point table; double-clicking clears it. On the score
// screen a click also adds the file to the sidebar's shown files.
function initPlot(target) {
  const figure = JSON.parse(document.getElementById(target.dataset.plot).textContent);
  const table = pointTables.get(target.dataset.pointTable);
  Plotly.newPlot(target, figure.data, figure.layout, { responsive: true }).then(() => {
    attachLegendOpacity(target);
    target.dataset.plotReady = "true";
    if (table) {
      target.on("plotly_selected", (event) => table.show(eventStems(event)));
      // A double click clears the table in every drag mode; only in the box
      // and lasso modes does Plotly also report a deselection.
      target.on("plotly_deselect", () => table.show([]));
      target.on("plotly_doubleclick", () => table.show([]));
    }
    target.on("plotly_click", (event) => {
      // A point of an inactive trace stays out of the way of the active ones.
      if (isInactiveTrace(target.data[event.points[0]?.curveNumber])) {
        return;
      }
      const stems = eventStems(event).slice(0, 1);
      if (stems.length === 0) {
        return;
      }
      table?.show(stems);
      if (target.dataset.selectUrl) {
        selectClickedFile(target, stems[0], stems);
      }
    });
  });
}

const SIDEBAR_KEY = "flat-pca:sidebar";
const WIDTH_KEY = "flat-pca:width";

// Keeps a layout choice on the root element and, across pages, in storage.
function keepLayoutChoice(key, name, value) {
  document.documentElement.dataset[name] = value;
  try {
    localStorage.setItem(key, value);
  } catch {
    // Without storage the choice lasts only for this page.
  }
}

// Redraws every Plotly figure at the width of its container.
function resizePlots() {
  requestAnimationFrame(() => {
    for (const plot of document.querySelectorAll(".js-plotly-plot")) {
      Plotly.Plots.resize(plot);
    }
  });
}

// Applies the kept layout choices to the root element, as base.html does
// before the first paint.
function applyKeptLayoutChoices() {
  for (const [key, name] of [[SIDEBAR_KEY, "sidebar"], [WIDTH_KEY, "width"]]) {
    let value = null;
    try {
      value = localStorage.getItem(key);
    } catch {
      // Without storage the choice shown on this page stays.
      continue;
    }
    if (value) {
      document.documentElement.dataset[name] = value;
    } else {
      delete document.documentElement.dataset[name];
    }
  }
}

// Shows the collapse button's label and pressed width choice for the layout
// applied before the first paint (see base.html).
function syncLayoutControls() {
  const root = document.documentElement;
  const collapsed = root.dataset.sidebar === "collapsed";
  const toggle = document.getElementById("sidebar-toggle");
  const label = collapsed ? "Expand sidebar" : "Collapse sidebar";
  toggle.setAttribute("aria-expanded", `${!collapsed}`);
  toggle.setAttribute("aria-label", label);
  toggle.title = label;
  const width = root.dataset.width === "compact" ? "compact" : "wide";
  for (const button of document.querySelectorAll("[data-width-choice]")) {
    button.setAttribute("aria-pressed", `${button.dataset.widthChoice === width}`);
  }
}

document.getElementById("sidebar-toggle").addEventListener("click", () => {
  const collapsed = document.documentElement.dataset.sidebar === "collapsed";
  keepLayoutChoice(SIDEBAR_KEY, "sidebar", collapsed ? "expanded" : "collapsed");
  syncLayoutControls();
  resizePlots();
});

for (const button of document.querySelectorAll("[data-width-choice]")) {
  button.addEventListener("click", () => {
    keepLayoutChoice(WIDTH_KEY, "width", button.dataset.widthChoice);
    syncLayoutControls();
    resizePlots();
  });
}

syncLayoutControls();

// A page restored by "back" from the bfcache shows the choices made since it
// was left, possibly on another page.
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) {
    return;
  }
  applyKeptLayoutChoices();
  syncLayoutControls();
  resizePlots();
});

// A figure resized while its card was collapsed is drawn at the card's width
// again when the card opens. The toggle event does not bubble, so it is
// caught on its way down.
document.addEventListener(
  "toggle",
  (event) => {
    if (event.target.matches("details.card") && event.target.open) {
      resizePlots();
    }
  },
  true,
);

// ---- Help tips (templates/macros/help_tip.html) ----

// The explanation opens below the icon toward the right; it is shifted left
// when it would leave the viewport.
function placeHelpTip(tip) {
  const text = tip.querySelector(".help-tip-text");
  text.style.removeProperty("--help-tip-shift");
  const overflow = text.getBoundingClientRect().right - (document.documentElement.clientWidth - 16);
  if (overflow > 0) {
    text.style.setProperty("--help-tip-shift", `-${overflow}px`);
  }
}

// Moving between the icon and the explanation keeps an Escape in effect;
// coming back from outside shows the explanation again.
function openHelpTip(event) {
  const tip = event.target.closest?.(".help-tip");
  if (tip && !tip.contains(event.relatedTarget)) {
    tip.classList.remove("help-tip-dismissed");
    placeHelpTip(tip);
  }
}

document.addEventListener("mouseover", openHelpTip);
document.addEventListener("focusin", openHelpTip);

// Escape hides the shown explanation without moving the focus or the pointer.
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") {
    return;
  }
  for (const tip of document.querySelectorAll(".help-tip:hover, .help-tip:focus-within")) {
    tip.classList.add("help-tip-dismissed");
  }
});
