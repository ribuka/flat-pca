// Bridges between page controls and htmx. Data processing stays on the server.

const FILE_QUERY_DELAY_MS = 300;

// The data selection's file selection over all pages and filters lives as a
// JSON array in the hidden input #selected-stems, outside the reloaded file
// table; Select sends it as one form field.
function selectedStems() {
  return new Set(JSON.parse(document.getElementById("selected-stems").value));
}

function setSelectedStems(stems) {
  document.getElementById("selected-stems").value = JSON.stringify([...stems]);
}

// Returns the stems of every file matching the file table's filters, on all
// of its pages.
function matchingStems(root) {
  return JSON.parse(root.querySelector("[data-matching-stems]").textContent);
}

// Shows the selection in the row checkboxes, and in the header checkbox
// whether all, some, or none of the files matching the filters are selected.
function syncFileChecks(root) {
  const stems = selectedStems();
  for (const box of root.querySelectorAll("[data-stem-check]")) {
    box.checked = stems.has(box.value);
  }
  const matching = matchingStems(root);
  const checked = matching.filter((stem) => stems.has(stem)).length;
  const header = root.querySelector("[data-check-all]");
  header.checked = matching.length > 0 && checked === matching.length;
  header.indeterminate = checked > 0 && checked < matching.length;
  header.disabled = matching.length === 0;
}

// A row checkbox selects its file; the header checkbox selects or clears
// every file matching the filters, shown on this page or not.
document.addEventListener("change", (event) => {
  const root = event.target.closest("#file-table");
  if (!root || !event.target.matches("[data-check-all], [data-stem-check]")) {
    return;
  }
  const stems = selectedStems();
  const targets = event.target.matches("[data-check-all]")
    ? matchingStems(root)
    : [event.target.value];
  for (const stem of targets) {
    if (event.target.checked) {
      stems.add(stem);
    } else {
      stems.delete(stem);
    }
  }
  setSelectedStems(stems);
  syncFileChecks(root);
});

let fileQueryTimer = null;

// Sets query values of the file table and reloads it from the server.
function reloadFileTable(root, values) {
  clearTimeout(fileQueryTimer);
  for (const [name, value] of Object.entries(values)) {
    root.querySelector(`[data-file-query][name="${name}"]`).value = value;
  }
  htmx.trigger(root, "file-query-changed");
}

// Typing in a filter reloads the first page after a pause; choosing a
// category reloads it at once.
document.addEventListener("input", (event) => {
  const root = event.target.closest("#file-table");
  if (!root || !event.target.matches("input[data-file-query]")) {
    return;
  }
  clearTimeout(fileQueryTimer);
  fileQueryTimer = setTimeout(() => reloadFileTable(root, { page: "1" }), FILE_QUERY_DELAY_MS);
});

document.addEventListener("change", (event) => {
  const root = event.target.closest("#file-table");
  if (root && event.target.matches("select[data-file-query]")) {
    reloadFileTable(root, { page: "1" });
  }
});

// A column name sorts by that column, ascending first and then toggling; the
// page links move between pages.
document.addEventListener("click", (event) => {
  const root = event.target.closest("#file-table");
  if (!root) {
    return;
  }
  const sort = event.target.closest("[data-sort]");
  if (sort) {
    const current = root.querySelector('[data-file-query][name="sort"]').value;
    const order = root.querySelector('[data-file-query][name="order"]').value;
    const descending = current === sort.dataset.sort && order === "asc";
    reloadFileTable(root, {
      sort: sort.dataset.sort,
      order: descending ? "desc" : "asc",
      page: "1",
    });
    return;
  }
  const page = event.target.closest("[data-page]");
  if (page) {
    reloadFileTable(root, { page: page.dataset.page });
  }
});

document.addEventListener("htmx:afterSettle", (event) => {
  if (event.detail.elt.id === "file-table") {
    syncFileChecks(event.detail.elt);
  }
});

const BUSY_SHOW_DELAY_MS = 300;
const BUSY_TICK_MS = 250;

// The page-wide overlay shown while a request is pending: it greys out the
// page, blocks input, and shows a spinner, the elapsed seconds, and a cancel
// button. `start` takes the action of the cancel button.
const busyOverlay = (() => {
  const overlay = document.getElementById("busy-overlay");
  const elapsed = document.getElementById("busy-elapsed");
  const layout = document.querySelector(".layout");
  let showTimer = null;
  let tickTimer = null;
  let cancelAction = null;

  function stop() {
    clearTimeout(showTimer);
    clearInterval(tickTimer);
    cancelAction = null;
    overlay.hidden = true;
    overlay.classList.remove("busy-visible");
    layout.inert = false;
  }

  function start(onCancel) {
    stop();
    cancelAction = onCancel;
    const startedAt = performance.now();
    elapsed.textContent = "0";
    // Input is blocked at once; the overlay is drawn only after the delay.
    layout.inert = true;
    overlay.hidden = false;
    showTimer = setTimeout(() => {
      overlay.classList.add("busy-visible");
      document.getElementById("busy-cancel").focus();
    }, BUSY_SHOW_DELAY_MS);
    tickTimer = setInterval(() => {
      elapsed.textContent = `${Math.floor((performance.now() - startedAt) / 1000)}`;
    }, BUSY_TICK_MS);
  }

  document.getElementById("busy-cancel").addEventListener("click", () => {
    const action = cancelAction;
    stop();
    if (action) {
      action();
    }
  });

  return { start, stop };
})();

// Submits a GET form whenever one of its controls changes.
document.addEventListener("change", (event) => {
  const form = event.target.closest("form[data-auto-submit]");
  if (form) {
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

// A page restored by "back" (from the bfcache or with restored form state)
// keeps neither the overlay nor the conditions that were never shown.
// A page from the bfcache also shows the sidebar choices the server holds now.
window.addEventListener("pageshow", (event) => {
  busyOverlay.stop();
  for (const form of document.querySelectorAll("form[data-auto-submit]")) {
    form.reset();
  }
  if (event.persisted && document.getElementById("view-selection")) {
    htmx.ajax("GET", "/sidebar/selection", { target: "#view-selection", swap: "innerHTML" });
  }
});

// The sidebar's run and shown-file choices: the server answers a change with
// HX-Refresh, so the overlay covers the page until it is reloaded. Cancelling
// stops the request or the reload and shows the choices the server holds.
document.addEventListener("htmx:beforeRequest", (event) => {
  const form = event.detail.elt.closest("[data-busy-reload]");
  if (!form) {
    return;
  }
  busyOverlay.start(() => {
    htmx.trigger(form, "htmx:abort");
    window.stop();
    htmx.ajax("GET", "/sidebar/selection", { target: "#view-selection", swap: "innerHTML" });
  });
});

document.addEventListener("htmx:afterRequest", (event) => {
  if (event.detail.elt.closest("[data-busy-reload]") && !event.detail.successful) {
    busyOverlay.stop();
  }
});

const VIEW_FILE_SEARCH_KEY = "flat-pca:view-file-search";

// Hides the shown-file choices whose names do not contain the search text.
function filterViewFiles(input) {
  const text = input.value.trim().toLowerCase();
  for (const item of document.querySelectorAll("#view-selection li[data-stem]")) {
    item.hidden = text !== "" && !item.dataset.stem.toLowerCase().includes(text);
  }
}

// The search text survives the reload that follows each choice in this tab.
document.addEventListener("input", (event) => {
  if (!event.target.matches("[data-view-file-search]")) {
    return;
  }
  try {
    sessionStorage.setItem(VIEW_FILE_SEARCH_KEY, event.target.value);
  } catch {
    // Without storage the search text is simply not kept.
  }
  filterViewFiles(event.target);
});

// Once settled, the sidebar's choices are processed by htmx and marked ready.
document.addEventListener("htmx:afterSettle", (event) => {
  if (event.detail.elt.id !== "view-selection") {
    return;
  }
  event.detail.elt.dataset.ready = "true";
  const input = event.detail.elt.querySelector("[data-view-file-search]");
  if (!input) {
    return;
  }
  try {
    input.value = sessionStorage.getItem(VIEW_FILE_SEARCH_KEY) ?? "";
  } catch {
    input.value = "";
  }
  filterViewFiles(input);
});

// Adds a clicked file to the sidebar's shown files, then reloads the page or
// opens `openUrl`. The request names the run of the figure, so a figure
// drawn before the sidebar's run changed adds nothing. At the limit of shown
// files, or for such a stale figure, a warning is shown instead.
async function selectClickedFile(target, stem) {
  const warning = document.getElementById("plot-select-warning");
  // The overlay blocks further clicks until the page is replaced.
  busyOverlay.start(() => window.stop());
  let result = {};
  try {
    const response = await fetch(target.dataset.selectUrl, {
      method: "POST",
      body: new URLSearchParams({ run: target.dataset.runId, stem }),
    });
    result = response.ok ? await response.json() : {};
  } catch {
    // A failed request or response is reported below like a refusal.
  }
  if (result.added) {
    if (target.dataset.openUrl) {
      window.location.href = target.dataset.openUrl;
    } else {
      window.location.reload();
    }
    return;
  }
  busyOverlay.stop();
  const message = result.message ?? `${stem} を表示ファイルに追加できませんでした。`;
  if (warning) {
    warning.textContent = message;
    warning.hidden = false;
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
// fetched from the server.
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
    await Plotly.react(byStepTime, trends.by_step_time.data, trends.by_step_time.layout);
    if (request !== latest) {
      return;
    }
    await Plotly.react(byWavelength, trends.by_wavelength.data, trends.by_wavelength.layout);
    if (request !== latest) {
      return;
    }
    root.dataset.trendWavelength = trends.wavelength;
    root.dataset.trendStepTime = trends.step_time;
  }

  function update() {
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
}

for (const root of document.querySelectorAll("[data-explore]")) {
  initExplore(root);
}

// Model, score, and T²/Q screens: draws each embedded figure; clicking a point
// that names a file adds the file to the sidebar's shown files.
function initPlot(target) {
  const figure = JSON.parse(document.getElementById(target.dataset.plot).textContent);
  Plotly.newPlot(target, figure.data, figure.layout, { responsive: true }).then(() => {
    target.dataset.plotReady = "true";
    if (!target.dataset.selectUrl) {
      return;
    }
    target.on("plotly_click", (event) => {
      const stem = event.points[0].customdata;
      if (stem === undefined) {
        return;
      }
      selectClickedFile(target, stem);
    });
  });
}

for (const target of document.querySelectorAll("[data-plot]")) {
  initPlot(target);
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

// Shows the collapse button's label and pressed width choice for the layout
// applied before the first paint (see base.html).
function syncLayoutControls() {
  const root = document.documentElement;
  const collapsed = root.dataset.sidebar === "collapsed";
  const toggle = document.getElementById("sidebar-toggle");
  const label = collapsed ? "サイドバーを広げる" : "サイドバーを折りたたむ";
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
