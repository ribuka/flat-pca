// Bridges between page controls and htmx. Data processing stays on the server.

// Shows in the header checkbox of a table whether all, some, or none of its
// row checkboxes are checked.
function syncCheckAll(table) {
  const header = table.querySelector("[data-check-all]");
  const boxes = [...table.querySelectorAll("tbody input[type=checkbox]")];
  const checked = boxes.filter((box) => box.checked).length;
  header.checked = boxes.length > 0 && checked === boxes.length;
  header.indeterminate = checked > 0 && checked < boxes.length;
  header.disabled = boxes.length === 0;
}

// The header checkbox checks or unchecks every shown row; a row checkbox
// updates the header.
document.addEventListener("change", (event) => {
  const table = event.target.closest("table[data-check-table]");
  if (!table) {
    return;
  }
  if (event.target.matches("[data-check-all]")) {
    for (const box of table.querySelectorAll("tbody input[type=checkbox]")) {
      box.checked = event.target.checked;
    }
  }
  syncCheckAll(table);
});

document.addEventListener("htmx:afterSettle", (event) => {
  for (const table of event.detail.elt.querySelectorAll("table[data-check-table]")) {
    syncCheckAll(table);
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
window.addEventListener("pageshow", () => {
  busyOverlay.stop();
  for (const form of document.querySelectorAll("form[data-auto-submit]")) {
    form.reset();
  }
});

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

// Places the vertical StepTime slider beside the heatmap. The slider spans
// the StepTimes inside the shown y range (all of them unless zoomed), and its
// thumb at either end lies on the row of that StepTime. Reads the plot area
// from Plotly's computed layout. Returns whether the slider value changed
// because it fell outside the shown range.
function alignStepTimeSlider(heatmap, slider, stepTimes) {
  const size = heatmap._fullLayout._size;
  const range = heatmap._fullLayout.yaxis.range;
  const bottom = Math.min(...range);
  const top = Math.max(...range);
  const shown = stepTimes
    .map((value, index) => ({ value, index }))
    .filter(({ value }) => value >= bottom && value <= top);
  if (shown.length === 0) {
    slider.hidden = true;
    return false;
  }
  slider.hidden = false;
  const before = slider.value;
  slider.min = shown[0].index;
  slider.max = shown[shown.length - 1].index;
  const toPixel = (value) => size.t + (size.h * (top - value)) / (top - bottom);
  const thumb = parseFloat(getComputedStyle(slider).getPropertyValue("--thumb-size"));
  const first = toPixel(shown[0].value);
  const last = toPixel(shown[shown.length - 1].value);
  slider.style.top = `${last - thumb / 2}px`;
  slider.style.height = `${first - last + thumb}px`;
  return slider.value !== before;
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
    const wavelength = axes.wavelengths[Number(wavelengthSlider.value)];
    const stepTime = axes.step_times[Number(stepTimeSlider.value)];
    root.querySelector("#explore-wavelength-value").textContent = `${wavelength}`;
    root.querySelector("#explore-step-time-value").textContent = `${stepTime}`;
    Plotly.relayout(heatmap, {
      shapes: crosshair(wavelength, stepTime),
      annotations: markers(wavelength, stepTime),
    });
    latest += 1;
    clearTimeout(timer);
    timer = setTimeout(() => fetchTrends(wavelength, stepTime), TREND_DEBOUNCE_MS);
  }

  Plotly.newPlot(heatmap, figure.data, figure.layout, { responsive: true }).then(() => {
    // Every redraw, including a resize or a zoom, may move the rows. A zoom
    // that hides the selected StepTime moves the point into the shown range.
    alignStepTimeSlider(heatmap, stepTimeSlider, axes.step_times);
    heatmap.on("plotly_afterplot", () => {
      if (alignStepTimeSlider(heatmap, stepTimeSlider, axes.step_times)) {
        update();
      }
    });
    heatmap.on("plotly_click", (event) => {
      const point = event.points[0];
      wavelengthSlider.value = nearestIndex(axes.wavelengths, point.x);
      stepTimeSlider.value = nearestIndex(axes.step_times, point.y);
      update();
    });
    wavelengthSlider.addEventListener("input", update);
    stepTimeSlider.addEventListener("input", update);
    update();
  });
}

for (const root of document.querySelectorAll("[data-explore]")) {
  initExplore(root);
}

// Model, score, and T²/Q screens: draws each embedded figure; clicking a point
// that names a file opens the spectral exploration of that file.
function initPlot(target) {
  const figure = JSON.parse(document.getElementById(target.dataset.plot).textContent);
  Plotly.newPlot(target, figure.data, figure.layout, { responsive: true }).then(() => {
    target.dataset.plotReady = "true";
    if (!target.dataset.exploreUrl) {
      return;
    }
    target.on("plotly_click", (event) => {
      const stem = event.points[0].customdata;
      if (stem === undefined) {
        return;
      }
      window.location.href = `${target.dataset.exploreUrl}&file=${encodeURIComponent(stem)}`;
    });
  });
}

for (const target of document.querySelectorAll("[data-plot]")) {
  initPlot(target);
}
