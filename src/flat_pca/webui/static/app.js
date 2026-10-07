// Bridges between page controls and htmx. Data processing stays on the server.
document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-check-all], [data-uncheck-all]");
  if (!button) {
    return;
  }
  const checked = button.hasAttribute("data-check-all");
  const container = document.querySelector(
    button.getAttribute(checked ? "data-check-all" : "data-uncheck-all"),
  );
  if (!container) {
    return;
  }
  for (const box of container.querySelectorAll("input[type=checkbox]")) {
    box.checked = checked;
  }
});

// Submits a GET form whenever one of its controls changes.
document.addEventListener("change", (event) => {
  const form = event.target.closest("form[data-auto-submit]");
  if (form) {
    form.requestSubmit();
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

// Spectral exploration: the heatmap click and the two sliders choose one
// point; the trends at that point are fetched from the server.
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
    Plotly.relayout(heatmap, { shapes: crosshair(wavelength, stepTime) });
    latest += 1;
    clearTimeout(timer);
    timer = setTimeout(() => fetchTrends(wavelength, stepTime), TREND_DEBOUNCE_MS);
  }

  Plotly.newPlot(heatmap, figure.data, figure.layout, { responsive: true }).then(() => {
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
