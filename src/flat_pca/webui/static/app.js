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
