"""The model and the kind of targets chosen on the transform screen."""

from __future__ import annotations

import threading


class TransformSettings:
    """Thread-safe holder of the transform screen's settings.

    The settings are shared by every browser tab and live for the lifetime
    of the workspace.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_run_id: str | None = None
        self._use_same_data = True

    @property
    def model_run_id(self) -> str | None:
        """Return the fit run whose model transforms the targets.

        Returns
        -------
        str | None
            The chosen fit run, or ``None`` for the latest succeeded fit run.
        """
        with self._lock:
            return self._model_run_id

    @property
    def use_same_data(self) -> bool:
        """Return whether the transform targets are the model's fit targets.

        Returns
        -------
        bool
            ``True`` (the default) to transform the files the model was
            fitted on; ``False`` to transform the files chosen from the
            catalog.
        """
        with self._lock:
            return self._use_same_data

    def update(self, model_run_id: str, use_same_data: bool) -> None:
        """Replace the settings.

        Parameters
        ----------
        model_run_id : str
            Fit run whose model transforms the targets, already validated by
            the caller.
        use_same_data : bool
            Whether the transform targets are the model's fit targets.
        """
        with self._lock:
            self._model_run_id = model_run_id
            self._use_same_data = use_same_data
