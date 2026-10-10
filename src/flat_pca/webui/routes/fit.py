"""Preprocessing and PCA screen: fit form, memory estimate, and fit runs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.datastructures import FormData

from ..jobs.fit_run import (
    AUTO_COMPONENT_CAP,
    AUTO_COMPONENT_CUMULATIVE,
    build_fit_config,
)
from ..run_config import run_config
from ..services.catalog_query import list_files, selection_ranges
from ..services.fit_estimate import estimate_fit
from ..services.fit_form import (
    IMPUTE_STRATEGIES,
    INTENSITY_TRANSFORMS,
    SCALING_STRATEGIES,
    FitForm,
    default_form_values,
    form_values_from_config,
    parse_fit_form,
)
from ..services.runs import get_run, latest_run, list_runs, run_status
from ..templating import templates
from ..workspace import FIT_JOB, Workspace
from .dependencies import get_workspace

router = APIRouter(prefix="/fit")
WorkspaceDependency = Annotated[Workspace, Depends(get_workspace)]


def _submitted_values(form: FormData) -> dict[str, object]:
    """Return submitted form values as text, with ``target_steps`` as a list.

    Parameters
    ----------
    form : FormData
        Parsed request form.

    Returns
    -------
    dict[str, object]
        Values keyed by input name.
    """
    values: dict[str, object] = {
        name: value for name, value in form.items() if isinstance(value, str)
    }
    values["target_steps"] = [
        value for value in form.getlist("target_steps") if isinstance(value, str)
    ]
    return values


def _form_context(
    workspace: Workspace, form: FitForm, *, run: dict[str, object] | None = None
) -> dict[str, object]:
    """Return the template context of the fit form.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    form : FitForm
        Form values and errors to render.
    run : dict[str, object] | None, default None
        Fit run whose status the page shows.

    Returns
    -------
    dict[str, object]
        Form, choices, catalog ranges, estimate, and run status.
    """
    stems = workspace.selection.stems
    valid_preprocess = {
        name: value for name, value in form.preprocess.items() if name not in form.errors
    }
    estimate = estimate_fit(workspace.database, stems, valid_preprocess, form.pca)
    return {
        "form": form,
        "values": form.values,
        "errors": form.errors,
        "selected_count": len(stems),
        "ranges": selection_ranges(workspace.database, stems),
        "intensity_transforms": INTENSITY_TRANSFORMS,
        "impute_strategies": IMPUTE_STRATEGIES,
        "scaling_strategies": SCALING_STRATEGIES,
        "auto_cumulative": AUTO_COMPONENT_CUMULATIVE,
        "auto_cap": AUTO_COMPONENT_CAP,
        "estimate": estimate,
        "memory_warn_gb": workspace.settings.jobs.memory_warn_gb,
        "status": run_status(run),
    }


def _initial_form(workspace: Workspace, run: dict[str, object] | None) -> FitForm:
    """Return the form values of a run, or the defaults without one.

    Parameters
    ----------
    workspace : Workspace
        Application workspace.
    run : dict[str, object] | None
        Fit run to reproduce, or ``None`` for the defaults (with the
        workspace's ``fit_defaults.toml`` values).

    Returns
    -------
    FitForm
        Parsed form holding the values (errors are not shown initially).
    """
    stems = workspace.selection.stems
    ranges = selection_ranges(workspace.database, stems)
    values = (
        default_form_values(ranges, workspace.fit_defaults.form_values())
        if run is None
        else form_values_from_config(run_config(run), ranges)
    )
    parsed = parse_fit_form(values, len(stems))
    return FitForm(
        values=parsed.values,
        preprocess=parsed.preprocess,
        pca=parsed.pca,
        mahalanobis=parsed.mahalanobis,
        spe=parsed.spe,
    )


@router.get("", response_class=HTMLResponse)
def fit_page(
    request: Request, workspace: WorkspaceDependency, run: str | None = None
) -> HTMLResponse:
    """Render the preprocessing and PCA page.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.
    run : str | None, default None
        Fit run to reopen: the form shows its settings and the status
        panel shows the run. Without it, the form shows the defaults and
        the status panel shows the latest fit run.

    Returns
    -------
    HTMLResponse
        Full page.

    Raises
    ------
    HTTPException
        With status 404 if ``run`` is not a fit run.
    """
    if run is None:
        shown = latest_run(workspace.database, FIT_JOB)
        form = _initial_form(workspace, None)
    else:
        shown = get_run(workspace.database, run)
        if shown is None or shown["kind"] != FIT_JOB:
            raise HTTPException(status_code=404, detail=f"fit run not found: {run}")
        form = _initial_form(workspace, shown)
    return templates.TemplateResponse(
        request, "pages/fit.html", _form_context(workspace, form, run=shown)
    )


@router.post("/estimate", response_class=HTMLResponse)
async def fit_estimate(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the feature-count and memory estimate of the submitted form.

    Parameters
    ----------
    request : Request
        Current request carrying the form.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Estimate partial.
    """
    values = _submitted_values(await request.form())
    form = parse_fit_form(values, len(workspace.selection.stems))
    return templates.TemplateResponse(
        request, "partials/fit_estimate.html", _form_context(workspace, form)
    )


@router.post("", response_class=HTMLResponse)
async def submit_fit(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Validate the form and queue a fit job.

    The estimate must not exceed ``jobs.memory_warn_gb`` unless the form
    confirms it with ``confirm_memory``.

    Parameters
    ----------
    request : Request
        Current request carrying the form.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        The form partial, with errors next to their fields when invalid.
        When a job is queued, the response also replaces the run status
        out of band and triggers ``fit-started``.
    """
    values = _submitted_values(await request.form())
    stems = workspace.selection.stems
    form = parse_fit_form(values, len(stems))
    context = _form_context(workspace, form)
    estimate = context["estimate"]
    if (
        form.is_valid
        and estimate.exceeds(workspace.settings.jobs.memory_warn_gb)  # type: ignore[attr-defined]
        and not values.get("confirm_memory")
    ):
        form.errors["confirm_memory"] = (
            "The memory estimate exceeds the limit. Check the confirmation box to run anyway."
        )
    if not form.is_valid:
        return templates.TemplateResponse(request, "partials/fit_form.html", context)

    selected = set(stems)
    files = [file for file in list_files(workspace.database) if file["stem"] in selected]
    run_id = workspace.submit_fit(
        build_fit_config(
            workspace.settings,
            files,
            form.preprocess,
            form.pca,
            form.mahalanobis,
            form.spe,
        )
    )
    context["status"] = run_status(get_run(workspace.database, run_id))
    context["oob_status"] = True
    response = templates.TemplateResponse(request, "partials/fit_form.html", context)
    response.headers["HX-Trigger"] = "fit-started"
    return response


@router.get("/runs", response_class=HTMLResponse)
def fit_runs(request: Request, workspace: WorkspaceDependency) -> HTMLResponse:
    """Render the list of fit runs.

    Parameters
    ----------
    request : Request
        Current request.
    workspace : Workspace
        Application workspace.

    Returns
    -------
    HTMLResponse
        Run list partial, newest first.
    """
    return templates.TemplateResponse(
        request,
        "partials/fit_runs.html",
        {"runs": list_runs(workspace.database, FIT_JOB)},
    )


@router.get("/runs/{run_id}/status", response_class=HTMLResponse)
def fit_run_status(
    request: Request, run_id: str, workspace: WorkspaceDependency, polling: bool = False
) -> HTMLResponse:
    """Render the status of one fit run, polled while it is active.

    Parameters
    ----------
    request : Request
        Current request.
    run_id : str
        Fit run to show.
    workspace : Workspace
        Application workspace.
    polling : bool, default False
        Whether the request comes from polling an active run. When such a
        poll finds the run finished, the response triggers ``fit-updated``
        so the run list reloads, and, if the run succeeded, a success icon
        replaces ``#fit-result`` next to the submit button out of band.

    Returns
    -------
    HTMLResponse
        Run status partial.

    Raises
    ------
    HTTPException
        With status 404 if the run is not a fit run.
    """
    run = get_run(workspace.database, run_id)
    if run is None or run["kind"] != FIT_JOB:
        raise HTTPException(status_code=404, detail=f"fit run not found: {run_id}")
    status = run_status(run)
    finished = polling and not status["active"]
    context = {
        "status": status,
        "finished_succeeded": finished and run["status"] == "succeeded",
    }
    response = templates.TemplateResponse(request, "partials/fit_run_status.html", context)
    if finished:
        response.headers["HX-Trigger"] = "fit-updated"
    return response
