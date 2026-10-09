"""Loading and validation of the Web UI ``settings.toml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    model_validator,
)

from .settings_files import (
    DEFAULT_SETTINGS_PATH,
    read_settings_document,
    select_settings_file,
)
from .settings_paths import expand_settings_paths

MetadataColumnType = Literal["category", "number", "datetime"]
# File-list and run-sample columns (samples.parquet) that metadata columns
# would overwrite.
RESERVED_METADATA_COLUMNS = frozenset(
    {"stem", "path", "n_rows", "n_steps", "n_segments", "source"}
)


class SettingsError(ValueError):
    """Raised when ``settings.toml`` cannot be read or fails validation."""


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys and is immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkspaceSettings(_StrictModel):
    """``[workspace]`` section.

    Attributes
    ----------
    dir : Path
        Directory holding the DuckDB file and the ``runs/`` directory.
    """

    dir: Path


class DataSettings(_StrictModel):
    """``[data]`` section.

    Attributes
    ----------
    root : Path
        Directory scanned for Parquet files.
    glob : str
        Glob pattern, relative to ``root``, selecting the Parquet files.
    """

    root: Path
    glob: str = Field(default="**/*.parquet", min_length=1)


class MetadataColumnSettings(_StrictModel):
    """One entry of ``[metadata.columns]``.

    Attributes
    ----------
    type : Literal["category", "number", "datetime"]
        Column type.
    format : str | None
        ``strptime`` format, allowed only for ``datetime`` columns. ``None``
        lets polars infer the format.
    """

    type: MetadataColumnType
    format: str | None = None

    @model_validator(mode="after")
    def _check_format(self) -> MetadataColumnSettings:
        """Reject ``format`` on non-datetime columns.

        Returns
        -------
        MetadataColumnSettings
            The validated settings.
        """
        if self.format is not None and self.type != "datetime":
            raise ValueError("format is only allowed for datetime columns")
        return self


class MetadataSettings(_StrictModel):
    """``[metadata]`` section.

    Attributes
    ----------
    csv : Path
        Metadata CSV path.
    key : str
        CSV column matched against each Parquet file's ``Path.stem``.
    columns : dict[str, MetadataColumnSettings]
        CSV columns to import and their types. Other columns are ignored.
    """

    csv: Path
    key: str = Field(min_length=1)
    columns: dict[str, MetadataColumnSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_columns(self) -> MetadataSettings:
        """Reject empty, reserved, and key column names.

        Returns
        -------
        MetadataSettings
            The validated settings.
        """
        for name in self.columns:
            if not name:
                raise ValueError("metadata column names must not be empty")
            if name in RESERVED_METADATA_COLUMNS:
                raise ValueError(f"metadata column name is reserved: {name!r}")
            if name == self.key:
                raise ValueError(f"metadata key column must not be imported: {name!r}")
        return self


class UiSettings(_StrictModel):
    """``[ui]`` section.

    Attributes
    ----------
    default_color_by : str | None
        Metadata column used to color scatter plots by default.
    default_x_axis : str | None
        Metadata column on the horizontal axis of the T²/Q control charts
        by default.
    heatmap_max_cells : int
        Maximum number of heatmap cells sent to the browser.
    explore_max_files : int
        Maximum number of shown files chosen in the sidebar, and so shown at
        once on the spectral exploration screen; also the number of
        prepared rows kept by the display cache.
    memory_poll_seconds : int
        Interval, in seconds, at which the sidebar refreshes the memory usage.
    """

    default_color_by: str | None = None
    default_x_axis: str | None = None
    heatmap_max_cells: PositiveInt = 1_200_000
    explore_max_files: PositiveInt = 20
    memory_poll_seconds: PositiveInt = 5


class JobsSettings(_StrictModel):
    """``[jobs]`` section.

    Attributes
    ----------
    artifact_dtype : Literal["float32", "float64"]
        Floating-point dtype of saved run matrices.
    memory_warn_gb : float
        Estimated memory above which the UI asks for confirmation.
    """

    artifact_dtype: Literal["float32", "float64"] = "float32"
    memory_warn_gb: PositiveFloat = 16


class Settings(_StrictModel):
    """Validated contents of ``settings.toml``.

    Attributes
    ----------
    workspace : WorkspaceSettings
        Workspace location.
    data : DataSettings
        Parquet discovery settings.
    metadata : MetadataSettings | None
        Metadata CSV settings, or ``None`` when no CSV is configured.
    ui : UiSettings
        Display defaults.
    jobs : JobsSettings
        Job execution settings.
    """

    workspace: WorkspaceSettings
    data: DataSettings
    metadata: MetadataSettings | None = None
    ui: UiSettings = UiSettings()
    jobs: JobsSettings = JobsSettings()

    @model_validator(mode="after")
    def _check_ui_columns(self) -> Settings:
        """Require UI default columns to be configured metadata columns.

        Returns
        -------
        Settings
            The validated settings.
        """
        columns = self.metadata_columns
        for field in ("default_color_by", "default_x_axis"):
            name = getattr(self.ui, field)
            if name is not None and name not in columns:
                raise ValueError(f"ui.{field} is not a metadata column: {name!r}")
        return self

    @property
    def metadata_columns(self) -> dict[str, MetadataColumnSettings]:
        """Return the configured metadata columns.

        Returns
        -------
        dict[str, MetadataColumnSettings]
            Column settings keyed by name; empty without ``[metadata]``.
        """
        return {} if self.metadata is None else dict(self.metadata.columns)

    @property
    def database_path(self) -> Path:
        """Return the DuckDB file path.

        Returns
        -------
        Path
            ``{workspace.dir}/flatpca.duckdb``.
        """
        return self.workspace.dir / "flatpca.duckdb"

    @property
    def runs_dir(self) -> Path:
        """Return the directory holding one subdirectory per run.

        Returns
        -------
        Path
            ``{workspace.dir}/runs``.
        """
        return self.workspace.dir / "runs"


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Read and validate ``settings.toml``.

    When a sibling ``settings.local.toml`` exists, it is read instead of
    ``path``. In ``workspace.dir``, ``data.root``, and ``metadata.csv``,
    environment variables (``%NAME%``, ``${NAME}``) and a leading ``~`` are
    expanded, ``{root}`` stands for ``data.root``, and relative paths are
    resolved against the directory containing the settings file.

    Parameters
    ----------
    path : str | Path, default ``config/settings.toml``
        Settings file path.

    Returns
    -------
    Settings
        Validated settings.

    Raises
    ------
    SettingsError
        If a file cannot be read, is not valid TOML, has a path that cannot be
        expanded, or fails validation.
    """
    settings_path = select_settings_file(Path(path).resolve())
    try:
        raw = read_settings_document(settings_path)
        expand_settings_paths(raw, settings_path.parent)
    except ValueError as error:
        raise SettingsError(str(error)) from error
    try:
        return Settings.model_validate(raw)
    except ValidationError as error:
        raise SettingsError(f"invalid settings in {settings_path}:\n{error}") from error
