"""Loading and validation of the Web UI ``settings.toml``."""

from __future__ import annotations

import tomllib
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

MetadataColumnType = Literal["category", "number", "datetime"]
RESERVED_METADATA_COLUMNS = frozenset({"stem"})


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
    default_order_by : str | None
        Metadata column ordering T²/Q control charts by default.
    heatmap_max_cells : int
        Maximum number of heatmap cells sent to the browser.
    """

    default_color_by: str | None = None
    default_order_by: str | None = None
    heatmap_max_cells: PositiveInt = 1_200_000


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
        for field in ("default_color_by", "default_order_by"):
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


def _resolve_relative_paths(raw: dict[str, object], base: Path) -> None:
    """Resolve relative path entries against the settings file directory.

    Parameters
    ----------
    raw : dict[str, object]
        Parsed TOML document, modified in place.
    base : Path
        Directory containing the settings file.
    """
    for section, key in (("workspace", "dir"), ("data", "root"), ("metadata", "csv")):
        table = raw.get(section)
        if isinstance(table, dict) and isinstance(table.get(key), str):
            table[key] = str(base / Path(table[key]))


def load_settings(path: str | Path) -> Settings:
    """Read and validate ``settings.toml``.

    Relative ``workspace.dir``, ``data.root``, and ``metadata.csv`` values are
    resolved against the directory containing the settings file.

    Parameters
    ----------
    path : str | Path
        Settings file path.

    Returns
    -------
    Settings
        Validated settings.

    Raises
    ------
    SettingsError
        If the file cannot be read, is not valid TOML, or fails validation.
    """
    settings_path = Path(path).resolve()
    try:
        with settings_path.open("rb") as file:
            raw = tomllib.load(file)
    except OSError as error:
        raise SettingsError(
            f"cannot read settings file {settings_path}: {error}"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise SettingsError(f"invalid TOML in {settings_path}: {error}") from error

    _resolve_relative_paths(raw, settings_path.parent)
    try:
        return Settings.model_validate(raw)
    except ValidationError as error:
        raise SettingsError(f"invalid settings in {settings_path}:\n{error}") from error
