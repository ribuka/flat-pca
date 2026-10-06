"""Tests for reading the metadata CSV."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from flat_pca.webui.services.metadata_csv import read_metadata_csv
from flat_pca.webui.settings import MetadataSettings, Settings


def _metadata(csv: Path, **columns: dict[str, str]) -> MetadataSettings:
    """Build metadata settings for a CSV."""
    return MetadataSettings.model_validate(
        {"csv": csv, "key": "file_name", "columns": columns}
    )


def test_read_metadata_csv_converts_configured_columns(settings: Settings) -> None:
    """Configured columns are typed and other columns are dropped."""
    assert settings.metadata is not None

    frame = read_metadata_csv(settings.metadata)

    assert frame.columns == ["stem", "lot", "date", "yield_pct"]
    assert frame.schema["date"] == pl.Datetime("us")
    assert frame.schema["yield_pct"] == pl.Float64
    assert frame.row(0) == (
        "run-1",
        "A",
        datetime.fromisoformat("2026-01-02T03:04:05"),
        91.5,
    )
    assert frame["yield_pct"][2] is None


def test_read_metadata_csv_rejects_duplicate_keys(tmp_path: Path) -> None:
    """A repeated key makes the join ambiguous."""
    csv = tmp_path / "m.csv"
    csv.write_text("file_name,lot\na,1\na,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="repeats"):
        read_metadata_csv(_metadata(csv, lot={"type": "category"}))


def test_read_metadata_csv_rejects_missing_columns(tmp_path: Path) -> None:
    """Every configured column must exist in the CSV."""
    csv = tmp_path / "m.csv"
    csv.write_text("file_name,lot\na,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        read_metadata_csv(
            _metadata(csv, lot={"type": "category"}, size={"type": "number"})
        )


def test_read_metadata_csv_rejects_unconvertible_values(tmp_path: Path) -> None:
    """A non-numeric value in a number column is an error."""
    csv = tmp_path / "m.csv"
    csv.write_text("file_name,size\na,big\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid metadata CSV values"):
        read_metadata_csv(_metadata(csv, size={"type": "number"}))


def test_read_metadata_csv_rejects_missing_file(tmp_path: Path) -> None:
    """A missing CSV is reported as ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        read_metadata_csv(_metadata(tmp_path / "none.csv"))
