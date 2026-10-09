"""Tests that the data table package depends only on what it may use."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from flat_pca.webui import datatable

PACKAGE_DIR = Path(datatable.__file__).parent
# Third-party packages the data table may import besides the standard library.
ALLOWED_PACKAGES = {"polars", "jinja2"}


def _imported_modules(path: Path) -> list[tuple[str, int]]:
    """Return each absolute module a file imports and each relative level."""
    imports: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, 0) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", node.level))
    return imports


def test_datatable_imports_only_the_standard_library_polars_and_jinja2() -> None:
    """No module of the package imports flat_pca or another package."""
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    assert files

    forbidden = [
        f"{path.relative_to(PACKAGE_DIR)}: {'.' * level}{module}"
        for path in files
        for module, level in _imported_modules(path)
        if (level > 1)
        or (
            level == 0
            and module.split(".")[0] not in sys.stdlib_module_names | ALLOWED_PACKAGES
        )
    ]

    assert forbidden == []
