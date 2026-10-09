"""Fetch the subset Material Symbols font bundled with the Web UI.

The Web UI works without a network, so it serves a Material Symbols
(Outlined) font that contains only the icons it uses. Add an icon name to
``ICON_NAMES`` and run ``uv run -m scripts.fetch_material_symbols`` from the
repository root to download the font again.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from pathlib import Path

ICON_NAMES = (
    "check_circle",
    "help",
    "left_panel_close",
    "left_panel_open",
    "progress_activity",
)
FAMILY = "Material Symbols Outlined"
AXES = "opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200"
CSS2_URL = "https://fonts.googleapis.com/css2"
# Google Fonts returns woff2 only to browsers that support it.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src/flat_pca/webui/static/vendor/material-symbols-outlined.woff2"
)


def css2_url(icon_names: tuple[str, ...]) -> str:
    """Build the CSS2 API URL of a font that contains only the given icons.

    Parameters
    ----------
    icon_names : tuple[str, ...]
        Material Symbols icon names, in any order.

    Returns
    -------
    str
        CSS2 API URL. ``icon_names`` is sorted alphabetically, as the API
        requires.
    """
    query = urllib.parse.urlencode(
        {
            "family": f"{FAMILY}:{AXES}",
            "icon_names": ",".join(sorted(set(icon_names))),
            "display": "block",
        },
        safe=":,@.",
    )
    return f"{CSS2_URL}?{query}"


def font_url(css: str) -> str:
    """Extract the woff2 URL from a CSS2 API stylesheet.

    Parameters
    ----------
    css : str
        Stylesheet returned by the CSS2 API.

    Returns
    -------
    str
        URL of the woff2 font.

    Raises
    ------
    ValueError
        If the stylesheet has no woff2 source.
    """
    match = re.search(r"src:\s*url\(([^)]+)\)\s*format\('woff2'\)", css)
    if match is None:
        raise ValueError(f"no woff2 source in the stylesheet:\n{css}")
    return match.group(1)


def _get(url: str) -> bytes:
    """Download a URL with a browser user agent.

    Parameters
    ----------
    url : str
        URL to download.

    Returns
    -------
    bytes
        Response body.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main() -> None:
    """Download the subset font of ``ICON_NAMES`` to ``OUTPUT``."""
    css = _get(css2_url(ICON_NAMES)).decode()
    font = _get(font_url(css))
    OUTPUT.write_bytes(font)
    print(f"{OUTPUT.name}: {len(font):,} bytes ({', '.join(sorted(ICON_NAMES))})")


if __name__ == "__main__":
    main()
