from __future__ import annotations

import re


def natural_keys(text: int | str | bool) -> list[int | str]:

    if isinstance(text, bool):
        text = str(text)

    if isinstance(text, int):
        return [text]

    def atoi(s: str) -> int | str:
        return int(s) if s.isdigit() else s

    return [atoi(c) for c in re.split(r"(\d+)", str(text))]
