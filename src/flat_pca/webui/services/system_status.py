"""Memory usage of the Web UI server and its machine, and the app version."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version

import psutil

PACKAGE_NAME = "flat_pca"
_GIB = 1024**3


@dataclass(frozen=True)
class MemoryUsage:
    """Memory used by the server process and by the whole machine.

    Attributes
    ----------
    process_rss : int
        Resident set size of the Web UI server process, in bytes.
    system_used : int
        Memory used on the machine (total minus available), in bytes.
    system_total : int
        Total physical memory of the machine, in bytes.
    """

    process_rss: int
    system_used: int
    system_total: int

    @property
    def system_percent(self) -> float:
        """Return the share of the machine's memory in use.

        Returns
        -------
        float
            ``system_used / system_total`` in percent.
        """
        return 100 * self.system_used / self.system_total


def read_memory_usage() -> MemoryUsage:
    """Read the memory usage of this process and of the machine.

    Returns
    -------
    MemoryUsage
        Current memory usage.
    """
    system = psutil.virtual_memory()
    return MemoryUsage(
        process_rss=psutil.Process().memory_info().rss,
        system_used=system.total - system.available,
        system_total=system.total,
    )


def format_gib(n_bytes: int) -> str:
    """Format a byte count in GiB with one decimal place.

    Parameters
    ----------
    n_bytes : int
        Byte count.

    Returns
    -------
    str
        For example ``"1.5 GiB"``.
    """
    return f"{n_bytes / _GIB:.1f} GiB"


def app_version() -> str:
    """Return the installed flat-pca version (the ``VERSION`` file).

    Returns
    -------
    str
        Version prefixed with ``v``, for example ``"v0.1.10"``.
    """
    return f"v{version(PACKAGE_NAME)}"
