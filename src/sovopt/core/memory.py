"""How much physical memory is free: the one number a refusal to start a
model that cannot fit is measured against.

A model past what the machine holds does not fail cleanly. It pages, the
machine slows to a crawl, and whatever was waiting for the answer -- the
page's request, a terminal -- sees the process vanish when it is closed or
killed. Test_Data's two years of hourly planning did exactly that on the
page. Asking the operating system first costs one call.

Nothing here allocates or measures the process; it reads what the
operating system reports as available to a new allocation without
swapping, and returns None when it cannot tell, in which case every check
that uses it stands aside.

References
----------
Microsoft, "GlobalMemoryStatusEx function (sysinfoapi.h)" and the
  MEMORYSTATUSEX structure, Win32 API documentation -- ``ullAvailPhys``.
Linux man-pages, proc_meminfo(5) -- ``MemAvailable``, the kernel's estimate
  of memory available to start new applications without swapping.
POSIX.1-2017, sysconf() -- ``_SC_AVPHYS_PAGES`` and ``_SC_PAGE_SIZE``, the
  fallback elsewhere.
"""

from __future__ import annotations

import os
import sys

__all__ = ["available_bytes", "gib"]


def available_bytes() -> int | None:
    """Physical memory available to a new allocation, in bytes; None if the
    operating system will not say."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD),
                            ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemoryStatusEx()
            st.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return int(st.ullAvailPhys)
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        pages, size = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return int(pages) * int(size)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def gib(n: float) -> str:
    """A byte count as a person reads it off a task manager: GB, or MB
    under one."""
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"
