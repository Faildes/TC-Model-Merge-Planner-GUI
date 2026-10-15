"""Native dialogs responsibilities."""
from __future__ import annotations

from pathlib import Path
from tkinter.filedialog import askopenfilename as _TCCM_REAL_ASKOPENFILENAME
from tkinter.filedialog import askopenfilenames as _TCCM_REAL_ASKOPENFILENAMES
from tkinter.filedialog import asksaveasfilename as _TCCM_REAL_ASKSAVEASFILENAME
from typing import Any
from typing import Dict
import sys


def _tccm_running_on_macos() -> bool:
    return sys.platform == "darwin"


def _tccm_safe_dialog_kwargs(kwargs: Dict[str, Any], *, save_dialog: bool = False) -> Dict[str, Any]:
    safe = dict(kwargs or {})
    if not _tccm_running_on_macos():
        return safe

    # This is the important part: do not send custom type filters to Tk on
    # macOS.  Tk's Cocoa file dialog bridge may translate some extensions to
    # nil UTTypes and abort before Python can catch anything.
    safe.pop("filetypes", None)
    safe.pop("typevariable", None)

    # Keep the dialog close to the first implementation: title/initialdir are
    # enough for the file picker, and extension handling is done after return.
    try:
        initialdir = str(safe.get("initialdir") or "").strip()
        if initialdir and not Path(initialdir).expanduser().exists():
            safe.pop("initialdir", None)
    except Exception:
        safe.pop("initialdir", None)

    # Multi-dot defaultextension values may also reach the native save-panel
    # type bridge on some Tk builds.  Use a simple extension in the dialog and
    # then force .tccm.json in Python where needed.
    if save_dialog:
        ext = str(safe.get("defaultextension") or "")
        if ext.count(".") > 1:
            safe["defaultextension"] = ".json"
    return safe


def _tccm_askopenfilename_safe(*args, **kwargs):
    return _TCCM_REAL_ASKOPENFILENAME(*args, **_tccm_safe_dialog_kwargs(kwargs, save_dialog=False))


def _tccm_askopenfilenames_safe(*args, **kwargs):
    return _TCCM_REAL_ASKOPENFILENAMES(*args, **_tccm_safe_dialog_kwargs(kwargs, save_dialog=False))


def _tccm_asksaveasfilename_safe(*args, **kwargs):
    return _TCCM_REAL_ASKSAVEASFILENAME(*args, **_tccm_safe_dialog_kwargs(kwargs, save_dialog=True))
