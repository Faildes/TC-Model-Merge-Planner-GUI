"""Paths responsibilities."""
from __future__ import annotations

from pathlib import Path
RESOURCE_ROOT = Path(__file__).resolve().parent.parent

from typing import List
import os


def _planner_dir() -> Path:
    return RESOURCE_ROOT


def _toolpath_candidates(base: Path | None = None) -> List[Path]:
    root = base or _planner_dir()
    return [
        root / "tools" / "chattiori_model_merge",
        root / "tools" / "chattiori_model_merger",
    ]


def _preferred_toolpath() -> str:
    candidates = _toolpath_candidates()
    for path in candidates:
        if (path / "merge.py").exists() or (path / "lora_bake.py").exists() or path.exists():
            return str(path)
    return str(candidates[0])


def _notebook_runtime_workpath(workpath: str) -> str:
    """Return the runtime root that should appear in generated scripts/notebooks."""
    raw = str(workpath or "").strip() or "."
    raw = raw.rstrip("/\\") or raw
    if os.path.normpath(raw) == os.path.normpath("/kaggle"):
        return "/kaggle/working"
    return raw
