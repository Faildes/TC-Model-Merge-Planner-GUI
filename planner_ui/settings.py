"""Settings responsibilities."""
from __future__ import annotations

from .constants import INIT_CONFIG
from .paths import RESOURCE_ROOT
from pathlib import Path
from typing import Any
from typing import Dict
import json
import os
import pickle

CONFIG_FILE = os.path.join(os.getcwd(), "config.tccm")


def save_config_to_disk(config):
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(config, f)


def _planner_tooltip_catalog_path_candidates() -> list[Path]:
    """Return tooltip JSON candidates in priority order."""
    here = Path((str(RESOURCE_ROOT / "main.py"))).resolve().parent
    return [
        here / "planner_tooltips.json",
        Path.cwd() / "planner_tooltips.json",
        here / "config" / "planner_tooltips.json",
    ]


def _planner_load_tooltip_catalog() -> Dict[str, Any]:
    for path in _planner_tooltip_catalog_path_candidates():
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as e:
            try:
                print(f"[tooltip] failed to load {path}: {type(e).__name__}: {e}")
            except Exception:
                pass
    return {"options": {}, "choices": {}, "modes": {}, "left_panel": {}, "right_panel": {}}


def load_config_from_disk():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "rb") as f:
                cfg = pickle.load(f)
            merged = INIT_CONFIG.copy()
            merged.update(cfg)
            if os.path.normpath(str(merged.get("workpath", "") or "")) == os.path.normpath("/kaggle"):
                merged["workpath"] = "/kaggle/working"
            ur = str(merged.get("UR", "") or "").strip()
            repo = str(merged.get("hf_repo_id", "") or "").strip()
            if repo and (not ur or ur == INIT_CONFIG.get("UR", "")):
                merged["UR"] = repo
            if str(merged.get("UR", "") or "").strip():
                merged["hf_repo_id"] = str(merged.get("UR", "") or "").strip()
            return merged
        except Exception:
            return INIT_CONFIG.copy()
    return INIT_CONFIG.copy()

PLANNER_TOOLTIP_CATALOG = _planner_load_tooltip_catalog()
LEFT_PANEL_FIELD_HELP = dict(PLANNER_TOOLTIP_CATALOG.get("left_panel") or {})
RIGHT_PANEL_FIELD_HELP = dict(PLANNER_TOOLTIP_CATALOG.get("right_panel") or {})
RIGHT_PANEL_FIELD_HELP.setdefault('LoRA Merge', {'short': 'Merge multiple LoRA/LyCORIS models into one LoRA without baking into a checkpoint.', 'detail': 'This is separate from LoRA Bake. It emits lora_bake.py with --merge_loras and registers the output as a LoRA so it can be baked or merged later.'})
RIGHT_PANEL_FIELD_HELP.setdefault('Unmapped Legacy Signatures', {'short': 'Legacy @/-- tokens that could not be represented by structured options.', 'detail': 'Additional Signatures are no longer shown as a separate editor. When old plans are imported, supported tokens are converted into Option controls; unknown or unsupported tokens are preserved here and appended during export/run.'})


# Derive help from the JSON loaded for this run, including user edits.
_TOOLTIP_CATALOG = PLANNER_TOOLTIP_CATALOG
_OPTION_HELP_TEXT = dict(_TOOLTIP_CATALOG.get("options") or {})
_CHOICE_HELP_TEXT = dict(_TOOLTIP_CATALOG.get("choices") or {})
_MODE_HELP_TEXT = dict(_TOOLTIP_CATALOG.get("modes") or {})
