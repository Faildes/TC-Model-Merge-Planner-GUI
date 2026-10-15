"""Discovery responsibilities."""
from __future__ import annotations

from .constants import BASE_MODEL_BLOCK_ATTRS
from .constants import BASE_MODEL_OPTIONS
from .constants import FALLBACK_MERGE_MODES
from .constants import SDXL_BLOCKS
from .paths import RESOURCE_ROOT
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
import importlib.util

_DISCOVER_BLOCKS_INSTALL_ATTEMPTED = False


def _coerce_block_names(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        seq = list(raw.keys())
    elif isinstance(raw, (list, tuple)):
        seq = list(raw)
    else:
        try:
            seq = list(raw)
        except Exception:
            seq = []
    return [str(x) for x in seq if str(x).strip()]


def _load_module_from_candidates(module_name: str, candidates: List[Path]):
    errors: list[str] = []
    for path in candidates:
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                errors.append(f"spec failed: {path}")
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, errors
        except Exception as e:
            errors.append(f"{path}: {type(e).__name__}: {e}")
    return None, errors


def discover_block_sets() -> Dict[str, List[str]]:
    global _DISCOVER_BLOCKS_INSTALL_ATTEMPTED
    fallback = {name: list(SDXL_BLOCKS) for name in BASE_MODEL_OPTIONS}
    # Anima is one Planner-facing family.  Use the 40-block 2.9B topology as
    # the superset even when merger Utils.py is temporarily unavailable;
    # merge.py remaps legacy/native 28-block ratios by the expansion manifest.
    fallback["Anima"] = ["BASE"] + [f"L{i:02d}" for i in range(40)] + ["VAE"]
    roots = list(dict.fromkeys([
        Path.cwd() / "tools/chattiori_model_merger",
        RESOURCE_ROOT / "tools/chattiori_model_merger",
    ]))
    # Block names do not need tensor engines. The new tables module uses only
    # the standard library; older installations still fall back to Utils.py.
    candidates = [root / "cmm_core/block_tables.py" for root in roots]
    candidates += [root / "Utils.py" for root in roots]
    utils_mod, load_errors = _load_module_from_candidates("planner_cmm_utils", candidates)

    if utils_mod is None and not _DISCOVER_BLOCKS_INSTALL_ATTEMPTED:
        _DISCOVER_BLOCKS_INSTALL_ATTEMPTED = True
        try:
            from install_planner_deps import install_or_update
            install_or_update(skip_pip=True, update_if_needed=True)
            importlib.invalidate_caches()
            utils_mod, retry_errors = _load_module_from_candidates("planner_cmm_utils_retry", candidates)
            load_errors.extend(retry_errors)
        except Exception as e:
            load_errors.append(f"installer: {type(e).__name__}: {e}")

    if utils_mod is None:
        print("[discover_block_sets] Falling back to SDXL block names.")
        for err in load_errors:
            print(f"[discover_block_sets] {err}")
        return fallback

    for base_model, attr_name in BASE_MODEL_BLOCK_ATTRS.items():
        names = _coerce_block_names(getattr(utils_mod, attr_name, None))
        if names:
            fallback[base_model] = names
    # Planner intentionally exposes one Anima family with the 40-block
    # superset.  This also keeps the UI correct if an older merger is still
    # installed for a moment during an upgrade; execution itself still
    # requires the matching merger version for 28<->40 compatibility.
    fallback["Anima"] = ["BASE"] + [f"L{i:02d}" for i in range(40)] + ["VAE"]
    return fallback


def discover_merge_modes() -> List[Dict[str, Any]]:
    try:
        # Keep startup usable even when the merger tool has not been installed
        # yet (or is temporarily unavailable/offline).  The previous import
        # happened outside this try block and could abort the whole Planner.
        from tools.chattiori_model_merger.merge_modes import theta_funcs, modes_need_m2, modes_need_beta
        modes: List[Dict[str, Any]] = []
        for key, value in theta_funcs.items():
            if key in {"CLIPXOR", "XDARE"}:
                continue
            label = "No Merge (Model 0)" if key == "NoIn" else key
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                label = str(value[2])
            elif isinstance(value, dict):
                label = str(value.get("label") or value.get("name") or key)
            modes.append({
                "key": key,
                "label": label,
                "needs_m2": key in modes_need_m2,
                "needs_beta": key in modes_need_beta,
            })
        if modes:
            return modes
    except Exception:
        pass
    return FALLBACK_MERGE_MODES
