"""Schema responsibilities."""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import shlex


def _infer_ratio_mode(value: str, allow_block_weight: bool = True, randomized: bool = False) -> str:
    s = str(value or "").strip()
    if randomized:
        return "Randomize"
    if not s:
        return "Single"
    if ":" in s:
        return "Elemental"
    if allow_block_weight and "," in s:
        return "Block weight"
    return "Single"


def default_ratio(mode: str = "Single") -> Dict[str, Any]:
    return {
        "mode": mode,
        "value": "0.5" if mode == "Single" else ("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" if mode == "Block weight" else ""),
    }


def _normalize_ratio_spec(spec: Any, *, allow_block_weight: bool = True, default_single: str = "0.5") -> Dict[str, Any]:
    if isinstance(spec, dict):
        mode = str(spec.get("mode") or _infer_ratio_mode(spec.get("value", ""), allow_block_weight=allow_block_weight))
        value = str(spec.get("value", "")).strip()
    else:
        value = str(spec or "").strip()
        mode = _infer_ratio_mode(value, allow_block_weight=allow_block_weight)
    if mode == "Single" and not value:
        value = default_single
    if mode == "Block weight" and not value:
        value = default_ratio("Block weight")["value"]
    return {"mode": mode, "value": value}


def make_entry_base(entry_type: str = "Checkpoint Merge") -> Dict[str, Any]:
    from .syntax import _uid

    base = {"id": _uid(), "type": entry_type}
    if entry_type == "Download Model":
        base.update({"model_name": "", "link": "", "model_type": "Checkpoint"})
    elif entry_type == "Local Model":
        base.update({"local_path": "", "model_name": "", "model_type": "Checkpoint"})
    elif entry_type == "Remove Model":
        base.update({"model": ""})
    elif entry_type == "Checkpoint Merge":
        base.update({
            "merge_mode": "WS",
            "model0": "",
            "model1": "",
            "model2": "",
            "alpha": default_ratio("Single"),
            "beta": default_ratio("Single"),
            "output_name": "",
            "precision": "",
            "additional_signatures": "",
            "raw_signatures": "",
        })
    elif entry_type == "LoRA Bake":
        base.update({
            "checkpoint": "",
            "loras": [],
            "output_name": "",
            "precision": "",
            "additional_signatures": "",
            "raw_signatures": "",
        })
    return base


def default_plan_original() -> Dict[str, Any]:
    return {"version": 2, "format": "planner-json", "final_memo": "", "history": [], "entries": [make_entry("Checkpoint Merge")]}






def make_entry_structured(entry_type: str = "Checkpoint Merge") -> Dict[str, Any]:
    from .schema import make_entry_base as _PLANNER_ORIG_MAKE_ENTRY
    from .options import _planner_default_cli_options
    from .syntax import _uid

    if entry_type == "LoRA Merge":
        return {
            "id": _uid(),
            "type": "LoRA Merge",
            "loras": [],
            "output_name": "",
            "precision": "",
            "additional_signatures": "",
            "raw_signatures": "",
            "cli_options": _planner_default_cli_options("LoRA Merge"),
        }
    entry = _PLANNER_ORIG_MAKE_ENTRY(entry_type)
    if entry_type in ("Checkpoint Merge", "LoRA Bake"):
        entry.setdefault("cli_options", _planner_default_cli_options(entry_type))
    return entry


def default_plan_structured() -> Dict[str, Any]:
    from .schema import default_plan_original as _PLANNER_ORIG_DEFAULT_PLAN

    plan = _PLANNER_ORIG_DEFAULT_PLAN()
    plan.setdefault("final_memo", "")
    plan.setdefault("history", [])
    plan.setdefault("meta", {})
    return plan




def _planner_quote_cli_value(value: Any) -> str:
    return shlex.quote(str(value))






def _lora_items_lines_from_entry(entry: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
    from .compiler import _json_literal

    parts = []
    for lora in entry.get("loras") or []:
        name = str(lora.get("name") or "").strip()
        if not name:
            continue
        ratio = _normalize_ratio_spec(lora.get("ratio"), allow_block_weight=False, default_single="1.0")["value"] or "1.0"
        parts.append((name, ratio))
    lines = ['lora_items = []']
    for name, ratio in parts:
        safe_ratio = str(ratio).replace('\\', '\\\\').replace('"', '\\"')
        lines.append('lora_items.append(f"{model_file(' + _json_literal(name).replace('"', "'") + ')}:" + "' + safe_ratio + '")')
    return lines, parts


def _entry_reference_aliases_base(entry: Dict[str, Any]) -> List[str]:
    from .compiler import _entry_reference_aliases as _PLANNER_ORIG_ENTRY_REFS

    if entry.get("type") != "LoRA Merge":
        return _PLANNER_ORIG_ENTRY_REFS(entry)
    refs: List[str] = []
    for lora in entry.get("loras", []) or []:
        name = str(lora.get("name") or "").strip()
        if name:
            refs.append(name)
    return refs


def _entry_produced_aliases_for_cleanup_base(entry: Dict[str, Any]) -> List[str]:
    from .compiler import _entry_produced_aliases_for_cleanup as _PLANNER_ORIG_ENTRY_PRODUCED

    if entry.get("type") == "LoRA Merge":
        name = str(entry.get("output_name") or "").strip()
        return [f"TEMP{name}" if name.startswith("_") else name] if name else []
    return _PLANNER_ORIG_ENTRY_PRODUCED(entry)


def make_entry(entry_type: str = "Checkpoint Merge") -> Dict[str, Any]:
    from .schema import make_entry_structured as _SIGLESS_PREV_MAKE_ENTRY
    from .options import _planner_default_cli_options

    entry = _SIGLESS_PREV_MAKE_ENTRY(entry_type)
    if entry_type in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
        entry.setdefault("cli_options", _planner_default_cli_options(entry_type))
        entry.setdefault("architecture", "")
        entry.setdefault("unmapped_signatures", "")
        entry["additional_signatures"] = ""
        entry["raw_signatures"] = ""
    return entry


def _ratio_text_base(spec: Dict[str, Any] | None) -> str:
    spec = _normalize_ratio_spec(spec, allow_block_weight=True, default_single="0.5")
    mode = str(spec.get("mode", "Single"))
    value = str(spec.get("value", "")).strip()
    if mode == "Randomize":
        return "@r " + (value.strip("\"'") or "0.0,1.0")
    if mode == "Single":
        return value or "0.5"
    value = f'"{value[0].strip("\'\"")}{value[1:-1]}{value[-1].strip("\'\"")}"' if value else '""'
    return value


def default_plan() -> Dict[str, Any]:
    from .schema import default_plan_structured as _tccm_adv_default_plan_base

    plan = _tccm_adv_default_plan_base()
    plan.setdefault("variants", [])
    plan.setdefault("run_profiles", [])
    plan.setdefault("current_variant", "")
    plan.setdefault("schema_version", 3)
    return plan




def normalize_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize current and legacy records in one pass.

    Preserve IDs, embedded sources, unknown row fields and project metadata.
    Legacy CLI tokens are migrated before final ratio/precision normalization.
    """
    from .options import (
        _display_arch_name, _display_precision_name, _planner_clean_cli_options,
        _sigless_apply_legacy_signatures_base,
    )

    plan = {
        "version": 2, "format": "planner-json", "final_memo": "", "history": [],
        "entries": [], "meta": {}, "variants": [], "run_profiles": [],
        "current_variant": "", "schema_version": 3,
    }
    if isinstance(data, dict):
        plan["version"] = data.get("version", 2)
        plan["format"] = data.get("format", "planner-json")
        plan["final_memo"] = str(data.get("final_memo", "") or "")
        history = data.get("history", [])
        plan["history"] = history if isinstance(history, list) else []
        if isinstance(data.get("meta"), dict):
            plan["meta"] = data["meta"]
        for key in ("variants", "run_profiles", "current_variant", "runtime",
                    "schema_version", "migrated_from", "migrated_at"):
            if key in data:
                plan[key] = data[key]
        for raw in data.get("entries", []):
            if not isinstance(raw, dict):
                continue
            entry_type = raw.get("type", "Checkpoint Merge")
            entry = make_entry(entry_type)
            entry.update(raw)
            if entry_type in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
                if entry_type != "Checkpoint Merge":
                    entry["loras"] = [
                        {
                            "name": lora.get("name", ""),
                            "ratio": _normalize_ratio_spec(
                                lora.get("ratio"), allow_block_weight=False,
                                default_single="1.0",
                            ),
                            **({"_source": lora["_source"]}
                               if isinstance(lora.get("_source"), dict) else {}),
                        }
                        for lora in entry.get("loras", [])
                        if isinstance(lora, dict)
                    ]
                entry["cli_options"] = _planner_clean_cli_options(
                    entry_type, entry.get("cli_options")
                )
                _sigless_apply_legacy_signatures_base(entry)
                entry["architecture"] = _display_arch_name(entry.get("architecture"))
                entry["precision"] = _display_precision_name(entry.get("precision"))
                if entry_type == "Checkpoint Merge":
                    for key in ("alpha", "beta"):
                        ratio = _normalize_ratio_spec(entry.get(key))
                        if ratio["mode"] == "Randomize":
                            ratio["value"] = ratio["value"].strip("\"'")
                        entry[key] = ratio
                    mode = str(entry.get("merge_mode") or "WS").strip().upper()
                    if mode in {"CLIPXOR", "XDARE"}:
                        entry["merge_mode"] = "NoIn" if mode == "CLIPXOR" else "DARE"
                        entry["cli_options"]["clipxor"] = True
            plan["entries"].append(entry)
    if not plan["entries"]:
        plan["entries"] = [make_entry("Checkpoint Merge")]
    return plan
