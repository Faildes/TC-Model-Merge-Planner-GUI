"""Variants ui controls and callbacks."""
from __future__ import annotations

from .variants import _apply_run_profile as _method_apply_run_profile
from .variants import _delete_run_profile as _method_delete_run_profile
from .variants import _delete_variant as _method_delete_variant
from .variants import _export_run_profiles_json as _method_export_run_profiles_json
from .variants import _export_variants_json as _method_export_variants_json
from .variants import _load_variant as _method_load_variant
from .variants import _save_current_as_run_profile as _method_save_current_as_run_profile
from .variants import _save_current_as_variant as _method_save_current_as_variant
from .variants import _tccm_choose_by_name as _method_tccm_choose_by_name
from plan import make_entry
from typing import Any
from typing import Dict
import copy


class VariantsUiMixin:
    """Variants ui behavior for the Planner."""

    _save_current_as_variant = _method_save_current_as_variant


    _load_variant = _method_load_variant


    _delete_variant = _method_delete_variant


    _export_variants_json = _method_export_variants_json


    _save_current_as_run_profile = _method_save_current_as_run_profile


    _apply_run_profile = _method_apply_run_profile


    _delete_run_profile = _method_delete_run_profile


    _export_run_profiles_json = _method_export_run_profiles_json


    def _cache_entry_variant(self, entry: Dict[str, Any]):
        entry_id = entry.get("id")
        entry_type = entry.get("type")
        if not entry_id or not entry_type:
            return
        self.line_variant_cache.setdefault(entry_id, {})[entry_type] = copy.deepcopy(entry)


    def _restore_entry_variant(self, current_entry: Dict[str, Any], new_type: str) -> Dict[str, Any]:
        entry_id = current_entry.get("id") or make_entry(new_type).get("id")
        cached = self.line_variant_cache.get(entry_id, {}).get(new_type)
        if cached is not None:
            restored = copy.deepcopy(cached)
            restored["id"] = entry_id
            restored["type"] = new_type
            return restored
        fresh = make_entry(new_type)
        fresh["id"] = entry_id
        return fresh


    _tccm_choose_by_name = _method_tccm_choose_by_name
