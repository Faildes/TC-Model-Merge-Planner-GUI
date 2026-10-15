"""Model editors controls and callbacks."""
from __future__ import annotations

from .editor_support import _add_lora_block_preserve_scroll as _method_add_lora_block
from .editor_support import _move_lora_block as _method_move_lora_block
from .editor_support import _planner_lora_compatibility_check as _method_planner_lora_compatibility_check
from .editor_support import _remove_lora_block_preserve_scroll as _method_remove_lora_block
from .source_support import _planner_lora_ref_resolves_final as _method_planner_lora_ref_resolves_final
from .source_support import _render_checkpoint_merge_entry_source_buttons_final as _method_render_checkpoint_merge_entry
from .source_support import _render_lora_bake_entry_source_buttons_hover_final as _method_render_lora_bake_entry
from .source_support import _render_lora_merge_entry_source_buttons_hover_final as _method_render_lora_merge_entry
from typing import Any
from typing import Dict
import copy
import tkinter as tk


class ModelEditorsMixin:
    """Model editors behavior for the Planner."""

    def _on_base_model_change(self, _event=None):
        self._schedule_config_save()
        if self.plan_data.get("entries"):
            canvas = getattr(self, "canvas", None)
            yview = None
            try:
                if canvas is not None and canvas.winfo_exists():
                    yview = canvas.yview()
            except Exception:
                yview = None
            self._render_current_line()
            if yview is not None:
                try:
                    if canvas is not None and canvas.winfo_exists():
                        canvas.update_idletasks()
                        canvas.yview_moveto(float(yview[0]))
                except Exception:
                    pass


    _render_checkpoint_merge_entry = _method_render_checkpoint_merge_entry


    def _resolve_merge_mode_key(self, raw_mode: str) -> str:
        raw = str(raw_mode or "").strip().lower()
        if not raw:
            return ""
        for item in self.merge_modes:
            key = str(item.get("key") or "").strip()
            if key and raw == key.lower():
                return key
        return ""


    _render_lora_bake_entry = _method_render_lora_bake_entry


    def _sync_lora_name(self, lora_entry: Dict[str, Any], name_var: tk.StringVar):
        lora_entry["name"] = name_var.get()
        self._after_entry_change()


    _add_lora_block = _method_add_lora_block


    _remove_lora_block = _method_remove_lora_block


    def _get_lora_source(self, lora_entry):
        value = lora_entry.get("_source")
        return value if isinstance(value, dict) else None


    def _set_lora_source(self, lora_entry, spec):
        if spec:
            lora_entry["_source"] = copy.deepcopy(spec)
        else:
            lora_entry.pop("_source", None)


    _move_lora_block = _method_move_lora_block


    _planner_lora_compatibility_check = _method_planner_lora_compatibility_check


    _render_lora_merge_entry = _method_render_lora_merge_entry


    _planner_lora_ref_resolves_final = _method_planner_lora_ref_resolves_final
