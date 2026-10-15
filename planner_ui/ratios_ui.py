"""Ratios ui controls and callbacks."""
from __future__ import annotations

from .editor_support import _build_ratio_value_widget_sigless as _method_build_ratio_value_widget
from .editor_support import _planner_build_ratio_preset_buttons as _method_planner_build_ratio_preset_buttons
from .editor_support import _planner_load_ratio_preset as _method_planner_load_ratio_preset
from .editor_support import _planner_ratio_to_lines as _method_planner_ratio_to_lines
from .editor_support import _planner_save_ratio_preset as _method_planner_save_ratio_preset
from .editor_support import _planner_show_ratio_diff as _method_planner_show_ratio_diff
from .editor_support import _safe_migration_tool as _method_safe_migration_tool
from .help_support import _build_ratio_section_dropdown_refined as _method_build_ratio_section
from tkinter import Frame
from typing import Any
from typing import Dict
from typing import List
import re
import tkinter as tk


class RatiosUiMixin:
    """Ratios ui behavior for the Planner."""

    _safe_migration_tool = _method_safe_migration_tool


    @staticmethod
    def _is_plain_number(value: str) -> bool:
        return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value or "").strip()))


    def _serialize_block_values(
        self,
        values: List[Any],
        expected_count: int | None = None,
        *,
        trailing_comma: bool = False,
    ) -> str:
        count = expected_count or len(self._current_block_names())
        normalized = [
            self._format_ratio_float(values[i] if i < len(values) else 0.0)
            for i in range(count)
        ]
        text = ",".join(normalized)
        return text + "," if trailing_comma else text


    def _parse_block_values(self, value: str, expected_count: int | None = None) -> List[float]:
        count = expected_count or len(self._current_block_names())
        raw = str(value or "").replace("，", ",").replace("\n", ",").strip()
        if not raw:
            return [0.0] * count

        parts = [p.strip() for p in raw.split(",")]
        vals: List[float] = []
        for part in parts:
            if not part:
                continue
            vals.append(self._coerce_ratio_float(part))

        if len(vals) < count:
            vals.extend([0.0] * (count - len(vals)))
        return vals[:count]


    _build_ratio_section = _method_build_ratio_section


    @staticmethod
    def _clamp_ratio_float(value: float) -> float:
        return max(0.0, min(1.0, value))


    def _coerce_ratio_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return self._clamp_ratio_float(float(str(value).strip()))
        except Exception:
            return self._clamp_ratio_float(default)


    def _format_ratio_float(self, value: Any) -> str:
        v = self._coerce_ratio_float(value)
        text = f"{v:.12f}".rstrip("0").rstrip(".")
        return text or "0"


    def _commit_block_ratio_entry(
        self,
        value_var: tk.StringVar,
        scale_var: tk.DoubleVar,
        frame: Frame,
        ratio: Dict[str, Any],
    ):
        value = self._coerce_ratio_float(value_var.get(), default=scale_var.get())
        formatted = self._format_ratio_float(value)
        if value_var.get() != formatted:
            value_var.set(formatted)
        if abs(scale_var.get() - value) > 1e-9:
            scale_var.set(value)
        self._sync_block_ratio(frame, ratio)


    def _sync_block_ratio_scale(
        self,
        value_var: tk.StringVar,
        scale_var: tk.DoubleVar,
        frame: Frame,
        ratio: Dict[str, Any],
    ):
        value = self._coerce_ratio_float(scale_var.get())
        formatted = self._format_ratio_float(value)
        if value_var.get() != formatted:
            value_var.set(formatted)
        self._sync_block_ratio(frame, ratio)


    _build_ratio_value_widget = _method_build_ratio_value_widget


    def _sync_block_ratio(self, frame: Frame, ratio: Dict[str, Any]):
        values: List[float] = []
        for child in frame.winfo_children():
            if isinstance(child, Frame):
                var = getattr(child, "_ratio_var", None)
                if var is not None:
                    values.append(self._coerce_ratio_float(var.get()))
        expected = len(self._current_block_names())
        if len(values) < expected:
            values.extend([0.0] * (expected - len(values)))
        ratio["value"] = self._serialize_block_values(
            values[:expected],
            expected_count=expected,
            trailing_comma=False,
        )
        self._after_entry_change()


    def _sync_ratio_value(self, ratio: Dict[str, Any], value: str):
        ratio["value"] = value
        self._after_entry_change()


    def _ratio_single_float(self, ratio, default_value: float = 0.5):
        if not isinstance(ratio, dict):
            return float(default_value)
        if str(ratio.get("mode") or "Single") != "Single":
            raise ValueError("Only Single ratios can be collapsed")
        try:
            return float(str(ratio.get("value") or default_value).strip())
        except Exception as exc:
            raise ValueError(f"Invalid Single ratio: {ratio!r}") from exc


    _planner_ratio_to_lines = _method_planner_ratio_to_lines


    _planner_show_ratio_diff = _method_planner_show_ratio_diff


    _planner_save_ratio_preset = _method_planner_save_ratio_preset


    _planner_load_ratio_preset = _method_planner_load_ratio_preset


    _planner_build_ratio_preset_buttons = _method_planner_build_ratio_preset_buttons
