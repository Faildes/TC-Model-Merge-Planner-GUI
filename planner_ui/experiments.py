"""Experiments controls and callbacks."""
from __future__ import annotations

from .developer_tools import _parse_sweep_values as _method_parse_sweep_values
from .developer_tools import _planner_dev_ensure_config as _method_planner_dev_ensure_config
from .developer_tools import _planner_dev_mode_enabled as _method_planner_dev_mode_enabled
from .developer_tools import _planner_dev_only as _method_planner_dev_only
from .developer_tools import _planner_model_anatomy_scan as _method_planner_model_anatomy_scan
from .developer_tools import _planner_optimize_plan_full as _method_planner_optimize_plan_full
from .developer_tools import _planner_show_image_compare as _method_planner_show_image_compare
from .developer_tools import _planner_show_versions as _method_planner_show_versions
from .disk_estimates import _planner_estimate_disk_runtime_robust as _method_planner_estimate_disk_runtime
from .editor_support import _planner_open_ratio_sweep as _method_planner_open_ratio_sweep


class ExperimentsMixin:
    """Experiments behavior for the Planner."""

    def _show_unused_models(self):
        analysis = self._planner_analysis()
        dead = sorted(analysis.get("dead_entries", set()))
        detail_lines = ["Dead / unused merge lines:", ""]
        if not dead:
            detail_lines.append("None")
        else:
            for idx in dead:
                detail_lines.append(f"line {idx + 1}: {self._line_summary(self.plan_data['entries'][idx])}")
        self._show_scrollable_text_dialog("Unused Models", "\n".join(detail_lines))


    _planner_dev_ensure_config = _method_planner_dev_ensure_config


    _planner_dev_mode_enabled = _method_planner_dev_mode_enabled


    _planner_dev_only = _method_planner_dev_only


    _planner_show_versions = _method_planner_show_versions


    _parse_sweep_values = _method_parse_sweep_values


    _planner_model_anatomy_scan = _method_planner_model_anatomy_scan


    _planner_show_image_compare = _method_planner_show_image_compare


    _planner_optimize_plan_full = _method_planner_optimize_plan_full


    _planner_open_ratio_sweep = _method_planner_open_ratio_sweep


    _planner_estimate_disk_runtime = _method_planner_estimate_disk_runtime
