"""Validation ui controls and callbacks."""
from __future__ import annotations

from .source_support import _planner_analysis_source_aware as _method_planner_analysis
from .source_support import _show_dependency_view_source_aware as _method_show_dependency_view
from .tools_support import _planner_show_dependency_graph as _method_planner_show_dependency_graph
from .tools_support import _show_dependency_chain_viewer as _method_show_dependency_chain_viewer
from .tools_support import _show_output_collision_checker as _method_show_output_collision_checker
from .tools_support import _show_plan_health_dashboard as _method_show_plan_health_dashboard
from .tools_support import _tccm_analyze_plan as _method_tccm_analyze_plan
from .validation_support import _plan_entry_problem_map_linear as _method_plan_entry_problem_map
from .validation_support import _planner_collect_issue_rows as _method_planner_collect_issue_rows
from .validation_support import _planner_problem_explanation_titleized as _method_planner_problem_explanation
from .validation_support import _planner_problem_severity_final as _method_planner_problem_severity
from .validation_support import _show_prevalidation_with_download_checks as _method_show_prevalidation
from pathlib import Path
import copy


class ValidationUiMixin:
    """Validation ui behavior for the Planner."""

    _planner_show_dependency_graph = _method_planner_show_dependency_graph


    _show_plan_health_dashboard = _method_show_plan_health_dashboard


    _show_dependency_chain_viewer = _method_show_dependency_chain_viewer


    _show_output_collision_checker = _method_show_output_collision_checker


    _plan_entry_problem_map = _method_plan_entry_problem_map


    def _planner_dependency_summary(self):
        analysis = self._planner_analysis()
        links = sum(len(v) for v in analysis['consumed_by_entry'].values())
        missing = len(analysis['missing_links'])
        dup = len(analysis['duplicate_aliases'])
        return f'Deps: {links} link(s) / {missing} missing / {dup} duplicate alias(es)'


    _show_dependency_view = _method_show_dependency_view


    _show_prevalidation = _method_show_prevalidation


    _planner_analysis = _method_planner_analysis


    _planner_problem_severity = _method_planner_problem_severity


    _planner_problem_explanation = _method_planner_problem_explanation


    def _validate_plan_entries(self):
        problems = []
        available = {'Checkpoint': set(), 'LoRA': set(), 'LyCORIS': set()}
        for idx, entry in enumerate(self.plan_data.get('entries', []), start=1):
            etype = entry.get('type')
            prefix = f"line {idx} ({etype})"
            if etype == 'Download Model':
                if not entry.get('model_name'): problems.append(f"{prefix}: model_name is empty")
                if not entry.get('link'): problems.append(f"{prefix}: link is empty")
                name = (entry.get('model_name') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip()
                if name: available.setdefault(kind, set()).add(name)
            elif etype == 'Local Model':
                if not entry.get('local_path'): problems.append(f"{prefix}: local_path is empty")
                path = (entry.get('local_path') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip()
                if path: available.setdefault(kind, set()).add(Path(path).stem)
            elif etype == 'Remove Model':
                if not entry.get('model'): problems.append(f"{prefix}: model is empty")
            elif etype == 'Checkpoint Merge':
                for k in ('model0','model1','output_name'):
                    if not entry.get(k): problems.append(f"{prefix}: {k} is empty")
                for ref in (entry.get('model0'), entry.get('model1'), entry.get('model2')):
                    if ref and ref not in available['Checkpoint']:
                        problems.append(f"{prefix}: checkpoint ref not available -> {ref}")
                if entry.get('output_name'): available['Checkpoint'].add(entry['output_name'])
            elif etype == 'LoRA Bake':
                if not entry.get('checkpoint'): problems.append(f"{prefix}: checkpoint is empty")
                elif entry.get('checkpoint') not in available['Checkpoint']:
                    problems.append(f"{prefix}: checkpoint ref not available -> {entry.get('checkpoint')}")
                if not entry.get('output_name'): problems.append(f"{prefix}: output_name is empty")
                for lora in entry.get('loras', []):
                    name = lora.get('name')
                    if not name:
                        problems.append(f"{prefix}: one LoRA name is empty")
                    elif name not in available['LoRA'] and name not in available['LyCORIS']:
                        problems.append(f"{prefix}: LoRA ref not available -> {name}")
                if entry.get('output_name'): available['Checkpoint'].add(entry['output_name'])
        return problems


    def _normalize_loaded_plan_entries(self, plan_data):
        data = copy.deepcopy(plan_data or {"entries": []})
        entries = data.get("entries") or []
        if not isinstance(entries, list):
            data["entries"] = []
            return data
        for entry in entries:
            self._absorb_mode_signature_from_entry(entry)
        data["entries"] = entries
        return data


    def _planner_effective_plan_data(self):
        effective_entries = []
        for entry in self.plan_data.get('entries', []):
            if self._entry_is_disabled(entry):
                continue
            etype = entry.get('type')
            if etype == 'Remove Model':
                continue
            if etype == 'Checkpoint Merge':
                for slot in ('model0', 'model1', 'model2'):
                    spec = self._get_entry_slot_source(entry, slot)
                    alias = str(entry.get(slot) or self._source_alias_from_spec(spec)).strip()
                    if spec and alias:
                        effective_entries.append(self._make_effective_source_entry(spec, alias))
                effective_entries.append(self._sanitize_effective_payload(entry))
            elif etype == 'LoRA Bake':
                checkpoint_spec = self._get_entry_slot_source(entry, 'checkpoint')
                checkpoint_alias = str(entry.get('checkpoint') or self._source_alias_from_spec(checkpoint_spec)).strip()
                if checkpoint_spec and checkpoint_alias:
                    effective_entries.append(self._make_effective_source_entry(checkpoint_spec, checkpoint_alias))
                for lora in entry.get('loras', []) or []:
                    spec = self._get_lora_source(lora)
                    alias = str(lora.get('name') or self._source_alias_from_spec(spec)).strip()
                    if spec and alias:
                        effective_entries.append(self._make_effective_source_entry(spec, alias))
                effective_entries.append(self._sanitize_effective_payload(entry))
            else:
                effective_entries.append(self._sanitize_effective_payload(entry))
        return self._append_auto_remove_lines({'entries': effective_entries})


    _planner_collect_issue_rows = _method_planner_collect_issue_rows


    _tccm_analyze_plan = _method_tccm_analyze_plan


    def _sanitize_effective_payload(self, value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if str(key).startswith("_"):
                    continue
                out[key] = self._sanitize_effective_payload(item)
            return out
        if isinstance(value, list):
            return [self._sanitize_effective_payload(x) for x in value]
        return copy.deepcopy(value)
