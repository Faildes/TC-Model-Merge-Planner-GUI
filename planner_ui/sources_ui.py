"""Sources ui controls and callbacks."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import LOCAL_TYPES
from .source_support import _build_local_selection_row_retained as _method_build_local_selection_row
from .source_support import _build_source_backed_ref_row_dropdown_refined as _method_build_source_backed_ref_row
from .source_support import _collect_available_models_dependency_safe_final as _method_collect_available_models
from .source_support import _entry_consumed_aliases_with_lora_merge as _method_entry_consumed_aliases
from .source_support import _entry_produced_aliases_dependency_safe_final as _method_entry_produced_aliases
from .source_support import _iter_embedded_sources_with_lora_merge as _method_iter_embedded_sources
from .source_support import _make_effective_source_entry_alias_preserving as _method_make_effective_source_entry
from .source_support import _planner_bucket_has_ref_final as _method_planner_bucket_has_ref_final
from .source_support import _planner_build_source_model_row_final as _method_planner_build_source_model_row_final
from .source_support import _planner_entry_can_produce_final as _method_planner_entry_can_produce_final
from .source_support import _planner_merge_requires_model2_final as _method_planner_merge_requires_model2_final
from .source_support import _planner_ref_resolves_final as _method_planner_ref_resolves_final
from .source_support import _planner_source_records as _method_planner_source_records
from .source_support import _planner_source_spec_is_usable_final as _method_planner_source_spec_is_usable_final
from .source_support import _prompt_local_source_dialog_retained as _method_prompt_local_source_dialog
from .source_support import _scan_local_model_choices_retained as _method_scan_local_model_choices
from pathlib import Path
from plan import normalize_plan
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import simpledialog
from tkinter import ttk
from typing import Any
from typing import Dict
from typing import List
import copy
import os
import re
import tkinter as tk


class SourcesUiMixin:
    """Sources ui behavior for the Planner."""

    _planner_source_records = _method_planner_source_records


    _planner_build_source_model_row_final = _method_planner_build_source_model_row_final


    _planner_source_spec_is_usable_final = _method_planner_source_spec_is_usable_final


    _collect_available_models = _method_collect_available_models


    def _candidate_local_model_roots(self) -> List[Path]:
        roots: List[Path] = []
        model_dir = self._normalize_user_path(self.entries.get("model_dir").get().strip() if self.entries.get("model_dir") else "")
        workpath = self._normalize_user_path(self.entries.get("workpath").get().strip() if self.entries.get("workpath") else "")
 
        if model_dir:
            roots.append(Path(model_dir))
        elif workpath:
            wp = Path(workpath)
            preferred = [wp / "tmp" / "models", wp / "models"]
            existing = [p for p in preferred if p.exists() and p.is_dir()]
            if existing:
                roots.extend(existing)
            else:
                roots.append(preferred[0])
 
        unique: List[Path] = []
        seen = set()
        for root in roots:
            root = Path(self._normalize_user_path(root))
            try:
                key = os.path.normcase(str(root.resolve()))
            except Exception:
                key = os.path.normcase(str(root))
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique


    _scan_local_model_choices = _method_scan_local_model_choices


    _build_local_selection_row = _method_build_local_selection_row


    def _browse_local_path(self, var: tk.StringVar):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.safetensors *.ckpt *.pt *.bin"), ("All", "*.*")])
        path = self._normalize_user_path(path)
        if path:
            var.set(path)


    def _render_local_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Local Model")
        self._build_local_selection_row(frame, entry)
        self._build_entry_row(frame, "Model Name", entry, "model_name")

        path_row = Frame(frame)
        path_row.pack(fill="x", pady=3)
        Label(path_row, text="Local Path", width=18, anchor="w").pack(side="left")
        path_var = tk.StringVar(value=str(entry.get("local_path", "")))
        path_entry = Entry(path_row, textvariable=path_var, state="readonly")
        path_entry.pack(side="left", fill="x", expand=True, padx=4)

        def sync_path(*_args):
            entry["local_path"] = path_var.get()
            if path_var.get().strip() and not str(entry.get("model_name") or "").strip():
                entry["model_name"] = Path(path_var.get()).stem
            self._after_entry_change()

        path_var.trace_add("write", sync_path)
        ttk.Button(path_row, text="📂", width=3, command=lambda: self._browse_local_path(path_var)).pack(side="left")

        self._build_combo_row(frame, "Type", entry, "model_type", LOCAL_TYPES)
        info = Frame(frame)
        info.pack(fill="x")
        Label(info, text="Local Selection lists models under Model Dir or Workspace Path/tmp/models (fallback: Workspace Path).", fg="#666666", wraplength=620, justify="left").pack(anchor="w", padx=4, pady=(4, 0))
        Label(info, text="Use the folder icon to pick a model outside those locations.", fg="#666666", wraplength=620, justify="left").pack(anchor="w", padx=4, pady=(2, 0))


    def _compute_aliases(self):
        aliases = []
        for idx, entry in enumerate(self.plan_data.get("entries", [])):
            produces = []
            consumes = []
            for _slot, alias, _kind, _spec in self._iter_embedded_sources(entry):
                produces.append(alias)
            produces.extend(self._entry_produced_aliases(entry))
            consumes.extend(self._entry_consumed_aliases(entry))
            aliases.append((idx, [x for x in produces if x], [x for x in consumes if x]))
        return aliases


    def _slot_sources_dict(self, entry):
        data = entry.get("_slot_sources")
        if not isinstance(data, dict):
            data = {}
            entry["_slot_sources"] = data
        return data


    def _get_entry_slot_source(self, entry, slot_key: str):
        data = entry.get("_slot_sources")
        if isinstance(data, dict):
            value = data.get(slot_key)
            if isinstance(value, dict):
                return value
        return None


    def _set_entry_slot_source(self, entry, slot_key: str, spec):
        data = self._slot_sources_dict(entry)
        if spec:
            data[slot_key] = copy.deepcopy(spec)
        else:
            data.pop(slot_key, None)
        if not data:
            entry.pop("_slot_sources", None)


    def _source_alias_from_spec(self, spec, fallback: str = "") -> str:
        if not isinstance(spec, dict):
            return str(fallback or "").strip()
        alias = str(spec.get("alias") or "").strip()
        if alias:
            return alias
        if str(spec.get("mode") or "") == "local":
            local_path = str(spec.get("local_path") or "").strip()
            if local_path:
                return Path(local_path).stem
        return str(fallback or "").strip()


    def _normalize_source_kind(self, raw_kind: str, kind_options):
        options = [str(x) for x in (kind_options or []) if str(x).strip()]
        if not options:
            return str(raw_kind or "").strip()
        raw = str(raw_kind or "").strip().lower()
        for option in options:
            if raw == option.lower():
                return option
        alias_map = {
            "lora": "LoRA",
            "lycoris": "LyCORIS",
            "checkpoint": "Checkpoint",
        }
        normalized = alias_map.get(raw)
        if normalized in options:
            return normalized
        return options[0]


    def _guess_alias_from_link(self, link: str, default_value: str = "") -> str:
        text = str(link or "").strip()
        if not text:
            return str(default_value or "").strip()
        tail = re.split(r"[?#]", text)[0].rstrip("/")
        name = Path(tail).name
        stem = Path(name).stem
        return stem or str(default_value or "").strip()


    def _format_source_spec_text(self, spec) -> str:
        if not isinstance(spec, dict):
            return ""
        mode = str(spec.get("mode") or "").strip().title()
        kind = str(spec.get("kind") or "").strip() or "Model"
        alias = self._source_alias_from_spec(spec)
        if str(spec.get("mode") or "") == "download":
            tail = str(spec.get("link") or "").strip()
            tail = tail[:72] + ("…" if len(tail) > 72 else "")
            return f"Embedded source: {mode} {kind} → {alias}  ({tail})"
        if str(spec.get("mode") or "") == "local":
            local_path = str(spec.get("local_path") or "").strip()
            local_name = Path(local_path).name if local_path else alias
            return f"Embedded source: {mode} {kind} → {alias}  ({local_name})"
        return f"Embedded source: {kind} → {alias}"


    def _ask_source_kind(self, title: str, kind_options, default_kind: str = ""):
        options = [str(x) for x in (kind_options or []) if str(x).strip()]
        if not options:
            return ""
        if len(options) == 1:
            return options[0]
        initial = self._normalize_source_kind(default_kind or options[0], options)
        answer = simpledialog.askstring(title, f"Type ({', '.join(options)}):", initialvalue=initial, parent=self.root)
        if answer is None:
            return None
        answer = answer.strip()
        if not answer:
            return initial
        return self._normalize_source_kind(answer, options)


    def _prompt_download_source_dialog(self, label: str, kind_options, current_alias: str = "", current_source=None):
        source = current_source if isinstance(current_source, dict) else {}
        kind = self._ask_source_kind(f"{label} Download", kind_options, str(source.get("kind") or ""))
        if not kind:
            return None
        initial_link = str(source.get("link") or "").strip()
        link = simpledialog.askstring(f"{label} Download", "Model URL:", initialvalue=initial_link, parent=self.root)
        if link is None:
            return None
        link = link.strip()
        if not link:
            return None
        alias_default = str(source.get("alias") or current_alias or self._guess_alias_from_link(link, kind)).strip()
        alias = simpledialog.askstring(f"{label} Download", "Alias / model name:", initialvalue=alias_default, parent=self.root)
        if alias is None:
            return None
        alias = alias.strip() or alias_default
        if not alias:
            return None
        return {
            "mode": "download",
            "kind": kind,
            "alias": alias,
            "link": link,
        }


    _prompt_local_source_dialog = _method_prompt_local_source_dialog


    def _restore_embedded_sources_into_entry(self, entry, raw_entry):
        if not isinstance(entry, dict) or not isinstance(raw_entry, dict):
            return entry
        raw_slot_sources = raw_entry.get("_slot_sources")
        if isinstance(raw_slot_sources, dict):
            cleaned_slot_sources = {}
            for key, spec in raw_slot_sources.items():
                if isinstance(spec, dict):
                    cleaned_slot_sources[str(key)] = copy.deepcopy(spec)
            if cleaned_slot_sources:
                entry["_slot_sources"] = cleaned_slot_sources
        raw_loras = raw_entry.get("loras") or []
        loras = entry.get("loras") or []
        if isinstance(raw_loras, list) and isinstance(loras, list):
            for idx, raw_lora in enumerate(raw_loras):
                if idx >= len(loras) or not isinstance(raw_lora, dict) or not isinstance(loras[idx], dict):
                    continue
                spec = raw_lora.get("_source")
                if isinstance(spec, dict):
                    loras[idx]["_source"] = copy.deepcopy(spec)
        return entry


    def _normalize_entry_preserving_embedded_sources(self, raw_entry):
        entry = normalize_plan({'entries': [copy.deepcopy(raw_entry)]})['entries'][0]
        entry = self._restore_embedded_sources_into_entry(entry, raw_entry)
        self._absorb_mode_signature_from_entry(entry)
        return entry


    def _normalize_plan_preserving_embedded_sources(self, raw_plan):
        plan = normalize_plan(copy.deepcopy(raw_plan))
        raw_entries = []
        if isinstance(raw_plan, dict):
            raw_entries = raw_plan.get('entries') or []
        entries = plan.get('entries') or []
        if isinstance(raw_entries, list) and isinstance(entries, list):
            for idx, raw_entry in enumerate(raw_entries):
                if idx >= len(entries):
                    break
                self._restore_embedded_sources_into_entry(entries[idx], raw_entry)
        return self._normalize_loaded_plan_entries(plan)


    _build_source_backed_ref_row = _method_build_source_backed_ref_row


    _iter_embedded_sources = _method_iter_embedded_sources


    _entry_consumed_aliases = _method_entry_consumed_aliases


    _entry_produced_aliases = _method_entry_produced_aliases


    _make_effective_source_entry = _method_make_effective_source_entry


    def _gather_selected_source_specs_by_alias(self, indices):
        mapping = {}
        for idx in indices:
            try:
                entry = self.plan_data["entries"][idx]
            except Exception:
                continue
            for _slot, alias, _kind, spec in self._iter_embedded_sources(entry):
                if alias and isinstance(spec, dict) and alias not in mapping:
                    mapping[alias] = copy.deepcopy(spec)
        return mapping


    _planner_entry_can_produce_final = _method_planner_entry_can_produce_final


    _planner_bucket_has_ref_final = _method_planner_bucket_has_ref_final


    _planner_ref_resolves_final = _method_planner_ref_resolves_final


    _planner_merge_requires_model2_final = _method_planner_merge_requires_model2_final
