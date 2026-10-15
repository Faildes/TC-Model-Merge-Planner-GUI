"""Project ui controls and callbacks."""
from __future__ import annotations

from . import dialogs as filedialog
from .document_support import _ensure_plan_path_project_json as _method_ensure_plan_path
from .document_support import _export_project_json_schema as _method_export_project_json_schema
from .document_support import _open_plan_dialog_project_json_macos_safe as _method_open_plan_dialog_project_json
from .document_support import _planner_restore_project_payload_advanced as _method_planner_restore_project_payload
from .document_support import _restore_session_state_universal_json as _method_restore_session_state
from .document_support import _suggest_export_txt_path_project_json as _method_suggest_export_txt_path
from .document_support import _tccm_atomic_autosave as _method_save_plan_to_file
from .document_support import _tccm_atomic_browse_plan_file as _method_browse_plan_file
from .document_support import _tccm_atomic_new_plan as _method_new_plan
from .document_support import _tccm_atomic_runtime_payload as _method_tccm_current_runtime_payload
from .document_support import _tccm_project_json_schema as _method_tccm_project_json_schema
from .document_support import _tccm_v2_apply_runtime as _method_tccm_apply_runtime_payload
from .document_support import _tccm_v2_load as _method_load_plan_from_path
from .document_support import _tccm_v2_project_payload as _method_planner_project_json_payload
from .document_support import _tccm_v2_save_button as _method_save_plan_text_button
from .document_support import _tccm_v2_save_project as _method_planner_save_project_json
from .document_support import _write_temp_plan_text_project_json as _method_write_temp_plan_text
from .tools_support import _copy_current_line_json as _method_copy_current_line_json
from .tools_support import _copy_effective_json as _method_copy_effective_json
from .tools_support import _copy_full_json as _method_copy_full_json
from .tools_support import _export_effective_json as _method_export_effective_json
from .tools_support import _export_full_json as _method_export_full_json
from .tools_support import _export_model_provenance_json as _method_export_model_provenance_json
from .tools_support import _load_preset_json_universal_json as _method_load_preset_json
from .tools_support import _planner_apply_loaded_meta_id_aware as _method_planner_apply_loaded_meta
from .tools_support import _planner_effective_json_payload as _method_planner_effective_json_payload
from .tools_support import _planner_export_json_common as _method_planner_export_json_common
from .tools_support import _planner_meta_payload_id_aware as _method_planner_meta_payload
from .tools_support import _save_preset_json_universal_json as _method_save_preset_json
from .tools_support import _show_model_provenance as _method_show_model_provenance
from .tools_support import _tccm_ensure_extension_state as _method_tccm_ensure_extension_state
from .tools_support import _tccm_provenance_payload as _method_tccm_provenance_payload
from pathlib import Path
from plan import create_plan
from plan import export_plan_records_txt
from plan import load_plan_records
from plan import normalize_plan
from tkinter import messagebox
import json
import os
import tkinter as tk


class ProjectUiMixin:
    """Project ui behavior for the Planner."""

    _planner_effective_json_payload = _method_planner_effective_json_payload


    _planner_restore_project_payload = _method_planner_restore_project_payload


    _planner_save_project_json = _method_planner_save_project_json


    _planner_export_json_common = _method_planner_export_json_common


    _export_full_json = _method_export_full_json


    _export_effective_json = _method_export_effective_json


    _copy_full_json = _method_copy_full_json


    _copy_effective_json = _method_copy_effective_json


    _copy_current_line_json = _method_copy_current_line_json


    _open_plan_dialog_project_json = _method_open_plan_dialog_project_json


    _tccm_current_runtime_payload = _method_tccm_current_runtime_payload


    _tccm_apply_runtime_payload = _method_tccm_apply_runtime_payload


    _tccm_provenance_payload = _method_tccm_provenance_payload


    _show_model_provenance = _method_show_model_provenance


    _export_model_provenance_json = _method_export_model_provenance_json


    _tccm_project_json_schema = _method_tccm_project_json_schema


    _export_project_json_schema = _method_export_project_json_schema


    def _normalize_user_path(self, value: str | os.PathLike | None) -> str:
        raw = os.fspath(value) if value is not None else ""
        raw = str(raw).strip()
        if not raw:
            return ""
        try:
            raw = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            pass
        try:
            raw = os.path.normpath(raw)
        except Exception:
            pass
        return raw


    def _paths_equal(self, left: str | os.PathLike | None, right: str | os.PathLike | None) -> bool:
        l = self._normalize_user_path(left)
        r = self._normalize_user_path(right)
        if not l or not r:
            return False
        try:
            return os.path.normcase(l) == os.path.normcase(r)
        except Exception:
            return l == r


    def _path_display_name(self, value: str | os.PathLike | None) -> str:
        normalized = self._normalize_user_path(value)
        return Path(normalized).name if normalized else ""


    _ensure_plan_path = _method_ensure_plan_path


    _write_temp_plan_text = _method_write_temp_plan_text


    _tccm_ensure_extension_state = _method_tccm_ensure_extension_state


    _restore_session_state = _method_restore_session_state


    _browse_plan_file = _method_browse_plan_file


    _load_plan_from_path = _method_load_plan_from_path


    _new_plan = _method_new_plan


    _save_plan_to_file = _method_save_plan_to_file


    _save_plan_text_button = _method_save_plan_text_button


    _suggest_export_txt_path = _method_suggest_export_txt_path


    def _export_as_txt(self):
        temp_plan_path = None
        try:
            temp_plan_path = self._write_temp_plan_text()
            initial = self.config.get("saveas") or self._suggest_export_txt_path()
            path = filedialog.asksaveasfilename(
                title="Export as txt",
                defaultextension=".txt",
                initialfile=Path(initial).name,
                initialdir=str(Path(initial).parent),
                filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            )
            if not path:
                return
            params = {key: widget.get() for key, widget in self.entries.items()}
            create_plan(
                filepath=temp_plan_path,
                workpath=params.get("workpath", ""),
                saveas=path,
                title=params.get("title", "merge_plan"),
                vae=params.get("vae", ""),
                vae_name=params.get("vae_name", "VAE"),
                bake_vae=bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get()),
                CivitAPI=params.get("CivitAPI", ""),
                HuggingAPI=params.get("HuggingAPI", ""),
                UR=params.get("UR", ""),
                model_dir=params.get("model_dir", ""),
                vae_dir=params.get("vae_dir", ""),
            )
            self.config["saveas"] = path
            self._save_current_state_to_config()
            self.status_label.config(text=f"Exported txt: {os.path.basename(path)}")
            messagebox.showinfo("Export", f"TXT exported successfully.\n\n{path}")
        except Exception as e:
            self._show_detailed_error("Export TXT Error", e, context="Operation: export_as_txt")
        finally:
            if temp_plan_path and os.path.exists(temp_plan_path):
                try:
                    os.remove(temp_plan_path)
                except Exception:
                    pass


    def _base_load_plan_from_path(self):
        path = self._normalize_user_path(self.entries["filepath"].get().strip())
        if not path:
            messagebox.showwarning("Plan Path", "Plan Text Path is empty.")
            return
        try:
            self.plan_data = normalize_plan(load_plan_records(path))
            self.current_index = 0
            self._refresh_line_selector()
            self._render_current_line()
            self.status_label.config(text=f"Loaded plan: {self._path_display_name(path)}")
        except Exception as e:
            self._show_detailed_error("Load Error", e)


    def _base_new_plan(self):
        current_filepath = self._normalize_user_path(self.entries["filepath"].get().strip()) if "filepath" in self.entries else ""
        current_title = self.entries["title"].get().strip() if "title" in self.entries else ""
        initial_name = Path(current_filepath).name if current_filepath else f"{(current_title or 'merge_plan')}.txt"
        initial_dir = str(Path(current_filepath).expanduser().parent) if current_filepath else os.getcwd()

        try:
            path = filedialog.asksaveasfilename(
                title="Create new Plan file",
                defaultextension=".txt",
                initialfile=initial_name,
                initialdir=initial_dir,
                filetypes=[("Plan Text", "*.txt"), ("All files", "*.*")],
            )
            path = self._normalize_user_path(path)
            if not path:
                return
            if not path.lower().endswith(".txt"):
                path += ".txt"

            self.plan_data = self._planner_default_visible_plan()
            self.current_index = 0
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            export_plan_records_txt(path, self.plan_data)

            self.entries["filepath"].delete(0, tk.END)
            self.entries["filepath"].insert(0, path)
            self.entries["title"].delete(0, tk.END)
            self.entries["title"].insert(0, Path(path).stem)
            self.notebook_path_var.set("")
            self.executed_notebook_path_var.set("")

            self._refresh_line_selector()
            self._render_current_line()
            self._save_current_state_to_config()
            self.status_label.config(text=f"Created new plan: {self._path_display_name(path)}")
        except Exception as e:
            self._show_detailed_error("New Plan Error", e, context="Operation: new_plan")


    def _planner_refresh_plan_meta(self):
        if hasattr(self, 'plan_summary_var'):
            count = len(self.plan_data.get('entries', []))
            sel = len(self._planner_get_selected_indices()) if count else 0
            self.plan_summary_var.set(f"{count} lines  |  {sel} selected")
        if hasattr(self, 'plan_deps_var'):
            self.plan_deps_var.set(self._planner_dependency_summary())
        if hasattr(self, 'plan_unused_var'):
            self.plan_unused_var.set(self._planner_unused_summary())


    _save_preset_json = _method_save_preset_json


    _load_preset_json = _method_load_preset_json


    def _planner_meta_sidecar_path(self, plan_path: str | None = None) -> Path | None:
        raw = plan_path
        if not raw:
            try:
                raw = self._normalize_user_path(self.entries['filepath'].get().strip())
            except Exception:
                raw = ''
        raw = str(raw or '').strip()
        if not raw:
            return None
        return Path(raw + '.planner_meta.json')


    _planner_meta_payload = _method_planner_meta_payload


    def _planner_save_meta_to_disk(self, plan_path: str | None = None):
        sidecar = self._planner_meta_sidecar_path(plan_path)
        if sidecar is None:
            return
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(self._planner_meta_payload(), ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass


    _planner_apply_loaded_meta = _method_planner_apply_loaded_meta


    def _planner_load_meta_from_disk(self, plan_path: str | None = None):
        self._planner_apply_entry_defaults()
        sidecar = self._planner_meta_sidecar_path(plan_path)
        if sidecar is None or not sidecar.exists():
            return
        try:
            payload = json.loads(sidecar.read_text(encoding='utf-8'))
        except Exception:
            return
        self._planner_apply_loaded_meta(payload)


    _planner_project_json_payload = _method_planner_project_json_payload
