"""Document support responsibilities."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import TCCM_PROJECT_EXTENSION
from .constants import TCCM_PROJECT_FORMAT
from pathlib import Path
from plan import export_plan_records_txt
from plan import load_plan_records
from plan import make_entry
from plan import normalize_plan
from tkinter import Frame
from tkinter import Label
from tkinter import Text
from tkinter import messagebox
from tkinter import ttk
from typing import Any
from typing import Dict
import copy
import difflib
import json
import os
import re
import tempfile
import time
import tkinter as tk


def _save_current_state_to_config(self):
    from .settings import save_config_to_disk

    for key, widget in self.entries.items():
        self.config[key] = widget.get()
    self.config["hf_repo_id"] = self.entries["UR"].get()
    self.config["last_notebook_path"] = self.notebook_path_var.get()
    self.config["last_executed_notebook_path"] = self.executed_notebook_path_var.get()
    self.config["base_model"] = self.base_model_var.get()
    self.config["active_view"] = getattr(self, "active_view", "split")
    self.config["theme_mode"] = getattr(self, "theme_mode", "dark")
    self.config["ignore_install_deps"] = bool(getattr(self, "ignore_install_deps_var", tk.BooleanVar(value=False)).get())
    self.config["use_online"] = bool(getattr(self, "use_online_var", tk.BooleanVar(value=False)).get())
    self.config["upload_after_merge"] = bool(getattr(self, "upload_after_merge_var", tk.BooleanVar(value=False)).get())
    self.config["run_t2i"] = bool(getattr(self, "run_t2i_var", tk.BooleanVar(value=False)).get())
    self.config["bake_vae"] = bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get())
    save_config_to_disk(self.config)


def _planner_norm_path_key(self, path: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(str(path or ''))))
    except Exception:
        return os.path.normcase(str(path or ''))


def _save_current_state_to_config_dev(self):
    from .document_support import _save_current_state_to_config as _dev_base_save_current_state

    self._planner_dev_ensure_config()
    if hasattr(self, "dev_mode_var"):
        try: self.config["dev_mode"] = bool(self.dev_mode_var.get())
        except Exception: pass
    return _dev_base_save_current_state(self)


def _planner_is_project_json_path(path: str | os.PathLike | None) -> bool:
    raw = str(path or "").strip().lower()
    return raw.endswith(TCCM_PROJECT_EXTENSION) or raw.endswith(".tccm")


def _planner_project_json_payload(self, *, include_runtime: bool = True) -> Dict[str, Any]:
    from .tools_support import _planner_timestamp

    apply_defaults = getattr(self, "_planner_apply_entry_defaults", None)
    if callable(apply_defaults):
        apply_defaults()
    entries = copy.deepcopy((getattr(self, "plan_data", {}) or {}).get("entries", []) or [])
    payload: Dict[str, Any] = {
        "version": 3,
        "format": TCCM_PROJECT_FORMAT,
        "saved_at": _planner_timestamp(),
        "current_index": int(getattr(self, "current_index", 0) or 0),
        "entries": entries,
    }
    for key in ("final_memo", "history", "meta"):
        if isinstance(getattr(self, "plan_data", None), dict) and key in self.plan_data:
            payload[key] = copy.deepcopy(self.plan_data.get(key))
    if include_runtime:
        cfg = {}
        try:
            for key in getattr(self, "NOTEBOOK_PARAMS_KEYS", []):
                if key in getattr(self, "entries", {}):
                    cfg[key] = self.entries[key].get()
                elif key in getattr(self, "config", {}):
                    cfg[key] = copy.deepcopy(self.config.get(key))
            cfg["base_model"] = self.base_model_var.get() if hasattr(self, "base_model_var") else self.config.get("base_model", "SDXL")
            cfg["bake_vae"] = bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get())
            cfg["ignore_install_deps"] = bool(getattr(self, "ignore_install_deps_var", tk.BooleanVar(value=False)).get())
            cfg["use_online"] = bool(getattr(self, "use_online_var", tk.BooleanVar(value=False)).get())
            cfg["upload_after_merge"] = bool(getattr(self, "upload_after_merge_var", tk.BooleanVar(value=False)).get())
            cfg["run_t2i"] = bool(getattr(self, "run_t2i_var", tk.BooleanVar(value=False)).get())
        except Exception:
            cfg = copy.deepcopy(getattr(self, "config", {}) or {})
        payload["runtime"] = cfg
    return payload


def _planner_restore_project_payload(self, payload, *, source_path: str = ""):
    if isinstance(payload, dict) and "plan_data" in payload:
        payload = payload.get("plan_data")
    if isinstance(payload, list):
        payload = {"version": 3, "format": TCCM_PROJECT_FORMAT, "entries": payload}
    if not isinstance(payload, dict):
        payload = {"entries": []}
    raw_entries = payload.get("entries") or []
    entries = []
    for raw_entry in raw_entries if isinstance(raw_entries, list) else []:
        if not isinstance(raw_entry, dict):
            continue
        try:
            entry = self._normalize_entry_preserving_embedded_sources(copy.deepcopy(raw_entry))
        except Exception:
            entry = copy.deepcopy(raw_entry)
            entry.setdefault("id", make_entry(entry.get("type", "Checkpoint Merge")).get("id"))
        entry["memo"] = str(raw_entry.get("memo") or entry.get("memo") or "")
        entry["_locked"] = bool(raw_entry.get("_locked", entry.get("_locked", False)))
        entry["_disabled"] = bool(raw_entry.get("_disabled", entry.get("_disabled", False)))
        entry["_row_color"] = str(raw_entry.get("_row_color", entry.get("_row_color", "")) or "")
        entries.append(entry)
    self.plan_data = {
        "version": 3,
        "format": str(payload.get("format") or TCCM_PROJECT_FORMAT),
        "entries": entries or [make_entry("Checkpoint Merge")],
    }
    for key in ("final_memo", "history", "meta"):
        if key in payload:
            self.plan_data[key] = copy.deepcopy(payload.get(key))
    apply_defaults = getattr(self, "_planner_apply_entry_defaults", None)
    if callable(apply_defaults):
        apply_defaults()
    try:
        idx = int(payload.get("current_index", 0) or 0)
    except Exception:
        idx = 0
    self.current_index = max(0, min(idx, len(self.plan_data.get("entries", [])) - 1))


def _ensure_plan_path_project_json(self) -> str:
    from .document_support import _planner_force_project_extension_macos_safe as _planner_force_project_extension
    from .tools_support import _planner_safe_title

    path = self._normalize_user_path(self.entries["filepath"].get().strip())
    if not path:
        base_dir = self._normalize_user_path(os.getcwd())
        title = _planner_safe_title(self)
        path = os.path.join(base_dir, f"{title}{TCCM_PROJECT_EXTENSION}")
    path = _planner_force_project_extension(path)
    if path != self.entries["filepath"].get().strip():
        self.entries["filepath"].delete(0, tk.END)
        self.entries["filepath"].insert(0, path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_plan_to_file_project_json(self):
    self._planner_save_project_json(reason="autosave")


def _write_temp_plan_text_project_json(self) -> str:
    fd, temp_path = tempfile.mkstemp(prefix="planner_effective_", suffix=".txt")
    os.close(fd)
    export_plan_records_txt(temp_path, self._planner_effective_plan_data())
    return temp_path


def _get_notebook_paths_project_json(self) -> tuple[str, str]:
    from .tools_support import _planner_safe_title

    plan_path = Path(self._ensure_plan_path())
    title = _planner_safe_title(self) or plan_path.stem.replace(".tccm", "") or "merge_plan"
    return str(plan_path.with_name(f"{title}.ipynb")), str(plan_path.with_name(f"{title}.executed.ipynb"))


def _suggest_export_txt_path_project_json(self) -> str:
    from .tools_support import _planner_safe_title

    plan_path = Path(self._ensure_plan_path())
    title = _planner_safe_title(self) or plan_path.stem.replace(".tccm", "") or "merge_plan"
    return str(plan_path.with_name(f"{title}.txt"))


def _new_plan_project_json(self):
    from .document_support import _planner_force_project_extension_macos_safe as _planner_force_project_extension
    from .tools_support import _planner_write_json_file

    current_filepath = self._normalize_user_path(self.entries["filepath"].get().strip()) if "filepath" in self.entries else ""
    current_title = self.entries["title"].get().strip() if "title" in self.entries else ""
    initial_name = Path(current_filepath).name if current_filepath else f"{(current_title or 'merge_plan')}{TCCM_PROJECT_EXTENSION}"
    if not initial_name.lower().endswith(TCCM_PROJECT_EXTENSION):
        initial_name = Path(initial_name).stem + TCCM_PROJECT_EXTENSION
    initial_dir = str(Path(current_filepath).expanduser().parent) if current_filepath else os.getcwd()
    try:
        path = filedialog.asksaveasfilename(
            title="Create new TCCM Project",
            defaultextension=TCCM_PROJECT_EXTENSION,
            initialfile=initial_name,
            initialdir=initial_dir,
            filetypes=[("TCCM Project", "*.tccm.json"), ("All files", "*.*")],
        )
        path = self._normalize_user_path(path)
        if not path:
            return
        path = _planner_force_project_extension(path)
        self.plan_data = self._planner_default_visible_plan()
        self.current_index = 0
        _planner_write_json_file(path, _planner_project_json_payload(self))
        self.entries["filepath"].delete(0, tk.END)
        self.entries["filepath"].insert(0, path)
        self.entries["title"].delete(0, tk.END)
        self.entries["title"].insert(0, Path(path).name.removesuffix(TCCM_PROJECT_EXTENSION))
        self.notebook_path_var.set("")
        self.executed_notebook_path_var.set("")
        self._refresh_line_selector()
        self._render_current_line()
        self._save_current_state_to_config()
        self._planner_maybe_create_backup(path, reason="new-project", force=True)
        self.status_label.config(text=f"Created new project: {self._path_display_name(path)}")
    except Exception as e:
        self._show_detailed_error("New Project Error", e, context="Operation: new_project")


def _load_plan_from_path_project_json(self):
    from .tools_support import _planner_read_json_or_none

    path = self._normalize_user_path(self.entries["filepath"].get().strip())
    if not path:
        messagebox.showwarning("Plan Path", "Plan path is empty.")
        return
    try:
        payload = _planner_read_json_or_none(path)
        is_project = _planner_is_project_json_path(path) or (isinstance(payload, dict) and payload.get("format") == TCCM_PROJECT_FORMAT)
        if is_project:
            self._planner_restore_project_payload(payload, source_path=path)
        else:
            raw = normalize_plan(load_plan_records(path))
            collapse = getattr(self, "_collapse_internal_plan_entries", None)
            self.plan_data = collapse(raw) if callable(collapse) else raw
            load_meta = getattr(self, "_planner_load_meta_from_disk", None)
            if callable(load_meta):
                load_meta(path)
            apply_defaults = getattr(self, "_planner_apply_entry_defaults", None)
            if callable(apply_defaults):
                apply_defaults()
            self.current_index = 0
        if hasattr(self, "_history_undo"):
            self._history_undo.clear()
        if hasattr(self, "_history_redo"):
            self._history_redo.clear()
        self._last_history_snapshot = self._planner_plan_snapshot() if hasattr(self, "_planner_plan_snapshot") else None
        self._refresh_line_selector()
        self._render_current_line()
        self.status_label.config(text=f"Loaded plan: {self._path_display_name(path)}")
        self.root.after(500, lambda p=path: self._planner_offer_newer_backup_restore(p))
    except Exception as e:
        self._show_detailed_error("Load Error", e)


def _planner_offer_newer_backup_restore(self, plan_path: str | None = None):
    try:
        if not plan_path or not Path(plan_path).exists():
            return
        root = self._planner_backup_root(plan_path)
        bundles = sorted(root.glob("*.planbundle.json"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
        if not bundles:
            return
        latest = bundles[0]
        if latest.stat().st_mtime <= Path(plan_path).stat().st_mtime + 1.0:
            return
        key = str(latest.resolve())
        if getattr(self, "_last_offered_backup", None) == key:
            return
        self._last_offered_backup = key
        if not messagebox.askyesno("Newer Backup Found", f"A newer backup exists.\n\n{latest.name}\n\nRestore this backup?"):
            return
        payload = json.loads(latest.read_text(encoding="utf-8"))
        raw_plan = payload.get("plan_data") if isinstance(payload, dict) else payload
        if hasattr(self, "_planner_push_history"):
            self._planner_push_history()
        self._planner_restore_project_payload(raw_plan, source_path=str(latest))
        self._planner_save_project_json(plan_path, create_backup=False)
        self._refresh_line_selector()
        self._render_current_line()
        self.status_label.config(text=f"Restored newer backup: {latest.name}")
    except Exception:
        pass


def _create_menu_project_json(self):
    from .lifecycle_support import _create_menu as _tccm_base_create_menu

    _tccm_base_create_menu(self)
    try:
        menu_bar = self.root.nametowidget(self.root.cget("menu"))
    except Exception:
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
    project_menu = tk.Menu(menu_bar, tearoff=0)
    try:
        menu_bar.insert_cascade(0, label="File", menu=project_menu)
    except Exception:
        menu_bar.add_cascade(label="File", menu=project_menu)
    project_menu.add_command(label="New Project (.tccm.json)…", command=self._new_plan)
    project_menu.add_command(label="Load Project / Legacy Plan…", command=self._load_plan_from_path)
    project_menu.add_command(label="Save Project", command=self._save_plan_text_button)
    project_menu.add_separator()
    project_menu.add_command(label="Export Full Project JSON…", command=self._export_full_json)
    project_menu.add_command(label="Export Effective JSON…", command=self._export_effective_json)
    project_menu.add_command(label="Copy Full Project JSON", command=self._copy_full_json)
    project_menu.add_command(label="Copy Effective JSON", command=self._copy_effective_json)
    project_menu.add_command(label="Copy Current Line JSON", command=self._copy_current_line_json)
    project_menu.add_separator()
    project_menu.add_command(label="Backup Manager…", command=self._show_backup_manager)


def _create_menu_project_json_open_dialog(self):
    from .document_support import _create_menu_project_json as _tccm_project_create_menu_prev

    _tccm_project_create_menu_prev(self)
    try:
        menu_bar = self.root.nametowidget(self.root.cget("menu"))
        # The first menu was added by the project patch.  Replace its second
        # command with an actual open-dialog workflow when Tk allows it.
        file_menu_name = menu_bar.entrycget(0, "menu")
        file_menu = self.root.nametowidget(file_menu_name)
        file_menu.entryconfigure(1, command=self._open_plan_dialog_project_json)
    except Exception:
        pass


def _tccm_current_runtime_payload(self) -> dict:
    payload: dict = {}
    try:
        for key in getattr(self, "NOTEBOOK_PARAMS_KEYS", []):
            if key in getattr(self, "entries", {}):
                payload[key] = self.entries[key].get()
            elif key in getattr(self, "config", {}):
                payload[key] = copy.deepcopy(self.config.get(key))
        if hasattr(self, "base_model_var"):
            payload["base_model"] = self.base_model_var.get()
        for key, var_name in (
            ("bake_vae", "bake_vae_var"),
            ("ignore_install_deps", "ignore_install_deps_var"),
            ("use_online", "use_online_var"),
            ("upload_after_merge", "upload_after_merge_var"),
            ("run_t2i", "run_t2i_var"),
        ):
            var = getattr(self, var_name, None)
            if var is not None:
                payload[key] = bool(var.get())
    except Exception:
        payload = copy.deepcopy(getattr(self, "config", {}) or {})
    return payload


def _tccm_apply_runtime_payload(self, runtime: dict):
    if not isinstance(runtime, dict):
        return
    for key, value in runtime.items():
        self.config[key] = value
        try:
            if key in getattr(self, "entries", {}):
                widget = self.entries[key]
                if hasattr(widget, "delete") and hasattr(widget, "insert"):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(value))
        except Exception:
            pass
    try:
        if "base_model" in runtime and hasattr(self, "base_model_var"):
            self.base_model_var.set(str(runtime.get("base_model") or self.base_model_var.get()))
    except Exception:
        pass
    for key, var_name in (
        ("bake_vae", "bake_vae_var"),
        ("ignore_install_deps", "ignore_install_deps_var"),
        ("use_online", "use_online_var"),
        ("upload_after_merge", "upload_after_merge_var"),
        ("run_t2i", "run_t2i_var"),
    ):
        try:
            if key in runtime and hasattr(self, var_name):
                getattr(self, var_name).set(bool(runtime.get(key)))
        except Exception:
            pass
    try:
        self._schedule_config_save()
    except Exception:
        pass


def _tccm_backup_pin_path(path: Path) -> Path:
    return Path(str(path) + ".pin")


def _tccm_backup_is_pinned(path: Path) -> bool:
    return _tccm_backup_pin_path(path).exists()


def _show_backup_diff_manager(self):
    from .tools_support import _tccm_timestamp

    try:
        root_dir = self._planner_backup_root(None)
    except Exception:
        root_dir = Path(os.getcwd()) / ".planner_backups" / "plan"
    win = tk.Toplevel(self.root)
    win.title("Backup Diff / Pin Manager")
    win.geometry("1180x760+130+130")
    outer = Frame(win, padx=8, pady=8)
    outer.pack(fill="both", expand=True)
    Label(outer, text=f"Backup folder: {root_dir}", anchor="w", justify="left").pack(fill="x", pady=(0, 6))
    panes = ttk.PanedWindow(outer, orient="horizontal")
    panes.pack(fill="both", expand=True)
    left = Frame(panes)
    right = Frame(panes)
    panes.add(left, weight=1)
    panes.add(right, weight=3)
    listbox = tk.Listbox(left, exportselection=False, activestyle="none")
    listbox.pack(fill="both", expand=True)
    toolbar = Frame(left)
    toolbar.pack(fill="x", pady=(6, 0))
    notebook = ttk.Notebook(right)
    notebook.pack(fill="both", expand=True)
    detail = Text(notebook, wrap="none", font=("Consolas", 10))
    diff = Text(notebook, wrap="none", font=("Consolas", 10))
    notebook.add(detail, text="Detail")
    notebook.add(diff, text="Diff vs Current")
    bundles: list[Path] = []

    def bundle_label(path: Path) -> str:
        pin = "📌 " if _tccm_backup_is_pinned(path) else ""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            saved_at = str(payload.get("saved_at") or "")
            reason = str(payload.get("reason") or "")
            plan = payload.get("plan_data") if isinstance(payload, dict) else {}
            count = len((plan or {}).get("entries") or []) if isinstance(plan, dict) else 0
            return f"{pin}{path.name} | {saved_at} | {reason} | {count} line(s)"
        except Exception:
            return pin + path.name

    def current_bundle():
        sel = listbox.curselection()
        if not sel:
            return None
        idx = int(sel[-1])
        return bundles[idx] if 0 <= idx < len(bundles) else None

    def refresh():
        nonlocal bundles
        listbox.delete(0, "end")
        if root_dir.exists():
            bundles = sorted(root_dir.glob("*.planbundle.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        else:
            bundles = []
        for p in bundles:
            listbox.insert("end", bundle_label(p))
        if bundles:
            listbox.selection_set(0)
        show_selected()

    def selected_payload(path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw

    def show_selected(_event=None):
        detail.configure(state="normal"); diff.configure(state="normal")
        detail.delete("1.0", "end"); diff.delete("1.0", "end")
        path = current_bundle()
        if not path:
            detail.insert("1.0", "No backup selected.")
            detail.configure(state="disabled"); diff.configure(state="disabled")
            return
        try:
            payload = selected_payload(path)
            detail.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))
            backup_plan = payload.get("plan_data") if isinstance(payload, dict) else payload
            current_plan = self._planner_project_json_payload()
            old = json.dumps(backup_plan, ensure_ascii=False, indent=2).splitlines()
            new = json.dumps(current_plan, ensure_ascii=False, indent=2).splitlines()
            d = difflib.unified_diff(old, new, fromfile=path.name, tofile="current project", lineterm="")
            diff.insert("1.0", "\n".join(d) or "No diff.")
        except Exception as exc:
            detail.insert("1.0", f"Failed to read backup: {type(exc).__name__}: {exc}")
        detail.configure(state="disabled"); diff.configure(state="disabled")

    def toggle_pin():
        path = current_bundle()
        if not path:
            return
        pin_path = _tccm_backup_pin_path(path)
        try:
            if pin_path.exists():
                pin_path.unlink()
                self.status_label.config(text=f"Unpinned backup: {path.name}")
            else:
                pin_path.write_text(json.dumps({"pinned_at": _tccm_timestamp(), "backup": path.name}, ensure_ascii=False, indent=2), encoding="utf-8")
                self.status_label.config(text=f"Pinned backup: {path.name}")
        except Exception as exc:
            self._show_detailed_error("Pin Backup Error", exc)
        refresh()

    def restore_selected():
        path = current_bundle()
        if not path:
            return
        if not messagebox.askyesno("Restore Backup", f"Restore backup '{path.name}' to the current project?"):
            return
        try:
            payload = selected_payload(path)
            raw_plan = payload.get("plan_data") if isinstance(payload, dict) else payload
            if hasattr(self, "_planner_push_history"):
                self._planner_push_history()
            self._planner_restore_project_payload(raw_plan, source_path=str(path))
            self._refresh_line_selector()
            self._render_current_line()
            self._planner_save_project_json(reason="backup-restore")
            self.status_label.config(text=f"Restored backup: {path.name}")
            show_selected()
        except Exception as exc:
            self._show_detailed_error("Restore Backup Error", exc)

    listbox.bind("<<ListboxSelect>>", show_selected)
    ttk.Button(toolbar, text="Refresh", command=refresh).pack(side="left")
    ttk.Button(toolbar, text="Pin / Unpin", command=toggle_pin).pack(side="left", padx=(4, 0))
    ttk.Button(toolbar, text="Restore", command=restore_selected).pack(side="left", padx=(4, 0))
    ttk.Button(toolbar, text="Close", command=win.destroy).pack(side="right")
    refresh()


def _tccm_project_json_schema(self) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://faildes.example/schemas/tccm-planner-project.schema.json",
        "title": "TCCM Planner Project",
        "type": "object",
        "required": ["version", "format", "entries"],
        "properties": {
            "version": {"type": "integer", "minimum": 3},
            "format": {"const": TCCM_PROJECT_FORMAT},
            "saved_at": {"type": "string"},
            "current_index": {"type": "integer", "minimum": 0},
            "runtime": {"type": "object", "additionalProperties": True},
            "entries": {"type": "array", "items": {"$ref": "#/$defs/entry"}},
            "variants": {"type": "array", "items": {"$ref": "#/$defs/variant"}},
            "run_profiles": {"type": "array", "items": {"$ref": "#/$defs/run_profile"}},
            "final_memo": {"type": "string"},
            "history": {"type": "array"},
            "meta": {"type": "object"},
        },
        "$defs": {
            "ratio": {
                "type": "object",
                "properties": {
                    "mode": {"enum": ["Single", "Block weight", "Elemental", "Randomize"]},
                    "value": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "lora_ref": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "ratio": {"$ref": "#/$defs/ratio"}},
                "additionalProperties": True,
            },
            "entry": {
                "type": "object",
                "required": ["id", "type"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "_disabled": {"type": "boolean"},
                    "_locked": {"type": "boolean"},
                    "_row_color": {"type": "string"},
                    "memo": {"type": "string"},
                    "output_name": {"type": "string"},
                    "model_name": {"type": "string"},
                    "merge_mode": {"type": "string"},
                    "model0": {"type": "string"},
                    "model1": {"type": "string"},
                    "model2": {"type": "string"},
                    "checkpoint": {"type": "string"},
                    "loras": {"type": "array", "items": {"$ref": "#/$defs/lora_ref"}},
                    "alpha": {"$ref": "#/$defs/ratio"},
                    "beta": {"$ref": "#/$defs/ratio"},
                },
                "additionalProperties": True,
            },
            "variant": {
                "type": "object",
                "required": ["name", "entries"],
                "properties": {"name": {"type": "string"}, "saved_at": {"type": "string"}, "current_index": {"type": "integer"}, "entries": {"type": "array", "items": {"$ref": "#/$defs/entry"}}},
                "additionalProperties": True,
            },
            "run_profile": {
                "type": "object",
                "required": ["name", "runtime"],
                "properties": {"name": {"type": "string"}, "saved_at": {"type": "string"}, "runtime": {"type": "object", "additionalProperties": True}},
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }


def _export_project_json_schema(self):
    path = filedialog.asksaveasfilename(title="Export TCCM JSON Schema", defaultextension=".schema.json", initialfile="tccm-planner-project.schema.json", filetypes=[("JSON Schema", "*.schema.json"), ("JSON", "*.json"), ("All files", "*.*")])
    path = self._normalize_user_path(path)
    if not path:
        return
    Path(path).write_text(json.dumps(self._tccm_project_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    self.status_label.config(text=f"Exported JSON schema: {self._path_display_name(path)}")


def _planner_project_json_payload_advanced(self, *, include_runtime: bool = True):
    from .document_support import _planner_project_json_payload as _tccm_prev_project_payload
    from .tools_support import _tccm_timestamp

    self._tccm_ensure_extension_state()
    if callable(_tccm_prev_project_payload):
        payload = _tccm_prev_project_payload(self, include_runtime=include_runtime)
    else:
        payload = {"version": 3, "format": TCCM_PROJECT_FORMAT, "saved_at": _tccm_timestamp(), "current_index": int(getattr(self, "current_index", 0) or 0), "entries": copy.deepcopy((getattr(self, "plan_data", {}) or {}).get("entries", []) or [])}
        if include_runtime:
            payload["runtime"] = self._tccm_current_runtime_payload()
    payload["schema_version"] = 3
    payload["variants"] = copy.deepcopy(getattr(self, "plan_variants", []) or [])
    payload["run_profiles"] = copy.deepcopy(getattr(self, "run_profiles", []) or [])
    payload["current_variant"] = str(getattr(self, "current_variant_name", "") or "")
    return payload


def _planner_restore_project_payload_advanced(self, payload, *, source_path: str = ""):
    from .document_support import _planner_restore_project_payload as _tccm_prev_restore_project

    raw = payload
    if isinstance(raw, dict) and "plan_data" in raw:
        raw = raw.get("plan_data")
    if callable(_tccm_prev_restore_project):
        _tccm_prev_restore_project(self, payload, source_path=source_path)
    else:
        self.plan_data = normalize_plan(raw if isinstance(raw, dict) else {"entries": []})
    self._tccm_ensure_extension_state()
    if isinstance(raw, dict):
        self.plan_variants = copy.deepcopy(raw.get("variants") or []) if isinstance(raw.get("variants"), list) else []
        self.run_profiles = copy.deepcopy(raw.get("run_profiles") or []) if isinstance(raw.get("run_profiles"), list) else []
        self.current_variant_name = str(raw.get("current_variant") or "")
        runtime = raw.get("runtime")
        if isinstance(runtime, dict):
            self._tccm_apply_runtime_payload(runtime)
    else:
        self.plan_variants = []
        self.run_profiles = []
        self.current_variant_name = ""


def _tccm_is_project_path(path: str) -> bool:
    raw = str(path or "").lower()
    return raw.endswith(TCCM_PROJECT_EXTENSION) or raw.endswith(".tccm") or (not raw.endswith(".txt"))


def _planner_save_project_json_advanced(self, path: str | None = None, *, create_backup: bool = True, reason: str = "autosave") -> str:
    from .document_support import _planner_force_project_extension_macos_safe as _tccm_force_project_extension

    path = self._normalize_user_path(path or self._ensure_plan_path())
    path = _tccm_force_project_extension(path)
    if _tccm_is_project_path(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self._planner_project_json_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        export_plan_records_txt(path, self._planner_effective_plan_data())
        save_meta = getattr(self, "_planner_save_meta_to_disk", None)
        if callable(save_meta):
            save_meta(path)
    if create_backup:
        maybe_backup = getattr(self, "_planner_maybe_create_backup", None)
        if callable(maybe_backup):
            maybe_backup(path, reason=reason)
    try:
        self._schedule_config_save()
    except Exception:
        pass
    return path


def _planner_maybe_create_backup_advanced(self, plan_path: str | None = None, *, reason: str = "autosave", force: bool = False):
    from .tools_support import _tccm_timestamp

    try:
        now = time.time()
        if not force and now - float(getattr(self, "_last_backup_time", 0.0) or 0.0) < 15.0:
            # Avoid deep-copying / JSON-serializing a long plan for backup
            # calls that cannot create a backup yet.
            return
        source_plan_path = ""
        try:
            source_plan_path = str(plan_path or (self.entries.get("filepath").get() if self.entries.get("filepath") else ""))
        except Exception:
            source_plan_path = str(plan_path or "")
        project_payload = self._planner_project_json_payload()
        payload = {
            "bundle_version": 3,
            "version": 3,
            "saved_at": _tccm_timestamp(),
            "reason": str(reason or "autosave"),
            "source_plan_path": source_plan_path,
            "current_index": int(getattr(self, "current_index", 0) or 0),
            "config": copy.deepcopy(getattr(self, "config", {}) or {}),
            "plan_data": project_payload,
            "effective_line_count": len((self._planner_effective_json_payload().get("entries") or [])) if hasattr(self, "_planner_effective_json_payload") else 0,
            "health": {k: v for k, v in self._tccm_analyze_plan().items() if k in ("entry_count", "enabled_count", "disabled_count", "errors", "warnings")},
        }
        snapshot = json.dumps(payload["plan_data"], ensure_ascii=False, sort_keys=True)
        if not force:
            if snapshot == getattr(self, "_last_backup_snapshot", None):
                return
            if now - float(getattr(self, "_last_backup_time", 0.0) or 0.0) < 15.0:
                return
        root = self._planner_backup_root(plan_path)
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(reason or "autosave")).strip("_") or "autosave"
        out = root / f"{stamp}_{safe_reason}.planbundle.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        bundles = sorted(root.glob("*.planbundle.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        keep = max(3, int(getattr(self, "backup_keep_generations", 20) or 20))
        removable = [p for p in bundles if not _tccm_backup_is_pinned(p)]
        for stale in removable[keep:]:
            try:
                stale.unlink()
                pin = _tccm_backup_pin_path(stale)
                if pin.exists():
                    pin.unlink()
            except Exception:
                pass
        self._last_backup_snapshot = snapshot
        self._last_backup_time = now
    except Exception:
        pass


def _new_plan_advanced(self):
    from .document_support import _new_plan_project_json as _tccm_prev_new_plan

    self.plan_variants = []
    self.run_profiles = []
    self.current_variant_name = ""
    if callable(_tccm_prev_new_plan):
        return _tccm_prev_new_plan(self)


def _load_plan_from_path_macos_safe(self):
    from .tools_support import _tccm_open_plan_dialog_no_native_filter
    from .document_support import _load_plan_from_path_project_json as _tccm_previous_load_plan_from_path

    path = ""
    try:
        path = self._normalize_user_path(self.entries["filepath"].get().strip())
    except Exception:
        path = ""
    if not path:
        path = _tccm_open_plan_dialog_no_native_filter(self)
        if not path:
            return
        self.entries["filepath"].delete(0, tk.END)
        self.entries["filepath"].insert(0, path)
        self._schedule_config_save()
    if callable(_tccm_previous_load_plan_from_path):
        return _tccm_previous_load_plan_from_path(self)


def _open_plan_dialog_project_json_macos_safe(self):
    from .tools_support import _tccm_open_plan_dialog_no_native_filter

    path = _tccm_open_plan_dialog_no_native_filter(self)
    if not path:
        return
    self.entries["filepath"].delete(0, tk.END)
    self.entries["filepath"].insert(0, path)
    self._schedule_config_save()
    return self._load_plan_from_path()


def _planner_force_project_extension_macos_safe(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return raw
    low = raw.lower()
    ext = TCCM_PROJECT_EXTENSION
    if low.endswith(ext) or low.endswith(".tccm") or low.endswith(".txt"):
        return raw
    if low.endswith(".json"):
        return raw[:-5] + ext
    if not Path(raw).suffix:
        return raw + ext
    return raw + ext


def _tccm_json_loader_read_json(path: str):
    try:
        raw = Path(path).read_text(encoding="utf-8")
        if not raw.lstrip().startswith(("{", "[")):
            return None
        return json.loads(raw)
    except Exception:
        return None


def _tccm_json_loader_is_entry_dict(value) -> bool:
    return isinstance(value, dict) and bool(str(value.get("type") or "").strip())


def _tccm_json_loader_is_project_dict(value) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("plan_data"), dict):
        return True
    fmt = str(value.get("format") or "").strip()
    if fmt == TCCM_PROJECT_FORMAT:
        return True
    if str(value.get("schema_version") or "").strip() and isinstance(value.get("entries"), list):
        return True
    # Treat dictionaries with project-only state as project files even if a
    # hand-edited file forgot the format marker.
    project_keys = {"runtime", "variants", "run_profiles", "current_variant", "saved_at"}
    return isinstance(value.get("entries"), list) and any(k in value for k in project_keys)


def _tccm_json_loader_extract_project_plan(value):
    if isinstance(value, dict) and isinstance(value.get("plan_data"), dict):
        return value.get("plan_data") or {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"version": 3, "format": TCCM_PROJECT_FORMAT, "entries": value}
    return {"version": 3, "format": TCCM_PROJECT_FORMAT, "entries": []}


def _tccm_json_loader_raw_entries_from_json(value):
    """Return a plain list of entry dictionaries from project or preset JSON."""
    if _tccm_json_loader_is_project_dict(value):
        plan_part = _tccm_json_loader_extract_project_plan(value)
        entries = plan_part.get("entries") if isinstance(plan_part, dict) else []
        return [copy.deepcopy(x) for x in (entries or []) if isinstance(x, dict)]
    if isinstance(value, list):
        return [copy.deepcopy(x) for x in value if isinstance(x, dict)]
    if _tccm_json_loader_is_entry_dict(value):
        return [copy.deepcopy(value)]
    if isinstance(value, dict) and isinstance(value.get("entries"), list):
        return [copy.deepcopy(x) for x in (value.get("entries") or []) if isinstance(x, dict)]
    return []


def _tccm_json_loader_normalize_entries_for_insert(self, raw_entries, *, regenerate_ids: bool):
    out = []
    for raw in raw_entries or []:
        if not isinstance(raw, dict):
            continue
        try:
            normalizer = getattr(self, "_normalize_entry_preserving_embedded_sources", None)
            if callable(normalizer):
                entry = normalizer(copy.deepcopy(raw))
            else:
                entry = normalize_plan({"entries": [copy.deepcopy(raw)]})["entries"][0]
        except Exception:
            entry = copy.deepcopy(raw)
            entry.setdefault("type", "Checkpoint Merge")
        if regenerate_ids:
            entry["id"] = make_entry(entry.get("type", "Checkpoint Merge")).get("id")
        else:
            entry.setdefault("id", make_entry(entry.get("type", "Checkpoint Merge")).get("id"))
        # Preserve editor-only fields when present in presets/projects.
        for key in ("memo", "_locked", "_disabled", "_row_color"):
            if key in raw:
                entry[key] = copy.deepcopy(raw.get(key))
        out.append(entry)
    return out


def _tccm_json_loader_plan_from_any_file(self, path: str):
    """Load a path as a project, preset JSON, entries JSON, or legacy text."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Plan file does not exist or is not a file: {source}")
    data = _tccm_json_loader_read_json(str(source))
    if data is not None:
        if _tccm_json_loader_is_project_dict(data):
            return {
                "kind": "project",
                "json": data,
                "plan": _tccm_json_loader_extract_project_plan(data),
                "entries": _tccm_json_loader_raw_entries_from_json(data),
            }
        raw_entries = _tccm_json_loader_raw_entries_from_json(data)
        if raw_entries:
            entries = _tccm_json_loader_normalize_entries_for_insert(self, raw_entries, regenerate_ids=False)
            return {
                "kind": "preset-json",
                "json": data,
                "plan": {"version": 3, "format": "preset-json-as-plan", "entries": entries},
                "entries": raw_entries,
            }
        # Fall through to load_plan_records for legacy structured JSON if
        # possible.  This keeps older planner JSON variants usable.
    raw_plan = normalize_plan(load_plan_records(str(source)))
    collapse = getattr(self, "_collapse_internal_plan_entries", None)
    plan = collapse(raw_plan) if callable(collapse) else raw_plan
    return {
        "kind": "legacy-text" if source.suffix.lower() == ".txt" else "legacy-json",
        "json": data,
        "plan": plan,
        "entries": copy.deepcopy(plan.get("entries", []) or []),
    }


def _restore_session_state_universal_json(self):
    filepath = ""
    try:
        filepath = self.entries["filepath"].get().strip()
    except Exception:
        filepath = ""
    filepath = self._normalize_user_path(filepath) if filepath else ""
    if filepath and os.path.exists(filepath):
        try:
            loaded = _tccm_json_loader_plan_from_any_file(self, filepath)
            if loaded.get("kind") == "project":
                restore = getattr(self, "_planner_restore_project_payload", None)
                if callable(restore):
                    restore(loaded.get("json"), source_path=filepath)
                else:
                    self.plan_data = normalize_plan(loaded.get("plan") or {"entries": []})
            else:
                self.plan_data = normalize_plan(loaded.get("plan") or {"entries": []})
                self.current_index = 0
        except Exception:
            self.plan_data = self._planner_default_visible_plan()
    else:
        self.plan_data = self._planner_default_visible_plan()
    try:
        if hasattr(self, "_planner_apply_entry_defaults"):
            self._planner_apply_entry_defaults()
    except Exception:
        pass


def _tccm_atomic_normalize_path(self, value: str) -> str:
    try:
        return self._normalize_user_path(str(value or "").strip())
    except Exception:
        return str(value or "").strip()


def _tccm_atomic_init(self, *args, **kwargs):
    from .lifecycle_support import _planner_safe_init as _tccm_atomic_prev_init

    _tccm_atomic_prev_init(self, *args, **kwargs)
    current = ""
    try:
        widget = getattr(self, "entries", {}).get("filepath")
        current = _tccm_atomic_normalize_path(self, widget.get() if widget is not None else "")
    except Exception:
        current = ""
    self._tccm_active_plan_path = current
    self._tccm_pending_plan_path = ""
    self._tccm_plan_switch_in_progress = False


def _tccm_atomic_runtime_payload(self) -> dict:
    from .document_support import _tccm_current_runtime_payload as _tccm_atomic_prev_runtime_payload

    payload = _tccm_atomic_prev_runtime_payload(self) if callable(_tccm_atomic_prev_runtime_payload) else {}
    payload = dict(payload or {})
    payload.pop("filepath", None)
    return payload


def _tccm_atomic_apply_runtime(self, runtime: dict):
    from .document_support import _tccm_apply_runtime_payload as _tccm_atomic_prev_apply_runtime

    clean = dict(runtime or {}) if isinstance(runtime, dict) else {}
    clean.pop("filepath", None)
    if callable(_tccm_atomic_prev_apply_runtime):
        return _tccm_atomic_prev_apply_runtime(self, clean)


def _tccm_atomic_project_payload(self, *args, **kwargs):
    from .document_support import _planner_project_json_payload_advanced as _tccm_atomic_prev_project_payload

    payload = _tccm_atomic_prev_project_payload(self, *args, **kwargs) if callable(_tccm_atomic_prev_project_payload) else {}
    if isinstance(payload, dict):
        runtime = payload.get("runtime")
        if isinstance(runtime, dict):
            runtime = copy.deepcopy(runtime)
            runtime.pop("filepath", None)
            payload["runtime"] = runtime
    return payload


def _tccm_atomic_set_filepath_widget(self, path: str):
    try:
        widget = getattr(self, "entries", {}).get("filepath")
        if widget is not None:
            widget.delete(0, tk.END)
            widget.insert(0, path)
    except Exception:
        pass


def _tccm_atomic_active_path(self) -> str:
    active = _tccm_atomic_normalize_path(self, getattr(self, "_tccm_active_plan_path", ""))
    if active:
        return active
    # During early startup the explicit active-path field does not exist yet.
    try:
        widget = getattr(self, "entries", {}).get("filepath")
        return _tccm_atomic_normalize_path(self, widget.get() if widget is not None else "")
    except Exception:
        return ""


def _tccm_atomic_cancel_or_flush_pending_edit(self):
    pending = getattr(self, "_after_entry_change_id", None)
    if pending is None:
        return
    try:
        self.root.after_cancel(pending)
    except Exception:
        pass
    self._after_entry_change_id = None
    # The existing editor is autosave-oriented.  Persist the final edit to
    # the *currently active* file before replacing the in-memory document.
    try:
        if _tccm_atomic_active_path(self):
            self._save_plan_to_file()
    except Exception:
        # Loading should still be possible if the pre-switch autosave fails;
        # the normal load error handling remains authoritative.
        pass


def _tccm_atomic_save_project(self, path: str | None = None, *, create_backup: bool = True, reason: str = "autosave") -> str:
    from .document_support import _planner_save_project_json_advanced as _tccm_atomic_prev_save_project

    explicit = bool(path)
    active = _tccm_atomic_active_path(self)
    # Non-manual writes are background/editor writes and must stay attached
    # to the loaded document even when the path entry currently shows a
    # newly browsed file that has not been loaded yet.
    if not explicit and reason != "manual-save" and active:
        path = active
    result = _tccm_atomic_prev_save_project(self, path, create_backup=create_backup, reason=reason)
    # Manual Save intentionally adopts the visible path as the active file
    # (Save-As semantics).  Explicit migration/export paths do not silently
    # retarget the current document unless their caller later loads them.
    if reason in {"manual-save", "migration-load"} and result:
        normalized = _tccm_atomic_normalize_path(self, result)
        self._tccm_active_plan_path = normalized
        self._tccm_pending_plan_path = ""
        _tccm_atomic_set_filepath_widget(self, normalized)
    return result


def _tccm_atomic_autosave(self):
    from .document_support import _save_plan_to_file_project_json as _tccm_atomic_prev_autosave

    if bool(getattr(self, "_tccm_plan_switch_in_progress", False)):
        return None
    active = _tccm_atomic_active_path(self)
    if active and callable(getattr(self, "_planner_save_project_json", None)):
        return self._planner_save_project_json(active, reason="autosave")
    if callable(_tccm_atomic_prev_autosave):
        return _tccm_atomic_prev_autosave(self)
    return None


def _tccm_atomic_save_config(self):
    from .document_support import _save_current_state_to_config_dev as _tccm_atomic_prev_save_config
    from .settings import save_config_to_disk

    result = _tccm_atomic_prev_save_config(self) if callable(_tccm_atomic_prev_save_config) else None
    active = _tccm_atomic_active_path(self)
    if active:
        try:
            self.config["filepath"] = active
            save_config_to_disk(self.config)
        except Exception:
            pass
    return result


def _tccm_atomic_browse_plan_file(self):
    from .tools_support import _tccm_open_plan_dialog_no_native_filter

    opener = _tccm_open_plan_dialog_no_native_filter
    if callable(opener):
        path = opener(self)
    else:
        path = filedialog.askopenfilename(
            title="Open TCCM Project / Legacy Plan",
            filetypes=[
                ("Planner files", "*.tccm.json *.tccm *.json *.txt"),
                ("All files", "*.*"),
            ],
        )
        path = _tccm_atomic_normalize_path(self, path)
    if not path:
        return
    path = _tccm_atomic_normalize_path(self, path)
    self._tccm_pending_plan_path = path
    _tccm_atomic_set_filepath_widget(self, path)
    # Do not persist the candidate path to config until Load succeeds.
    try:
        self.status_label.config(text=f"Selected plan path (not loaded): {self._path_display_name(path)}")
    except Exception:
        pass


def _tccm_atomic_new_plan(self):
    from .document_support import _new_plan_advanced as _tccm_atomic_prev_new_plan

    before_active = _tccm_atomic_active_path(self)
    _tccm_atomic_cancel_or_flush_pending_edit(self)
    self._tccm_plan_switch_in_progress = True
    try:
        result = _tccm_atomic_prev_new_plan(self) if callable(_tccm_atomic_prev_new_plan) else None
        try:
            widget = getattr(self, "entries", {}).get("filepath")
            current = _tccm_atomic_normalize_path(self, widget.get() if widget is not None else "")
        except Exception:
            current = ""
        # The New dialog writes the new file before returning.  Adopt it only
        # when the visible path actually changed to an existing file.
        if current and current != before_active and os.path.exists(current):
            self._tccm_active_plan_path = current
            self._tccm_pending_plan_path = ""
            try:
                self.config["filepath"] = current
                self._schedule_config_save()
            except Exception:
                pass
        return result
    finally:
        self._tccm_plan_switch_in_progress = False


def _tccm_v2_norm(self, value: str) -> str:
    try:
        return self._normalize_user_path(str(value or "").strip())
    except Exception:
        return str(value or "").strip()


def _tccm_v2_active_path(self) -> str:
    # Once __init__ has established document identity, never infer it from
    # the visible path widget.  That widget may contain a pending load target.
    if hasattr(self, "_tccm_active_plan_path"):
        return _tccm_v2_norm(self, getattr(self, "_tccm_active_plan_path", ""))
    try:
        w = getattr(self, "entries", {}).get("filepath")
        return _tccm_v2_norm(self, w.get() if w is not None else "")
    except Exception:
        return ""


def _tccm_v2_set_path_widget(self, path: str) -> None:
    try:
        w = getattr(self, "entries", {}).get("filepath")
        if w is not None:
            w.delete(0, tk.END)
            w.insert(0, path)
    except Exception:
        pass


def _tccm_v2_set_title_widget(self, title: str) -> None:
    try:
        w = getattr(self, "entries", {}).get("title")
        if w is not None:
            w.delete(0, tk.END)
            w.insert(0, str(title or ""))
    except Exception:
        pass


def _tccm_v2_default_title_for_path(path: str) -> str:
    name = Path(path).name
    low = name.lower()
    if low.endswith(TCCM_PROJECT_EXTENSION):
        return name[:-len(TCCM_PROJECT_EXTENSION)] or "merge_plan"
    if low.endswith(".tccm"):
        return name[:-5] or "merge_plan"
    return Path(name).stem or "merge_plan"


def _tccm_v2_project_payload(self, *args, **kwargs):
    from .document_support import _tccm_atomic_project_payload as _tccm_v2_prev_payload

    payload = _tccm_v2_prev_payload(self, *args, **kwargs) if callable(_tccm_v2_prev_payload) else {}
    if not isinstance(payload, dict):
        payload = {}
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        runtime = copy.deepcopy(runtime)
        runtime.pop("filepath", None)
        payload["runtime"] = runtime
    # Store document UI identity separately from generic runtime state.
    # This prevents a missing runtime.title from inheriting another file's
    # currently displayed Notebook Title on the next load.
    title = ""
    try:
        title = self.entries["title"].get().strip()
    except Exception:
        title = ""
    payload["document"] = {"title": title}
    return payload


def _tccm_v2_apply_runtime(self, runtime: dict):
    from .document_support import _tccm_atomic_apply_runtime as _tccm_v2_prev_runtime_apply

    clean = dict(runtime or {}) if isinstance(runtime, dict) else {}
    clean.pop("filepath", None)
    if callable(_tccm_v2_prev_runtime_apply):
        return _tccm_v2_prev_runtime_apply(self, clean)


def _tccm_v2_save_project(self, path: str | None = None, *, create_backup: bool = True, reason: str = "autosave") -> str:
    from .document_support import _tccm_atomic_save_project as _tccm_v2_prev_save_project

    explicit_path = _tccm_v2_norm(self, path) if path else ""
    active = _tccm_v2_active_path(self)

    # No implicit Save-As: both autosave and the Save button save the active
    # loaded document.  Only callers that pass an explicit path (migration,
    # export, New creation, etc.) may target another file.
    target = explicit_path or active
    if not target:
        try:
            visible = self.entries["filepath"].get().strip()
        except Exception:
            visible = ""
        target = _tccm_v2_norm(self, visible)
    if not target:
        target = self._ensure_plan_path()
        target = _tccm_v2_norm(self, target)

    if not callable(_tccm_v2_prev_save_project):
        raise RuntimeError("Planner save implementation is unavailable")
    result = _tccm_v2_prev_save_project(self, target, create_backup=create_backup, reason=reason)
    result = _tccm_v2_norm(self, result)

    # Saving the current document keeps the UI pinned to its real path.
    # Explicit helper writes must not silently retarget document identity.
    if not explicit_path or reason in {"manual-save", "new-project", "migration-load"}:
        self._tccm_active_plan_path = result
        self._tccm_pending_plan_path = ""
        _tccm_v2_set_path_widget(self, result)
        try:
            self.config["filepath"] = result
        except Exception:
            pass
    return result


def _tccm_v2_save_button(self):
    try:
        active = _tccm_v2_active_path(self)
        if not active:
            # A project that has never been loaded/created may adopt the
            # visible path once, but a pending browse target never overrides
            # an existing active document.
            try:
                active = _tccm_v2_norm(self, self.entries["filepath"].get().strip())
            except Exception:
                active = ""
        path = self._planner_save_project_json(active or None, reason="manual-save")
        label = "project JSON" if _planner_is_project_json_path(path) else "runner text"
        self.status_label.config(text=f"Saved {label}: {self._path_display_name(path)}")
        messagebox.showinfo("Save Plan", f"Saved {label} successfully.\n\n{path}")
    except Exception as exc:
        self._show_detailed_error("Save Plan Error", exc, context="Operation: save_plan")


def _tccm_v2_load(self):
    from .tools_support import _tccm_open_plan_dialog_no_native_filter

    try:
        requested = _tccm_v2_norm(self, self.entries["filepath"].get().strip())
    except Exception:
        requested = ""
    if not requested:
        opener = _tccm_open_plan_dialog_no_native_filter
        requested = opener(self) if callable(opener) else _tccm_v2_norm(
            self, filedialog.askopenfilename(title="Open TCCM Project / Preset / Legacy Plan")
        )
        requested = _tccm_v2_norm(self, requested)
        if not requested:
            return None

    old_active = _tccm_v2_active_path(self)
    old_title = ""
    try:
        old_title = self.entries["title"].get()
    except Exception:
        pass

    # Commit any delayed edit only to the document that is actually active.
    pending = getattr(self, "_after_entry_change_id", None)
    if pending is not None:
        try:
            self.root.after_cancel(pending)
        except Exception:
            pass
        self._after_entry_change_id = None
        if old_active:
            try:
                self._planner_save_project_json(old_active, reason="autosave")
            except Exception:
                pass

    self._tccm_plan_switch_in_progress = True
    try:
        loaded = _tccm_json_loader_plan_from_any_file(self, requested)
        kind = loaded.get("kind", "")
        raw_json = loaded.get("json") if isinstance(loaded, dict) else None

        if kind == "project":
            restore = getattr(self, "_planner_restore_project_payload", None)
            if callable(restore):
                restore(raw_json, source_path=requested)
            else:
                self.plan_data = normalize_plan(loaded.get("plan") or {"entries": []})
        else:
            self.plan_data = normalize_plan(loaded.get("plan") or {"entries": []})
            self.current_index = 0
            if kind == "preset-json":
                if hasattr(self, "_tccm_ensure_extension_state"):
                    self._tccm_ensure_extension_state()
                self.plan_variants = []
                self.run_profiles = []
                self.current_variant_name = ""
            load_meta = getattr(self, "_planner_load_meta_from_disk", None)
            if callable(load_meta):
                try:
                    load_meta(requested)
                except Exception:
                    pass

        # Resolve Notebook Title from the loaded document, never from the
        # previously open file.  New files store document.title.  For older
        # files runtime.title remains supported.  If neither exists, derive
        # a deterministic title from the filename instead of leaking old UI.
        loaded_title = ""
        if isinstance(raw_json, dict):
            doc = raw_json.get("document")
            if isinstance(doc, dict):
                loaded_title = str(doc.get("title") or "").strip()
            if not loaded_title:
                rt = raw_json.get("runtime")
                if isinstance(rt, dict):
                    loaded_title = str(rt.get("title") or "").strip()
        if not loaded_title:
            loaded_title = _tccm_v2_default_title_for_path(requested)
        _tccm_v2_set_title_widget(self, loaded_title)
        try:
            self.config["title"] = loaded_title
        except Exception:
            pass

        # Only now is the requested file the active document identity.
        self._tccm_active_plan_path = requested
        self._tccm_pending_plan_path = ""
        _tccm_v2_set_path_widget(self, requested)
        try:
            self.config["filepath"] = requested
        except Exception:
            pass

        if hasattr(self, "_history_undo"):
            self._history_undo.clear()
        if hasattr(self, "_history_redo"):
            self._history_redo.clear()
        try:
            self._last_history_snapshot = self._planner_plan_snapshot() if hasattr(self, "_planner_plan_snapshot") else None
        except Exception:
            self._last_history_snapshot = None
        self.current_index = max(0, min(int(getattr(self, "current_index", 0) or 0), len(self.plan_data.get("entries", []) or [None]) - 1))
        self._refresh_line_selector()
        self._render_current_line()
        try:
            self._save_current_state_to_config()
        except Exception:
            pass
        label = "project" if kind == "project" else ("preset as plan" if kind == "preset-json" else "plan")
        self.status_label.config(text=f"Loaded {label}: {self._path_display_name(requested)}")
        try:
            offer = getattr(self, "_planner_offer_newer_backup_restore", None)
            if callable(offer):
                self.root.after(500, lambda p=requested: offer(p))
        except Exception:
            pass
        return loaded
    except Exception as exc:
        # Restore only UI identity.  In-memory replacement happens after a
        # successful parser/restore, so most load failures leave old content.
        self._tccm_active_plan_path = old_active
        self._tccm_pending_plan_path = ""
        if old_active:
            _tccm_v2_set_path_widget(self, old_active)
        if old_title:
            _tccm_v2_set_title_widget(self, old_title)
        self._show_detailed_error("Load Error", exc)
        return None
    finally:
        self._tccm_plan_switch_in_progress = False
