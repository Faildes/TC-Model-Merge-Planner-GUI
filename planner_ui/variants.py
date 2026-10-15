"""Variants responsibilities."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import TCCM_PROJECT_FORMAT
from pathlib import Path
from plan import make_entry
from tkinter import Frame
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk
import copy
import json
import tkinter as tk


def _tccm_choose_by_name(self, title: str, items: list[dict], *, label_key: str = "name"):
    if not items:
        messagebox.showinfo(title, "No items are available.")
        return None
    result = {"item": None}
    win = tk.Toplevel(self.root)
    win.title(title)
    win.geometry("560x420+180+180")
    outer = Frame(win, padx=8, pady=8)
    outer.pack(fill="both", expand=True)
    listbox = tk.Listbox(outer, exportselection=False, activestyle="none")
    listbox.pack(fill="both", expand=True)
    for item in items:
        name = str(item.get(label_key) or item.get("name") or "(unnamed)")
        saved = str(item.get("saved_at") or "")
        listbox.insert("end", f"{name}  {saved}".strip())
    if items:
        listbox.selection_set(0)
    buttons = Frame(outer)
    buttons.pack(fill="x", pady=(8, 0))
    def ok(_event=None):
        sel = listbox.curselection()
        if sel:
            result["item"] = items[int(sel[-1])]
        win.destroy()
    def cancel():
        result["item"] = None
        win.destroy()
    listbox.bind("<Double-Button-1>", ok)
    ttk.Button(buttons, text="OK", command=ok).pack(side="right", padx=(4, 0))
    ttk.Button(buttons, text="Cancel", command=cancel).pack(side="right")
    win.transient(self.root)
    win.grab_set()
    self.root.wait_window(win)
    return result["item"]


def _save_current_as_variant(self):
    from .tools_support import _tccm_timestamp

    self._tccm_ensure_extension_state()
    name = simpledialog.askstring("Save Variant", "Variant name:", parent=self.root)
    name = str(name or "").strip()
    if not name:
        return
    existing = next((v for v in self.plan_variants if str(v.get("name")) == name), None)
    if existing is not None and not messagebox.askyesno("Overwrite Variant", f"Variant '{name}' already exists. Overwrite it?"):
        return
    variant = {
        "name": name,
        "saved_at": _tccm_timestamp(),
        "current_index": int(getattr(self, "current_index", 0) or 0),
        "entries": copy.deepcopy((getattr(self, "plan_data", {}) or {}).get("entries", []) or []),
        "final_memo": copy.deepcopy((getattr(self, "plan_data", {}) or {}).get("final_memo", "")),
    }
    if existing is None:
        self.plan_variants.append(variant)
    else:
        existing.clear(); existing.update(variant)
    self.current_variant_name = name
    self._planner_save_project_json(reason="variant-save")
    self.status_label.config(text=f"Saved variant: {name}")


def _load_variant(self):
    self._tccm_ensure_extension_state()
    item = self._tccm_choose_by_name("Load Variant", self.plan_variants)
    if not item:
        return
    if not messagebox.askyesno("Load Variant", f"Replace the current visible plan with variant '{item.get('name')}'?"):
        return
    try:
        if hasattr(self, "_planner_push_history"):
            self._planner_push_history()
        self.plan_data = {
            "version": 3,
            "format": TCCM_PROJECT_FORMAT,
            "entries": copy.deepcopy(item.get("entries") or []),
            "final_memo": copy.deepcopy(item.get("final_memo", "")),
        }
        if not self.plan_data["entries"]:
            self.plan_data["entries"] = [make_entry("Checkpoint Merge")]
        self.current_index = max(0, min(int(item.get("current_index", 0) or 0), len(self.plan_data["entries"]) - 1))
        self.current_variant_name = str(item.get("name") or "")
        if hasattr(self, "_planner_apply_entry_defaults"):
            self._planner_apply_entry_defaults()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_save_project_json(reason="variant-load")
        self.status_label.config(text=f"Loaded variant: {self.current_variant_name}")
    except Exception as exc:
        self._show_detailed_error("Load Variant Error", exc)


def _delete_variant(self):
    self._tccm_ensure_extension_state()
    item = self._tccm_choose_by_name("Delete Variant", self.plan_variants)
    if not item:
        return
    name = str(item.get("name") or "")
    if not messagebox.askyesno("Delete Variant", f"Delete variant '{name}'?"):
        return
    self.plan_variants = [v for v in self.plan_variants if v is not item and str(v.get("name") or "") != name]
    if getattr(self, "current_variant_name", "") == name:
        self.current_variant_name = ""
    self._planner_save_project_json(reason="variant-delete")
    self.status_label.config(text=f"Deleted variant: {name}")


def _export_variants_json(self):
    from .tools_support import _tccm_timestamp

    self._tccm_ensure_extension_state()
    payload = {"version": 1, "format": "tccm-variants", "saved_at": _tccm_timestamp(), "variants": copy.deepcopy(self.plan_variants)}
    path = filedialog.asksaveasfilename(title="Export Variants JSON", defaultextension=".json", initialfile="variants.json", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
    path = self._normalize_user_path(path)
    if not path:
        return
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    self.status_label.config(text=f"Exported variants: {self._path_display_name(path)}")


def _save_current_as_run_profile(self):
    from .tools_support import _tccm_timestamp

    self._tccm_ensure_extension_state()
    name = simpledialog.askstring("Save Run Profile", "Profile name:", parent=self.root)
    name = str(name or "").strip()
    if not name:
        return
    existing = next((p for p in self.run_profiles if str(p.get("name")) == name), None)
    if existing is not None and not messagebox.askyesno("Overwrite Run Profile", f"Profile '{name}' already exists. Overwrite it?"):
        return
    profile = {"name": name, "saved_at": _tccm_timestamp(), "runtime": self._tccm_current_runtime_payload()}
    if existing is None:
        self.run_profiles.append(profile)
    else:
        existing.clear(); existing.update(profile)
    self._planner_save_project_json(reason="run-profile-save")
    self.status_label.config(text=f"Saved run profile: {name}")


def _apply_run_profile(self):
    self._tccm_ensure_extension_state()
    item = self._tccm_choose_by_name("Apply Run Profile", self.run_profiles)
    if not item:
        return
    self._tccm_apply_runtime_payload(item.get("runtime") or {})
    self._planner_save_project_json(reason="run-profile-apply")
    self.status_label.config(text=f"Applied run profile: {item.get('name')}")


def _delete_run_profile(self):
    self._tccm_ensure_extension_state()
    item = self._tccm_choose_by_name("Delete Run Profile", self.run_profiles)
    if not item:
        return
    name = str(item.get("name") or "")
    if not messagebox.askyesno("Delete Run Profile", f"Delete run profile '{name}'?"):
        return
    self.run_profiles = [p for p in self.run_profiles if p is not item and str(p.get("name") or "") != name]
    self._planner_save_project_json(reason="run-profile-delete")
    self.status_label.config(text=f"Deleted run profile: {name}")


def _export_run_profiles_json(self):
    from .tools_support import _tccm_timestamp

    self._tccm_ensure_extension_state()
    payload = {"version": 1, "format": "tccm-run-profiles", "saved_at": _tccm_timestamp(), "run_profiles": copy.deepcopy(self.run_profiles)}
    path = filedialog.asksaveasfilename(title="Export Run Profiles JSON", defaultextension=".json", initialfile="run_profiles.json", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
    path = self._normalize_user_path(path)
    if not path:
        return
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    self.status_label.config(text=f"Exported run profiles: {self._path_display_name(path)}")
