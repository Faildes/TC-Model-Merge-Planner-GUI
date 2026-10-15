"""Shortcuts responsibilities."""
from __future__ import annotations

from .constants import _DEV_DEFAULT_SHORTCUTS
from tkinter import Canvas
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import Scrollbar
from tkinter import messagebox
from tkinter import ttk
import copy
import tkinter as tk


def _planner_shortcut_registry(self):
    return {
        "undo": ("Undo", self._shortcut_undo),
        "redo": ("Redo", self._shortcut_redo),
        "copy_lines": ("Copy Lines", self._shortcut_copy_lines),
        "paste_lines": ("Paste Lines", self._shortcut_paste_lines),
        "delete_lines": ("Delete Lines", self._shortcut_remove_line),
        "duplicate_lines": ("Duplicate Lines", self._shortcut_duplicate_lines),
        "add_line": ("Add Line", self._shortcut_add_line),
        "move_up": ("Move Lines Up", self._shortcut_move_lines_up),
        "move_down": ("Move Lines Down", self._shortcut_move_lines_down),
        "validate": ("Validate", self._shortcut_validate),
        "save_preset": ("Save Preset", self._shortcut_save_preset),
        "load_preset": ("Load Preset", self._shortcut_load_preset),
        "run_notebook": ("Run Notebook", lambda e=None: (self._run_target_notebook(), "break")[1]),
        "save_plan": ("Save Plan Text", lambda e=None: (self._save_plan_text_button(), "break")[1]),
        "export_notebook": ("Export Notebook", lambda e=None: (self._export_as_notebook(), "break")[1]),
        "diff": ("Plan Diff", lambda e=None: (self._show_plan_diff_preview(), "break")[1]),
        "dependency_graph": ("Dependency Graph", lambda e=None: (self._planner_show_dependency_graph(), "break")[1]),
        "optimize_plan": ("Optimize Plan", lambda e=None: (self._planner_optimize_plan_full(), "break")[1]),
        "t2i_settings": ("T2I Settings", lambda e=None: (self._planner_dev_only(self._planner_open_t2i_settings), "break")[1]),
        "ratio_sweep": ("Ratio Sweep", lambda e=None: (self._planner_dev_only(self._planner_open_ratio_sweep), "break")[1]),
        "shortcut_settings": ("Shortcut Settings", lambda e=None: (self._planner_show_shortcut_settings(), "break")[1]),
    }


def _planner_bind_shortcuts_configurable(self):
    self._planner_dev_ensure_config()
    tag = 'PlannerGlobalShortcuts'
    self._planner_shortcut_bindtag = tag
    registry = self._planner_shortcut_registry()
    bindings = copy.deepcopy(_DEV_DEFAULT_SHORTCUTS)
    custom = self.config.get("shortcut_bindings") or {}
    if isinstance(custom, dict):
        for action, seqs in custom.items():
            if action in registry:
                if isinstance(seqs, str):
                    seqs = [s.strip() for s in seqs.split(",") if s.strip()]
                if isinstance(seqs, list):
                    bindings[action] = [str(s).strip() for s in seqs if str(s).strip()]
    self._planner_bound_shortcut_sequences = []
    for action, seqs in bindings.items():
        if action not in registry:
            continue
        _label, handler = registry[action]
        for sequence in seqs:
            try:
                self.root.bind_class(tag, sequence, handler, add='+')
                self._planner_bound_shortcut_sequences.append((tag, sequence))
            except Exception:
                pass
    self.root.bind_all('<FocusIn>', self._planner_register_shortcut_bindtag_on_focus, add='+')
    self._planner_register_shortcut_bindtag_on_tree(self.root)


def _planner_show_shortcut_settings(self):
    from .settings import save_config_to_disk

    self._planner_dev_ensure_config()
    win = tk.Toplevel(self.root)
    win.title("Shortcut Settings")
    win.geometry("840x640+180+120")
    outer = Frame(win, padx=10, pady=10)
    outer.pack(fill="both", expand=True)
    Label(outer, text="Enter Tk key sequences. Multiple shortcuts can be comma-separated.", anchor="w").pack(fill="x")
    canvas = Canvas(outer, highlightthickness=0)
    scroll = Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = Frame(canvas)
    canvas.configure(yscrollcommand=scroll.set)
    canvas.create_window((0,0), window=inner, anchor="nw", tags=("inner",))
    canvas.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfigure("inner", width=e.width))
    registry = self._planner_shortcut_registry()
    current = copy.deepcopy(self.config.get("shortcut_bindings") or _DEV_DEFAULT_SHORTCUTS)
    vars_map = {}
    for row_idx, (action, (label, _handler)) in enumerate(registry.items()):
        row = Frame(inner)
        row.pack(fill="x", pady=3)
        Label(row, text=label, width=24, anchor="w").pack(side="left")
        val = current.get(action, _DEV_DEFAULT_SHORTCUTS.get(action, []))
        if isinstance(val, list): val = ", ".join(val)
        var = tk.StringVar(value=str(val or ""))
        Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
        vars_map[action] = var
    def save():
        out = {}
        for action, var in vars_map.items():
            out[action] = [s.strip() for s in var.get().split(",") if s.strip()]
        self.config["shortcut_bindings"] = out
        save_config_to_disk(self.config)
        messagebox.showinfo("Shortcut Settings", "Saved. Restart the planner to fully reset existing class-level shortcut bindings.")
        win.destroy()
    def reset():
        for action, var in vars_map.items():
            var.set(", ".join(_DEV_DEFAULT_SHORTCUTS.get(action, [])))
    btns = Frame(outer)
    btns.pack(fill="x", pady=(8,0))
    ttk.Button(btns, text="Reset Defaults", command=reset).pack(side="left")
    ttk.Button(btns, text="Save", command=save).pack(side="right", padx=(4,0))
    ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")
