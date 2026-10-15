"""Lifecycle controls and callbacks."""
from __future__ import annotations

from .constants import INIT_CONFIG
from .document_support import _tccm_atomic_init as _method__init__
from .document_support import _tccm_atomic_save_config as _method_save_current_state_to_config
from .lifecycle_support import _initialize_widgets as _method_base_init
from tkinter import Entry
from tkinter import ttk
import tkinter as tk


class LifecycleMixin:
    """Lifecycle behavior for the Planner."""

    __init__ = _method__init__


    def _on_closing(self):
        self._save_current_state_to_config()
        self.root.destroy()


    def _schedule_config_save(self):
        if self.save_after_id:
            self.root.after_cancel(self.save_after_id)
        self.save_after_id = self.root.after(350, self._save_current_state_to_config)


    def _on_bake_vae_toggle(self, save: bool = True):
        enabled = bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get())
        for widget in getattr(self, "vae_field_widgets", []):
            try:
                if isinstance(widget, Entry):
                    widget.configure(state="normal" if enabled else "disabled")
                elif isinstance(widget, ttk.Button):
                    widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass
        if save:
            self._schedule_config_save()


    _save_current_state_to_config = _method_save_current_state_to_config


    _base_init = _method_base_init


    def _planner_init_extra_state(self):
        from .settings import load_config_from_disk

        cfg = getattr(self, 'config', None)
        if not isinstance(cfg, dict):
            try:
                cfg = load_config_from_disk()
            except Exception:
                cfg = INIT_CONFIG.copy()
            self.config = cfg
        self.plan_clipboard_entries = []
        self.visible_entry_indices = []
        self.plan_search_var = tk.StringVar(master=self.root, value="")
        self.plan_filter_var = tk.StringVar(master=self.root, value="All")
        self.plan_listbox = None
        self.selection_var = tk.StringVar(master=self.root, value="")
        self.selection_combo = None
        self.plan_add_button = None
        self.plan_remove_button = None
        self.plan_move_up_button = None
        self.plan_move_down_button = None
        self.plan_reload_button = None
        self.plan_summary_var = tk.StringVar(master=self.root, value="0 lines")
        self.plan_deps_var = tk.StringVar(master=self.root, value="Deps: 0 links")
        self.plan_unused_var = tk.StringVar(master=self.root, value="Unused: 0")
        self._selection_guard = False
        self._drag_start_visible = None
        self._drag_selected_model_indices = []
        self._drag_requires_primary = False
        self._history_undo = []
        self._history_redo = []
        self._history_after_id = None
        self._last_history_snapshot = None
        self._section_open_state = {
            "left_main_config": True,
            "left_notebook_run_options": True,
            "plan_view": True,
        }
        self._plan_hover_tip = None
        self._plan_hover_index = None
        self._plan_hover_text = ""
        self._plan_problem_map_cache = {}
        self._plan_context_menu = None
        self._plan_context_popup = None
        self._plan_context_root_bind_ids = {}
        self._plan_listbox_render_cache = None
        self._last_responsive_layout_state = None
        self._after_entry_change_id = None
        self._plan_drag_guide = None
        self._plan_drag_guide_target = None
        self._last_backup_snapshot = None
        self._last_backup_time = 0.0
        self.backup_keep_generations = int(self.config.get("backup_keep_generations", 20) or 20)
