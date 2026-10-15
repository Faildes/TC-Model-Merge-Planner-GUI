"""Lifecycle support responsibilities."""
from __future__ import annotations

from .constants import INIT_CONFIG
from pathlib import Path
from tkinter import Canvas
from tkinter import Entry
from tkinter import Frame
from tkinter import Scrollbar
from tkinter import Text
from typing import Any
from typing import Dict
from typing import List
import os
import threading
import tkinter as tk


def _initialize_widgets(self, root):
    from .runner import RunnerConsoleWindow
    from .discovery import discover_block_sets
    from .discovery import discover_merge_modes
    from .settings import load_config_from_disk

    self.root = root
    self.config = load_config_from_disk()
    self.entries: Dict[str, Entry] = {}
    self.block_sets = discover_block_sets()
    self.base_model_var = tk.StringVar(value=self.config.get("base_model", "SDXL"))
    self.merge_modes = discover_merge_modes()
    self.merge_mode_map = {m["key"]: m for m in self.merge_modes}
    self.plan_data = self._planner_default_visible_plan()
    self.current_index = 0
    self.render_after_id = None
    self.save_after_id = None
    self.plan_save_after_id = None
    self.progress_indicator = None
    self.console = RunnerConsoleWindow(root)
    self.stop_uploader_event = threading.Event()
    self.current_editor_widgets: List[tk.Widget] = []
    self.line_variant_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    self.left_canvas: Canvas | None = None
    self.left_scroll_frame: Frame | None = None
    self._active_multiline_text: Text | None = None
    self._elemental_alias_hover_popup: tk.Toplevel | None = None
    self._elemental_alias_hover_key = None
    self._autocomplete_popup: tk.Toplevel | None = None
    self._autocomplete_listbox: tk.Listbox | None = None
    self._autocomplete_target = None
    self._autocomplete_apply = None
    self._autocomplete_items: List[Dict[str, str]] = []
    self._autocomplete_row_frames: List[Frame] = []
    self._autocomplete_selected_index = 0
    self._autocomplete_canvas: Canvas | None = None
    self._autocomplete_inner: Frame | None = None
    self._autocomplete_scrollbar: Scrollbar | None = None
    self._rerender_after_id = None
    self.vae_field_widgets: List[tk.Widget] = []
    self.bake_vae_var = tk.BooleanVar(
        value=bool(self.config.get("bake_vae", bool(str(self.config.get("vae", "")).strip())))
    )

    self.root.geometry("1420x840+50+50")
    self.root.title("Model Planner")
    self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    self._create_menu()
    self._create_layout()
    self._restore_session_state()
    self._refresh_line_selector()
    self._render_current_line()


def _create_menu(self):
    menu_bar = tk.Menu(self.root)
    self.root.config(menu=menu_bar)
    utils_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Utilities", menu=utils_menu)
    utils_menu.add_command(label="Image Converter...", command=self._run_image_converter)
    utils_menu.add_command(label="Check aria2 Installation", command=self._check_aria2)
    help_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", command=self._show_about)


def _initialize_application(self, root):
    # Extended planner state is needed before the base layout is built,
    # but some of the later-added state initializers also read self.config.
    # Ensure config exists first so override stacks do not fail when methods
    # with the same name are replaced later in the class body.
    from .settings import load_config_from_disk

    self.root = root
    if not hasattr(self, "config") or not isinstance(getattr(self, "config", None), dict):
        try:
            self.config = load_config_from_disk()
        except Exception:
            self.config = INIT_CONFIG.copy()
    self._planner_init_extra_state()
    self._base_init(root)
    if not hasattr(self, "config") or not isinstance(getattr(self, "config", None), dict):
        self.config = INIT_CONFIG.copy()
    self.active_view = str(self.config.get("active_view", "split") or "split")
    self.theme_mode = str(self.config.get("theme_mode", "dark") or "dark")
    self.compact_breakpoint = int(self.config.get("compact_breakpoint", 1180) or 1180)
    self.outer_frame = getattr(self, "outer_frame", None)
    self.content_container = getattr(self, "content_container", None)
    self.top_fixed_bar = getattr(self, "top_fixed_bar", None)
    self.left_outer = getattr(self, "left_outer", None)
    self.right_outer = getattr(self, "right_outer", None)
    self.sep = getattr(self, "sep", None)
    self.view_switch_button = getattr(self, "view_switch_button", None)
    self.theme_toggle_button = getattr(self, "theme_toggle_button", None)
    self.fixed_title_label = getattr(self, "fixed_title_label", None)
    self._responsive_after_id = None
    self._planner_bind_shortcuts()
    self._planner_build_fixed_status_bar()
    self._last_history_snapshot = self._planner_plan_snapshot()
    self._refresh_line_selector()
    self._update_current_line_indicator()
    self._select_model_indices([self.current_index] if self.plan_data.get('entries') else [])
    self._planner_refresh_plan_meta()
    self._update_view_switch_label()
    self._update_responsive_layout()
    self._apply_theme_mode()


_planner_final_init = _initialize_application
_planner_prev_init = _initialize_widgets

def _planner_safe_init(self, root, _impl=_planner_final_init, _prev=_planner_prev_init):
    from .settings import load_config_from_disk

    self.root = root
    if not hasattr(self, 'config') or not isinstance(getattr(self, 'config', None), dict):
        try:
            self.config = load_config_from_disk()
        except Exception:
            self.config = INIT_CONFIG.copy()
    return _impl(self, root)


def _create_menu_advanced_tools(self):
    from .document_support import _create_menu_project_json_open_dialog as _tccm_prev_create_menu

    if callable(_tccm_prev_create_menu):
        _tccm_prev_create_menu(self)
    self._tccm_ensure_extension_state()
    try:
        menu_bar = self.root.nametowidget(self.root.cget("menu"))
    except Exception:
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
    plan_tools = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Plan Tools", menu=plan_tools)
    plan_tools.add_command(label="Plan Health Dashboard…", command=self._show_plan_health_dashboard)
    plan_tools.add_command(label="Dependency Chain Viewer…", command=self._show_dependency_chain_viewer)
    plan_tools.add_command(label="Output Name Collision Checker…", command=self._show_output_collision_checker)
    plan_tools.add_command(label="Model Line Provenance…", command=self._show_model_provenance)
    plan_tools.add_command(label="Export Provenance JSON…", command=self._export_model_provenance_json)
    plan_tools.add_separator()
    plan_tools.add_command(label="Backup Diff / Pin Manager…", command=self._show_backup_diff_manager)
    plan_tools.add_separator()
    variant_menu = tk.Menu(plan_tools, tearoff=0)
    plan_tools.add_cascade(label="Variants", menu=variant_menu)
    variant_menu.add_command(label="Save Current as Variant…", command=self._save_current_as_variant)
    variant_menu.add_command(label="Load Variant…", command=self._load_variant)
    variant_menu.add_command(label="Delete Variant…", command=self._delete_variant)
    variant_menu.add_command(label="Export Variants JSON…", command=self._export_variants_json)
    profile_menu = tk.Menu(plan_tools, tearoff=0)
    plan_tools.add_cascade(label="Run Profiles", menu=profile_menu)
    profile_menu.add_command(label="Save Current as Run Profile…", command=self._save_current_as_run_profile)
    profile_menu.add_command(label="Apply Run Profile…", command=self._apply_run_profile)
    profile_menu.add_command(label="Delete Run Profile…", command=self._delete_run_profile)
    profile_menu.add_command(label="Export Run Profiles JSON…", command=self._export_run_profiles_json)
    plan_tools.add_separator()
    plan_tools.add_command(label="Export TCCM JSON Schema…", command=self._export_project_json_schema)
    plan_tools.add_command(label="Safe Migration to .tccm.json…", command=self._safe_migration_tool)


def _tccm_plan_dialog_initialdir(self) -> str:
    try:
        raw = self._normalize_user_path(self.entries["filepath"].get().strip())
    except Exception:
        raw = ""
    try:
        if raw:
            p = Path(raw).expanduser()
            if p.is_dir():
                return str(p)
            if p.parent.exists():
                return str(p.parent)
    except Exception:
        pass
    return os.getcwd()


def _create_menu_macos_safe_open(self):
    from .lifecycle_support import _create_menu_advanced_tools as _tccm_prev_create_menu_for_safe_open

    if callable(_tccm_prev_create_menu_for_safe_open):
        _tccm_prev_create_menu_for_safe_open(self)
    pass  # Final methods are composed in app.py.
    pass  # Final methods are composed in app.py.
    pass  # Final methods are composed in app.py.
    try:
        self.root.bind("<Control-o>", lambda _e: (self._open_plan_dialog_project_json(), "break")[1], add="+")
        self.root.bind("<Command-o>", lambda _e: (self._open_plan_dialog_project_json(), "break")[1], add="+")
    except Exception:
        pass
    # Add an explicit safe Open item without relying on fragile menu indices.
    try:
        menu_bar = self.root.nametowidget(self.root.cget("menu"))
        file_menu_name = menu_bar.entrycget(0, "menu")
        file_menu = self.root.nametowidget(file_menu_name)
        file_menu.insert_command(0, label="Open Project / Legacy Plan…", command=self._open_plan_dialog_project_json)
        file_menu.insert_separator(1)
    except Exception:
        pass
