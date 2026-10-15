"""Layout ui controls and callbacks."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import LINE_TYPES
from .editor_support import _build_left_panel_refined_dev_buttons as _method_build_left_panel
from .help_support import _build_combo_row_dropdown_refined as _method_build_combo_row
from .lifecycle_support import _create_menu_macos_safe_open as _method_create_menu
from .tools_support import _planner_combobox_listbox as _method_planner_combobox_listbox
from PIL import Image
from tkinter import Canvas
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import LabelFrame
from tkinter import Scrollbar
from tkinter import Text
from tkinter import messagebox
from tkinter import ttk
from typing import Any
from typing import Dict
import os
import subprocess
import sys
import tkinter as tk


class LayoutUiMixin:
    """Layout ui behavior for the Planner."""

    _planner_combobox_listbox = _method_planner_combobox_listbox


    def _build_status_bar(self, parent):
        from .widgets import ProgressWindow

        status_frame = Frame(parent, bd=1, relief=tk.SUNKEN)
        status_frame.pack(side="bottom", fill="x")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = Label(status_frame, text="Status: Idle", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(8, 10), pady=4)

        self.progress_bar = ttk.Progressbar(status_frame, mode="indeterminate", length=240)
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=4)

        self._attach_tooltip([status_frame, self.status_label, self.progress_bar], self._left_help("Status").get("detail", ""))
        self.progress_indicator = ProgressWindow(self.status_label, self.progress_bar)
        self._add_inline_help(parent, "Status", padx=6, wraplength=540)


    def _build_collapsible_section(
        self,
        parent,
        title: str,
        *,
        key: str | None = None,
        default_open: bool = True,
        body_fill: str = "x",
        body_expand: bool = False,
        padx: int = 6,
        pady: int = 6,
    ):
        section_key = key or title
        expanded = bool(getattr(self, "_section_open_state", {}).get(section_key, default_open))
        outer = Frame(parent, bd=1, relief="groove")
        outer.pack(fill="both" if body_fill == "both" else "x", expand=body_expand, padx=padx, pady=pady, anchor="nw")
        header = Frame(outer, bg="#f2f2f2")
        header.pack(fill="x")
        text_var = tk.StringVar()
        body = Frame(outer, padx=8, pady=8)

        def apply_state():
            is_open = bool(self._section_open_state.get(section_key, default_open))
            text_var.set(("▼ " if is_open else "▶ ") + title)
            if is_open:
                if not body.winfo_manager():
                    body.pack(fill=body_fill, expand=body_expand)
            else:
                if body.winfo_manager():
                    body.pack_forget()

        def toggle():
            self._section_open_state[section_key] = not bool(self._section_open_state.get(section_key, default_open))
            apply_state()

        btn = ttk.Button(header, textvariable=text_var, command=toggle)
        try:
            btn.configure(style=self._button_style_for_text(title), takefocus=False)
        except Exception:
            pass
        try:
            btn.configure(compound='left')
        except Exception:
            pass
        btn.pack(side="left", fill="x", expand=True, padx=2, pady=2)
        apply_state()
        return body


    def _check_aria2(self):
        try:
            result = subprocess.run(["aria2c", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                messagebox.showinfo("aria2 Status", f"✓ aria2 is installed\n\n{result.stdout.splitlines()[0]}")
            else:
                messagebox.showwarning("aria2 Status", "aria2 is not installed or not in PATH.")
        except FileNotFoundError:
            messagebox.showwarning("aria2 Status", "aria2 is not installed or not in PATH.")


    def _play_notification_sound(self):
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"], check=False)
        except Exception:
            pass


    def _build_labeled_frame(self, parent, title: str) -> LabelFrame:
        frame = LabelFrame(parent, text=title, padx=8, pady=8)
        frame.pack(fill="x", padx=6, pady=6)
        self._add_right_inline_help(frame, title, padx=4, wraplength=720)
        return frame


    def _build_text_row(self, parent, label: str, entry: Dict[str, Any], key: str, height: int = 4):
        row = Frame(parent)
        row.pack(fill="both", expand=True, pady=4)
        label_widget = Label(row, text=label, width=18, anchor="nw")
        label_widget.pack(side="left")
        text_widget = Text(row, height=height, font=("Consolas", 12), undo=True, autoseparators=True, maxundo=-1)
        text_widget.pack(side="left", fill="both", expand=True, padx=4)
        self._configure_syntax_tags(text_widget)
        text_widget.insert("1.0", entry.get(key, ""))

        def on_modified(_event=None):
            entry[key] = text_widget.get("1.0", "end-1c")
            self._after_entry_change()
        text_widget.bind("<KeyRelease>", on_modified)
        text_widget.bind("<Button-1>", lambda _e: text_widget.after_idle(text_widget.focus_set), add="+")
        self._bind_multiline_text_scroll_priority(text_widget)
        self._attach_tooltip(label_widget, self._right_help(label).get("detail", ""))
        self._add_right_inline_help(parent, label)
        if label == "Additional Signatures":
            self._bind_text_autocomplete(
                text_widget,
                self._signature_autocomplete_candidates,
                mode="signature",
                validator=self._refresh_signature_invalid_highlight,
            )
            self._refresh_signature_invalid_highlight(text_widget)
        return text_widget


    @staticmethod
    def _show_about():
        messagebox.showinfo(
            "About",
            "Model Planner 2026\n\n"
            "Features:\n"
            "• Plan Creator based editing\n"
            "• JSON/legacy plan loading\n"
            "• Auto notebook generation before run\n"
            "• Dedicated execution console\n"
            "• HuggingFace upload helper",
        )


    @staticmethod
    def _run_image_converter():
        paths = filedialog.askopenfilenames(filetypes=(("Images", "*.png *.jpg *.webp"), ("All", "*.*")))
        if not paths:
            return
        count = 0
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.convert("RGB").save(os.path.splitext(p)[0] + ".jpg", "jpeg", quality=85)
                count += 1
            except Exception:
                pass
        messagebox.showinfo("Done", f"Converted {count} images.")


    _create_menu = _method_create_menu


    def _create_layout(self):
        outer = Frame(self.root)
        outer.pack(fill="both", expand=True)
        self.outer_frame = outer

        self.top_fixed_bar = Frame(outer, padx=10, pady=8)
        self.top_fixed_bar.pack(fill="x")

        self.left_header_frame = Frame(self.top_fixed_bar)
        self.left_header_frame.pack(fill="x")
        self._build_left_fixed_header(self.left_header_frame)

        container = Frame(outer, padx=10, pady=10)
        container.pack(fill="both", expand=True)
        self.content_container = container

        self.left_outer = Frame(container)
        self.left_outer.pack(side="left", fill="both", expand=True, padx=(0, 10))

        left_body = Frame(self.left_outer)
        left_body.pack(fill="both", expand=True)
        self.left_body = left_body

        self.left_canvas = Canvas(left_body, width=600, highlightthickness=0)
        left_scrollbar = Scrollbar(left_body, orient="vertical", command=self.left_canvas.yview)
        self.left_frame = Frame(self.left_canvas)

        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        self.left_canvas.create_window((0, 0), window=self.left_frame, anchor="nw", tags=("left_inner",))
        self.left_frame.bind("<Configure>", lambda _e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")))
        self.left_canvas.bind(
            "<Configure>",
            lambda e: self.left_canvas.itemconfigure("left_inner", width=e.width),
        )
        self._install_panel_scroll_support()

        self.sep = Frame(container, width=2, bg="#cccccc")
        self.sep.pack(side="left", fill="y", padx=5)
        self.right_outer = Frame(container)
        self.right_outer.pack(side="left", fill="both", expand=True)
        self.right_frame = self.right_outer

        self._build_left_panel(self.left_frame)
        self._build_right_panel(self.right_outer)
        self._build_status_bar(outer)

        self.root.bind("<Configure>", self._on_root_resize, add="+")
        self.root.after_idle(self._update_view_switch_label)
        self.root.after_idle(self._update_responsive_layout)
        self.root.after_idle(self._apply_theme_mode)


    _build_left_panel = _method_build_left_panel


    def _build_left_fixed_header(self, parent):
        controls_col = Frame(parent)
        controls_col.pack(side="left", anchor="n", padx=(0, 10))
        self.view_switch_button = ttk.Button(controls_col, text="⇄ View", command=self._cycle_active_view)
        self.view_switch_button.pack(anchor="w", fill="x")
        self.theme_toggle_button = ttk.Button(controls_col, text="☾ Dark", command=self._toggle_theme_mode)
        self.theme_toggle_button.pack(anchor="w", fill="x", pady=(6, 0))

        title_col = Frame(parent)
        title_col.pack(side="left", anchor="w", fill="x", expand=True)
        title_row = Frame(title_col)
        title_row.pack(anchor="nw", fill="x")
        title_label = Label(title_row, text="Planner & Runner", font=("MS Gothic", 16, "bold"), anchor="w")
        title_label.pack(side="left", anchor="w")
        self.current_line_indicator_var = tk.StringVar(value="Line 1 / 1 : ")
        disk_btn = ttk.Button(title_row, text="Disk Estimate", command=lambda: self._planner_estimate_disk_runtime())
        disk_btn.pack(side="right", anchor="e", padx=(6, 0))
        current_line_label = Label(title_row, textvariable=self.current_line_indicator_var, font=("MS Gothic", 10, "bold"), anchor="e", fg="#555555")
        current_line_label.pack(side="right", anchor="e")
        self.fixed_title_label = title_label
        self.current_line_indicator_label = current_line_label
        self.disk_estimate_button = disk_btn

        self._attach_tooltip(self.view_switch_button, "Switch between split view and single-panel planner/editor views depending on window width.")
        self._attach_tooltip(self.theme_toggle_button, "Toggle light and dark appearance for the planner UI.")
        self._attach_tooltip(title_label, "Main control area for plan files, notebook execution, exports, uploads, and runtime options.")


    def _build_right_panel(self, parent):
        top_frame = Frame(parent)
        top_frame.pack(anchor='nw', fill='x', pady=(0, 8))
        title_label = Label(top_frame, text='Plan Creator', font=('MS Gothic', 16, 'bold'), fg='#225588')
        title_label.pack(side='left')
        reset_btn = ttk.Button(top_frame, text='Reset Plan', command=self._reset_plan)
        reset_btn.pack(side='right', padx=5)
        plan_versions_btn = ttk.Button(top_frame, text='Plan Versions', command=lambda: self._planner_show_versions())
        plan_versions_btn.pack(side='right', padx=(0, 5))
        self.plan_versions_button = plan_versions_btn
        self._attach_tooltip(title_label, self._right_help('Plan Creator').get('detail', ''))
        self._attach_tooltip(reset_btn, self._right_help('Reset Plan').get('detail', ''))
        self._attach_tooltip(plan_versions_btn, 'Open plan change history / experiment versions.')
        self._add_right_inline_help(parent, 'Plan Creator', padx=4, wraplength=760)

        plan_body = self._build_collapsible_section(
            parent,
            'Plan View',
            key='plan_view',
            default_open=True,
            body_fill='both',
            body_expand=False,
            padx=0,
            pady=6,
        )

        selector_frame = Frame(plan_body)
        selector_frame.pack(fill='x', pady=(0, 6))
        selector_frame.grid_columnconfigure(1, weight=1)
        target_label = Label(selector_frame, text='Target Line', width=10, anchor='w')
        target_label.grid(row=0, column=0, sticky='w', padx=(0, 6))
        self.selection_combo = ttk.Combobox(selector_frame, textvariable=self.selection_var, state='readonly', width=34)
        self._bind_combobox_mousewheel_passthrough(self.selection_combo)
        self.selection_combo.grid(row=0, column=1, sticky='ew', padx=(0, 6))
        self.selection_combo.bind('<<ComboboxSelected>>', self._on_line_selection_change)

        action_frame = Frame(selector_frame)
        action_frame.grid(row=0, column=2, sticky='e')
        self.plan_add_button = ttk.Button(action_frame, text='+', width=3, command=self._add_line)
        self.plan_add_button.pack(side='left', padx=(0, 2))
        self.plan_remove_button = ttk.Button(action_frame, text='-', width=3, command=self._remove_line)
        self.plan_remove_button.pack(side='left', padx=(0, 2))
        self.plan_move_up_button = ttk.Button(action_frame, text='↑', width=3, command=self._move_line_up)
        self.plan_move_up_button.pack(side='left', padx=(0, 2))
        self.plan_move_down_button = ttk.Button(action_frame, text='↓', width=3, command=self._move_line_down)
        self.plan_move_down_button.pack(side='left', padx=(0, 2))
        self.plan_reload_button = ttk.Button(action_frame, text='↻', width=3, command=self._load_plan_from_path)
        self.plan_reload_button.pack(side='left')

        self._attach_tooltip([target_label, self.selection_combo], self._right_help('Target Line').get('detail', ''))
        self._attach_tooltip(self.plan_add_button, self._right_help('Add Line').get('detail', ''))
        self._attach_tooltip(self.plan_remove_button, self._right_help('Remove Line').get('detail', ''))
        self._attach_tooltip(self.plan_move_up_button, self._right_help('Move Line Up').get('detail', ''))
        self._attach_tooltip(self.plan_move_down_button, self._right_help('Move Line Down').get('detail', ''))
        self._attach_tooltip(self.plan_reload_button, self._right_help('Reload Plan File').get('detail', ''))

        search_row = Frame(plan_body)
        search_row.pack(fill='x', pady=(0, 4))
        Label(search_row, text='Search', width=8, anchor='w').pack(side='left')
        search_entry = Entry(search_row, textvariable=self.plan_search_var)
        search_entry.pack(side='left', fill='x', expand=True, padx=(0, 6))
        search_entry.bind('<KeyRelease>', lambda _e: self._refresh_line_selector())
        Label(search_row, text='Filter', width=8, anchor='w').pack(side='left')
        filter_values = ['All'] + LINE_TYPES
        filter_combo = ttk.Combobox(search_row, textvariable=self.plan_filter_var, values=filter_values, state='readonly', width=18)
        self._bind_combobox_mousewheel_passthrough(filter_combo)
        filter_combo.pack(side='left')
        filter_combo.bind('<<ComboboxSelected>>', lambda _e: self._refresh_line_selector())

        button_grid = Frame(plan_body)
        button_grid.pack(fill='x', pady=(0, 4))
        for col in range(4):
            button_grid.grid_columnconfigure(col, weight=1)
        action_specs = [
            ('Add Below', self._add_line, 'Ctrl(Command)+Shift+;'),
            ('Remove', self._remove_line, 'Delete / Backspace'),
            ('Optimize Plan', self._planner_optimize_plan_full, ''),
            ('Shortcuts', self._planner_show_shortcut_settings, ''),
            ('Diff', self._show_plan_diff_preview, 'Ctrl(Command)+Shift+:'),
            ('Validate', self._show_prevalidation, 'Ctrl(Command)+Shift+V'),
            ('Save Preset', self._save_preset_json, 'Ctrl(Command)+Alt(Option)+S'),
            ('Load Preset', self._load_preset_json, 'Ctrl(Command)+Alt(Option)+L'),
        ]
        for idx, (label, cmd, shortcut) in enumerate(action_specs):
            btn = ttk.Button(button_grid, text=label, command=cmd)
            btn.grid(row=idx // 4, column=idx % 4, sticky='ew', padx=3, pady=3)
            self._attach_tooltip(btn, f'{label}\nShortcut: {shortcut}')

        info_row = Frame(plan_body)
        info_row.pack(fill='x', pady=(0, 4))
        deps_label = Label(info_row, textvariable=self.plan_deps_var, anchor='w', fg='#555555', cursor='hand2')
        deps_label.pack(side='left', fill='x', expand=True)
        deps_label.bind('<Button-1>', lambda _e: self._planner_show_dependency_graph(), add='+')
        self._attach_tooltip(deps_label, 'Dependency summary. Click to open the interactive dependency graph.\nOther shortcuts: Undo Ctrl+Z, Redo Ctrl+Shift+Z, Copy Ctrl+C, Paste Ctrl+V, Duplicate Ctrl+D, Delete Backspace/Delete, Reorder Shift+Up/Down or drag the selected lines.')
        unused_label = Label(info_row, textvariable=self.plan_unused_var, anchor='e', fg='#555555', cursor='hand2')
        unused_label.pack(side='right', padx=(8, 0))
        unused_label.bind('<Button-1>', lambda _e: self._show_unused_models(), add='+')
        self._attach_tooltip(unused_label, 'Unused model summary. Click to open the detailed unused-model report.')

        list_row = Frame(plan_body, height=220)
        list_row.pack(fill='both', expand=False)
        list_row.pack_propagate(False)
        self.plan_listbox = tk.Listbox(
            list_row,
            selectmode=tk.EXTENDED,
            height=10,
            exportselection=False,
            activestyle='none',
            bg='#fbfbfb',
            selectbackground='#3567d6',
            selectforeground='#ffffff',
        )
        plan_scroll = Scrollbar(list_row, orient='vertical', command=self.plan_listbox.yview)
        self.plan_listbox.configure(yscrollcommand=plan_scroll.set)
        self.plan_listbox.pack(side='left', fill='both', expand=True)
        plan_scroll.pack(side='right', fill='y')
        self.plan_listbox.bind('<<ListboxSelect>>', self._on_plan_list_select)
        self.plan_listbox.bind('<ButtonPress-1>', self._plan_list_drag_start, add='+')
        self.plan_listbox.bind('<ButtonRelease-1>', self._plan_list_drag_end, add='+')
        self.plan_listbox.bind('<Double-Button-1>', self._on_plan_list_select, add='+')
        self.plan_listbox.bind('<Shift-Up>', self._shortcut_move_lines_up, add='+')
        self.plan_listbox.bind('<Shift-Down>', self._shortcut_move_lines_down, add='+')
        self.plan_listbox.bind('<Motion>', self._on_plan_list_motion, add='+')
        self.plan_listbox.bind('<Leave>', self._hide_plan_item_hover, add='+')
        self.plan_listbox.bind('<ButtonPress-2>', self._show_plan_context_menu, add='+')
        self.plan_listbox.bind('<ButtonPress-3>', self._show_plan_context_menu, add='+')
        self.plan_listbox.bind('<Control-Button-1>', self._show_plan_context_menu, add='+')
        self.plan_listbox.bind('<Button-1>', self._on_plan_help_click, add='+')

        summary = Label(plan_body, textvariable=self.plan_summary_var, anchor='w', fg='#666666')
        summary.pack(fill='x', pady=(4, 0))

        self.canvas_container = Frame(parent)
        self.canvas_container.pack(fill='both', expand=True, pady=5)
        self.canvas = Canvas(self.canvas_container, highlightthickness=0)
        self.scrollbar = Scrollbar(self.canvas_container, orient='vertical', command=self.canvas.yview)
        self.scroll_frame = Frame(self.canvas)
        self.scroll_frame.bind('<Configure>', lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor='nw', tags=('right_inner',))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure('right_inner', width=e.width))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.pack(side='right', fill='y')


    def _is_compact_layout(self) -> bool:
        try:
            return int(self.root.winfo_width() or 0) < int(getattr(self, "compact_breakpoint", 1180))
        except Exception:
            return False


    def _set_active_view(self, view: str):
        allowed = {"split", "left", "right"}
        self.active_view = view if view in allowed else "split"
        self.config["active_view"] = self.active_view
        self._update_view_switch_label()
        self._update_responsive_layout()
        self._schedule_config_save()


    def _cycle_active_view(self):
        current = getattr(self, "active_view", "split")
        if self._is_compact_layout():
            cycle = ["left", "right"]
            current = current if current in cycle else "left"
            next_view = cycle[(cycle.index(current) + 1) % len(cycle)]
            self._set_active_view(next_view)
            return
        cycle = ["split", "left", "right"]
        current = current if current in cycle else "split"
        next_view = cycle[(cycle.index(current) + 1) % len(cycle)]
        self._set_active_view(next_view)


    def _update_view_switch_label(self):
        btn = getattr(self, "view_switch_button", None)
        if btn is None:
            return
        current = getattr(self, "active_view", "split")
        if self._is_compact_layout():
            label = "⇄ View: Planner" if current != "right" else "⇄ View: Editor"
        else:
            label = {
                "split": "⇄ View: Split",
                "left": "⇄ View: Planner",
                "right": "⇄ View: Editor",
            }.get(current, "⇄ View")
        try:
            btn.configure(text=label)
        except Exception:
            pass


    def _update_responsive_layout(self):
        compact = self._is_compact_layout()
        current = getattr(self, "active_view", "split")
        view = "left" if compact and current == "split" else current
        layout_state = (compact, view)
        left_outer = getattr(self, "left_outer", None)
        right_outer = getattr(self, "right_outer", None)
        sep = getattr(self, "sep", None)
        if left_outer is None or right_outer is None:
            return
        if layout_state == getattr(self, "_last_responsive_layout_state", None):
            self._update_view_switch_label()
            return
        self._last_responsive_layout_state = layout_state
        for widget in (left_outer, sep, right_outer):
            if widget is None:
                continue
            try:
                widget.pack_forget()
            except Exception:
                pass
        if compact:
            if view == "right":
                right_outer.pack(side="left", fill="both", expand=True)
            else:
                left_outer.pack(side="left", fill="both", expand=True)
        else:
            if view == "left":
                left_outer.pack(side="left", fill="both", expand=True)
            elif view == "right":
                right_outer.pack(side="left", fill="both", expand=True)
            else:
                left_outer.pack(side="left", fill="both", expand=True, padx=(0, 10))
                if sep is not None:
                    sep.pack(side="left", fill="y", padx=5)
                right_outer.pack(side="left", fill="both", expand=True)
        self._update_view_switch_label()


    def _schedule_responsive_layout(self):
        if getattr(self, "_responsive_after_id", None):
            try:
                self.root.after_cancel(self._responsive_after_id)
            except Exception:
                pass
        self._responsive_after_id = self.root.after(80, self._run_responsive_layout_update)


    def _run_responsive_layout_update(self):
        self._responsive_after_id = None
        self._update_responsive_layout()


    _build_combo_row = _method_build_combo_row


    def _planner_build_fixed_status_bar(self):
        from .widgets import ProgressWindow

        try:
            old = self.status_label.master if getattr(self, 'status_label', None) is not None else None
            if old is not None and old.winfo_exists():
                old.destroy()
        except Exception:
            pass
        status_frame = Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_frame.pack(side="bottom", fill="x")
        status_frame.grid_columnconfigure(0, weight=1)
        self.status_label = Label(status_frame, text="Status: Idle", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(8, 10), pady=4)
        self.progress_bar = ttk.Progressbar(status_frame, mode="indeterminate", length=260)
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=4)
        self._attach_tooltip([status_frame, self.status_label, self.progress_bar], self._left_help("Status").get("detail", ""))
        self.progress_indicator = ProgressWindow(self.status_label, self.progress_bar)


    def _sync_target_line_combo(self):
        combo = getattr(self, 'selection_combo', None)
        var = getattr(self, 'selection_var', None)
        if combo is None or var is None:
            return
        entries = self.plan_data.get('entries', [])
        values = [f"{idx + 1:02d} • {self._line_summary(entry)}" for idx, entry in enumerate(entries)]
        try:
            combo['values'] = values
        except Exception:
            pass
        if values:
            current_idx = max(0, min(int(getattr(self, 'current_index', 0) or 0), len(values) - 1))
            self.current_index = current_idx
            var.set(values[current_idx])
        else:
            var.set('')


    def _build_entry_row(self, parent, label: str, entry: Dict[str, Any], key: str, width: int = 52, browse: bool = False):
        row = Frame(parent)
        row.pack(fill="x", pady=3)
        label_widget = Label(row, text=label, width=18, anchor="w")
        label_widget.pack(side="left")
        var = tk.StringVar(value=str(entry.get(key, "")))
        ent = Entry(row, textvariable=var, width=width)
        ent.pack(side="left", fill="x", expand=True, padx=4)

        def sync(*_args):
            entry[key] = var.get()
            self._after_entry_change()
        var.trace_add("write", sync)

        widgets = [row, label_widget, ent]
        if browse:
            btn = ttk.Button(row, text="📂", width=3, command=lambda: self._browse_local_path(var))
            btn.pack(side="left")
            widgets.append(btn)
        self._attach_tooltip(label_widget, self._right_help(label).get("detail", ""))
        self._add_right_inline_help(parent, label)
        return var
