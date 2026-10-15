"""Appearance controls and callbacks."""
from __future__ import annotations

from .editor_support import _entry_type_color_with_lora_merge as _method_entry_type_color
from .tools_support import _apply_plan_listbox_item_styles_final as _method_apply_plan_listbox_item_styles
from .tools_support import _entry_row_color_final as _method_entry_row_color
from .tools_support import _show_row_color_dialog_final as _method_show_row_color_dialog
from tkinter import ttk


class AppearanceMixin:
    """Appearance behavior for the Planner."""

    _show_row_color_dialog = _method_show_row_color_dialog


    def _planner_disable_widget_subtree(self, widget, exempt_roots):
        if widget is None:
            return
        for root in exempt_roots:
            if widget is root or self._planner_widget_is_descendant(widget, root):
                return
        try:
            cls = widget.winfo_class()
        except Exception:
            cls = ''
        try:
            if cls in {'Entry', 'TEntry'}:
                widget.configure(state='disabled')
            elif cls == 'Text':
                widget.configure(state='disabled')
            elif cls == 'TCombobox':
                widget.configure(state='disabled')
            elif cls in {'Button', 'TButton', 'Checkbutton', 'TCheckbutton', 'Radiobutton', 'TRadiobutton'}:
                widget.configure(state='disabled')
        except Exception:
            pass
        try:
            children = widget.winfo_children()
        except Exception:
            children = []
        for child in children:
            self._planner_disable_widget_subtree(child, exempt_roots)


    _entry_type_color = _method_entry_type_color


    _apply_plan_listbox_item_styles = _method_apply_plan_listbox_item_styles


    def _toggle_theme_mode(self):
        self.theme_mode = "light" if str(getattr(self, "theme_mode", "dark") or "dark").lower() == "dark" else "dark"
        self.config["theme_mode"] = self.theme_mode
        self._apply_theme_mode()
        self._schedule_config_save()


    def _theme_colors(self):
        if str(getattr(self, "theme_mode", "dark") or "dark").lower() == "light":
            return {
                "bg": "#edf1fb",
                "panel": "#f6f8fe",
                "surface": "#eef2fb",
                "subtle": "#e3e9f8",
                "text": "#1c2340",
                "muted": "#66739a",
                "accent": "#5f63d9",
                "accent_text": "#ffffff",
                "entry_bg": "#ffffff",
                "entry_fg": "#17203a",
                "canvas": "#f2f5fd",
                "border": "#c9d3ee",
                "button_bg": "#e7ecfb",
                "button_fg": "#25315b",
                "button_hover": "#dce4fb",
                "button_pressed": "#ced9f7",
                "primary_bg": "#5f63d9",
                "primary_hover": "#555bcc",
                "danger_bg": "#f0dbe5",
                "danger_hover": "#e4c9d6",
                "danger_fg": "#5f2940",
                "select_bg": "#5b67dd",
                "select_fg": "#ffffff",
                "scrollbar_bg": "#bcc9ea",
                "scrollbar_trough": "#e7ecfb",
                "scrollbar_active": "#98a9db",
            }
        return {
            "bg": "#070b14",
            "panel": "#0d1222",
            "surface": "#131a30",
            "subtle": "#19213b",
            "text": "#edf2ff",
            "muted": "#97a6d2",
            "accent": "#6c73ff",
            "accent_text": "#ffffff",
            "entry_bg": "#0a1020",
            "entry_fg": "#f3f6ff",
            "canvas": "#080d18",
            "border": "#263050",
            "button_bg": "#18203b",
            "button_fg": "#edf2ff",
            "button_hover": "#212b4c",
            "button_pressed": "#2a345c",
            "primary_bg": "#5964ea",
            "primary_hover": "#515ddb",
            "danger_bg": "#4b2437",
            "danger_hover": "#623048",
            "danger_fg": "#ffeaf2",
            "select_bg": "#4250d8",
            "select_fg": "#ffffff",
            "scrollbar_bg": "#2a355d",
            "scrollbar_trough": "#0a1020",
            "scrollbar_active": "#3a4a82",
        }


    def _button_style_for_text(self, text: str) -> str:
        label = str(text or "").strip().lower()
        if not label:
            return "TButton"
        if "theme" in label or "view:" in label or label.startswith("⇄ view"):
            return "Toolbar.TButton"
        if label in {"-", "reset plan"} or "remove" in label or "delete" in label or label.startswith("■ stop"):
            return "Danger.TButton"
        if label in {"+", "new", "load", "add"} or "run merge" in label or "upload latest" in label:
            return "Primary.TButton"
        if "export" in label or "show console" in label or "save" in label or "reload" in label or label in {"↑", "↓"}:
            return "Muted.TButton"
        return "TButton"


    def _apply_theme_mode(self):
        colors = self._theme_colors()
        try:
            self.root.configure(bg=colors["bg"])
            self.root._planner_theme_colors = colors
            self.root._planner_theme_mode = str(getattr(self, "theme_mode", "dark") or "dark")
        except Exception:
            pass
        try:
            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            style.configure(".", background=colors["panel"], foreground=colors["text"])
            style.configure("TFrame", background=colors["panel"])
            style.configure("TLabel", background=colors["panel"], foreground=colors["text"])
            style.configure("TLabelframe", background=colors["panel"], bordercolor=colors["border"], relief="solid")
            style.configure("TLabelframe.Label", background=colors["panel"], foreground=colors["text"])
            style.configure("TCheckbutton", background=colors["panel"], foreground=colors["text"])
            style.map("TCheckbutton", background=[("active", colors["panel"])], foreground=[("active", colors["text"]), ("selected", colors["text"])])
            style.configure("TRadiobutton", background=colors["panel"], foreground=colors["text"])
            style.map("TRadiobutton", background=[("active", colors["panel"])], foreground=[("active", colors["text"]), ("selected", colors["text"])])
            style.configure("TCombobox", fieldbackground=colors["entry_bg"], background=colors["entry_bg"], foreground=colors["entry_fg"], arrowcolor=colors["muted"], bordercolor=colors["border"], padding=4)
            style.map("TCombobox", fieldbackground=[("readonly", colors["entry_bg"])], selectbackground=[("readonly", colors["select_bg"])], selectforeground=[("readonly", colors["select_fg"])], foreground=[("readonly", colors["entry_fg"])], background=[("active", colors["entry_bg"])])
            style.configure("TButton", background=colors["button_bg"], foreground=colors["button_fg"], bordercolor=colors["border"], focusthickness=1, focuscolor=colors["accent"], padding=(10, 6), relief="flat")
            style.map("TButton", background=[("pressed", colors["button_pressed"]), ("active", colors["button_hover"])], foreground=[("pressed", colors["button_fg"]), ("active", colors["button_fg"])], bordercolor=[("active", colors["accent"]), ("pressed", colors["accent"])])
            style.configure("Primary.TButton", background=colors["primary_bg"], foreground=colors["accent_text"], bordercolor=colors["primary_bg"], focusthickness=1, focuscolor=colors["accent"], padding=(10, 6), relief="flat")
            style.map("Primary.TButton", background=[("pressed", colors["accent"]), ("active", colors["primary_hover"])], foreground=[("pressed", colors["accent_text"]), ("active", colors["accent_text"])], bordercolor=[("active", colors["primary_hover"]), ("pressed", colors["accent"])])
            style.configure("Muted.TButton", background=colors["surface"], foreground=colors["text"], bordercolor=colors["border"], focusthickness=1, focuscolor=colors["accent"], padding=(10, 6), relief="flat")
            style.map("Muted.TButton", background=[("pressed", colors["button_pressed"]), ("active", colors["button_hover"])], foreground=[("pressed", colors["text"]), ("active", colors["text"])], bordercolor=[("active", colors["accent"]), ("pressed", colors["accent"])])
            style.configure("Danger.TButton", background=colors["danger_bg"], foreground=colors["danger_fg"], bordercolor=colors["danger_bg"], focusthickness=1, focuscolor=colors["accent"], padding=(10, 6), relief="flat")
            style.map("Danger.TButton", background=[("pressed", colors["danger_hover"]), ("active", colors["danger_hover"])], foreground=[("pressed", colors["danger_fg"]), ("active", colors["danger_fg"])], bordercolor=[("active", colors["danger_hover"]), ("pressed", colors["danger_hover"])])
            style.configure("Toolbar.TButton", background=colors["surface"], foreground=colors["accent"], bordercolor=colors["border"], focusthickness=1, focuscolor=colors["accent"], padding=(10, 6), relief="flat")
            style.map("Toolbar.TButton", background=[("pressed", colors["button_pressed"]), ("active", colors["button_hover"])], foreground=[("pressed", colors["accent"]), ("active", colors["accent"])], bordercolor=[("active", colors["accent"]), ("pressed", colors["accent"])])
            style.configure("Menu.TButton", background=colors["entry_bg"], foreground=colors["entry_fg"], bordercolor=colors["border"], focusthickness=1, focuscolor=colors["accent"], padding=(12, 8), relief="flat")
            style.map("Menu.TButton", background=[("disabled", colors["subtle"]), ("pressed", colors["button_pressed"]), ("active", colors["select_bg"])], foreground=[("disabled", colors["muted"]), ("pressed", colors["entry_fg"]), ("active", colors["select_fg"])], bordercolor=[("disabled", colors["border"]), ("active", colors["accent"]), ("pressed", colors["accent"])])
            style.configure("MenuDanger.TButton", background=colors["danger_bg"], foreground=colors["danger_fg"], bordercolor=colors["danger_bg"], focusthickness=1, focuscolor=colors["accent"], padding=(12, 8), relief="flat")
            style.map("MenuDanger.TButton", background=[("disabled", colors["subtle"]), ("pressed", colors["danger_hover"]), ("active", colors["danger_hover"])], foreground=[("disabled", colors["muted"]), ("pressed", colors["danger_fg"]), ("active", colors["danger_fg"])], bordercolor=[("disabled", colors["border"]), ("active", colors["danger_hover"]), ("pressed", colors["danger_hover"])])
        except Exception:
            pass
        btn = getattr(self, "theme_toggle_button", None)
        if btn is not None:
            try:
                btn.configure(text=("☀ Light" if str(getattr(self, "theme_mode", "dark") or "dark").lower() == "dark" else "☾ Dark"), style="Toolbar.TButton")
            except Exception:
                pass
        view_btn = getattr(self, "view_switch_button", None)
        if view_btn is not None:
            try:
                view_btn.configure(style="Toolbar.TButton")
            except Exception:
                pass
        for widget in self.root.winfo_children():
            self._apply_theme_to_children(widget, colors)
        self._apply_scrollbar_palette(colors)
        try:
            self._style_autocomplete_popup(colors)
        except Exception:
            pass
        try:
            tip = getattr(self, "_plan_hover_tip", None)
            if tip is not None and tip.winfo_exists():
                self._hide_plan_item_hover()
        except Exception:
            pass
        try:
            self.console.apply_theme()
        except Exception:
            pass


    def _apply_theme_to_children(self, widget, colors):
        try:
            cls = widget.winfo_class()
        except Exception:
            cls = ""
        try:
            if cls in {"Frame", "Labelframe", "LabelFrame"}:
                bg = colors["panel"]
                try:
                    if str(widget.cget("relief")) in {"sunken", "groove", "ridge"}:
                        bg = colors["surface"]
                except Exception:
                    pass
                widget.configure(bg=bg, highlightbackground=colors["border"])
                try:
                    if cls in {"Labelframe", "LabelFrame"}:
                        widget.configure(fg=colors["text"])
                except Exception:
                    pass
            elif cls == "Label":
                fg = colors["text"]
                try:
                    current_fg = str(widget.cget("fg")).lower()
                    current_text = str(widget.cget("text") or "")
                    if current_fg in {"#666666", "gray", "grey", "#555555"}:
                        fg = colors["muted"]
                    elif current_fg in {"#225588", "#1f5fbf"} or current_text in {"Plan Creator", "Planner & Runner"}:
                        fg = colors["accent"]
                except Exception:
                    pass
                widget.configure(bg=colors["panel"], fg=fg)
            elif cls == "Entry":
                widget.configure(bg=colors["entry_bg"], fg=colors["entry_fg"], insertbackground=colors["entry_fg"], disabledbackground=colors["subtle"], disabledforeground=colors["muted"], readonlybackground=colors["subtle"], highlightbackground=colors["border"], highlightcolor=colors["accent"])
            elif cls == "Text":
                widget.configure(bg=colors["entry_bg"], fg=colors["entry_fg"], insertbackground=colors["entry_fg"], selectbackground=colors["select_bg"], selectforeground=colors["select_fg"], highlightbackground=colors["border"], highlightcolor=colors["accent"])
            elif cls == "Listbox":
                widget.configure(bg=colors["entry_bg"], fg=colors["entry_fg"], selectbackground=colors["select_bg"], selectforeground=colors["select_fg"], highlightbackground=colors["border"], highlightcolor=colors["accent"])
            elif cls == "Canvas":
                widget.configure(bg=colors["canvas"], highlightbackground=colors["border"])
            elif cls == "Scrollbar":
                self._style_scrollbar_widget(widget, colors)
            elif cls == "TButton":
                try:
                    widget.configure(style=self._button_style_for_text(widget.cget("text")))
                except Exception:
                    pass
            elif cls == "TCheckbutton":
                try:
                    widget.configure(style="TCheckbutton")
                except Exception:
                    pass
            elif cls == "TCombobox":
                try:
                    widget.configure(style="TCombobox")
                except Exception:
                    pass
        except Exception:
            pass
        for child in widget.winfo_children():
            self._apply_theme_to_children(child, colors)


    def _planner_row_color_presets(self):
        light = {
            'Default': None,
            'Blue': '#e7f0ff',
            'Green': '#e7f7ed',
            'Purple': '#f1e7ff',
            'Amber': '#fff2de',
            'Rose': '#ffe8ef',
            'Gray': '#eef1f5',
        }
        dark = {
            'Default': None,
            'Blue': '#13233c',
            'Green': '#112a1e',
            'Purple': '#24163b',
            'Amber': '#3a2812',
            'Rose': '#3a1625',
            'Gray': '#1a2230',
        }
        return light if str(getattr(self, 'theme_mode', 'dark') or 'dark').lower() == 'light' else dark


    _entry_row_color = _method_entry_row_color
