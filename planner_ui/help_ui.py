"""Help ui controls and callbacks."""
from __future__ import annotations

from .editor_support import _on_plan_help_click_sigless as _method_on_plan_help_click
from .help_support import _build_plan_item_tooltip_text_with_lora_merge as _method_build_plan_item_tooltip_text
from .help_support import _planner_attach_combobox_dropdown_tooltips as _method_planner_attach_combobox_dropdown_tooltips
from .help_support import _planner_attach_dynamic_tooltip as _method_planner_attach_dynamic_tooltip
from .help_support import _planner_choice_dropdown_item_tooltip as _method_planner_choice_dropdown_item_tooltip
from .help_support import _planner_hide_dropdown_tooltip as _method_planner_hide_dropdown_tooltip
from .help_support import _planner_line_type_tooltip_text as _method_planner_line_type_tooltip_text
from .help_support import _planner_mode_choice_tooltip_compact as _method_planner_mode_choice_tooltip
from .help_support import _planner_mode_dropdown_item_tooltip as _method_planner_mode_dropdown_item_tooltip
from .help_support import _planner_model_choice_tooltip_text as _method_planner_model_choice_tooltip_text
from .help_support import _planner_option_tooltip_text as _method_planner_option_tooltip_text
from .help_support import _planner_show_dropdown_tooltip as _method_planner_show_dropdown_tooltip
from .settings import LEFT_PANEL_FIELD_HELP
from .settings import RIGHT_PANEL_FIELD_HELP
from .tools_support import _planner_attach_merge_mode_help_final as _method_planner_attach_merge_mode_help_final
from tkinter import Frame
from tkinter import Label
from tkinter import Text
from typing import Dict
from typing import List
import re
import tkinter as tk
import webbrowser


class HelpUiMixin:
    """Help ui behavior for the Planner."""

    _on_plan_help_click = _method_on_plan_help_click


    _planner_attach_dynamic_tooltip = _method_planner_attach_dynamic_tooltip


    _planner_option_tooltip_text = _method_planner_option_tooltip_text


    _planner_model_choice_tooltip_text = _method_planner_model_choice_tooltip_text


    _planner_mode_choice_tooltip = _method_planner_mode_choice_tooltip


    _planner_mode_dropdown_item_tooltip = _method_planner_mode_dropdown_item_tooltip


    _planner_line_type_tooltip_text = _method_planner_line_type_tooltip_text


    _planner_choice_dropdown_item_tooltip = _method_planner_choice_dropdown_item_tooltip


    _planner_hide_dropdown_tooltip = _method_planner_hide_dropdown_tooltip


    _planner_show_dropdown_tooltip = _method_planner_show_dropdown_tooltip


    _planner_attach_combobox_dropdown_tooltips = _method_planner_attach_combobox_dropdown_tooltips


    _planner_attach_merge_mode_help_final = _method_planner_attach_merge_mode_help_final


    def _extract_elemental_hover_token(self, widget: Text, event) -> tuple[str, str, str, str]:
        try:
            index = widget.index(f"@{event.x},{event.y}")
            line_idx, col_idx = map(int, index.split("."))
            line_text = widget.get(f"{line_idx}.0", f"{line_idx}.end")
        except Exception:
            return "", "", "", ""
        token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")
        for match in token_re.finditer(line_text):
            if match.start() <= col_idx <= match.end():
                return match.group(0), f"{line_idx}.{match.start()}", f"{line_idx}.{match.end()}", index
        return "", "", "", ""


    def _hide_elemental_alias_hover(self, _event=None):
        win = getattr(self, "_elemental_alias_hover_popup", None)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
        self._elemental_alias_hover_popup = None
        self._elemental_alias_hover_key = None


    def _show_elemental_alias_hover_popup(self, widget: Text, alias_entries: List[Dict[str, str]], event):
        if not alias_entries:
            self._hide_elemental_alias_hover()
            return
        alias_name = str(alias_entries[0].get("alias") or "").strip()
        entries_key = tuple(sorted((str(item.get("canonical") or "").lower(), str(item.get("effect") or "")) for item in alias_entries))
        key = (alias_name.lower(), entries_key)
        if getattr(self, "_elemental_alias_hover_key", None) == key and getattr(self, "_elemental_alias_hover_popup", None) is not None:
            return
        self._hide_elemental_alias_hover()
        try:
            x = int(getattr(event, "x_root", 0) or 0) + 16
            y = int(getattr(event, "y_root", 0) or 0) + 18
        except Exception:
            return
        popup = tk.Toplevel(widget)
        popup.wm_overrideredirect(True)
        popup.wm_geometry(f"+{x}+{y}")
        try:
            colors = self._theme_colors()
            bg_panel = colors.get("panel", "#0f1528")
            bg_surface = colors.get("surface", "#131a30")
            fg_text = colors.get("text", "#eef2ff")
            fg_muted = colors.get("muted", "#b7c2ea")
            border = colors.get("border", "#2a355d")
        except Exception:
            bg_panel = "#0f1528"
            bg_surface = "#131a30"
            fg_text = "#eef2ff"
            fg_muted = "#b7c2ea"
            border = "#2a355d"
        try:
            popup.configure(bg=bg_panel)
        except Exception:
            pass
        card = Frame(popup, bg=bg_surface, highlightthickness=1, highlightbackground=border, padx=10, pady=8)
        card.pack(fill="both", expand=True)
        Label(card, text=alias_name, anchor="w", justify="left", font=("MS Gothic", 11, "bold"), bg="#252b45", fg="#f4f7ff").pack(fill="x", anchor="w")
        Label(card, text="Alias target elements", anchor="w", justify="left", font=("MS Gothic", 9), bg="#252b45", fg="#c6cff5").pack(fill="x", anchor="w", pady=(2, 6))
        for item in alias_entries:
            canonical = str(item.get("canonical") or "").strip()
            effect = str(item.get("effect") or "").strip()
            row = Frame(card, bg="#252b45")
            row.pack(fill="x", anchor="w", pady=(0, 4))
            Label(row, text=canonical or "?", anchor="w", justify="left", font=("MS Gothic", 10, "bold"), bg="#252b45", fg="#f4f7ff").pack(fill="x", anchor="w")
            if effect:
                Label(row, text=effect, anchor="w", justify="left", wraplength=420, font=("MS Gothic", 9), bg="#252b45", fg="#c6cff5").pack(fill="x", anchor="w")
        self._elemental_alias_hover_popup = popup
        self._elemental_alias_hover_key = key


    def _hide_plan_item_hover(self, _event=None):
        tip = getattr(self, "_plan_hover_tip", None)
        if tip is not None:
            try:
                tip.destroy()
            except Exception:
                pass
        self._plan_hover_tip = None
        self._plan_hover_index = None
        self._plan_hover_text = ""


    def _show_plan_item_hover(self, event, text: str):
        self._hide_plan_item_hover()
        if not text:
            return
        colors = self._theme_colors()
        tip = tk.Toplevel(self.root)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        x = int(getattr(event, "x_root", 0) or 0) + 14
        y = int(getattr(event, "y_root", 0) or 0) + 14
        tip.wm_geometry(f"+{x}+{y}")
        try:
            tip.configure(bg=colors["panel"])
        except Exception:
            pass
        card = Frame(
            tip,
            bg=colors["surface"],
            highlightthickness=1,
            highlightbackground=colors["border"],
            padx=2,
            pady=2,
        )
        card.pack(fill="both", expand=True)
        body = Label(
            card,
            text=text,
            justify="left",
            anchor="w",
            wraplength=680,
            bg=colors["surface"],
            fg=colors["text"],
            padx=14,
            pady=12,
            font=("MS Gothic", 11),
        )
        body.pack(fill="both", expand=True)
        self._plan_hover_tip = tip


    def _left_help(self, key: str) -> Dict[str, str]:
        return LEFT_PANEL_FIELD_HELP.get(key, {"short": "", "detail": ""})


    def _attach_tooltip(self, widgets, detail: str):
        from .widgets import Tooltip

        if not detail:
            return
        if not isinstance(widgets, (list, tuple, set)):
            widgets = [widgets]
        for widget in widgets:
            if widget is not None:
                Tooltip(widget, detail)


    def _open_help_link(self, event, url: str):
        if not url or not self._event_has_open_modifier(event):
            return None
        try:
            webbrowser.open_new_tab(url)
            self.status_label.config(text=f"Opened link: {url}")
        except Exception as e:
            self._show_detailed_error("Open Link Error", e, context=f"URL: {url}", show_messagebox=False)
        return "break"


    def _add_inline_help(self, parent, key: str, *, padx: int = 4, pady=(1, 4), wraplength: int = 520):
        help_info = self._left_help(key)
        short = help_info.get("short", "").strip()
        if not short:
            return None
        link = str(help_info.get("link", "") or "").strip()
        fg = "#1f5fbf" if link else "#666666"
        cursor = "hand2" if link else ""
        lbl = Label(parent, text=short, fg=fg, justify="left", wraplength=wraplength, anchor="w", cursor=cursor)
        if link:
            lbl.bind("<Button-1>", lambda event, url=link: self._open_help_link(event, url), add="+")
        lbl.pack(anchor="w", padx=padx, pady=pady, fill="x")
        return lbl


    def _right_help(self, key: str) -> Dict[str, str]:
        normalized = re.sub(r"\s+\d+$", "", str(key or "").strip())
        return RIGHT_PANEL_FIELD_HELP.get(normalized, {"short": "", "detail": ""})


    def _add_right_inline_help(self, parent, key: str, *, padx: int = 22, pady=(1, 4), wraplength: int = 760):
        help_info = self._right_help(key)
        short = help_info.get("short", "").strip()
        if not short:
            return None
        lbl = Label(parent, text=short, fg="#666666", justify="left", wraplength=wraplength, anchor="w")
        lbl.pack(anchor="w", padx=padx, pady=pady, fill="x")
        return lbl


    _build_plan_item_tooltip_text = _method_build_plan_item_tooltip_text
