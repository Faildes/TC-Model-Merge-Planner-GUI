"""Keyboard controls and callbacks."""
from __future__ import annotations

from .shortcuts import _planner_bind_shortcuts_configurable as _method_planner_bind_shortcuts
from .shortcuts import _planner_shortcut_registry as _method_planner_shortcut_registry
from .shortcuts import _planner_show_shortcut_settings as _method_planner_show_shortcut_settings
from tkinter import ttk
import tkinter as tk


class KeyboardMixin:
    """Keyboard behavior for the Planner."""

    def _planner_widget_state(self, widget):
        if widget is None:
            return ""
        try:
            return str(widget.cget("state") or "").strip().lower()
        except Exception:
            return ""


    def _planner_is_editable_text_widget(self, widget):
        if widget is None:
            return False
        state = self._planner_widget_state(widget)
        try:
            cls = str(widget.winfo_class() or "")
        except Exception:
            cls = ""
        if isinstance(widget, tk.Text) or cls == "Text":
            return state != "disabled"
        if isinstance(widget, (tk.Entry, ttk.Entry)) or cls in {"Entry", "TEntry"}:
            return state not in {"readonly", "disabled"}
        if isinstance(widget, ttk.Combobox) or cls == "TCombobox":
            return state not in {"readonly", "disabled"}
        return False


    def _planner_widget_is_descendant(self, widget, ancestor):
        if widget is None or ancestor is None:
            return False
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            try:
                current = current.master
            except Exception:
                break
        return False


    def _planner_focus_in_plan_scope(self):
        focus = self.root.focus_get()
        if focus is None:
            return False
        return (
            focus is self.plan_listbox
            or self._planner_widget_is_descendant(focus, self.plan_listbox)
            or self._planner_widget_is_descendant(focus, getattr(self, "right_outer", None))
        )


    def _planner_is_text_focus(self):
        return self._planner_is_editable_text_widget(self.root.focus_get())


    _planner_bind_shortcuts = _method_planner_bind_shortcuts


    def _planner_register_shortcut_bindtag(self, widget):
        tag = getattr(self, '_planner_shortcut_bindtag', None)
        if widget is None or not tag:
            return
        try:
            if widget.winfo_toplevel() is not self.root:
                return
            tags = list(widget.bindtags())
        except Exception:
            return
        if tag in tags:
            if tags[0] != tag:
                tags.remove(tag)
                tags.insert(0, tag)
                try:
                    widget.bindtags(tuple(tags))
                except Exception:
                    pass
            return
        tags.insert(0, tag)
        try:
            widget.bindtags(tuple(tags))
        except Exception:
            pass


    def _planner_register_shortcut_bindtag_on_focus(self, event=None):
        widget = getattr(event, 'widget', None) if event is not None else None
        self._planner_register_shortcut_bindtag(widget)


    def _planner_register_shortcut_bindtag_on_tree(self, widget):
        if widget is None:
            return
        self._planner_register_shortcut_bindtag(widget)
        try:
            children = widget.winfo_children()
        except Exception:
            children = []
        for child in children:
            self._planner_register_shortcut_bindtag_on_tree(child)


    def _shortcut_show_diff(self, event=None):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        self._show_plan_diff_preview()
        return 'break'


    def _shortcut_validate(self, event=None):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        self._show_prevalidation()
        return 'break'


    def _shortcut_save_preset(self, event=None):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        self._save_preset_json()
        return 'break'


    def _shortcut_load_preset(self, event=None):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        self._load_preset_json()
        return 'break'


    def _set_widget_state_if_present(self, widget, state: str):
        if widget is None:
            return
        try:
            if widget.winfo_exists():
                widget.configure(state=state)
        except Exception:
            pass


    _planner_shortcut_registry = _method_planner_shortcut_registry


    _planner_show_shortcut_settings = _method_planner_show_shortcut_settings


    def _planner_focus_allows_line_shortcuts(self):
        focus = self.root.focus_get()
        if focus is None:
            return True
        if self._planner_is_editable_text_widget(focus):
            return False
        try:
            return focus.winfo_toplevel() is self.root
        except Exception:
            return False


    def _shortcut_undo(self, event=None):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        if self._history_after_id is not None:
            self._planner_commit_history_checkpoint()
        if not self._history_undo:
            return 'break'
        current = (self._planner_plan_snapshot(), self.current_index)
        payload, index = self._history_undo.pop()
        self._history_redo.append(current)
        self._last_history_snapshot = payload
        self._planner_restore_snapshot(payload, index)
        self.status_label.config(text='Undo applied')
        return 'break'


    def _shortcut_redo(self, event=None):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        if not self._history_redo:
            return 'break'
        current = (self._planner_plan_snapshot(), self.current_index)
        payload, index = self._history_redo.pop()
        self._history_undo.append(current)
        self._last_history_snapshot = payload
        self._planner_restore_snapshot(payload, index)
        self.status_label.config(text='Redo applied')
        return 'break'


    def _shortcut_copy_lines(self, event=None):
        if self.plan_listbox is None or not self._planner_focus_allows_line_shortcuts():
            return None
        self._planner_restore_plan_list_focus()
        self._copy_selected_lines()
        return 'break'


    def _shortcut_paste_lines(self, event=None):
        if self.plan_listbox is None or not self._planner_focus_allows_line_shortcuts():
            return None
        self._planner_restore_plan_list_focus()
        self._paste_copied_lines()
        return 'break'


    def _shortcut_delete_lines(self, event=None):
        if self.plan_listbox is None or not self._planner_focus_allows_line_shortcuts():
            return None
        self._planner_restore_plan_list_focus()
        self._delete_selected_lines()
        return 'break'


    def _shortcut_duplicate_lines(self, event=None):
        if self.plan_listbox is None or not self._planner_focus_allows_line_shortcuts():
            return None
        self._planner_restore_plan_list_focus()
        self._duplicate_selected_lines()
        return 'break'


    def _shortcut_add_line(self, event=None):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        self._add_line()
        return 'break'


    def _shortcut_move_lines_up(self, event=None):
        if self.plan_listbox is None:
            return None
        return self._invoke_plan_reorder_button('up')


    def _shortcut_move_lines_down(self, event=None):
        if self.plan_listbox is None:
            return None
        return self._invoke_plan_reorder_button('down')


    def _shortcut_remove_line(self, event=None):
        if self.plan_listbox is None or not self._planner_focus_allows_line_shortcuts():
            return None
        self._remove_line()
        return 'break'
