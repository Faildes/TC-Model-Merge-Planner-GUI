"""Scrolling controls and callbacks."""
from __future__ import annotations

from tkinter import Canvas
from tkinter import Frame
from tkinter import Scrollbar
from tkinter import Text
from tkinter import ttk
import tkinter as tk


class ScrollingMixin:
    """Scrolling behavior for the Planner."""

    def _apply_scrollbar_palette(self, colors):
        try:
            self.root.option_add("*Scrollbar.background", colors["scrollbar_bg"])
            self.root.option_add("*Scrollbar.troughColor", colors["scrollbar_trough"])
            self.root.option_add("*Scrollbar.activeBackground", colors["scrollbar_active"])
            self.root.option_add("*Scrollbar.highlightBackground", colors["border"])
            self.root.option_add("*Scrollbar.highlightColor", colors["border"])
            self.root.option_add("*Scrollbar.width", 12)
        except Exception:
            pass

        def walk(widget):
            if widget is None:
                return
            try:
                if widget.winfo_class() == "Scrollbar":
                    self._style_scrollbar_widget(widget, colors)
            except Exception:
                pass
            try:
                children = widget.winfo_children()
            except Exception:
                children = []
            for child in children:
                walk(child)

        walk(self.root)
        self._style_scrollbar_widget(getattr(self, "_autocomplete_scrollbar", None), colors)
        self._style_scrollbar_widget(getattr(self, "scrollbar", None), colors)


    def _canvas_contains_widget(self, canvas: Canvas | None, widget) -> bool:
        if canvas is None or widget is None:
            return False
        while widget is not None:
            if widget == canvas:
                return True
            widget = getattr(widget, "master", None)
        return False


    def _canvas_for_event(self, event):
        widget = getattr(event, "widget", None)
        if self._canvas_contains_widget(self.left_canvas, widget):
            return self.left_canvas
        if self._canvas_contains_widget(self.canvas, widget):
            return self.canvas
        try:
            target = self.root.winfo_containing(event.x_root, event.y_root)
        except Exception:
            target = None
        if self._canvas_contains_widget(self.left_canvas, target):
            return self.left_canvas
        if self._canvas_contains_widget(self.canvas, target):
            return self.canvas
        return None


    def _canvas_can_scroll(self, canvas: Canvas | None) -> bool:
        if canvas is None:
            return False
        try:
            first, last = canvas.yview()
        except Exception:
            return False
        return not (abs(first) < 1e-9 and abs(last - 1.0) < 1e-9)


    def _scroll_canvas_by_event(self, event, fallback_units: int | None = None):
        canvas = self._canvas_for_event(event)
        if canvas is None:
            return None
        if not self._canvas_can_scroll(canvas):
            return "break"
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            units = int(-1 * (delta / 120))
            if units == 0:
                units = -1 if delta > 0 else 1
        else:
            units = fallback_units or 0
        widget = getattr(self, "_active_multiline_text", None)
        if widget is not None:
            try:
                if not widget.winfo_exists():
                    self._active_multiline_text = None
                    widget = None
            except Exception:
                self._active_multiline_text = None
                widget = None
        if widget is not None:
            target = getattr(event, "widget", None)
            probe = target
            while probe is not None:
                if probe is widget:
                    return self._scroll_text_widget(widget, event)
                probe = getattr(probe, "master", None)
        if units:
            canvas.yview_scroll(units, "units")
            return "break"
        return None


    def _scroll_text_widget(self, widget: Text | None, event, fallback_units: int | None = None):
        if widget is None:
            return None
        try:
            if not widget.winfo_exists():
                self._active_multiline_text = None
                return None
        except Exception:
            self._active_multiline_text = None
            return None
        try:
            target = self.root.winfo_containing(event.x_root, event.y_root)
        except Exception:
            target = getattr(event, "widget", None)
        inside = False
        cur = target
        while cur is not None:
            if cur == widget:
                inside = True
                break
            try:
                parent_name = cur.winfo_parent()
            except Exception:
                break
            if not parent_name:
                break
            try:
                cur = cur._nametowidget(parent_name)
            except Exception:
                break
        if not inside:
            return None
        try:
            first, last = widget.yview()
        except Exception:
            return None
        delta = int(getattr(event, "delta", 0) or 0)
        if delta:
            units = int(-1 * (delta / 120))
            if units == 0:
                units = -1 if delta > 0 else 1
        else:
            units = fallback_units or 0
        if units:
            can_scroll = not (abs(first) < 1e-9 and abs(last - 1.0) < 1e-9)
            if can_scroll:
                widget.yview_scroll(units, "units")
            return "break"
        return None


    def _install_panel_scroll_support(self):
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel_up, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel_down, add="+")


    def _on_mousewheel(self, event):
        handled = self._scroll_text_widget(getattr(self, "_active_multiline_text", None), event)
        if handled == "break":
            return handled
        return self._scroll_canvas_by_event(event)


    def _on_mousewheel_up(self, event):
        handled = self._scroll_text_widget(getattr(self, "_active_multiline_text", None), event, fallback_units=-1)
        if handled == "break":
            return handled
        return self._scroll_canvas_by_event(event, fallback_units=-1)


    def _on_mousewheel_down(self, event):
        handled = self._scroll_text_widget(getattr(self, "_active_multiline_text", None), event, fallback_units=1)
        if handled == "break":
            return handled
        return self._scroll_canvas_by_event(event, fallback_units=1)


    def _show_scrollable_text_dialog(self, title: str, detail: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("1080x760+90+90")

        outer = Frame(win, padx=8, pady=8)
        outer.pack(fill="both", expand=True)

        text_frame = Frame(outer)
        text_frame.pack(fill="both", expand=True)

        yscroll = Scrollbar(text_frame, orient="vertical")
        xscroll = Scrollbar(text_frame, orient="horizontal")
        text = Text(
            text_frame,
            wrap="none",
            font=("Consolas", 11),
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
        )
        yscroll.config(command=text.yview)
        xscroll.config(command=text.xview)

        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        text.insert("1.0", detail)
        text.configure(padx=14, pady=12, spacing1=3, spacing3=3)
        text.focus_set()

        button_bar = Frame(outer)
        button_bar.pack(fill="x", pady=(8, 0))
        ttk.Button(button_bar, text="Copy All", command=lambda: (win.clipboard_clear(), win.clipboard_append(detail))).pack(side="left")
        ttk.Button(button_bar, text="Close", command=win.destroy).pack(side="right")
        try:
            colors = self._theme_colors()
            win.configure(bg=colors["bg"])
            self._apply_theme_to_children(win, colors)
        except Exception:
            pass


    def _style_scrollbar_widget(self, widget, colors):
        if widget is None:
            return
        try:
            widget.configure(
                background=colors["scrollbar_bg"],
                troughcolor=colors["scrollbar_trough"],
                activebackground=colors["scrollbar_active"],
                highlightbackground=colors["border"],
                highlightcolor=colors["border"],
                bd=0,
                relief="flat",
                activerelief="flat",
                elementborderwidth=0,
                width=12,
            )
        except Exception:
            pass


    def _bind_combobox_mousewheel_passthrough(self, combo: ttk.Combobox):
        def handler(event, fallback_units: int | None = None):
            self._scroll_canvas_by_event(event, fallback_units=fallback_units)
            return "break"
        combo.bind("<MouseWheel>", lambda e: handler(e), add="+")
        combo.bind("<Button-4>", lambda e: handler(e, -1), add="+")
        combo.bind("<Button-5>", lambda e: handler(e, 1), add="+")
        return combo


    def _bind_multiline_text_scroll_priority(self, widget: Text):
        widget.bind("<Enter>", lambda _e, w=widget: self._set_active_multiline_text(w), add="+")
        widget.bind("<Leave>", lambda _e: self._set_active_multiline_text(None), add="+")
        widget.bind("<FocusIn>", lambda _e, w=widget: self._set_active_multiline_text(w), add="+")
        widget.bind("<FocusOut>", lambda _e: self._set_active_multiline_text(None), add="+")
        widget.bind("<Button-1>", lambda _e, w=widget: widget.after_idle(lambda: self._set_active_multiline_text(w)), add="+")
