"""Widgets responsibilities."""
from __future__ import annotations

from tkinter import Frame
from tkinter import Label
import time
import tkinter as tk


class ProgressWindow:
    def __init__(self, status_label, progress_bar):
        self.status_label = status_label
        self.progress_bar = progress_bar
        self.start_time = None
        self.total_items = 0
        self.current_item = 0
        self.is_active = False


    def start(self, total_items, operation_name="Operation"):
        self.start_time = time.time()
        self.total_items = max(1, total_items)
        self.current_item = 0
        self.is_active = True
        self.operation_name = operation_name
        self.progress_bar.start()
        self._update_status()


    def update(self, current_item, current_name=""):
        self.current_item = current_item
        if self.is_active:
            self._update_status(current_name)


    def _update_status(self, current_name=""):
        if not self.is_active:
            return
        elapsed = time.time() - (self.start_time or time.time())
        if self.current_item > 0:
            eta = (elapsed / self.current_item) * max(0, self.total_items - self.current_item)
            eta_str = time.strftime("%M:%S", time.gmtime(eta))
            percent = (self.current_item / self.total_items) * 100
            status = f"{self.operation_name}: {self.current_item}/{self.total_items} ({percent:.0f}%) ETA: {eta_str}"
            if current_name:
                status += f" - {current_name}"
            self.status_label.config(text=status)
        else:
            self.status_label.config(text=f"{self.operation_name}: Starting...")


    def finish(self, success=True):
        self.is_active = False
        self.progress_bar.stop()
        if success:
            elapsed = time.time() - (self.start_time or time.time())
            self.status_label.config(text=f"✅ Completed in {time.strftime('%M:%S', time.gmtime(elapsed))}")
        else:
            self.status_label.config(text="❌ Operation failed")



class Tooltip:
    def __init__(self, widget, text: str, *, delay: int = 700, wraplength: int = 420):
        self.widget = widget
        self.text = str(text or "").strip()
        self.delay = delay
        self.wraplength = wraplength
        self.tipwindow: tk.Toplevel | None = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")


    def _schedule(self, _event=None):
        self._cancel()
        if not self.text:
            return
        self._after_id = self.widget.after(self.delay, self._show)


    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None


    def _theme_colors(self) -> dict:
        try:
            top = self.widget.winfo_toplevel()
            colors = getattr(top, "_planner_theme_colors", None)
            if isinstance(colors, dict):
                return colors
        except Exception:
            pass
        return {
            "surface": "#131a30",
            "panel": "#0f1528",
            "text": "#eef2ff",
            "muted": "#b7c2ea",
            "border": "#2a355d",
        }


    def _show(self):
        self._cancel()
        if self.tipwindow is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        except Exception:
            return
        colors = self._theme_colors()
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        try:
            tip.configure(bg=colors.get("panel", "#0f1528"))
        except Exception:
            pass
        card = Frame(
            tip,
            bg=colors.get("surface", "#131a30"),
            highlightthickness=1,
            highlightbackground=colors.get("border", "#2a355d"),
            padx=2,
            pady=2,
        )
        card.pack(fill="both", expand=True)
        label = Label(
            card,
            text=self.text,
            justify="left",
            anchor="w",
            wraplength=max(int(self.wraplength or 0), 760),
            bg=colors.get("surface", "#131a30"),
            fg=colors.get("text", "#eef2ff"),
            padx=14,
            pady=12,
            font=("MS Gothic", 11),
        )
        label.pack(fill="both", expand=True)
        self.tipwindow = tip


    def _hide(self, _event=None):
        self._cancel()
        if self.tipwindow is not None:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
            self.tipwindow = None



class _PlannerDynamicTooltip(Tooltip):
    def __init__(self, widget, text_provider, *, delay: int = 700, wraplength: int = 760):
        self._text_provider = text_provider
        super().__init__(widget, "", delay=delay, wraplength=wraplength)


    def _schedule(self, _event=None):
        try:
            text = self._text_provider() if callable(self._text_provider) else str(self._text_provider or "")
        except Exception:
            text = ""
        self.text = str(text or "").strip()
        if not self.text:
            return
        return super()._schedule(_event)
