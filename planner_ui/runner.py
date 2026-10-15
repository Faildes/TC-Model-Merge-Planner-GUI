"""Runner responsibilities."""
from __future__ import annotations

from .constants import CONSOLE_BURST_POLL_MS
from .constants import CONSOLE_IDLE_POLL_MS
from .constants import CONSOLE_MAX_BATCH_CHARS
from .constants import CONSOLE_MAX_BATCH_ITEMS
from .constants import CONSOLE_MAX_BATCH_LINES
from .constants import CONSOLE_MAX_IDLE_LINES
from .constants import CONSOLE_MAX_OUTPUT_LINES
from .constants import CONSOLE_MAX_PROCESS_ROWS
from .constants import CONSOLE_NOTEBOOK_REFRESH_MS
from .constants import CONSOLE_POLL_MS
from .constants import CONSOLE_QUEUE_MAX
from PIL import Image
from PIL import ImageTk
from pathlib import Path
from tkinter import Canvas
from tkinter import Frame
from tkinter import Label
from tkinter import LabelFrame
from tkinter import Scrollbar
from tkinter import Text
from tkinter import ttk
from typing import Any
from typing import Dict
from typing import List
import base64
import io
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import tkinter as tk


class RunnerConsoleWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.win: tk.Toplevel | None = None
        self.queue: queue.Queue = queue.Queue(maxsize=CONSOLE_QUEUE_MAX)
        self.output_text: Text | None = None
        self.idle_text: Text | None = None
        self.output_canvas: Canvas | None = None
        self.output_inner: Frame | None = None
        self.state_var = tk.StringVar(value="IDLE")
        self.current_step_var = tk.StringVar(value="Waiting")
        self.progress_var = tk.StringVar(value="")
        self.progress_pct_var = tk.DoubleVar(value=0.0)
        self.operation_var = tk.StringVar(value="Idle")
        self.active_process_var = tk.StringVar(value="No active process")
        self._polling = False
        self._poll_after_id = None
        self._queue_lock = threading.Lock()
        self._pending_progress: str | None = None
        self._pending_progress_fraction: tuple[float | None, str] | None = None
        self._dropped_console_items = 0
        self._last_notebook_refresh = 0.0
        self._final_notebook_render_pending = False
        self._live_notebook_render = str(os.environ.get("PLANNER_LIVE_NOTEBOOK_RENDER", "0")).strip().lower() in {"1", "true", "yes", "on"}
        self._last_progress_text = ""
        self._last_progress_line = ""
        self._executed_notebook_path = ""
        self._proc: subprocess.Popen | None = None
        self._stop_requested = False
        self._stop_btn: ttk.Button | None = None
        self._progressbar: ttk.Progressbar | None = None
        self.process_tree: ttk.Treeview | None = None
        self._process_row_seq = 0
        self._process_rows_by_pid: Dict[str, str] = {}
        self._notebook_render_sig = None
        self._image_refs: List[Any] = []
        self._task_cards: List[Any] = []
        self._task_accent_labels: List[tuple[Any, str]] = []
        self._task_title_label: Any | None = None
        self._task_progress_label: Any | None = None
        self._task_notebook: Any | None = None
        self._applied_task_theme_mode: str | None = None


    def _task_palette(self) -> Dict[str, Any]:
        base = getattr(self.root, "_planner_theme_colors", None)
        if not isinstance(base, dict):
            mode = str(getattr(self.root, "_planner_theme_mode", "dark") or "dark").lower()
            if mode == "light":
                base = {
                    "bg": "#edf1fb", "panel": "#f6f8fe", "surface": "#eef2fb",
                    "subtle": "#e3e9f8", "text": "#1c2340", "muted": "#66739a",
                    "accent": "#5f63d9", "entry_bg": "#ffffff", "entry_fg": "#17203a",
                    "canvas": "#f2f5fd", "border": "#c9d3ee",
                    "select_bg": "#5b67dd", "select_fg": "#ffffff",
                    "scrollbar_bg": "#bcc9ea", "scrollbar_trough": "#e7ecfb",
                    "scrollbar_active": "#98a9db",
                }
            else:
                base = {
                    "bg": "#070b14", "panel": "#0d1222", "surface": "#131a30",
                    "subtle": "#19213b", "text": "#edf2ff", "muted": "#97a6d2",
                    "accent": "#6c73ff", "entry_bg": "#0a1020", "entry_fg": "#f3f6ff",
                    "canvas": "#080d18", "border": "#263050",
                    "select_bg": "#4250d8", "select_fg": "#ffffff",
                    "scrollbar_bg": "#2a355d", "scrollbar_trough": "#0a1020",
                    "scrollbar_active": "#3a4a82",
                }
        mode = str(getattr(self.root, "_planner_theme_mode", "dark") or "dark").lower()
        light = mode == "light"
        semantic = {
            "state": "#0369a1" if light else "#7dd3fc",
            "operation": "#6d28d9" if light else "#c4b5fd",
            "step": "#be185d" if light else "#f9a8d4",
            "process": "#15803d" if light else "#86efac",
            "error": "#b91c1c" if light else "#ff6b7a",
            "warning": "#b45309" if light else "#ffb454",
            "info": "#0369a1" if light else "#8ecaff",
            "success": "#15803d" if light else "#76e6a1",
            "merge": "#a21caf" if light else "#e879f9",
            "download": "#0284c7" if light else "#38bdf8",
            "save": "#a16207" if light else "#facc15",
            "upload": "#15803d" if light else "#4ade80",
            "cell": "#7c3aed" if light else "#c4b5fd",
            "live_progress": "#1f2937" if light else "#f8fafc",
            "notebook_header": "#1d4ed8" if light else "#9ad1ff",
            "image_error": "#b91c1c" if light else "#ff8a8a",
        }
        out = dict(base)
        out.update(semantic)
        out["is_light"] = light
        out["notebook_bg"] = base.get("entry_bg", "#ffffff" if light else "#111111")
        out["notebook_block"] = base.get("surface", "#f6f8fe" if light else "#131a30")
        return out


    def apply_theme(self):
        if self.win is None:
            return
        try:
            if not self.win.winfo_exists():
                return
        except Exception:
            return
        colors = self._task_palette()
        mode = "light" if colors.get("is_light") else "dark"
        theme_changed = self._applied_task_theme_mode not in (None, mode)
        self._applied_task_theme_mode = mode
        try:
            self.win.configure(bg=colors["bg"])
        except Exception:
            pass

        # Task Center gets its own ttk styles so changing its palette never
        # compromises the main planner widgets.
        try:
            style = ttk.Style()
            style.configure(
                "Task.Treeview",
                background=colors["entry_bg"], fieldbackground=colors["entry_bg"],
                foreground=colors["entry_fg"], bordercolor=colors["border"], rowheight=24,
            )
            style.map(
                "Task.Treeview",
                background=[("selected", colors["select_bg"])],
                foreground=[("selected", colors["select_fg"])],
            )
            style.configure(
                "Task.Treeview.Heading",
                background=colors["surface"], foreground=colors["text"],
                bordercolor=colors["border"], relief="flat",
            )
            style.map("Task.Treeview.Heading", background=[("active", colors["subtle"])])
            style.configure("Task.TNotebook", background=colors["bg"], borderwidth=0)
            style.configure(
                "Task.TNotebook.Tab", background=colors["surface"], foreground=colors["muted"],
                padding=(12, 7), borderwidth=0,
            )
            style.map(
                "Task.TNotebook.Tab",
                background=[("selected", colors["panel"]), ("active", colors["subtle"])],
                foreground=[("selected", colors["accent"]), ("active", colors["text"])],
            )
            style.configure(
                "Task.Horizontal.TProgressbar",
                troughcolor=colors["subtle"], background=colors["accent"],
                bordercolor=colors["border"], lightcolor=colors["accent"], darkcolor=colors["accent"],
            )
        except Exception:
            pass

        def parent_bg(widget):
            try:
                parent = widget.master
                value = parent.cget("bg")
                return value or colors["panel"]
            except Exception:
                return colors["panel"]

        def walk(widget):
            try:
                cls = widget.winfo_class()
            except Exception:
                cls = ""
            try:
                if cls == "Toplevel":
                    widget.configure(bg=colors["bg"])
                elif cls in {"Frame", "Labelframe", "LabelFrame"}:
                    widget.configure(bg=colors["panel"], highlightbackground=colors["border"])
                    if cls in {"Labelframe", "LabelFrame"}:
                        widget.configure(fg=colors["text"])
                elif cls == "Label":
                    widget.configure(bg=parent_bg(widget), fg=colors["text"])
                elif cls == "Text":
                    widget.configure(
                        bg=colors["entry_bg"], fg=colors["entry_fg"],
                        insertbackground=colors["entry_fg"],
                        selectbackground=colors["select_bg"], selectforeground=colors["select_fg"],
                        highlightbackground=colors["border"], highlightcolor=colors["accent"],
                    )
                elif cls == "Canvas":
                    widget.configure(bg=colors["notebook_bg"], highlightbackground=colors["border"])
                elif cls == "Scrollbar":
                    widget.configure(
                        background=colors.get("scrollbar_bg", colors["border"]),
                        troughcolor=colors.get("scrollbar_trough", colors["bg"]),
                        activebackground=colors.get("scrollbar_active", colors["accent"]),
                        highlightbackground=colors["border"], highlightcolor=colors["border"],
                        bd=0, relief="flat", activerelief="flat", elementborderwidth=0, width=12,
                    )
            except Exception:
                pass
            try:
                children = widget.winfo_children()
            except Exception:
                children = []
            for child in children:
                walk(child)

        walk(self.win)

        # Restore the card hierarchy and semantic/Jupyter-like accents after the
        # generic pass.  This keeps the window colorful without hard-coding a
        # dark-only surface.
        for card in list(self._task_cards):
            try:
                if card.winfo_exists():
                    card.configure(bg=colors["surface"], highlightbackground=colors["border"])
                    for child in card.winfo_children():
                        if child.winfo_class() == "Label":
                            child.configure(bg=colors["surface"])
            except Exception:
                pass
        try:
            if self._task_title_label is not None and self._task_title_label.winfo_exists():
                self._task_title_label.configure(fg=colors["text"])
            if self._task_progress_label is not None and self._task_progress_label.winfo_exists():
                self._task_progress_label.configure(fg=colors["muted"])
        except Exception:
            pass
        role_colors = {
            "state": colors["state"], "operation": colors["operation"],
            "step": colors["step"], "process": colors["process"],
        }
        for label, role in list(self._task_accent_labels):
            try:
                if label.winfo_exists():
                    label.configure(fg=role_colors.get(role, colors["accent"]))
            except Exception:
                pass

        if self.process_tree is not None:
            try:
                self.process_tree.configure(style="Task.Treeview")
                self.process_tree.tag_configure("running", foreground=colors["info"])
                self.process_tree.tag_configure("done", foreground=colors["success"])
                self.process_tree.tag_configure("failed", foreground=colors["error"])
                self.process_tree.tag_configure("cell", foreground=colors["cell"])
            except Exception:
                pass
        if self._task_notebook is not None:
            try:
                self._task_notebook.configure(style="Task.TNotebook")
            except Exception:
                pass
        if self._progressbar is not None:
            try:
                self._progressbar.configure(style="Task.Horizontal.TProgressbar")
            except Exception:
                pass

        tag_colors = {k: colors[k] for k in (
            "error", "warning", "info", "success", "process", "merge",
            "download", "save", "upload", "cell", "live_progress"
        )}
        for widget in (self.idle_text, self.output_text):
            if widget is None:
                continue
            for tag, color in tag_colors.items():
                try:
                    widget.tag_configure(tag, foreground=color)
                except Exception:
                    pass

        # Rebuild already-rendered notebook output only when the user actually
        # switches light/dark mode.  Theme changes are rare, so this preserves
        # the low steady-state CPU policy while keeping cell/output surfaces
        # visually consistent with the rest of the Task Center.
        if theme_changed and self._executed_notebook_path and self._notebook_render_sig is not None:
            self._notebook_render_sig = None
            self._refresh_notebook_view(force=True)


    def _make_scrolled_text(self, parent, *, wrap: str, bg: str, fg: str):
        holder = Frame(parent)
        holder.pack(fill="both", expand=True)
        text = Text(holder, wrap=wrap, font=("Consolas", 10), bg=bg, fg=fg, undo=False, maxundo=0)
        yscroll = Scrollbar(holder, orient="vertical", command=text.yview)
        xscroll = Scrollbar(holder, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        holder.grid_rowconfigure(0, weight=1)
        holder.grid_columnconfigure(0, weight=1)
        return text


    def show(self):
        if self.win is None or not self.win.winfo_exists():
            colors = self._task_palette()
            self.win = tk.Toplevel(self.root)
            self.win.title("Planner Task Center")
            self.win.geometry("1240x820+100+90")
            self.win.configure(bg="#0b1020")

            header = Frame(self.win, padx=10, pady=10, bg="#0b1020")
            header.pack(fill="x")
            top = Frame(header, bg="#0b1020")
            top.pack(fill="x")
            self._task_title_label = Label(
                top,
                text="Execution / Transfer Monitor",
                font=("MS Gothic", 15, "bold"),
                bg="#0b1020",
                fg="#f3f6ff",
            )
            self._task_title_label.pack(side="left", anchor="w")
            self._stop_btn = ttk.Button(top, text="■ Stop", command=self.request_stop, state="disabled")
            self._stop_btn.pack(side="right")

            cards = Frame(header, bg="#0b1020")
            cards.pack(fill="x", pady=(9, 7))

            def add_card(title: str, variable: tk.StringVar, accent: str):
                card = Frame(
                    cards,
                    bg="#121a31",
                    highlightthickness=1,
                    highlightbackground="#263457",
                    padx=10,
                    pady=7,
                )
                card.pack(side="left", fill="x", expand=True, padx=(0, 7))
                self._task_cards.append(card)
                Label(card, text=title, bg="#121a31", fg="#7f8fb5", font=("Consolas", 8, "bold")).pack(anchor="w")
                value_label = Label(card, textvariable=variable, bg="#121a31", fg=accent, font=("Consolas", 10, "bold"), anchor="w")
                value_label.pack(fill="x", anchor="w")
                self._task_accent_labels.append((value_label, title.lower().replace("current ", "").replace(" ", "_")))

            add_card("STATE", self.state_var, "#7dd3fc")
            add_card("OPERATION", self.operation_var, "#c4b5fd")
            add_card("CURRENT STEP", self.current_step_var, "#f9a8d4")
            add_card("PROCESS", self.active_process_var, "#86efac")

            # Normalize the generated role names to the palette keys.
            self._task_accent_labels = [
                (label, "step" if role == "step" else role)
                for label, role in self._task_accent_labels
            ]
            self._task_progress_label = Label(header, textvariable=self.progress_var, fg="#b8c4e3", bg="#0b1020", anchor="w")
            self._task_progress_label.pack(fill="x", anchor="w")
            pb_holder = Frame(header, bg="#0b1020")
            pb_holder.pack(fill="x", pady=(4, 0))
            self._progressbar = ttk.Progressbar(pb_holder, variable=self.progress_pct_var, maximum=100.0, mode="determinate", style="Task.Horizontal.TProgressbar")
            self._progressbar.pack(fill="x", expand=True)

            notebook = ttk.Notebook(self.win, style="Task.TNotebook")
            self._task_notebook = notebook
            notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

            activity_frame = Frame(notebook, bg="#0d1428")
            output_frame = Frame(notebook, bg="#111111")
            raw_frame = Frame(notebook, bg="#0a0d14")
            notebook.add(activity_frame, text="Activity")
            notebook.add(output_frame, text="Notebook")
            notebook.add(raw_frame, text="Raw Console")

            proc_holder = LabelFrame(
                activity_frame,
                text=" Processes / Cells ",
                bg="#0d1428",
                fg="#dbeafe",
                padx=6,
                pady=6,
            )
            proc_holder.pack(fill="x", padx=8, pady=(8, 4))
            columns = ("status", "kind", "pid", "detail")
            self.process_tree = ttk.Treeview(proc_holder, columns=columns, show="headings", height=7, style="Task.Treeview")
            self.process_tree.heading("status", text="Status")
            self.process_tree.heading("kind", text="Kind")
            self.process_tree.heading("pid", text="PID/Cell")
            self.process_tree.heading("detail", text="Process / command")
            self.process_tree.column("status", width=90, stretch=False)
            self.process_tree.column("kind", width=110, stretch=False)
            self.process_tree.column("pid", width=90, stretch=False)
            self.process_tree.column("detail", width=760, stretch=True)
            proc_scroll = Scrollbar(proc_holder, orient="vertical", command=self.process_tree.yview)
            self.process_tree.configure(yscrollcommand=proc_scroll.set)
            self.process_tree.pack(side="left", fill="x", expand=True)
            proc_scroll.pack(side="right", fill="y")
            try:
                self.process_tree.tag_configure("running", foreground="#0ea5e9")
                self.process_tree.tag_configure("done", foreground="#22c55e")
                self.process_tree.tag_configure("failed", foreground="#ef4444")
                self.process_tree.tag_configure("cell", foreground="#a78bfa")
            except Exception:
                pass

            activity_log = LabelFrame(
                activity_frame,
                text=" Activity timeline ",
                bg="#0d1428",
                fg="#dbeafe",
                padx=5,
                pady=5,
            )
            activity_log.pack(fill="both", expand=True, padx=8, pady=(4, 8))
            self.idle_text = self._make_scrolled_text(activity_log, wrap="word", bg="#0d1428", fg="#dbeafe")

            self.output_canvas = Canvas(output_frame, bg="#111111", highlightthickness=0)
            y_scroll = Scrollbar(output_frame, orient="vertical", command=self.output_canvas.yview)
            x_scroll = Scrollbar(output_frame, orient="horizontal", command=self.output_canvas.xview)
            self.output_canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            self.output_canvas.pack(side="left", fill="both", expand=True)
            y_scroll.pack(side="right", fill="y")
            x_scroll.pack(side="bottom", fill="x")

            self.output_inner = Frame(self.output_canvas, bg="#111111")
            self.output_canvas.create_window((0, 0), window=self.output_inner, anchor="nw", tags=("inner",))
            self.output_inner.bind("<Configure>", self._on_output_frame_configure)
            self.output_canvas.bind("<Configure>", self._on_output_canvas_configure)

            placeholder = Label(
                self.output_inner,
                text="Notebook render appears here after execution. Live full-notebook rendering stays disabled by default to keep CPU/UI load low.",
                anchor="w",
                justify="left",
                font=("Consolas", 10),
                bg=colors["notebook_bg"],
                fg=colors["muted"],
                padx=10,
                pady=10,
            )
            placeholder.pack(fill="x", anchor="w")

            self.output_text = self._make_scrolled_text(raw_frame, wrap="word", bg="#070b12", fg="#f3f3f3")

            tag_colors = {
                "error": "#ff6b7a",
                "warning": "#ffb454",
                "info": "#8ecaff",
                "success": "#76e6a1",
                "process": "#67e8f9",
                "merge": "#e879f9",
                "download": "#38bdf8",
                "save": "#facc15",
                "upload": "#4ade80",
                "cell": "#c4b5fd",
                "live_progress": "#f8fafc",
            }
            for widget in (self.idle_text, self.output_text):
                if widget is None:
                    continue
                for tag, color in tag_colors.items():
                    widget.tag_configure(tag, foreground=color)

            self.idle_text.insert("end", "Task monitor initialized.\n", "success")
            self.output_text.insert("end", "Raw process output will appear here.\n", "info")
            self.idle_text.see("end")
            self.output_text.see("end")
            self.win.protocol("WM_DELETE_WINDOW", self._on_window_close)
        else:
            self.win.deiconify()
            self.win.lift()
        self.apply_theme()
        self._schedule_poll(CONSOLE_POLL_MS)


    def _schedule_poll(self, delay: int = CONSOLE_POLL_MS):
        if self.win is None or not self.win.winfo_exists():
            return
        if self._poll_after_id is not None:
            return
        self._polling = True
        self._poll_after_id = self.root.after(max(1, int(delay)), self._poll_queue)


    def _enqueue(self, item, *, critical: bool = False):
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            if not critical:
                self._dropped_console_items += 1
                return False
            # Critical state/proc events must win over old log noise.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(item)
                return True
            except queue.Full:
                self._dropped_console_items += 1
                return False


    def _on_window_close(self):
        try:
            if self.win is not None and self.win.winfo_exists():
                self.win.destroy()
        except tk.TclError:
            pass
        finally:
            if self._poll_after_id is not None:
                try:
                    self.root.after_cancel(self._poll_after_id)
                except Exception:
                    pass
            self._poll_after_id = None
            self._polling = False
            self.win = None
            self.output_text = None
            self.idle_text = None
            self.output_canvas = None
            self.output_inner = None
            self._stop_btn = None
            self._progressbar = None
            self.process_tree = None
            self._task_cards.clear()
            self._task_accent_labels.clear()
            self._task_title_label = None
            self._task_progress_label = None
            self._task_notebook = None
            self._applied_task_theme_mode = None
            self._process_rows_by_pid.clear()
            self._proc = None


    def _safe_configure_stop_button(self, *, state: str | None = None) -> bool:
        btn = getattr(self, "_stop_btn", None)
        if btn is None:
            return False
        try:
            if not btn.winfo_exists():
                self._stop_btn = None
                return False
            if state is not None:
                btn.configure(state=state)
            return True
        except tk.TclError:
            self._stop_btn = None
            return False


    def _on_output_frame_configure(self, _event=None):
        if self.output_canvas is not None:
            self.output_canvas.configure(scrollregion=self.output_canvas.bbox("all"))


    def _on_output_canvas_configure(self, event=None):
        if self.output_canvas is not None:
            width = max(200, event.width if event else self.output_canvas.winfo_width())
            self.output_canvas.itemconfigure("inner", width=width)


    def bind_notebook(self, notebook_path: str):
        self._executed_notebook_path = notebook_path or ""
        self._notebook_render_sig = None
        self._last_notebook_refresh = 0.0
        self._final_notebook_render_pending = True
        self._enqueue(("reset_notebook_view",), critical=True)


    def set_state(self, text: str):
        self._enqueue(("state", text), critical=True)


    def set_step(self, text: str):
        self._enqueue(("step", text), critical=True)


    def set_operation(self, text: str):
        self._enqueue(("operation", text), critical=True)


    def set_progress_mode(self, mode: str):
        mode = "indeterminate" if str(mode).lower().startswith("ind") else "determinate"
        self._enqueue(("progress_mode", mode), critical=True)


    def process_event(self, status: str, kind: str, ident: str, detail: str = ""):
        self._enqueue(("process_event", str(status), str(kind), str(ident), str(detail)), critical=True)


    def set_progress(self, text: str):
        # Progress is a latest-value signal, not a log.  Coalescing prevents
        # tqdm updates from flooding Tk's event loop during long plans.
        with self._queue_lock:
            self._pending_progress = str(text)


    def set_progress_fraction(self, fraction: float | None, text: str = ""):
        with self._queue_lock:
            self._pending_progress_fraction = (fraction, str(text or ""))


    def attach_process(self, proc: subprocess.Popen | None, label: str = "Notebook runner"):
        self._enqueue(("proc", proc, str(label)), critical=True)


    def log(self, text: str, kind: str = "info"):
        if text:
            self._enqueue(("log", text, kind), critical=False)


    def idle(self, text: str):
        if text:
            self._enqueue(("idle", text), critical=False)


    def clear_output(self):
        self._enqueue(("clear",), critical=True)


    def request_stop(self):
        self._stop_requested = True
        self._safe_configure_stop_button(state="disabled")
        self._enqueue(("state", "STOPPING"), critical=True)
        self._enqueue(("step", "Stopping notebook process"), critical=True)
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
        except Exception as e:
            self._enqueue(("log", f"Stop request failed: {e}", "error"), critical=True)


    @staticmethod
    def _text_at_bottom(widget: Text | None) -> bool:
        if widget is None:
            return True
        try:
            return widget.yview()[1] >= 0.98
        except Exception:
            return True


    @staticmethod
    def _trim_text(widget: Text | None, max_lines: int):
        if widget is None:
            return
        try:
            total = int(widget.index("end-1c").split(".")[0])
        except Exception:
            return
        if total > max_lines:
            widget.delete("1.0", f"{total - max_lines + 1}.0")


    @staticmethod
    def _extract_progress_fraction(text: str) -> float | None:
        s = str(text or "").strip()
        if not s:
            return None
        m = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)%", s)
        if m:
            try:
                pct = float(m.group(1))
            except Exception:
                pct = -1.0
            if 0.0 <= pct <= 100.0:
                return pct / 100.0
        m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", s)
        if m:
            try:
                cur = float(m.group(1))
                total = float(m.group(2))
            except Exception:
                return None
            if total > 0 and 0 <= cur <= total:
                return cur / total
        return None


    @staticmethod
    def _looks_like_progress_line(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        lower = stripped.lower()
        return (
            stripped.startswith("[planner-progress]")
            or stripped.startswith("[#")
            or "%|" in stripped
            or "it/s" in lower
            or "s/it" in lower
            or ("dl:" in lower and ("eta:" in lower or "%" in stripped))
        )


    def _clear_live_progress(self, widget: Text | None):
        if widget is None:
            return
        try:
            ranges = widget.tag_ranges("live_progress")
            while ranges:
                widget.delete(ranges[0], ranges[1])
                ranges = widget.tag_ranges("live_progress")
        except Exception:
            return


    def _sync_live_progress(self, widget: Text | None, text: str, max_lines: int, tag: str = "info"):
        if widget is None:
            return
        at_bottom = self._text_at_bottom(widget)
        self._clear_live_progress(widget)
        text = str(text or "").strip()
        if text:
            try:
                if widget.index("end-1c") != "1.0" and widget.get("end-2c", "end-1c") != "\n":
                    widget.insert("end", "\n")
            except Exception:
                pass
            widget.insert("end", f"[progress] {text}\n", (tag, "live_progress"))
            self._trim_text(widget, max_lines)
            if at_bottom:
                widget.see("end")


    @staticmethod
    def _normalize_stream_text(text: str) -> str:
        lines: List[str] = []
        current = ""
        for ch in str(text):
            if ch == "\\r":
                current = ""
            elif ch == "\\n":
                lines.append(current)
                current = ""
            else:
                current += ch
        if current:
            lines.append(current)
        return "\\n".join(lines).strip()


    @staticmethod
    def _extract_text_output(output: Dict[str, Any]) -> str:
        otype = output.get("output_type")
        if otype == "stream":
            return RunnerConsoleWindow._normalize_stream_text(output.get("text", ""))
        if otype == "error":
            tb = output.get("traceback") or []
            if tb:
                return "\\n".join(str(x) for x in tb)
            return f"{output.get('ename', 'Error')}: {output.get('evalue', '')}".strip()
        data = output.get("data") or {}
        if isinstance(data, dict):
            if "text/plain" in data:
                val = data["text/plain"]
                if isinstance(val, list):
                    val = "".join(str(x) for x in val)
                return RunnerConsoleWindow._normalize_stream_text(str(val))
            if "text/markdown" in data:
                val = data["text/markdown"]
                if isinstance(val, list):
                    val = "".join(str(x) for x in val)
                return str(val).strip()
            if "application/json" in data:
                return json.dumps(data["application/json"], ensure_ascii=False, indent=2)
        return ""


    def _render_output_block(self, parent: Frame, output: Dict[str, Any]):
        colors = self._task_palette()
        bg = colors["notebook_block"]
        block = Frame(parent, bg=bg, bd=0, highlightthickness=1, highlightbackground=colors["border"], padx=8, pady=6)
        block.pack(fill="x", expand=True, anchor="w", padx=6, pady=4)

        text = self._extract_text_output(output)
        data = output.get("data") or {}
        image_data = data.get("image/png") if isinstance(data, dict) else None
        if image_data:
            try:
                if isinstance(image_data, list):
                    image_data = "".join(str(x) for x in image_data)
                pil = Image.open(io.BytesIO(base64.b64decode(image_data)))
                max_w = 960
                if pil.width > max_w:
                    ratio = max_w / float(pil.width)
                    pil = pil.resize((int(pil.width * ratio), max(1, int(pil.height * ratio))), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                self._image_refs.append(photo)
                img_label = Label(block, image=photo, bg=bg)
                img_label.pack(anchor="w", pady=(0, 6))
            except Exception as e:
                err = Label(
                    block,
                    text=f"[image render failed] {e}",
                    justify="left",
                    anchor="w",
                    font=("Consolas", 10),
                    bg=bg,
                    fg=colors["image_error"],
                )
                err.pack(fill="x", anchor="w")
        if text:
            if text.startswith("<PIL.") and image_data:
                text = ""
            if text:
                lbl = Label(
                    block,
                    text=text,
                    justify="left",
                    anchor="w",
                    font=("Consolas", 10),
                    bg=bg,
                    fg=colors["entry_fg"] if output.get("output_type") != "error" else colors["error"],
                    wraplength=max(300, (self.output_canvas.winfo_width() - 40) if self.output_canvas else 900),
                )
                lbl.pack(fill="x", anchor="w")


    def _render_notebook_outputs(self, nb: Dict[str, Any]):
        if self.output_inner is None:
            return
        colors = self._task_palette()
        try:
            self.output_inner.configure(bg=colors["notebook_bg"])
            if self.output_canvas is not None:
                self.output_canvas.configure(bg=colors["notebook_bg"])
        except Exception:
            pass
        for child in list(self.output_inner.winfo_children()):
            child.destroy()
        self._image_refs.clear()

        cells = nb.get("cells") or []
        visible = 0
        for idx, cell in enumerate(cells, start=1):
            outputs = cell.get("outputs") or []
            if not outputs:
                continue
            visible += 1
            cell_frame = Frame(self.output_inner, bg=colors["notebook_bg"], padx=6, pady=6)
            cell_frame.pack(fill="x", expand=True, anchor="w")
            source = "".join(cell.get("source") or [])
            header_text = f"Cell {idx}"
            first_line = next((ln.strip() for ln in source.splitlines() if ln.strip()), "")
            if first_line:
                header_text += f"  ·  {first_line[:100]}"
            header = Label(
                cell_frame,
                text=header_text,
                anchor="w",
                justify="left",
                font=("Consolas", 10, "bold"),
                bg=colors["notebook_bg"],
                fg=colors["notebook_header"],
            )
            header.pack(fill="x", anchor="w", padx=6, pady=(0, 2))
            for output in outputs:
                self._render_output_block(cell_frame, output)

        if visible == 0:
            empty = Label(
                self.output_inner,
                text="Notebook has started, but no rendered outputs are available yet.",
                anchor="w",
                justify="left",
                font=("Consolas", 10),
                bg="#111111",
                fg="#d0d0d0",
                padx=10,
                pady=10,
            )
            empty.pack(fill="x", anchor="w")

        self._on_output_frame_configure()
        if self.output_canvas is not None:
            self.output_canvas.yview_moveto(1.0)


    def _refresh_notebook_view(self, *, force: bool = False):
        path = self._executed_notebook_path
        if not path:
            return
        now = time.monotonic()
        if (not force) and (now - self._last_notebook_refresh) < (CONSOLE_NOTEBOOK_REFRESH_MS / 1000.0):
            return
        self._last_notebook_refresh = now
        p = Path(path)
        if not p.exists():
            return
        try:
            stat = p.stat()
            sig = (int(stat.st_mtime_ns), int(stat.st_size))
            if sig == self._notebook_render_sig:
                return
            raw = p.read_text(encoding="utf-8")
            nb = json.loads(raw)
        except Exception:
            return
        self._notebook_render_sig = sig
        self._render_notebook_outputs(nb)


    def _reset_notebook_widgets(self):
        if self.output_inner is not None:
            colors = self._task_palette()
            for child in list(self.output_inner.winfo_children()):
                child.destroy()
            self._image_refs.clear()
            try:
                self.output_inner.configure(bg=colors["notebook_bg"])
                if self.output_canvas is not None:
                    self.output_canvas.configure(bg=colors["notebook_bg"])
            except Exception:
                pass
            placeholder = Label(
                self.output_inner,
                text="Notebook render will appear here after execution (live render is disabled by default for performance).",
                anchor="w",
                justify="left",
                font=("Consolas", 10),
                bg=colors["notebook_bg"],
                fg=colors["muted"],
                padx=10,
                pady=10,
            )
            placeholder.pack(fill="x", anchor="w")
            self._on_output_frame_configure()


    def _insert_idle_batch(self, chunks: list[str]):
        if self.idle_text is None or not chunks:
            return
        at_bottom = self._text_at_bottom(self.idle_text)
        payload = []
        for text in chunks:
            payload.append(text if text.endswith("\n") else text + "\n")
        self.idle_text.insert("end", "".join(payload))
        self._trim_text(self.idle_text, CONSOLE_MAX_IDLE_LINES)
        if at_bottom:
            self.idle_text.see("end")


    def _insert_log_batch(self, chunks: list[tuple[str, str]]):
        if self.output_text is None or not chunks:
            return
        at_bottom = self._text_at_bottom(self.output_text)
        last_kind = None
        buffer: list[str] = []

        def flush():
            nonlocal last_kind, buffer
            if not buffer:
                return
            self.output_text.insert("end", "".join(buffer), last_kind or "info")
            buffer = []

        for text, kind in chunks:
            t = text if text.endswith("\n") else text + "\n"
            if kind != last_kind and buffer:
                flush()
            last_kind = kind
            buffer.append(t)
        flush()
        self._trim_text(self.output_text, CONSOLE_MAX_OUTPUT_LINES)
        if at_bottom:
            self.output_text.see("end")

            
    def _append_text(self, widget: Text | None, payloads: list[tuple[str, str]], max_lines: int):
        if widget is None or not payloads:
            return
        at_bottom = self._text_at_bottom(widget)
        current_tag = None
        chunk = []
        for text, tag in payloads:
            if not text.endswith("\n"):
                text += "\n"
            if current_tag is None:
                current_tag = tag
            if tag != current_tag and chunk:
                widget.insert("end", "".join(chunk), current_tag)
                chunk = []
                current_tag = tag
            chunk.append(text)
        if chunk:
            widget.insert("end", "".join(chunk), current_tag or "info")
        self._trim_text(widget, max_lines)
        if at_bottom:
            widget.see("end")


    def _update_process_tree(self, status: str, kind: str, ident: str, detail: str):
        tree = self.process_tree
        if tree is None:
            return
        try:
            key = str(ident or "-")
            status_u = str(status or "").upper()
            existing = self._process_rows_by_pid.get(key)
            tag = "failed" if status_u in {"FAILED", "ERROR"} else ("done" if status_u in {"DONE", "END", "EXITED"} else ("cell" if str(kind).lower() == "cell" else "running"))
            values = (status_u or "RUNNING", kind or "process", key, detail[:500])
            if existing and tree.exists(existing):
                tree.item(existing, values=values, tags=(tag,))
                row_id = existing
            else:
                self._process_row_seq += 1
                row_id = tree.insert("", "end", values=values, tags=(tag,))
                self._process_rows_by_pid[key] = row_id
            if status_u in {"RUNNING", "START", "BUILDING", "UPLOADING"}:
                self.active_process_var.set(f"{kind}: {detail[:80]}" if detail else f"{kind}: {key}")
            elif status_u in {"FAILED", "ERROR"}:
                self.active_process_var.set(f"Failed: {detail[:80]}" if detail else f"Failed: {key}")
            children = tree.get_children("")
            if len(children) > CONSOLE_MAX_PROCESS_ROWS:
                for old in children[: len(children) - CONSOLE_MAX_PROCESS_ROWS]:
                    vals = tree.item(old, "values")
                    if len(vals) >= 3:
                        old_key = str(vals[2])
                        if self._process_rows_by_pid.get(old_key) == old:
                            self._process_rows_by_pid.pop(old_key, None)
                    tree.delete(old)
            tree.see(row_id)
        except tk.TclError:
            return


    def _poll_queue(self):
        self._poll_after_id = None
        self._polling = False
        if self.win is None or not self.win.winfo_exists():
            return

        idle_payloads: list[tuple[str, str]] = []
        output_payloads: list[tuple[str, str]] = []
        processed = 0
        processed_lines = 0
        processed_chars = 0

        with self._queue_lock:
            pending_progress = self._pending_progress
            pending_fraction = self._pending_progress_fraction
            self._pending_progress = None
            self._pending_progress_fraction = None

        if pending_progress is not None:
            text = str(pending_progress).strip()
            self.progress_var.set(text)
            self._last_progress_text = text
            fraction = self._extract_progress_fraction(text)
            if fraction is not None:
                self.progress_pct_var.set(max(0.0, min(100.0, float(fraction) * 100.0)))

        if pending_fraction is not None:
            fraction, text = pending_fraction
            if fraction is None:
                self.progress_pct_var.set(0.0)
            else:
                self.progress_pct_var.set(max(0.0, min(100.0, float(fraction) * 100.0)))
            text = str(text).strip()
            if text:
                self.progress_var.set(text)
                self._last_progress_text = text

        try:
            while processed < CONSOLE_MAX_BATCH_ITEMS:
                item = self.queue.get_nowait()
                processed += 1
                op = item[0]

                if op == "state":
                    self.state_var.set(str(item[1]))
                elif op == "step":
                    self.current_step_var.set(str(item[1]))
                elif op == "operation":
                    self.operation_var.set(str(item[1]))
                elif op == "progress_mode":
                    mode = str(item[1])
                    if self._progressbar is not None:
                        try:
                            if mode == "indeterminate":
                                self._progressbar.configure(mode="indeterminate")
                                self._progressbar.start(12)
                            else:
                                self._progressbar.stop()
                                self._progressbar.configure(mode="determinate")
                        except tk.TclError:
                            pass
                elif op == "process_event":
                    self._update_process_tree(str(item[1]), str(item[2]), str(item[3]), str(item[4]))
                elif op == "proc":
                    self._proc = item[1]
                    label = str(item[2]) if len(item) > 2 else "Process"
                    state = "normal" if (self._proc is not None and self._proc.poll() is None) else "disabled"
                    self._safe_configure_stop_button(state=state)
                    if self._proc is not None:
                        pstatus = "RUNNING" if self._proc.poll() is None else "DONE"
                        self._update_process_tree(pstatus, "runner", str(getattr(self._proc, "pid", "-")), label)
                elif op == "idle":
                    text = str(item[1])
                    idle_payloads.append((text, "info"))
                    processed_chars += len(text)
                    processed_lines += max(1, text.count("\n") + 1)
                elif op == "log":
                    text = str(item[1])
                    output_payloads.append((text, str(item[2])))
                    processed_chars += len(text)
                    processed_lines += max(1, text.count("\n") + 1)
                elif op == "reset_notebook_view":
                    self._reset_notebook_widgets()
                elif op == "clear":
                    if self.output_text is not None:
                        self.output_text.delete("1.0", "end")
                    if self.idle_text is not None:
                        self.idle_text.delete("1.0", "end")
                    if self.process_tree is not None:
                        try:
                            for row in self.process_tree.get_children(""):
                                self.process_tree.delete(row)
                        except tk.TclError:
                            pass
                    self._process_rows_by_pid.clear()
                    self.active_process_var.set("No active process")
                    # Progress is coalesced outside the queue.  Do not erase a
                    # newer progress update that arrived immediately after the
                    # queued clear operation.
                    if pending_progress is None and pending_fraction is None:
                        self.progress_var.set("")
                        self.progress_pct_var.set(0.0)
                        self._last_progress_text = ""
                    if self._progressbar is not None:
                        try:
                            self._progressbar.stop()
                            self._progressbar.configure(mode="determinate")
                        except tk.TclError:
                            pass

                if processed_lines >= CONSOLE_MAX_BATCH_LINES or processed_chars >= CONSOLE_MAX_BATCH_CHARS:
                    break
        except queue.Empty:
            pass
        except Exception as e:
            output_payloads.append((f"[console error] {type(e).__name__}: {e}", "error"))
        finally:
            if self._dropped_console_items:
                output_payloads.append((
                    f"[console] dropped {self._dropped_console_items} old log update(s) to keep the UI responsive.",
                    "info",
                ))
                self._dropped_console_items = 0
            if idle_payloads:
                self._append_text(self.idle_text, idle_payloads, CONSOLE_MAX_IDLE_LINES)
            if output_payloads:
                self._append_text(self.output_text, output_payloads, CONSOLE_MAX_OUTPUT_LINES)
            self._sync_live_progress(self.idle_text, self._last_progress_text, CONSOLE_MAX_IDLE_LINES)
            self._sync_live_progress(self.output_text, self._last_progress_text, CONSOLE_MAX_OUTPUT_LINES)

            running = self._proc is not None and self._proc.poll() is None
            if running:
                if self._live_notebook_render:
                    self._refresh_notebook_view(force=False)
            elif self._final_notebook_render_pending and self.state_var.get() in {"DONE", "FAILED", "STOPPED"}:
                # Render the executed notebook once after the process has
                # finished.  Rebuilding every widget during execution was one
                # of the largest sources of planner UI load.
                self._refresh_notebook_view(force=True)
                self._final_notebook_render_pending = False

            if not running:
                self._safe_configure_stop_button(state="disabled")

            with self._queue_lock:
                has_pending_progress = self._pending_progress is not None or self._pending_progress_fraction is not None
            if not self.queue.empty() or has_pending_progress:
                delay = CONSOLE_BURST_POLL_MS
            elif running:
                delay = CONSOLE_POLL_MS
            else:
                delay = CONSOLE_IDLE_POLL_MS
            self._schedule_poll(delay)
