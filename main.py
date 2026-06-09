from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import pickle
import queue
import re
import subprocess
import sys
import threading
import tempfile
import time
import traceback
import signal
import copy
import difflib
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter import Canvas, Entry, Frame, Label, LabelFrame, Scrollbar, Text
import nbformat

from IPython.core.interactiveshell import InteractiveShell

from huggingface_hub import HfApi, upload_file
from PIL import Image, ImageTk

from plan import (
    SDXL_BLOCKS,
    PlanCompileError,
    create_plan,
    create_plan_ipynb,
    default_plan,
    default_ratio,
    export_plan_records_txt,
    load_plan_records,
    make_entry,
    normalize_plan,
)

CONFIG_FILE = os.path.join(os.getcwd(), "config.tccm")
CHECK_INTERVAL_MS = 2500
CONSOLE_POLL_MS = 150
CONSOLE_BURST_POLL_MS = 35
CONSOLE_MAX_BATCH_ITEMS = 500
CONSOLE_MAX_BATCH_LINES = 120
CONSOLE_MAX_BATCH_CHARS = 65536
CONSOLE_MAX_OUTPUT_LINES = 4000
CONSOLE_MAX_IDLE_LINES = 1800
PLAN_UI_REFRESH_MS = 120
PLAN_AUTOSAVE_MS = 900
LINE_TYPES = ["Checkpoint Merge", "LoRA Bake", "LoRA Merge"]
INTERNAL_LINE_TYPES = ["Download Model", "Local Model", "Remove Model"]
RATIO_MODES = ["Single", "Block weight", "Elemental", "Randomize"]
DOWNLOAD_TYPES = ["Checkpoint", "LoRA", "LyCORIS"]
LOCAL_TYPES = ["Checkpoint", "LoRA", "LyCORIS"]
BASE_MODEL_OPTIONS = ["SD1.5", "SDXL", "Flux", "ZImage", "Anima"]
BASE_MODEL_BLOCK_ATTRS = {
    "SD1.5": "BLOCKID",
    "SDXL": "BLOCKIDXLL",
    "Flux": "BLOCKIDFLUX",
    "ZImage": "BLOCKIDZI",
    "Anima": "BLOCKIDAM",
}
_DISCOVER_BLOCKS_INSTALL_ATTEMPTED = False


def _coerce_block_names(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        seq = list(raw.keys())
    elif isinstance(raw, (list, tuple)):
        seq = list(raw)
    else:
        try:
            seq = list(raw)
        except Exception:
            seq = []
    return [str(x) for x in seq if str(x).strip()]


def _load_module_from_candidates(module_name: str, candidates: List[Path]):
    errors: list[str] = []
    for path in candidates:
        if not path.exists():
            errors.append(f"missing: {path}")
            continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                errors.append(f"spec failed: {path}")
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, errors
        except Exception as e:
            errors.append(f"{path}: {type(e).__name__}: {e}")
    return None, errors


def discover_block_sets() -> Dict[str, List[str]]:
    global _DISCOVER_BLOCKS_INSTALL_ATTEMPTED
    fallback = {name: list(SDXL_BLOCKS) for name in BASE_MODEL_OPTIONS}
    candidates = [
        Path("tools/chattiori_model_merger/Utils.py"),
        Path(__file__).resolve().parent / "tools/chattiori_model_merger/Utils.py",
    ]
    utils_mod, load_errors = _load_module_from_candidates("planner_cmm_utils", candidates)

    if utils_mod is None and not _DISCOVER_BLOCKS_INSTALL_ATTEMPTED:
        _DISCOVER_BLOCKS_INSTALL_ATTEMPTED = True
        try:
            from install_planner_deps import install_or_update
            install_or_update(skip_pip=True, update_if_needed=True)
            importlib.invalidate_caches()
            utils_mod, retry_errors = _load_module_from_candidates("planner_cmm_utils_retry", candidates)
            load_errors.extend(retry_errors)
        except Exception as e:
            load_errors.append(f"installer: {type(e).__name__}: {e}")

    if utils_mod is None:
        print("[discover_block_sets] Falling back to SDXL block names.")
        for err in load_errors:
            print(f"[discover_block_sets] {err}")
        return fallback

    for base_model, attr_name in BASE_MODEL_BLOCK_ATTRS.items():
        names = _coerce_block_names(getattr(utils_mod, attr_name, None))
        if names:
            fallback[base_model] = names
    return fallback


FALLBACK_MERGE_MODES = [
    {"key": "WS", "label": "Weighted Sum", "needs_m2": False, "needs_beta": False},
    {"key": "ST", "label": "Smooth Add", "needs_m2": True, "needs_beta": True},
    {"key": "TRS", "label": "Triple Sum", "needs_m2": True, "needs_beta": True},
    {"key": "DARE", "label": "DARE", "needs_m2": False, "needs_beta": True},
    {"key": "AD", "label": "Add Difference", "needs_m2": True, "needs_beta": False},
    {"key": "SWAP", "label": "Swap", "needs_m2": False, "needs_beta": False},
    {"key": "CLIPXOR", "label": "CLIPXOR", "needs_m2": False, "needs_beta": False},
    {"key": "TF", "label": "Trim and Fill", "needs_m2": False, "needs_beta": False},
    {"key": "FWM", "label": "FWM", "needs_m2": False, "needs_beta": False},
]


def save_config_to_disk(config):
    with open(CONFIG_FILE, "wb") as f:
        pickle.dump(config, f)


INIT_CONFIG = {
    "filepath": "",
    "workpath": "/kaggle/working",
    "title": "merge_plan",
    "vae": "",
    "vae_name": "VAE",
    "bake_vae": False,
    "model_dir": "",
    "vae_dir": "",
    "CivitAPI": "",
    "HuggingAPI": "",
    "UR": "User/Repo",
    "hf_repo_id": "",
    "hf_folder_path": "",
    "last_notebook_path": "",
    "last_executed_notebook_path": "",
    "saveas": "",
    "base_model": "SDXL",
    "ignore_install_deps": False,
    "use_online": False,
    "upload_after_merge": True,
    "run_t2i": False,
    "theme_mode": "dark",
}


def _planner_tooltip_catalog_path_candidates() -> list[Path]:
    """Return tooltip JSON candidates in priority order."""
    here = Path(__file__).resolve().parent
    return [
        here / "planner_tooltips.json",
        Path.cwd() / "planner_tooltips.json",
        here / "config" / "planner_tooltips.json",
    ]


def _planner_load_tooltip_catalog() -> Dict[str, Any]:
    for path in _planner_tooltip_catalog_path_candidates():
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as e:
            try:
                print(f"[tooltip] failed to load {path}: {type(e).__name__}: {e}")
            except Exception:
                pass
    return {"options": {}, "choices": {}, "modes": {}, "left_panel": {}, "right_panel": {}}


PLANNER_TOOLTIP_CATALOG = _planner_load_tooltip_catalog()
LEFT_PANEL_FIELD_HELP = dict(PLANNER_TOOLTIP_CATALOG.get("left_panel") or {})
RIGHT_PANEL_FIELD_HELP = dict(PLANNER_TOOLTIP_CATALOG.get("right_panel") or {})


ELEMENTAL_CANDIDATE_JSON_FILES = {
    "SD1.5": "elemental_candidates_sd15.json",
    "SDXL": "elemental_candidates_sdxl.json",
    "Flux": "elemental_candidates_flux.json",
    "ZImage": "elemental_candidates_zimage.json",
    "Anima": "elemental_candidates_anima.json",
}

DEFAULT_ELEMENTAL_ELEMENTS = {
    "attn": "attention and prompt-conditioned feature routing",
    "res": "residual feature mixing and style carry-over",
    "proj": "projection and feature conversion",
    "conv": "convolutional local texture and shape response",
    "ff": "feed-forward feature shaping",
    "norm": "normalization and activation balance",
    "to_q": "query attention projection",
    "to_k": "key attention projection",
    "to_v": "value attention projection",
    "to_out": "attention output projection",
}



def load_config_from_disk():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "rb") as f:
                cfg = pickle.load(f)
            merged = INIT_CONFIG.copy()
            merged.update(cfg)
            if os.path.normpath(str(merged.get("workpath", "") or "")) == os.path.normpath("/kaggle"):
                merged["workpath"] = "/kaggle/working"
            ur = str(merged.get("UR", "") or "").strip()
            repo = str(merged.get("hf_repo_id", "") or "").strip()
            if repo and (not ur or ur == INIT_CONFIG.get("UR", "")):
                merged["UR"] = repo
            if str(merged.get("UR", "") or "").strip():
                merged["hf_repo_id"] = str(merged.get("UR", "") or "").strip()
            return merged
        except Exception:
            return INIT_CONFIG.copy()
    return INIT_CONFIG.copy()


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


class RunnerConsoleWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.win: tk.Toplevel | None = None
        self.queue: queue.Queue = queue.Queue()
        self.output_text: Text | None = None
        self.idle_text: Text | None = None
        self.output_canvas: Canvas | None = None
        self.output_inner: Frame | None = None
        self.state_var = tk.StringVar(value="IDLE")
        self.current_step_var = tk.StringVar(value="Waiting")
        self.progress_var = tk.StringVar(value="")
        self.progress_pct_var = tk.DoubleVar(value=0.0)
        self._polling = False
        self._last_progress_text = ""
        self._last_progress_line = ""
        self._executed_notebook_path = ""
        self._proc: subprocess.Popen | None = None
        self._stop_requested = False
        self._stop_btn: ttk.Button | None = None
        self._notebook_render_sig = None
        self._image_refs: List[Any] = []

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
            self.win = tk.Toplevel(self.root)
            self.win.title("Planner Runner")
            self.win.geometry("1180x760+120+120")

            header = Frame(self.win, padx=8, pady=8)
            header.pack(fill="x")
            top = Frame(header)
            top.pack(fill="x")
            Label(top, text="Execution Console", font=("MS Gothic", 14, "bold")).pack(side="left", anchor="w")
            self._stop_btn = ttk.Button(top, text="■ Stop", command=self.request_stop, state="disabled")
            self._stop_btn.pack(side="right")

            Label(header, textvariable=self.state_var, fg="#225588", font=("Consolas", 10, "bold")).pack(anchor="w")
            Label(header, textvariable=self.current_step_var, fg="#333333").pack(anchor="w")
            Label(header, textvariable=self.progress_var, fg="#666666").pack(anchor="w")
            pb_holder = Frame(header)
            pb_holder.pack(fill="x", pady=(4, 0))
            ttk.Progressbar(pb_holder, variable=self.progress_pct_var, maximum=100.0, mode="determinate").pack(fill="x", expand=True)

            notebook = ttk.Notebook(self.win)
            notebook.pack(fill="both", expand=True, padx=8, pady=8)

            idle_frame = Frame(notebook)
            output_frame = Frame(notebook, bg="#111111")
            notebook.add(idle_frame, text="IDLE")
            notebook.add(output_frame, text="Jupyter Output")

            self.idle_text = Text(idle_frame, wrap="word", font=("Consolas", 10), bg="#101820", fg="#d6f5ff")
            self.idle_text.pack(fill="both", expand=True)

            output_pane = ttk.PanedWindow(output_frame, orient="vertical")
            output_pane.pack(fill="both", expand=True)

            render_frame = Frame(output_pane, bg="#111111")
            raw_frame = Frame(output_pane)
            output_pane.add(render_frame, weight=4)
            output_pane.add(raw_frame, weight=1)

            self.output_canvas = Canvas(render_frame, bg="#111111", highlightthickness=0)
            y_scroll = Scrollbar(render_frame, orient="vertical", command=self.output_canvas.yview)
            x_scroll = Scrollbar(render_frame, orient="horizontal", command=self.output_canvas.xview)
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
                text="Notebook render will appear here while the notebook executes.",
                anchor="w",
                justify="left",
                font=("Consolas", 10),
                bg="#111111",
                fg="#d0d0d0",
                padx=10,
                pady=10,
            )
            placeholder.pack(fill="x", anchor="w")

            self.output_text = Text(raw_frame, wrap="word", font=("Consolas", 10), bg="#0f0f0f", fg="#f3f3f3", height=8)
            self.output_text.pack(fill="both", expand=True)
            self.output_text.tag_configure("error", foreground="#ff8a8a")
            self.output_text.tag_configure("info", foreground="#9ad1ff")
            self.output_text.tag_configure("success", foreground="#8dff8d")
            self.idle_text.insert("end", "Planner console initialized.\n")
            self.output_text.insert("end", "Jupyter output stream will appear here.\n")
            self.idle_text.see("end")
            self.output_text.see("end")
            self.win.protocol("WM_DELETE_WINDOW", self._on_window_close)
        else:
            self.win.deiconify()
            self.win.lift()
        if not self._polling:
            self._polling = True
            self.root.after(CONSOLE_POLL_MS, self._poll_queue)

    def _on_window_close(self):
        try:
            if self.win is not None and self.win.winfo_exists():
                self.win.destroy()
        except tk.TclError:
            pass
        finally:
            self.win = None
            self.output_text = None
            self.idle_text = None
            self.output_canvas = None
            self.output_inner = None
            self._stop_btn = None
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
        self.queue.put(("reset_notebook_view",))

    def set_state(self, text: str):
        self.queue.put(("state", text))

    def set_step(self, text: str):
        self.queue.put(("step", text))

    def set_progress(self, text: str):
        self.queue.put(("progress", text))

    def set_progress_fraction(self, fraction: float | None, text: str = ""):
        self.queue.put(("progress_fraction", fraction, text))

    def attach_process(self, proc: subprocess.Popen | None):
        self.queue.put(("proc", proc))

    def log(self, text: str, kind: str = "info"):
        if text:
            self.queue.put(("log", text, kind))

    def idle(self, text: str):
        if text:
            self.queue.put(("idle", text))

    def clear_output(self):
        self.queue.put(("clear",))

    def request_stop(self):
        self._stop_requested = True
        self._safe_configure_stop_button(state="disabled")
        self.queue.put(("state", "STOPPING"))
        self.queue.put(("step", "Stopping notebook process"))
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
            self.queue.put(("log", f"Stop request failed: {e}", "error"))

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
        bg = "#111111"
        block = Frame(parent, bg=bg, bd=0, highlightthickness=1, highlightbackground="#2b2b2b", padx=8, pady=6)
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
                    fg="#ff8a8a",
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
                    fg="#f3f3f3" if output.get("output_type") != "error" else "#ff9a9a",
                    wraplength=max(300, (self.output_canvas.winfo_width() - 40) if self.output_canvas else 900),
                )
                lbl.pack(fill="x", anchor="w")

    def _render_notebook_outputs(self, nb: Dict[str, Any]):
        if self.output_inner is None:
            return
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
            cell_frame = Frame(self.output_inner, bg="#111111", padx=6, pady=6)
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
                bg="#111111",
                fg="#9ad1ff",
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

    def _refresh_notebook_view(self):
        path = self._executed_notebook_path
        if not path:
            return
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
            for child in list(self.output_inner.winfo_children()):
                child.destroy()
            self._image_refs.clear()
            placeholder = Label(
                self.output_inner,
                text="Notebook render will appear here while the notebook executes.",
                anchor="w",
                justify="left",
                font=("Consolas", 10),
                bg="#111111",
                fg="#d0d0d0",
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

    def _poll_queue(self):
        self._polling = False
        idle_payloads: list[tuple[str, str]] = []
        output_payloads: list[tuple[str, str]] = []
        processed = 0
        try:
            while processed < CONSOLE_MAX_BATCH_LINES:
                item = self.queue.get_nowait()
                processed += 1
                self.show()
                op = item[0]
                if op == "state":
                    self.state_var.set(str(item[1]))
                elif op == "step":
                    self.current_step_var.set(str(item[1]))
                elif op == "progress":
                    text = str(item[1]).strip()
                    self.progress_var.set(text)
                    self._last_progress_text = text
                    fraction = self._extract_progress_fraction(text)
                    if fraction is not None:
                        self.progress_pct_var.set(max(0.0, min(100.0, float(fraction) * 100.0)))
                elif op == "progress_fraction":
                    fraction = item[1]
                    text = str(item[2]).strip() if len(item) > 2 else ""
                    if fraction is None:
                        self.progress_pct_var.set(0.0)
                    else:
                        self.progress_pct_var.set(max(0.0, min(100.0, float(fraction) * 100.0)))
                    if text:
                        self.progress_var.set(text)
                        self._last_progress_text = text
                elif op == "proc":
                    self._proc = item[1]
                    state = "normal" if (self._proc is not None and self._proc.poll() is None) else "disabled"
                    self._safe_configure_stop_button(state=state)
                elif op == "idle":
                    idle_payloads.append((str(item[1]), "info"))
                elif op == "log":
                    output_payloads.append((str(item[1]), str(item[2])))
                elif op == "reset_notebook_view":
                    self._reset_notebook_widgets()
                elif op == "clear":
                    if self.output_text is not None:
                        self.output_text.delete("1.0", "end")
                    if self.idle_text is not None:
                        self.idle_text.delete("1.0", "end")
                    self.progress_var.set("")
                    self.progress_pct_var.set(0.0)
                    self._last_progress_text = ""
        except queue.Empty:
            pass
        except Exception as e:
            output_payloads.append((f"[console error] {type(e).__name__}: {e}", "error"))
        finally:
            if idle_payloads:
                self._append_text(self.idle_text, idle_payloads, CONSOLE_MAX_IDLE_LINES)
            if output_payloads:
                self._append_text(self.output_text, output_payloads, CONSOLE_MAX_OUTPUT_LINES)
            self._sync_live_progress(self.idle_text, self._last_progress_text, CONSOLE_MAX_IDLE_LINES)
            self._sync_live_progress(self.output_text, self._last_progress_text, CONSOLE_MAX_OUTPUT_LINES)
            self._refresh_notebook_view()
            if self._proc is None or self._proc.poll() is not None:
                self._safe_configure_stop_button(state="disabled")
            if self.win is not None and self.win.winfo_exists():
                self._polling = True
                delay = CONSOLE_BURST_POLL_MS if not self.queue.empty() else CONSOLE_POLL_MS
                self.root.after(delay, self._poll_queue)


def discover_merge_modes() -> List[Dict[str, Any]]:
    from tools.chattiori_model_merger.merge_modes import theta_funcs, modes_need_m2, modes_need_beta
    try:
        modes: List[Dict[str, Any]] = []
        for key, value in theta_funcs.items():
            if key == "NoIn":
                continue
            label = key
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                label = str(value[2])
            elif isinstance(value, dict):
                label = str(value.get("label") or value.get("name") or key)
            modes.append({
                "key": key,
                "label": label,
                "needs_m2": key in modes_need_m2,
                "needs_beta": key in modes_need_beta,
            })
        if modes:
            return modes
    except Exception:
        pass
    return FALLBACK_MERGE_MODES


class ModelPlannerApp:
    NOTEBOOK_PARAMS_KEYS = [
        "filepath", "workpath", "title", "vae", "vae_name", "bake_vae", "CivitAPI", "HuggingAPI", "UR", "model_dir", "vae_dir", "ignore_install_deps", "use_online", "upload_after_merge", "run_t2i"
    ]

    def __init__(self, root):
        self.root = root
        self.config = load_config_from_disk()
        self.entries: Dict[str, Entry] = {}
        self.block_sets = discover_block_sets()
        self.base_model_var = tk.StringVar(value=self.config.get("base_model", "SDXL"))
        self.merge_modes = discover_merge_modes()
        self.merge_mode_map = {m["key"]: m for m in self.merge_modes}
        self.block_sets = discover_block_sets()
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

    def _left_help(self, key: str) -> Dict[str, str]:
        return LEFT_PANEL_FIELD_HELP.get(key, {"short": "", "detail": ""})

    def _attach_tooltip(self, widgets, detail: str):
        if not detail:
            return
        if not isinstance(widgets, (list, tuple, set)):
            widgets = [widgets]
        for widget in widgets:
            if widget is not None:
                Tooltip(widget, detail)

    def _event_has_open_modifier(self, event) -> bool:
        state = int(getattr(event, "state", 0) or 0)
        return bool(state & 0x0004 or state & 0x0008 or state & 0x0010 or state & 0x0080)

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

    def _elemental_candidate_json_filename(self) -> str:
        base_model = self.base_model_var.get() if hasattr(self, "base_model_var") else self.config.get("base_model", "SDXL")
        return ELEMENTAL_CANDIDATE_JSON_FILES.get(base_model, "elemental_candidates_sdxl.json")

    def _elemental_candidate_json_path(self) -> Path:
        return Path(__file__).resolve().parent / self._elemental_candidate_json_filename()

    @staticmethod
    def _normalize_named_effect_items(raw) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        if isinstance(raw, dict):
            iterable = [{"name": k, "effect": v} for k, v in raw.items()]
        elif isinstance(raw, list):
            iterable = raw
        else:
            iterable = []
        for item in iterable:
            if isinstance(item, str):
                name = item.strip()
                effect = ""
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("key") or item.get("id") or item.get("label") or "").strip()
                effect = str(item.get("effect") or item.get("what") or item.get("description") or item.get("detail") or item.get("help") or "").strip()
            else:
                continue
            if name:
                items.append({"name": name, "effect": effect})
        return items

    @staticmethod
    def _element_lookup_from_items(items: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        lookup: Dict[str, List[Dict[str, str]]] = {}
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            entry = {
                "name": name,
                "effect": str(item.get("effect", "")).strip(),
            }
            aliases = item.get("aliases")
            if isinstance(aliases, list) and aliases:
                entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
            lookup.setdefault(name, []).append(entry)
        return lookup

    def _resolve_layer_element_items(self, raw_items, element_lookup: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        resolved: List[Dict[str, str]] = []
        seen = set()
        for raw in raw_items or []:
            candidates: List[Dict[str, str]] = []
            if isinstance(raw, str):
                candidates = list(element_lookup.get(raw.strip(), []))
            elif isinstance(raw, dict):
                raw_name = str(raw.get("name", "")).strip()
                if raw_name and raw_name in element_lookup:
                    candidates = list(element_lookup.get(raw_name, []))
                elif raw_name:
                    entry = {
                        "name": raw_name,
                        "effect": str(raw.get("effect", "")).strip(),
                    }
                    aliases = raw.get("aliases")
                    if isinstance(aliases, list) and aliases:
                        entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
                    candidates = [entry]
            for item in candidates:
                key = (str(item.get("name", "")).strip().lower(), str(item.get("effect", "")).strip())
                if key in seen or not key[0]:
                    continue
                seen.add(key)
                resolved.append(dict(item))
        return resolved


    def _block_effect_hint_fallback(self, block_name: str) -> str:
        name = str(block_name or "").upper()
        if name == "BASE":
            return "global base behavior, broad style baseline, and shared foundation layers"
        if name.startswith("IN"):
            try:
                idx = int(name[2:])
            except Exception:
                idx = 0
            if idx <= 2:
                return "very coarse composition, silhouette, camera distance, and large structure"
            if idx <= 5:
                return "mid-scale shapes, pose flow, anatomy rhythm, and scene layout"
            return "deeper feature extraction, local forms, and transition into detailed latent features"
        if name.startswith("MID"):
            return "bottleneck/global mixing, concept fusion, and whole-image latent coordination"
        if name.startswith("OUT"):
            try:
                idx = int(name[3:])
            except Exception:
                idx = 0
            if idx <= 2:
                return "large decoded structures, major reconstruction, and broad rendering direction"
            if idx <= 5:
                return "surface shaping, material feel, lighting transitions, and regional detail"
            return "fine finishing, micro-detail, crisp edges, and final texture cleanup"
        return "model-specific feature group; effect depends on architecture and training"

    def _block_effect_hint(self, block_name: str) -> str:
        name = self._normalize_elemental_layer_name(block_name)
        path = self._elemental_candidate_json_path()
        cache_sig = None
        try:
            if path.exists():
                stat = path.stat()
                cache_sig = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            cache_sig = None

        effect_map = getattr(self, "_block_effect_hint_cache", None)
        if not isinstance(effect_map, dict) or getattr(self, "_block_effect_hint_cache_sig", None) != cache_sig:
            effect_map = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    layer_items = self._normalize_named_effect_items(
                        data.get("layers") or data.get("blocks") or data.get("hierarchy") or []
                    )
                    for item in layer_items:
                        layer_name = self._normalize_elemental_layer_name(item.get("name", ""))
                        effect = str(item.get("effect") or "").strip()
                        if layer_name and effect:
                            effect_map[layer_name] = effect
                except Exception:
                    effect_map = {}
            self._block_effect_hint_cache = effect_map
            self._block_effect_hint_cache_sig = cache_sig

        effect = str(effect_map.get(name, "") or "").strip() if isinstance(effect_map, dict) else ""
        if effect:
            return effect
        return self._block_effect_hint_fallback(name)

    def _load_elemental_catalog(self) -> Dict[str, Any]:
        path = self._elemental_candidate_json_path()
        cache_sig = None
        try:
            if path.exists():
                stat = path.stat()
                cache_sig = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            cache_sig = None
        if cache_sig is not None and getattr(self, "_elemental_catalog_cache_sig", None) == cache_sig:
            cached = getattr(self, "_elemental_catalog_cache", None)
            if isinstance(cached, dict):
                return cached

        layers: List[Dict[str, str]] = []
        elements: List[Dict[str, str]] = []
        layer_elements: Dict[str, List[Dict[str, str]]] = {}
        layer_aliases: Dict[str, str] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                layers = self._normalize_named_effect_items(data.get("layers") or data.get("blocks") or data.get("hierarchy") or [])
                elements = self._normalize_named_effect_items(data.get("elements") or data.get("elementals") or data.get("parts") or [])
                raw_map = data.get("layer_elements") or data.get("elements_by_layer") or {}
                element_lookup = self._element_lookup_from_items(elements)
                if isinstance(raw_map, dict):
                    for layer_name, raw_items in raw_map.items():
                        norm = self._normalize_elemental_layer_name(layer_name)
                        if not norm:
                            continue
                        layer_elements[norm] = self._resolve_layer_element_items(raw_items, element_lookup)
                raw_aliases = data.get("layer_aliases") or {}
                if isinstance(raw_aliases, dict):
                    for alias_name, canonical in raw_aliases.items():
                        alias_norm = str(alias_name or "").strip().upper()
                        canonical_norm = str(canonical or "").strip().upper()
                        if alias_norm and canonical_norm:
                            layer_aliases[alias_norm] = canonical_norm
            except Exception:
                layers = []
                elements = []
                layer_elements = {}
                layer_aliases = {}
        if not layers:
            layers = [{"name": name, "effect": self._block_effect_hint(name)} for name in (self._current_block_names() or list(SDXL_BLOCKS))]
        if not layer_elements:
            fallback_items = [{"name": name, "effect": effect} for name, effect in DEFAULT_ELEMENTAL_ELEMENTS.items()]
            for layer in layers:
                layer_name = self._normalize_elemental_layer_name(layer.get("name", ""))
                if layer_name:
                    layer_elements[layer_name] = [dict(item) for item in fallback_items]
        if not elements:
            if layer_elements:
                dedup: Dict[str, Dict[str, str]] = {}
                for items in layer_elements.values():
                    for item in items:
                        key = str(item.get("name", "")).strip().lower()
                        if key and key not in dedup:
                            dedup[key] = {
                                "name": str(item.get("name", "")).strip(),
                                "effect": str(item.get("effect", "")).strip(),
                            }
                elements = list(dedup.values())
            else:
                elements = [{"name": name, "effect": effect} for name, effect in DEFAULT_ELEMENTAL_ELEMENTS.items()]

        catalog: Dict[str, Any] = {
            "layers": layers,
            "elements": elements,
            "layer_elements": layer_elements,
            "layer_aliases": layer_aliases,
            "path": [{"name": self._elemental_candidate_json_filename(), "effect": "base-model-specific candidate file loaded for elemental popup suggestions"}],
        }
        self._elemental_catalog_cache = catalog
        self._elemental_catalog_cache_sig = cache_sig
        return catalog

    def _normalize_elemental_layer_name(self, name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        upper = raw.upper()
        alias_map = {"MID": "M00", "MID00": "M00", "MIDDLE": "M00"}
        cached = getattr(self, "_elemental_catalog_cache", None)
        if isinstance(cached, dict):
            for alias_name, canonical in (cached.get("layer_aliases") or {}).items():
                alias_map[str(alias_name).strip().upper()] = str(canonical).strip().upper()
        upper = alias_map.get(upper, upper)
        known = {str(x).strip().upper() for x in (self._current_block_names() or [])}
        known.update({"BASE", "VAE", "M00"})
        if upper in known:
            return upper
        return upper

    def _elemental_valid_elements_for_layer(self, layer_name: str) -> List[Dict[str, str]]:
        catalog = self._load_elemental_catalog()
        norm = self._normalize_elemental_layer_name(layer_name)
        layer_map = catalog.get("layer_elements") or {}
        if isinstance(layer_map, dict) and norm in layer_map:
            return list(layer_map.get(norm) or [])
        return list(catalog.get("elements") or [])

    def _elemental_alias_lookup(self, layer_name: str | None = None) -> Dict[str, List[Dict[str, str]]]:
        items = self._elemental_valid_elements_for_layer(layer_name) if layer_name else list((self._load_elemental_catalog().get("elements") or []))
        lookup: Dict[str, List[Dict[str, str]]] = {}
        seen = set()
        for item in items:
            canonical = str(item.get("name", "")).strip()
            effect = str(item.get("effect", "")).strip()
            aliases = item.get("aliases") if isinstance(item, dict) else None
            if not canonical or not isinstance(aliases, list):
                continue
            for alias in aliases:
                alias_str = str(alias).strip()
                if not alias_str or alias_str.lower() == canonical.lower():
                    continue
                key = (alias_str.lower(), canonical.lower(), effect)
                if key in seen:
                    continue
                seen.add(key)
                lookup.setdefault(alias_str.lower(), []).append({
                    "alias": alias_str,
                    "canonical": canonical,
                    "effect": effect,
                })
        return lookup

    def _elemental_layer_order(self) -> List[str]:
        catalog = self._load_elemental_catalog()
        out: List[str] = []
        for item in catalog.get("layers", []):
            name = self._normalize_elemental_layer_name(item.get("name", ""))
            if name and name not in out:
                out.append(name)
        if not out:
            out = [self._normalize_elemental_layer_name(x) for x in self._current_block_names()]
        return out

    def _expand_elemental_layer_token(self, token: str) -> List[str]:
        token = str(token or "").strip()
        if not token:
            return []
        order = self._elemental_layer_order()
        if "-" in token and not token.startswith("-") and not token.endswith("-"):
            left, right = token.split("-", 1)
            left_n = self._normalize_elemental_layer_name(left)
            right_n = self._normalize_elemental_layer_name(right)
            if left_n in order and right_n in order:
                li = order.index(left_n)
                ri = order.index(right_n)
                if li <= ri:
                    return order[li:ri + 1]
                return order[ri:li + 1]
        norm = self._normalize_elemental_layer_name(token)
        return [norm] if norm in order else []

    def _resolve_elemental_layers_expr(self, expr: str) -> List[str]:
        expr = str(expr or "")
        out: List[str] = []
        seen = set()
        for token in expr.split():
            for layer_name in self._expand_elemental_layer_token(token):
                key = layer_name.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(layer_name)
        return out

    def _elemental_valid_elements_for_layers(self, layer_names: List[str]) -> List[Dict[str, str]]:
        catalog = self._load_elemental_catalog()
        layer_map = catalog.get("layer_elements") or {}
        out: List[Dict[str, str]] = []
        seen = set()
        if layer_names:
            for layer_name in layer_names:
                items = layer_map.get(self._normalize_elemental_layer_name(layer_name), []) if isinstance(layer_map, dict) else []
                for item in items:
                    name = str(item.get("name", "")).strip()
                    if not name:
                        continue
                    key = name.lower()
                    if key not in seen:
                        seen.add(key)
                        entry = {"name": name, "effect": str(item.get("effect", "")).strip()}
                        aliases = item.get("aliases") if isinstance(item, dict) else None
                        if isinstance(aliases, list) and aliases:
                            entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
                        out.append(entry)
        else:
            for item in catalog.get("elements", []):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    entry = {"name": name, "effect": str(item.get("effect", "")).strip()}
                    aliases = item.get("aliases") if isinstance(item, dict) else None
                    if isinstance(aliases, list) and aliases:
                        entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
                    out.append(entry)
        return out

    @staticmethod
    def _autocomplete_item(insert: str, effect: str = "", *, label: str = "") -> Dict[str, str]:
        display = label or (f"{insert}    — {effect}" if effect else insert)
        return {"insert": insert, "display": display, "effect": effect}

    def _elemental_separator_for_widget(self, widget: Text | None) -> str:
        sep = str(getattr(widget, "_elemental_separator", ",") or ",") if widget is not None else ","
        return sep if sep in {",", ";"} else ","

    def _elemental_context(self, widget: Text) -> Dict[str, Any]:
        sep = self._elemental_separator_for_widget(widget)
        before = widget.get("1.0", "insert")
        abs_end = len(before)
        seg_start = max(before.rfind(sep), before.rfind("\n"))
        segment = before[seg_start + 1:]
        base_abs = seg_start + 1
        first_colon = segment.find(":")
        second_colon = segment.find(":", first_colon + 1) if first_colon >= 0 else -1

        def token_info(text_part: str, offset: int):
            if not text_part or text_part.endswith((" ", "\t")):
                pos = f"1.0+{abs_end}c"
                return "", pos, pos
            m = re.search(r"(\S+)$", text_part)
            if not m:
                pos = f"1.0+{abs_end}c"
                return "", pos, pos
            start_abs = base_abs + offset + m.start(1)
            return m.group(1), f"1.0+{start_abs}c", f"1.0+{abs_end}c"

        if first_colon < 0:
            probe, start, end = token_info(segment, 0)
            return {
                "stage": "layer",
                "probe": probe,
                "start": start,
                "end": end,
                "segment": segment,
                "layer_part": segment,
                "layer_expr": segment,
                "element_part": "",
                "target_layers": self._resolve_elemental_layers_expr(segment),
            }

        layer_part = segment[:first_colon]
        if second_colon < 0:
            element_part = segment[first_colon + 1:]
            probe, start, end = token_info(element_part, first_colon + 1)
            return {
                "stage": "element",
                "probe": probe,
                "start": start,
                "end": end,
                "segment": segment,
                "layer_part": layer_part,
                "layer_expr": layer_part,
                "element_part": element_part,
                "target_layers": self._resolve_elemental_layers_expr(layer_part),
            }

        return {
            "stage": "strength",
            "probe": "",
            "start": None,
            "end": None,
            "segment": segment,
            "layer_part": layer_part,
            "layer_expr": layer_part,
            "element_part": segment[first_colon + 1:second_colon],
            "target_layers": self._resolve_elemental_layers_expr(layer_part),
        }

    def _elemental_suggestion_items(self, widget: Text) -> List[Dict[str, str]]:
        catalog = self._load_elemental_catalog()
        ctx = self._elemental_context(widget)
        probe = str(ctx.get("probe") or "").strip().lower()

        if ctx.get("stage") == "layer":
            out = []
            for item in catalog.get("layers", []):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                if probe and probe not in name.lower():
                    continue
                out.append(self._autocomplete_item(name, item.get("effect", ""), label=f"layer  {name}    — {item.get('effect', '')}"))
            return out[:40]

        if ctx.get("stage") == "element":
            out = []
            for item in self._elemental_valid_elements_for_layers(ctx.get("target_layers") or []):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                aliases = [str(x).strip().lower() for x in (item.get("aliases") or []) if str(x).strip()]
                haystack = [name.lower()] + aliases
                if probe and not any(probe in x for x in haystack):
                    continue
                out.append(self._autocomplete_item(name, item.get("effect", ""), label=f"element {name}    — {item.get('effect', '')}"))
            return out[:40]

        return []

    def _signature_autocomplete_candidates(self) -> List[str]:
        modes = [f"@m {m['key']}" for m in self.merge_modes]
        items = [
            "@p half",
            "@p bhalf",
            "@p quarter",
            "@p fp32",
            "@c 0",
            "@c 1",
            "@c 2",
            "@f 1,2,3",
            "@seed 1234",
            "@rank 32",
            "@arch sdxl",
            "@arch flux",
            "--bake_fp32",
            "--save_half",
            "--save_bhalf",
            "--save_quarter",
            "--save_full",
            "--prune",
            "--save_safetensors",
        ]
        items.extend(modes)
        return items

    def _hide_autocomplete(self, _event=None):
        target = self._autocomplete_target
        if self._autocomplete_popup is not None:
            try:
                self._autocomplete_popup.destroy()
            except Exception:
                pass
        self._autocomplete_popup = None
        self._autocomplete_listbox = None
        self._autocomplete_canvas = None
        self._autocomplete_inner = None
        self._autocomplete_scrollbar = None
        self._autocomplete_row_frames = []
        self._autocomplete_selected_index = 0
        self._autocomplete_target = None
        self._autocomplete_apply = None
        self._autocomplete_items = []
        try:
            if target is not None and target.winfo_exists():
                target.focus_set()
        except Exception:
            pass

    def _normalize_autocomplete_suggestions(self, suggestions) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for item in suggestions or []:
            if isinstance(item, dict):
                insert = str(item.get("insert") or "").strip()
                display = str(item.get("display") or insert).strip()
                effect = str(item.get("effect") or "").strip()
                title = str(item.get("title") or insert).strip()
            else:
                insert = str(item).strip()
                display = insert
                effect = ""
                title = insert
            if not effect and display and display != insert and "—" in display:
                try:
                    effect = display.split("—", 1)[1].strip()
                except Exception:
                    effect = effect
            if insert:
                out.append({"insert": insert, "display": display or insert, "title": title or insert, "effect": effect})
        return out

    def _autocomplete_has_items(self) -> bool:
        return bool(getattr(self, "_autocomplete_items", []))

    def _style_autocomplete_popup(self, colors=None):
        popup = getattr(self, "_autocomplete_popup", None)
        if popup is None:
            return
        if colors is None:
            colors = self._theme_colors()
        try:
            popup.configure(bg=colors["panel"])
        except Exception:
            pass
        canvas = getattr(self, "_autocomplete_canvas", None)
        if canvas is not None:
            try:
                canvas.configure(bg=colors["panel"], highlightbackground=colors["border"], highlightcolor=colors["accent"])
            except Exception:
                pass
        scrollbar = getattr(self, "_autocomplete_scrollbar", None)
        if scrollbar is not None:
            try:
                scrollbar.configure(
                    bg=colors["scrollbar_bg"],
                    troughcolor=colors["scrollbar_trough"],
                    activebackground=colors["scrollbar_active"],
                    highlightbackground=colors["border"],
                    highlightcolor=colors["border"],
                    bd=0,
                    relief="flat",
                    activerelief="flat",
                    elementborderwidth=0,
                )
            except Exception:
                pass
        selected = int(getattr(self, "_autocomplete_selected_index", 0) or 0)
        rows = list(getattr(self, "_autocomplete_row_frames", []) or [])
        for idx, row in enumerate(rows):
            row_bg = colors["select_bg"] if idx == selected else colors["entry_bg"]
            title_fg = colors["select_fg"] if idx == selected else colors["entry_fg"]
            effect_fg = colors["select_fg"] if idx == selected else colors["muted"]
            border = colors["accent"] if idx == selected else colors["border"]
            try:
                row.configure(bg=row_bg, highlightbackground=border)
            except Exception:
                pass
            for child in row.winfo_children():
                try:
                    role = getattr(child, "_autocomplete_role", "")
                    fg = title_fg if role == "title" else effect_fg
                    child.configure(bg=row_bg, fg=fg)
                except Exception:
                    pass

    def _select_autocomplete_index(self, idx: int):
        if not self._autocomplete_has_items():
            return
        idx = max(0, min(len(self._autocomplete_items) - 1, int(idx)))
        self._autocomplete_selected_index = idx
        self._style_autocomplete_popup()
        canvas = getattr(self, "_autocomplete_canvas", None)
        rows = getattr(self, "_autocomplete_row_frames", None) or []
        inner = getattr(self, "_autocomplete_inner", None)
        if canvas is not None and inner is not None and 0 <= idx < len(rows):
            try:
                canvas.update_idletasks()
                row = rows[idx]
                total_h = max(1, int(inner.winfo_height()))
                view_h = max(1, int(canvas.winfo_height()))
                y = int(row.winfo_y())
                h = max(1, int(row.winfo_height()))
                top_f, bottom_f = canvas.yview()
                top = top_f * total_h
                bottom = bottom_f * total_h
                if y < top:
                    canvas.yview_moveto(max(0.0, y / total_h))
                elif y + h > bottom:
                    canvas.yview_moveto(max(0.0, min(1.0, (y + h - view_h) / total_h)))
            except Exception:
                pass

    def _show_autocomplete(self, widget, suggestions, apply_func):
        items = self._normalize_autocomplete_suggestions(suggestions)
        if not items:
            self._hide_autocomplete()
            return
        if self._autocomplete_target is not widget:
            self._hide_autocomplete()
        popup = self._autocomplete_popup
        if popup is None:
            popup = tk.Toplevel(self.root)
            popup.wm_overrideredirect(True)
            popup.attributes("-topmost", True)
            holder = Frame(popup)
            holder.pack(fill="both", expand=True)
            canvas = Canvas(holder, width=560, height=320, highlightthickness=1, bd=0)
            scrollbar = Scrollbar(holder, orient="vertical", command=canvas.yview)
            inner = Frame(canvas)
            canvas.create_window((0, 0), window=inner, anchor="nw", tags=("popup_inner",))
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.bind("<Configure>", lambda e: canvas.itemconfigure("popup_inner", width=max(320, e.width - 2)))

            def _popup_mousewheel(event):
                try:
                    delta = getattr(event, "delta", 0)
                    if delta:
                        canvas.yview_scroll(-1 * int(delta / 120), "units")
                    elif getattr(event, "num", None) == 4:
                        canvas.yview_scroll(-1, "units")
                    elif getattr(event, "num", None) == 5:
                        canvas.yview_scroll(1, "units")
                except Exception:
                    pass
                return "break"

            for w in (canvas, inner):
                w.bind("<MouseWheel>", _popup_mousewheel, add="+")
                w.bind("<Button-4>", _popup_mousewheel, add="+")
                w.bind("<Button-5>", _popup_mousewheel, add="+")

            self._autocomplete_popup = popup
            self._autocomplete_canvas = canvas
            self._autocomplete_inner = inner
            self._autocomplete_scrollbar = scrollbar
            self._autocomplete_listbox = None
        else:
            canvas = self._autocomplete_canvas
            inner = self._autocomplete_inner

        self._autocomplete_row_frames = []
        for child in list(inner.winfo_children()):
            child.destroy()

        display_items = items[:40]
        for idx, item in enumerate(display_items):
            row = Frame(inner, padx=12, pady=8, cursor="hand2", highlightthickness=1, bd=0)
            row.pack(fill="x", expand=True, padx=4, pady=3)
            title = Label(row, text=item.get("title") or item["insert"], font=("MS Gothic", 11, "bold"), anchor="w", justify="left")
            title._autocomplete_role = "title"
            title.pack(fill="x", anchor="w")
            effect_text = str(item.get("effect") or "").strip()
            if effect_text:
                effect = Label(row, text=effect_text, font=("MS Gothic", 9), anchor="w", justify="left", wraplength=520)
                effect._autocomplete_role = "effect"
                effect.pack(fill="x", anchor="w", pady=(2, 0))
                widgets = (row, title, effect)
            else:
                widgets = (row, title)

            for bound in widgets:
                bound.bind("<Enter>", lambda _e, i=idx: self._select_autocomplete_index(i), add="+")
                bound.bind("<Button-1>", lambda _e, i=idx: (self._select_autocomplete_index(i), "break")[1], add="+")
                bound.bind("<Double-Button-1>", lambda _e, i=idx: (self._select_autocomplete_index(i), self._accept_autocomplete()), add="+")
            self._autocomplete_row_frames.append(row)

        try:
            bbox = widget.bbox("insert") if hasattr(widget, "bbox") else None
        except Exception:
            bbox = None
        try:
            root_x = widget.winfo_rootx()
            root_y = widget.winfo_rooty()
            x = root_x + (bbox[0] if bbox else 0) + 8
            y = root_y + (bbox[1] + bbox[3] if bbox else widget.winfo_height()) + 8
            popup.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass

        visible_rows = min(8, max(1, len(display_items)))
        try:
            self._autocomplete_canvas.configure(height=max(72, visible_rows * 58))
        except Exception:
            pass

        self._autocomplete_target = widget
        self._autocomplete_apply = apply_func
        self._autocomplete_items = display_items
        self._autocomplete_selected_index = 0
        self._style_autocomplete_popup(self._theme_colors())
        self._select_autocomplete_index(0)
        try:
            widget.after_idle(widget.focus_set)
        except Exception:
            pass

    def _move_autocomplete(self, step: int):
        if not self._autocomplete_has_items():
            return "break"
        idx = int(getattr(self, "_autocomplete_selected_index", 0) or 0)
        self._select_autocomplete_index(idx + step)
        return "break"

    def _accept_autocomplete(self, _event=None):
        if not self._autocomplete_has_items() or self._autocomplete_apply is None:
            return None
        idx = max(0, min(len(self._autocomplete_items) - 1, int(getattr(self, "_autocomplete_selected_index", 0) or 0)))
        value = self._autocomplete_items[idx]["insert"]
        target = self._autocomplete_target
        try:
            self._autocomplete_apply(value)
        finally:
            self._hide_autocomplete()
            try:
                if target is not None and target.winfo_exists():
                    target.focus_set()
            except Exception:
                pass
        return "break"

    @staticmethod
    def _is_plain_number(value: str) -> bool:
        return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value or "").strip()))

    @staticmethod
    def _is_shift_pressed(event) -> bool:
        state = int(getattr(event, "state", 0) or 0)
        return bool(state & 0x0001)

    def _configure_syntax_tags(self, widget: Text):
        widget.tag_configure("invalid_token", foreground="#cc2222")
        widget.tag_configure("signature_name", foreground="#8a2be2")
        widget.tag_configure("signature_value", foreground="#1f7a1f")
        widget.tag_configure("elemental_layer", foreground="#2457c5")
        widget.tag_configure("elemental_element", foreground="#0f8a8a")
        widget.tag_configure("numeric_value", foreground="#b36b00")
        widget.tag_configure("delimiter", foreground="#888888")

    def _clear_syntax_highlight(self, widget: Text):
        for tag in (
            "invalid_token",
            "signature_name",
            "signature_value",
            "elemental_layer",
            "elemental_element",
            "numeric_value",
            "delimiter",
        ):
            try:
                widget.tag_remove(tag, "1.0", "end")
            except Exception:
                pass

    def _add_tag_for_span(self, widget: Text, tag: str, start: int, end: int):
        if end <= start:
            return
        widget.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")

    def _schedule_rerender_current_line(self):
        if self._rerender_after_id is not None:
            try:
                self.root.after_cancel(self._rerender_after_id)
            except Exception:
                pass
        self._rerender_after_id = self.root.after(1, self._perform_scheduled_rerender)

    def _perform_scheduled_rerender(self):
        self._rerender_after_id = None
        canvas = getattr(self, "canvas", None)
        yview = None
        try:
            if canvas is not None and canvas.winfo_exists():
                yview = canvas.yview()
        except Exception:
            yview = None
        self._render_current_line()
        self._refresh_line_selector()
        if yview is not None:
            def _restore_scroll(c=canvas, view=yview):
                try:
                    if c is not None and c.winfo_exists():
                        c.update_idletasks()
                        c.yview_moveto(float(view[0]))
                except Exception:
                    pass
            self.root.after_idle(_restore_scroll)

    def _configure_syntax_tags(self, widget: Text):
        widget.tag_configure("invalid_token", foreground="#cc2222")
        widget.tag_configure("signature_name", foreground="#8a2be2")
        widget.tag_configure("signature_value", foreground="#1f7a1f")
        widget.tag_configure("elemental_layer", foreground="#2457c5")
        widget.tag_configure("elemental_element", foreground="#0f8a8a")
        widget.tag_configure("numeric_value", foreground="#b36b00")
        widget.tag_configure("delimiter", foreground="#888888")

    def _clear_syntax_highlight(self, widget: Text):
        for tag in (
            "invalid_token",
            "signature_name",
            "signature_value",
            "elemental_layer",
            "elemental_element",
            "numeric_value",
            "delimiter",
        ):
            try:
                widget.tag_remove(tag, "1.0", "end")
            except Exception:
                pass

    def _add_tag_for_span(self, widget: Text, tag: str, start: int, end: int):
        if end <= start:
            return
        widget.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")

    def _schedule_rerender_current_line(self):
        if self._rerender_after_id is not None:
            try:
                self.root.after_cancel(self._rerender_after_id)
            except Exception:
                pass
        self._rerender_after_id = self.root.after(1, self._perform_scheduled_rerender)

    def _perform_scheduled_rerender(self):
        self._rerender_after_id = None
        canvas = getattr(self, "canvas", None)
        yview = None
        try:
            if canvas is not None and canvas.winfo_exists():
                yview = canvas.yview()
        except Exception:
            yview = None
        self._render_current_line()
        self._refresh_line_selector()
        if yview is not None:
            def _restore_scroll(c=canvas, view=yview):
                try:
                    if c is not None and c.winfo_exists():
                        c.update_idletasks()
                        c.yview_moveto(float(view[0]))
                except Exception:
                    pass
            self.root.after_idle(_restore_scroll)

    def _apply_signature_completion(self, widget: Text, start, end, value: str):
        widget.delete(start, end)
        insert_text = value if str(value).endswith(" ") else f"{value} "
        widget.insert(start, insert_text)
        widget.mark_set("insert", f"{start}+{len(insert_text)}c")
        self._refresh_signature_invalid_highlight(widget)

    def _apply_elemental_completion(self, widget: Text, start, end, value: str):
        sep = self._elemental_separator_for_widget(widget)
        before = widget.get("1.0", start)
        stripped_before = before.rstrip()
        adjusted_start = start
        adjusted_end = end
        if stripped_before and not stripped_before.endswith((sep, ":", "[", "{", "\n")):
            last_piece = re.split(r"[\s,;]+", stripped_before)[-1]
            if self._is_plain_number(last_piece):
                widget.insert(start, sep)
                adjusted_start = widget.index(f"{start}+1c")
                adjusted_end = widget.index(f"{end}+1c")
        widget.delete(adjusted_start, adjusted_end)
        widget.insert(adjusted_start, value)
        widget.mark_set("insert", f"{adjusted_start}+{len(value)}c")
        self._refresh_elemental_invalid_highlight(widget)

    def _refresh_signature_invalid_highlight(self, widget: Text):
        self._clear_syntax_highlight(widget)
        try:
            text = widget.get("1.0", "end-1c")
        except Exception:
            return
        valid_signatures = {
            "@c", "@cosine", "@f", "@fine", "@s", "@seed",
            "@m", "@mode", "@p", "@precision", "@rank", "@arch",
        }
        valid_values = {
            "@m": {m["key"].lower() for m in self.merge_modes},
            "@mode": {m["key"].lower() for m in self.merge_modes},
            "@p": {"half", "bhalf", "bf16", "quarter", "fp8", "fp32", "full"},
            "@precision": {"half", "bhalf", "bf16", "quarter", "fp8", "fp32", "full"},
            "@arch": {"sd15", "sd1.5", "sdxl", "flux", "zimage", "anima"},
        }
        expecting = None
        for match in re.finditer(r"\S+", text):
            token = match.group(0)
            lower = token.lower()
            start, end = match.start(), match.end()
            if token.startswith("@"):
                if lower in valid_signatures:
                    self._add_tag_for_span(widget, "signature_name", start, end)
                    expecting = lower
                else:
                    self._add_tag_for_span(widget, "invalid_token", start, end)
                    expecting = None
                continue
            if token.startswith("--"):
                self._add_tag_for_span(widget, "signature_name", start, end)
                expecting = None
                continue
            if self._is_plain_number(token):
                self._add_tag_for_span(widget, "numeric_value", start, end)
                expecting = None
                continue
            if expecting in valid_values:
                if lower in valid_values[expecting]:
                    self._add_tag_for_span(widget, "signature_value", start, end)
                else:
                    self._add_tag_for_span(widget, "invalid_token", start, end)
            expecting = None

    def _refresh_elemental_invalid_highlight(self, widget: Text):
        self._clear_syntax_highlight(widget)
        try:
            text = widget.get("1.0", "end-1c")
        except Exception:
            return
        catalog = self._load_elemental_catalog()
        valid_layers = {str(item.get("name", "")).lower() for item in catalog.get("layers", []) if str(item.get("name", "")).strip()}
        token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-*]*")
        sep = self._elemental_separator_for_widget(widget)

        for m in re.finditer(r"[:,;\[\]{}]", text):
            self._add_tag_for_span(widget, "delimiter", m.start(), m.end())
        for m in re.finditer(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
            self._add_tag_for_span(widget, "numeric_value", m.start(), m.end())

        split_re = re.compile(r"[\n" + re.escape(sep) + r"]")
        segment_start = 0
        for split in list(split_re.finditer(text)) + [None]:
            segment_end = split.start() if split is not None else len(text)
            segment = text[segment_start:segment_end]
            if not segment:
                segment_start = (split.end() if split is not None else len(text))
                continue

            first_colon = segment.find(":")
            second_colon = segment.find(":", first_colon + 1) if first_colon >= 0 else -1

            if first_colon < 0:
                for match in re.finditer(r"\S+", segment):
                    token = match.group(0)
                    global_start = segment_start + match.start()
                    global_end = segment_start + match.end()
                    resolved = self._resolve_elemental_layers_expr(token)
                    if resolved:
                        self._add_tag_for_span(widget, "elemental_layer", global_start, global_end)
                    else:
                        for inner in token_re.finditer(token):
                            inner_token = inner.group(0)
                            inner_norm = self._normalize_elemental_layer_name(inner_token).lower()
                            inner_start = global_start + inner.start()
                            inner_end = global_start + inner.end()
                            if inner_norm in valid_layers:
                                self._add_tag_for_span(widget, "elemental_layer", inner_start, inner_end)
                            else:
                                self._add_tag_for_span(widget, "invalid_token", inner_start, inner_end)
            else:
                layer_part = segment[:first_colon]
                layer_names: List[str] = []
                for match in re.finditer(r"\S+", layer_part):
                    token = match.group(0)
                    global_start = segment_start + match.start()
                    global_end = segment_start + match.end()
                    resolved = self._resolve_elemental_layers_expr(token)
                    if resolved:
                        self._add_tag_for_span(widget, "elemental_layer", global_start, global_end)
                        for layer_name in resolved:
                            if layer_name not in layer_names:
                                layer_names.append(layer_name)
                    else:
                        for inner in token_re.finditer(token):
                            inner_token = inner.group(0)
                            inner_norm = self._normalize_elemental_layer_name(inner_token).lower()
                            inner_start = global_start + inner.start()
                            inner_end = global_start + inner.end()
                            if inner_norm in valid_layers:
                                self._add_tag_for_span(widget, "elemental_layer", inner_start, inner_end)
                            else:
                                self._add_tag_for_span(widget, "invalid_token", inner_start, inner_end)

                element_end = len(segment) if second_colon < 0 else second_colon
                element_part = segment[first_colon + 1:element_end]
                layer_items = self._elemental_valid_elements_for_layers(layer_names)
                valid_element_names = {
                    str(item.get("name", "")).lower()
                    for item in layer_items
                    if str(item.get("name", "")).strip()
                }
                valid_element_aliases = set()
                for layer_name in layer_names:
                    valid_element_aliases.update(self._elemental_alias_lookup(layer_name).keys())

                for match in token_re.finditer(element_part):
                    token = match.group(0)
                    token_lower = token.lower()
                    global_start = segment_start + first_colon + 1 + match.start()
                    global_end = segment_start + first_colon + 1 + match.end()
                    if token_lower in valid_element_names or token_lower in valid_element_aliases:
                        self._add_tag_for_span(widget, "elemental_element", global_start, global_end)
                    else:
                        self._add_tag_for_span(widget, "invalid_token", global_start, global_end)

            segment_start = (split.end() if split is not None else len(text))

    def _entry_token_bounds(self, widget: Entry):
        try:
            text = widget.get()
            pos = int(widget.index("insert"))
        except Exception:
            return "", 0, 0
        left = re.split(r"[\s,]", text[:pos])[-1]
        start = pos - len(left)
        return left, start, pos

    def _text_token_bounds(self, widget: Text):
        try:
            insert = widget.index("insert")
            line, col = map(int, insert.split("."))
            before = widget.get(f"{line}.0", insert)
        except Exception:
            return "", None, None
        token = re.split(r"[\s,]", before)[-1]
        start = f"{line}.{max(0, len(before) - len(token))}"
        return token, start, insert

    def _filter_candidates(self, token: str, candidates) -> List[str]:
        raw = str(token or "")
        probe = raw.lower()
        probe2 = probe.lstrip("[{")
        if not probe:
            return []
        out = []
        for item in candidates:
            value = item.get("insert") if isinstance(item, dict) else item
            low = str(value).lower()
            if low.startswith(probe) or (probe2 and (low.startswith(probe2) or low.startswith("[" + probe2) or low.startswith("{" + probe2))):
                out.append(item)
        return out[:20]

    def _bind_entry_autocomplete(self, widget: Entry, provider):
        def refresh(_event=None):
            token, start, end = self._entry_token_bounds(widget)
            suggestions = self._filter_candidates(token, provider())
            if not suggestions:
                self._hide_autocomplete()
                return
            def apply(value: str):
                widget.delete(start, end)
                widget.insert(start, value)
            self._show_autocomplete(widget, suggestions, apply)
        widget.bind("<KeyRelease>", refresh, add="+")
        widget.bind("<Tab>", self._accept_autocomplete, add="+")
        widget.bind("<Return>", self._accept_autocomplete, add="+")
        widget.bind("<Escape>", self._hide_autocomplete, add="+")
        widget.bind("<Down>", lambda _e: self._move_autocomplete(1), add="+")
        widget.bind("<Up>", lambda _e: self._move_autocomplete(-1), add="+")
        widget.bind("<FocusOut>", self._hide_autocomplete, add="+")

    def _bind_text_autocomplete(self, widget: Text, provider, *, mode: str = "plain", validator=None):
        def run_validator():
            if validator is not None:
                try:
                    validator(widget)
                except Exception:
                    pass

        def refresh(event=None):
            keysym = getattr(event, "keysym", "")
            run_validator()
            if keysym in {"Up", "Down", "Return", "Escape", "Tab"}:
                return

            token, start, end = self._text_token_bounds(widget)
            if mode == "elemental":
                ctx = self._elemental_context(widget)
                start = ctx.get("start")
                end = ctx.get("end")
                try:
                    suggestions = provider(widget, token)
                except TypeError:
                    suggestions = provider(widget)
            else:
                try:
                    suggestions = provider(widget, token)
                except TypeError:
                    suggestions = self._filter_candidates(token, provider())

            if not suggestions or start is None or end is None:
                self._hide_autocomplete()
                return

            def apply(value: str):
                if mode == "signature":
                    self._apply_signature_completion(widget, start, end, value)
                elif mode == "elemental":
                    self._apply_elemental_completion(widget, start, end, value)
                else:
                    widget.delete(start, end)
                    widget.insert(start, value)
                    widget.mark_set("insert", f"{start}+{len(value)}c")
                try:
                    widget.focus_force()
                    widget.event_generate("<KeyRelease>")
                except Exception:
                    pass

            self._show_autocomplete(widget, suggestions, apply)

        def on_tab(_event=None):
            if self._autocomplete_target is widget and self._autocomplete_has_items():
                return self._accept_autocomplete()
            widget.insert("insert", "    ")
            run_validator()
            return "break"

        def on_return(event=None):
            if self._autocomplete_target is widget and self._autocomplete_has_items():
                return self._accept_autocomplete()
            if self._is_shift_pressed(event):
                widget.insert("insert", "\n")
                run_validator()
            return "break"

        widget.bind("<KeyRelease>", refresh, add="+")
        widget.bind("<Tab>", on_tab, add="+")
        widget.bind("<Return>", on_return, add="+")
        widget.bind("<Escape>", self._hide_autocomplete, add="+")
        widget.bind("<Down>", lambda _e: self._move_autocomplete(1) if self._autocomplete_target is widget else None, add="+")
        widget.bind("<Up>", lambda _e: self._move_autocomplete(-1) if self._autocomplete_target is widget else None, add="+")
        widget.bind("<FocusOut>", self._hide_autocomplete, add="+")
        widget.bind("<FocusIn>", lambda _e: widget.after_idle(run_validator), add="+")
        widget.bind("<Button-1>", lambda _e: widget.after(1, widget.focus_force), add="+")
        widget.bind("<ButtonRelease-1>", lambda _e: widget.after_idle(run_validator), add="+")
        widget.bind("<<Paste>>", lambda _e: widget.after_idle(run_validator), add="+")

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

    def _on_elemental_text_motion(self, widget: Text, event):
        token, start, _end, _index = self._extract_elemental_hover_token(widget, event)
        if not token:
            self._hide_elemental_alias_hover()
            return None
        try:
            full_text = widget.get("1.0", "end-1c")
            seg_abs = widget.count("1.0", start, "chars")[0]
        except Exception:
            self._hide_elemental_alias_hover()
            return None
        left_text = full_text[:seg_abs]
        segment_start = max(left_text.rfind(","), left_text.rfind(";"), left_text.rfind("\n")) + 1
        segment_left = full_text[segment_start:seg_abs]
        if segment_left.count(":") != 1:
            self._hide_elemental_alias_hover()
            return None
        layer_expr = segment_left.split(":", 1)[0]
        layer_names = self._resolve_elemental_layers_expr(layer_expr)
        alias_info = self._elemental_alias_lookup(layer_names).get(token.lower()) or self._elemental_alias_lookup().get(token.lower())
        if alias_info:
            self._show_elemental_alias_hover_popup(widget, alias_info, event)
        else:
            self._hide_elemental_alias_hover()
        return None

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

    def _build_left_panel(self, parent):
        main_cfg = self._build_collapsible_section(
            parent,
            "Main Config",
            key="left_main_config",
            default_open=True,
            body_fill="x",
            body_expand=False,
        )

        base_model_frame = Frame(main_cfg)
        base_model_frame.pack(anchor="nw", fill="x", pady=(2, 0))
        base_model_label = Label(base_model_frame, text="Base Model", font=("MS Gothic", 9), width=18, anchor="w")
        base_model_label.pack(side="left")
        self.base_model_var = tk.StringVar(value=self.config.get("base_model", "SDXL"))
        self.base_model_combo = ttk.Combobox(base_model_frame, textvariable=self.base_model_var, values=BASE_MODEL_OPTIONS, state="readonly")
        self._bind_combobox_mousewheel_passthrough(self.base_model_combo)
        self.base_model_combo.pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.base_model_combo.bind("<<ComboboxSelected>>", self._on_base_model_change)
        self._attach_tooltip(base_model_label, self._left_help("Base Model").get("detail", ""))
        self._add_inline_help(main_cfg, "Base Model")

        fields_frame = Frame(main_cfg)
        fields_frame.pack(fill="x", pady=(2, 10))

        def add_left_entry_row(parent_widget, key, text, has_button=False, command=None):
            frame = Frame(parent_widget)
            frame.pack(anchor="nw", fill="x", pady=(2, 0))
            row_widgets = [frame]
            label_widget = Label(frame, text=text, font=("MS Gothic", 9), width=18, anchor="w")
            label_widget.pack(side="left")
            row_widgets.append(label_widget)
            if key == "filepath":
                new_btn = ttk.Button(frame, text="New", width=3, command=self._new_plan)
                new_btn.pack(side="left", padx=(4, 0))
                row_widgets.append(new_btn)
            entry = Entry(frame)
            if "Token" in text or "API" in text:
                entry.config(show="*")
            entry.insert(0, self.config.get(key, INIT_CONFIG.get(key, "")))
            entry.pack(side="left", expand=True, fill="x", padx=(5, 5 if has_button or key == "filepath" else 0))
            row_widgets.append(entry)
            self.entries[key] = entry
            entry.bind("<KeyRelease>", lambda _e: self._schedule_config_save())
            if key == "filepath":
                load_btn = ttk.Button(frame, text="Load", width=4, command=self._load_plan_from_path)
                load_btn.pack(side="right", padx=(0, 4))
                row_widgets.append(load_btn)
            if has_button:
                browse_btn = ttk.Button(frame, text="📂", width=3, command=command)
                browse_btn.pack(side="right")
                row_widgets.append(browse_btn)
            self._attach_tooltip(label_widget, self._left_help(text).get("detail", ""))
            self._add_inline_help(parent_widget, text)
            return row_widgets

        fields = [
            ("HuggingAPI", "HuggingFace Token", False, None),
            ("CivitAPI", "CivitAI API", False, None),
            ("filepath", "Plan Text Path", True, self._browse_plan_file),
            ("workpath", "Workspace Path", False, None),
            ("model_dir", "Model Dir (Opt.)", True, lambda: self._select_directory("model_dir")),
            ("title", "Notebook Title", False, None),
            ("UR", "User/Repo ID", False, None),
        ]

        for key, text, has_button, command in fields:
            add_left_entry_row(fields_frame, key, text, has_button, command)

        vae_outer = Frame(fields_frame)
        vae_outer.pack(anchor="nw", fill="x", pady=(8, 2))
        vae_check_col = Frame(vae_outer)
        vae_check_col.pack(side="left", anchor="n", padx=(0, 6), pady=(16, 0))
        vae_check = ttk.Checkbutton(
            vae_check_col,
            text="Bake",
            variable=self.bake_vae_var,
            command=self._on_bake_vae_toggle,
        )
        vae_check.pack(anchor="n")
        self._attach_tooltip(vae_check, self._left_help("Bake VAE").get("detail", ""))
        vae_box = LabelFrame(vae_outer, text="VAE", padx=6, pady=6)
        vae_box.pack(side="left", fill="x", expand=True)
        self._attach_tooltip(vae_box, self._left_help("Bake VAE").get("detail", ""))
        self.vae_field_widgets = []
        for key, text, has_button, command in [
            ("vae_dir", "VAE Dir (Opt.)", True, lambda: self._select_directory("vae_dir")),
            ("vae", "VAE URL", False, None),
            ("vae_name", "VAE Name", False, None),
        ]:
            self.vae_field_widgets.extend(add_left_entry_row(vae_box, key, text, has_button, command))
        self._add_inline_help(fields_frame, "Bake VAE")
        self._on_bake_vae_toggle(save=False)

        notebook_frame = LabelFrame(parent, text="Notebook Output", padx=5, pady=5)
        notebook_frame.pack(fill="x", pady=(5, 10))
        self._attach_tooltip(notebook_frame, self._left_help("Notebook Output").get("detail", ""))
        self.notebook_path_var = tk.StringVar(value=self.config.get("last_notebook_path", ""))
        self.executed_notebook_path_var = tk.StringVar(value=self.config.get("last_executed_notebook_path", ""))
        source_label = Label(notebook_frame, text="Source:", width=10, anchor="w")
        source_label.grid(row=0, column=0, sticky="w")
        source_entry = Entry(notebook_frame, textvariable=self.notebook_path_var, state="readonly")
        source_entry.grid(row=0, column=1, sticky="ew", padx=4)
        executed_label = Label(notebook_frame, text="Executed:", width=10, anchor="w")
        executed_label.grid(row=1, column=0, sticky="w")
        executed_entry = Entry(notebook_frame, textvariable=self.executed_notebook_path_var, state="readonly")
        executed_entry.grid(row=1, column=1, sticky="ew", padx=4)
        self._attach_tooltip([source_label, executed_label], self._left_help("Notebook Output").get("detail", ""))
        notebook_frame.grid_columnconfigure(1, weight=1)
        self._add_inline_help(parent, "Notebook Output", padx=10, wraplength=540)

        install_opts = self._build_collapsible_section(
            parent,
            "Notebook Option",
            key="left_notebook_option",
            default_open=True,
            body_fill="x",
            body_expand=False,
        )
        self.ignore_install_deps_var = tk.BooleanVar(value=bool(self.config.get("ignore_install_deps", False)))
        self.use_online_var = tk.BooleanVar(value=bool(self.config.get("use_online", False)))
        self.upload_after_merge_var = tk.BooleanVar(value=bool(self.config.get("upload_after_merge", False)))
        self.run_t2i_var = tk.BooleanVar(value=bool(self.config.get("run_t2i", False)))
        use_online_cb = ttk.Checkbutton(
            install_opts,
            text="Use online",
            variable=self.use_online_var,
            command=self._schedule_config_save,
        )
        use_online_cb.pack(anchor="w")
        self._add_inline_help(install_opts, "Use Online")
        ignore_cb = ttk.Checkbutton(
            install_opts,
            text="Ignore Install Deps",
            variable=self.ignore_install_deps_var,
            command=self._schedule_config_save,
        )
        ignore_cb.pack(anchor="w")
        self._add_inline_help(install_opts, "Ignore Install Deps")
        upload_cb = ttk.Checkbutton(
            install_opts,
            text="Upload After Merge",
            variable=self.upload_after_merge_var,
            command=self._schedule_config_save,
        )
        upload_cb.pack(anchor="w")
        self._add_inline_help(install_opts, "Upload After Merge")
        run_t2i_cb = ttk.Checkbutton(
            install_opts,
            text="Run T2I",
            variable=self.run_t2i_var,
            command=self._schedule_config_save,
        )
        run_t2i_cb.pack(anchor="w")
        self._add_inline_help(install_opts, "Run T2I")

        run_frame = Frame(parent)
        run_frame.pack(fill="x", pady=(8, 0))
        for col in range(2):
            run_frame.grid_columnconfigure(col, weight=1)
        action_buttons = [
            ("▶ Run Merge Notebook", self._run_target_notebook),
            ("💾 Save Plan Text", self._save_plan_text_button),
            ("📝 Export as notebook", self._export_as_notebook),
            ("📄 Export as txt", self._export_as_txt),
            ("📺 Show Console", self.console.show),
            ("⬆ Upload Latest Model", self._upload_merge_result),
        ]
        for idx, (text, command) in enumerate(action_buttons):
            row, col = divmod(idx, 2)
            grid_row = row * 2
            columnspan = 2 if idx == len(action_buttons) - 1 else 1
            btn = ttk.Button(run_frame, text=text, command=command)
            btn.grid(row=grid_row, column=col, columnspan=columnspan, sticky="ew", padx=4, pady=4)
            self._attach_tooltip(btn, self._left_help(text).get("detail", ""))
            help_container = Frame(run_frame)
            help_container.grid(row=grid_row + 1, column=col, columnspan=columnspan, sticky="ew", padx=4)
            self._add_inline_help(help_container, text, padx=0, pady=(0, 4), wraplength=240 if columnspan == 1 else 520)

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
    def _build_status_bar(self, parent):
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

    @staticmethod
    def _memo_preview_text(text: str, head: int = 12, tail: int = 10) -> str:
        raw = " ".join(str(text or "").split())
        if not raw:
            return ""
        if len(raw) <= head + tail + 3:
            return raw
        return f"{raw[:head]}…{raw[-tail:]}"

    def _build_plan_item_tooltip_text(self, model_idx: int, entry: Dict[str, Any], problems: List[str] | None = None) -> str:
        problems = problems or []
        lines = [f"Line {model_idx + 1}", self._line_summary(entry)]
        if problems:
            lines += ["", "Issues:"] + [f"• {problem}" for problem in problems]
        memo = str(entry.get("memo") or "").strip()
        if memo:
            lines += ["", "Memo:", memo]
        return "\n".join(lines)

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

    def _on_plan_list_motion(self, event):
        if self.plan_listbox is None or not self.visible_entry_indices:
            self._hide_plan_item_hover()
            return
        idx = self.plan_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.visible_entry_indices):
            self._hide_plan_item_hover()
            return
        bbox = self.plan_listbox.bbox(idx)
        if not bbox:
            self._hide_plan_item_hover()
            return
        bx, by, bw, bh = bbox
        if not (by <= event.y <= by + bh):
            self._hide_plan_item_hover()
            return
        model_idx = self.visible_entry_indices[idx]
        try:
            font = tkfont.Font(font=self.plan_listbox.cget("font"))
            hover_width = max(18, font.measure(f"{model_idx + 1:02d}") + 8)
        except Exception:
            hover_width = 28
        if not (bx <= event.x <= min(bx + bw, bx + hover_width)):
            self._hide_plan_item_hover()
            return
        entry = self.plan_data.get("entries", [])[model_idx]
        problems = getattr(self, "_plan_problem_map_cache", {}).get(model_idx, [])
        text = self._build_plan_item_tooltip_text(model_idx, entry, problems)
        if idx == getattr(self, "_plan_hover_index", None) and text == getattr(self, "_plan_hover_text", ""):
            return
        self._plan_hover_index = idx
        self._plan_hover_text = text
        self._show_plan_item_hover(event, text)

    def _entry_type_color(self, etype: str) -> str:
        return {
            "Download Model": "#1f5fbf",
            "Local Model": "#2e7d32",
            "Remove Model": "#6b7280",
            "Checkpoint Merge": "#7c3aed",
            "LoRA Bake": "#d97706",
        }.get(str(etype or ""), "#333333")

    def _plan_entry_problem_map(self) -> Dict[int, List[str]]:
        problems_by_idx: Dict[int, List[str]] = {}
        available = {"Checkpoint": set(), "LoRA": set(), "LyCORIS": set()}
        for idx, entry in enumerate(self.plan_data.get("entries", []), start=1):
            etype = entry.get("type")
            prefix = f"line {idx} ({etype})"
            problems: List[str] = []
            if etype == "Download Model":
                if not entry.get("model_name"):
                    problems.append(f"{prefix}: model_name is empty")
                if not entry.get("link"):
                    problems.append(f"{prefix}: link is empty")
                name = (entry.get("model_name") or "").strip()
                kind = (entry.get("model_type") or "Checkpoint").strip()
                if name:
                    available.setdefault(kind, set()).add(name)
            elif etype == "Local Model":
                if not entry.get("local_path"):
                    problems.append(f"{prefix}: local_path is empty")
                path = (entry.get("local_path") or "").strip()
                kind = (entry.get("model_type") or "Checkpoint").strip()
                if path:
                    available.setdefault(kind, set()).add(Path(path).stem)
            elif etype == "Remove Model":
                if not entry.get("model"):
                    problems.append(f"{prefix}: model is empty")
            elif etype == "Checkpoint Merge":
                for req_key in ("model0", "model1", "output_name"):
                    if not entry.get(req_key):
                        problems.append(f"{prefix}: {req_key} is empty")
                for ref in (entry.get("model0"), entry.get("model1"), entry.get("model2")):
                    if ref and ref not in available["Checkpoint"]:
                        problems.append(f"{prefix}: checkpoint ref not available -> {ref}")
                if entry.get("output_name"):
                    available["Checkpoint"].add(entry["output_name"])
            elif etype == "LoRA Bake":
                if not entry.get("checkpoint"):
                    problems.append(f"{prefix}: checkpoint is empty")
                elif entry.get("checkpoint") not in available["Checkpoint"]:
                    problems.append(f"{prefix}: checkpoint ref not available -> {entry.get('checkpoint')}")
                if not entry.get("output_name"):
                    problems.append(f"{prefix}: output_name is empty")
                for lora in entry.get("loras", []):
                    name = lora.get("name")
                    if not name:
                        problems.append(f"{prefix}: one LoRA name is empty")
                    elif name not in available["LoRA"] and name not in available["LyCORIS"]:
                        problems.append(f"{prefix}: LoRA ref not available -> {name}")
                if entry.get("output_name"):
                    available["Checkpoint"].add(entry["output_name"])
            problems_by_idx[idx - 1] = problems
        return problems_by_idx

    def _apply_plan_listbox_item_styles(self):
        if self.plan_listbox is None:
            return
        problem_map = getattr(self, "_plan_problem_map_cache", {})
        for vis_idx, model_idx in enumerate(self.visible_entry_indices):
            entry = self.plan_data.get("entries", [])[model_idx]
            problems = problem_map.get(model_idx, [])
            fg = "#cc2222" if problems else self._entry_type_color(entry.get("type", ""))
            try:
                self.plan_listbox.itemconfig(vis_idx, fg=fg)
            except Exception:
                try:
                    self.plan_listbox.itemconfigure(vis_idx, foreground=fg)
                except Exception:
                    pass

    def _build_right_panel(self, parent):
        top_frame = Frame(parent)
        top_frame.pack(anchor="nw", fill="x", pady=(0, 10))
        title_label = Label(top_frame, text="Plan Creator", font=("MS Gothic", 16, "bold"), fg="#225588")
        title_label.pack(side="left")
        reset_btn = ttk.Button(top_frame, text="Reset Plan", command=self._reset_plan)
        reset_btn.pack(side="right", padx=5)
        self._attach_tooltip(title_label, self._right_help("Plan Creator").get("detail", ""))
        self._attach_tooltip(reset_btn, self._right_help("Reset Plan").get("detail", ""))
        self._add_right_inline_help(parent, "Plan Creator", padx=4, wraplength=760)

        selector_frame = Frame(parent, bg="#eeeeee", pady=6)
        selector_frame.pack(fill="x")
        target_label = Label(selector_frame, text="Target Line:", bg="#eeeeee")
        target_label.grid(row=0, column=0, rowspan=2, sticky="w", padx=5)
        self.selection_var = tk.StringVar(master=self.root, value="")
        self.selection_combo = ttk.Combobox(selector_frame, textvariable=self.selection_var, state="readonly", width=30)
        self.selection_combo.grid(row=0, column=1, rowspan=2, sticky="ew", padx=5, pady=1)
        self.selection_combo.bind("<<ComboboxSelected>>", self._on_line_selection_change)

        add_remove_frame = Frame(selector_frame, bg="#eeeeee")
        add_remove_frame.grid(row=0, column=2, rowspan=2, sticky="ns", padx=2)
        add_btn = ttk.Button(add_remove_frame, text="+", width=3, command=self._add_line)
        add_btn.pack(fill="x", pady=(0, 2))
        remove_btn = ttk.Button(add_remove_frame, text="-", width=3, command=self._remove_line)
        remove_btn.pack(fill="x")

        move_frame = Frame(selector_frame, bg="#eeeeee")
        move_frame.grid(row=0, column=3, rowspan=2, sticky="ns", padx=2)
        up_btn = ttk.Button(move_frame, text="↑", width=3, command=self._move_line_up)
        up_btn.pack(fill="x", pady=(0, 2))
        down_btn = ttk.Button(move_frame, text="↓", width=3, command=self._move_line_down)
        down_btn.pack(fill="x")
        self.plan_move_up_button = up_btn
        self.plan_move_down_button = down_btn

        reload_btn = ttk.Button(selector_frame, text="↻ Reload File", command=self._load_plan_from_path)
        reload_btn.grid(row=0, column=4, rowspan=2, sticky="e", padx=5)

        self._attach_tooltip([target_label, self.selection_combo], self._right_help("Target Line").get("detail", ""))
        self._attach_tooltip(add_btn, self._right_help("Add Line").get("detail", ""))
        self._attach_tooltip(remove_btn, self._right_help("Remove Line").get("detail", ""))
        self._attach_tooltip(up_btn, self._right_help("Move Line Up").get("detail", ""))
        self._attach_tooltip(down_btn, self._right_help("Move Line Down").get("detail", ""))
        self._attach_tooltip(reload_btn, self._right_help("Reload Plan File").get("detail", ""))

        selector_frame.grid_columnconfigure(1, weight=1)
        self._add_right_inline_help(parent, "Target Line", padx=4, wraplength=760)

        self.canvas_container = Frame(parent)
        self.canvas_container.pack(fill="both", expand=True, pady=5)
        self.canvas = Canvas(self.canvas_container)
        self.scrollbar = Scrollbar(self.canvas_container, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = Frame(self.canvas)
        self.scroll_frame.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

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

    def _on_root_resize(self, event=None):
        if event is not None and getattr(event, "widget", None) is not self.root:
            return
        self._schedule_responsive_layout()
    # ---------------- utility ----------------
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

    def _normalize_user_path(self, value: str | os.PathLike | None) -> str:
        raw = os.fspath(value) if value is not None else ""
        raw = str(raw).strip()
        if not raw:
            return ""
        try:
            raw = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            pass
        try:
            raw = os.path.normpath(raw)
        except Exception:
            pass
        return raw

    def _paths_equal(self, left: str | os.PathLike | None, right: str | os.PathLike | None) -> bool:
        l = self._normalize_user_path(left)
        r = self._normalize_user_path(right)
        if not l or not r:
            return False
        try:
            return os.path.normcase(l) == os.path.normcase(r)
        except Exception:
            return l == r

    def _path_display_name(self, value: str | os.PathLike | None) -> str:
        normalized = self._normalize_user_path(value)
        return Path(normalized).name if normalized else ""

    def _bind_combobox_mousewheel_passthrough(self, combo: ttk.Combobox):
        def handler(event, fallback_units: int | None = None):
            self._scroll_canvas_by_event(event, fallback_units=fallback_units)
            return "break"
        combo.bind("<MouseWheel>", lambda e: handler(e), add="+")
        combo.bind("<Button-4>", lambda e: handler(e, -1), add="+")
        combo.bind("<Button-5>", lambda e: handler(e, 1), add="+")
        return combo

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

    def _set_active_multiline_text(self, widget: Text | None):
        try:
            if widget is not None and widget.winfo_exists():
                self._active_multiline_text = widget
                return
        except Exception:
            pass
        self._active_multiline_text = None

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

    def _bind_multiline_text_scroll_priority(self, widget: Text):
        widget.bind("<Enter>", lambda _e, w=widget: self._set_active_multiline_text(w), add="+")
        widget.bind("<Leave>", lambda _e: self._set_active_multiline_text(None), add="+")
        widget.bind("<FocusIn>", lambda _e, w=widget: self._set_active_multiline_text(w), add="+")
        widget.bind("<FocusOut>", lambda _e: self._set_active_multiline_text(None), add="+")
        widget.bind("<Button-1>", lambda _e, w=widget: widget.after_idle(lambda: self._set_active_multiline_text(w)), add="+")
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

    def _save_current_state_to_config(self):
        for key, widget in self.entries.items():
            self.config[key] = widget.get()
        self.config["hf_repo_id"] = self.entries["UR"].get()
        self.config["last_notebook_path"] = self.notebook_path_var.get()
        self.config["last_executed_notebook_path"] = self.executed_notebook_path_var.get()
        self.config["base_model"] = self.base_model_var.get()
        self.config["active_view"] = getattr(self, "active_view", "split")
        self.config["theme_mode"] = getattr(self, "theme_mode", "dark")
        self.config["ignore_install_deps"] = bool(getattr(self, "ignore_install_deps_var", tk.BooleanVar(value=False)).get())
        self.config["use_online"] = bool(getattr(self, "use_online_var", tk.BooleanVar(value=False)).get())
        self.config["upload_after_merge"] = bool(getattr(self, "upload_after_merge_var", tk.BooleanVar(value=False)).get())
        self.config["run_t2i"] = bool(getattr(self, "run_t2i_var", tk.BooleanVar(value=False)).get())
        self.config["bake_vae"] = bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get())
        save_config_to_disk(self.config)
    def _restore_session_state(self):
        filepath = self.entries["filepath"].get().strip()
        if filepath and os.path.exists(filepath):
            try:
                self.plan_data = normalize_plan(load_plan_records(filepath))
            except Exception:
                self.plan_data = self._planner_default_visible_plan()
        else:
            self.plan_data = self._planner_default_visible_plan()

    def _browse_plan_file(self):
        path = filedialog.askopenfilename(filetypes=[("Plan File", "*.json *.txt"), ("All", "*.*")])
        path = self._normalize_user_path(path)
        if path:
            self.entries["filepath"].delete(0, tk.END)
            self.entries["filepath"].insert(0, path)
            self._schedule_config_save()

    def _load_plan_from_path(self):
        path = self._normalize_user_path(self.entries["filepath"].get().strip())
        if not path:
            messagebox.showwarning("Plan Path", "Plan Text Path is empty.")
            return
        try:
            self.plan_data = normalize_plan(load_plan_records(path))
            self.current_index = 0
            self._refresh_line_selector()
            self._render_current_line()
            self.status_label.config(text=f"Loaded plan: {self._path_display_name(path)}")
        except Exception as e:
            self._show_detailed_error("Load Error", e)

    def _new_plan(self):
        current_filepath = self._normalize_user_path(self.entries["filepath"].get().strip()) if "filepath" in self.entries else ""
        current_title = self.entries["title"].get().strip() if "title" in self.entries else ""
        initial_name = Path(current_filepath).name if current_filepath else f"{(current_title or 'merge_plan')}.txt"
        initial_dir = str(Path(current_filepath).expanduser().parent) if current_filepath else os.getcwd()

        try:
            path = filedialog.asksaveasfilename(
                title="Create new Plan file",
                defaultextension=".txt",
                initialfile=initial_name,
                initialdir=initial_dir,
                filetypes=[("Plan Text", "*.txt"), ("All files", "*.*")],
            )
            path = self._normalize_user_path(path)
            if not path:
                return
            if not path.lower().endswith(".txt"):
                path += ".txt"

            self.plan_data = self._planner_default_visible_plan()
            self.current_index = 0
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            export_plan_records_txt(path, self.plan_data)

            self.entries["filepath"].delete(0, tk.END)
            self.entries["filepath"].insert(0, path)
            self.entries["title"].delete(0, tk.END)
            self.entries["title"].insert(0, Path(path).stem)
            self.notebook_path_var.set("")
            self.executed_notebook_path_var.set("")

            self._refresh_line_selector()
            self._render_current_line()
            self._save_current_state_to_config()
            self.status_label.config(text=f"Created new plan: {self._path_display_name(path)}")
        except Exception as e:
            self._show_detailed_error("New Plan Error", e, context="Operation: new_plan")
        
    def _reset_plan(self):
        self.plan_data = self._planner_default_visible_plan()
        self.current_index = 0
        self._refresh_line_selector()
        self._render_current_line()
        self.status_label.config(text=f"Reset the Plan")

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
        text = Text(text_frame, wrap="none", font=("Consolas", 11), yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.config(command=text.yview)
        xscroll.config(command=text.xview)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        text.insert("1.0", detail)
        text.configure(padx=14, pady=12, spacing1=3, spacing3=3)
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

    def _show_detailed_error(self, title: str, exc: Exception, *, context: str = ""):
        detail_parts = []
        if context:
            detail_parts.append(context)
        try:
            entry = self.plan_data.get("entries", [])[self.current_index]
            detail_parts.append("Current Plan Entry:\n" + json.dumps(entry, ensure_ascii=False, indent=2))
        except Exception:
            pass
        detail_parts.append(f"Exception: {type(exc).__name__}: {exc}")
        detail_parts.append("Traceback:\n" + traceback.format_exc())
        detail = "\n\n".join(detail_parts)
        try:
            self.console.show()
            self.console.log(detail, "error")
        except Exception:
            pass
        self._show_scrollable_text_dialog(title, detail)

    def _ensure_plan_path(self) -> str:
        path = self._normalize_user_path(self.entries["filepath"].get().strip())
        if not path:
            base_dir = self._normalize_user_path(os.getcwd())
            title = self.entries["title"].get().strip() or "merge_plan"
            path = os.path.join(base_dir, f"{title}.txt")
            self.entries["filepath"].delete(0, tk.END)
            self.entries["filepath"].insert(0, path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save_plan_to_file(self):
        path = self._ensure_plan_path()
        export_plan_records_txt(path, self.plan_data)
        self._schedule_config_save()

    def _save_plan_text_button(self):
        try:
            self._save_plan_to_file()
            path = self._ensure_plan_path()
            self.status_label.config(text=f"Saved plan text: {self._path_display_name(path)}")
            messagebox.showinfo("Save Plan Text", f"Plan text saved successfully.\n\n{path}")
        except Exception as e:
            self._show_detailed_error("Save Plan Text Error", e, context="Operation: save_plan_text")

    def _write_temp_plan_text(self) -> str:
        fd, temp_path = tempfile.mkstemp(prefix="planner_", suffix=".txt")
        # print(temp_path)
        os.close(fd)
        export_plan_records_txt(temp_path, self.plan_data)
        return temp_path

    def _get_notebook_paths(self) -> tuple[str, str]:
        plan_path = Path(self._ensure_plan_path())
        title = (self.entries["title"].get().strip() or plan_path.stem or "merge_plan").replace(" ", "_")
        notebook_path = str(plan_path.with_name(f"{title}.ipynb"))
        executed_path = str(plan_path.with_name(f"{title}.executed.ipynb"))
        return notebook_path, executed_path

    def _suggest_export_txt_path(self) -> str:
        plan_path = Path(self._ensure_plan_path())
        title = (self.entries["title"].get().strip() or plan_path.stem or "merge_plan").replace(" ", "_")
        return str(plan_path.with_name(f"{title}.txt"))

    def _current_entry_context(self) -> str:
        try:
            entries = self.plan_data.get("entries", [])
            if not entries:
                return ""
            entry = entries[self.current_index]
            return "Current Plan Entry:\n" + json.dumps(entry, ensure_ascii=False, indent=2)
        except Exception:
            return ""

    def _show_detailed_error(self, title: str, exc: Exception, *, context: str = "", show_messagebox: bool = True):
        tb = traceback.format_exc()
        detail_parts = []
        if context:
            detail_parts.append(context)
        current_entry = self._current_entry_context()
        if current_entry:
            detail_parts.append(current_entry)
        detail_parts.append(f"Exception: {exc!r}")
        detail_parts.append("Traceback:\n" + tb)
        detail = "\n\n".join(detail_parts)
        try:
            self.console.show()
            self.console.log(detail, "error")
        except Exception:
            pass
        if show_messagebox:
            self._show_scrollable_text_dialog(title, detail)

    def _export_as_txt(self):
        temp_plan_path = None
        try:
            temp_plan_path = self._write_temp_plan_text()
            initial = self.config.get("saveas") or self._suggest_export_txt_path()
            path = filedialog.asksaveasfilename(
                title="Export as txt",
                defaultextension=".txt",
                initialfile=Path(initial).name,
                initialdir=str(Path(initial).parent),
                filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            )
            if not path:
                return
            params = {key: widget.get() for key, widget in self.entries.items()}
            create_plan(
                filepath=temp_plan_path,
                workpath=params.get("workpath", ""),
                saveas=path,
                title=params.get("title", "merge_plan"),
                vae=params.get("vae", ""),
                vae_name=params.get("vae_name", "VAE"),
                bake_vae=bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get()),
                CivitAPI=params.get("CivitAPI", ""),
                HuggingAPI=params.get("HuggingAPI", ""),
                UR=params.get("UR", ""),
                model_dir=params.get("model_dir", ""),
                vae_dir=params.get("vae_dir", ""),
            )
            self.config["saveas"] = path
            self._save_current_state_to_config()
            self.status_label.config(text=f"Exported txt: {os.path.basename(path)}")
            messagebox.showinfo("Export", f"TXT exported successfully.\n\n{path}")
        except Exception as e:
            self._show_detailed_error("Export TXT Error", e, context="Operation: export_as_txt")
        finally:
            if temp_plan_path and os.path.exists(temp_plan_path):
                try:
                    os.remove(temp_plan_path)
                except Exception:
                    pass

    def _export_as_notebook(self):
        try:
            temp_plan_path = self._write_temp_plan_text()
            default_nb_path, _ = self._get_notebook_paths()
            path = filedialog.asksaveasfilename(
                title="Export notebook",
                defaultextension=".ipynb",
                initialfile=Path(default_nb_path).name,
                initialdir=str(Path(default_nb_path).parent),
                filetypes=[("Jupyter Notebook", "*.ipynb"), ("All files", "*.*")],
            )
            if not path:
                return
            params = {key: widget.get() for key, widget in self.entries.items()}
            create_plan_ipynb(
                filepath=temp_plan_path,
                workpath=params.get("workpath", ""),
                saveas=path,
                title=params.get("title", "merge_plan"),
                vae=params.get("vae", ""),
                vae_name=params.get("vae_name", "VAE"),
                bake_vae=bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get()),
                CivitAPI=params.get("CivitAPI", ""),
                HuggingAPI=params.get("HuggingAPI", ""),
                UR=params.get("UR", ""),
                model_dir=params.get("model_dir", ""),
                vae_dir=params.get("vae_dir", ""),
                ignore_install_deps=bool(getattr(self, "ignore_install_deps_var", tk.BooleanVar(value=False)).get()),
                use_online=bool(getattr(self, "use_online_var", tk.BooleanVar(value=False)).get()),
                upload_after_merge=bool(getattr(self, "upload_after_merge_var", tk.BooleanVar(value=False)).get()),
                run_t2i=bool(getattr(self, "run_t2i_var", tk.BooleanVar(value=False)).get()),
                t2i_settings=self._planner_get_t2i_settings() if hasattr(self, "_planner_get_t2i_settings") else None
            )
            self.notebook_path_var.set(path)
            self.config["saveas"] = path
            self._save_current_state_to_config()
            self.status_label.config(text=f"Exported notebook: {os.path.basename(path)}")
            messagebox.showinfo("Export", f"Notebook exported successfully.\n\n{path}")
        except Exception as e:
            self._show_detailed_error("Export Notebook Error", e, context="Operation: export_as_notebook")
        finally:
            if temp_plan_path and os.path.exists(temp_plan_path):
                try:
                    os.remove(temp_plan_path)
                except Exception:
                    pass

    def _on_base_model_change(self, _event=None):
        self._schedule_config_save()
        if self.plan_data.get("entries"):
            canvas = getattr(self, "canvas", None)
            yview = None
            try:
                if canvas is not None and canvas.winfo_exists():
                    yview = canvas.yview()
            except Exception:
                yview = None
            self._render_current_line()
            if yview is not None:
                try:
                    if canvas is not None and canvas.winfo_exists():
                        canvas.update_idletasks()
                        canvas.yview_moveto(float(yview[0]))
                except Exception:
                    pass

    def _current_block_names(self) -> List[str]:
        base_model = self.base_model_var.get() if hasattr(self, "base_model_var") else self.config.get("base_model", "SDXL")
        names = self.block_sets.get(base_model) or self.block_sets.get("SDXL") or list(SDXL_BLOCKS)
        return [str(name) for name in names]

    def _check_aria2(self):
        try:
            result = subprocess.run(["aria2c", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                messagebox.showinfo("aria2 Status", f"✓ aria2 is installed\n\n{result.stdout.splitlines()[0]}")
            else:
                messagebox.showwarning("aria2 Status", "aria2 is not installed or not in PATH.")
        except FileNotFoundError:
            messagebox.showwarning("aria2 Status", "aria2 is not installed or not in PATH.")

    def _select_directory(self, entry_key):
        directory_path = filedialog.askdirectory(title=f"Select {entry_key.replace('_', ' ').title()}")
        directory_path = self._normalize_user_path(directory_path)
        if directory_path:
            self.entries[entry_key].delete(0, tk.END)
            self.entries[entry_key].insert(0, directory_path)
            self._schedule_config_save()

    def _play_notification_sound(self):
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"], check=False)
        except Exception:
            pass

    def _format_exception_text(self, title: str, exc: Exception, *, plan_entry: Dict[str, Any] | None = None) -> str:
        chunks = [title, "", f"Exception: {type(exc).__name__}: {exc}"]

        if isinstance(exc, PlanCompileError):
            if getattr(exc, "entry_index", None) is not None:
                chunks += ["", f"Plan Entry Index: {exc.entry_index}"]
            if getattr(exc, "entry_type", None):
                chunks += [f"Plan Entry Type: {exc.entry_type}"]
            if getattr(exc, "entry_id", None):
                chunks += [f"Plan Entry ID: {exc.entry_id}"]
            if getattr(exc, "cause", None) is not None:
                cause = exc.cause
                chunks += [f"Cause: {type(cause).__name__}: {cause}"]
            payload = getattr(exc, "entry_payload", None) or getattr(exc, "entry", None)
            if payload:
                try:
                    chunks += ["", "Plan Compile Payload:", json.dumps(payload, ensure_ascii=False, indent=2)]
                except Exception:
                    chunks += ["", "Plan Compile Payload:", repr(payload)]
            source_lines = getattr(exc, "source_lines", None) or []
            if source_lines:
                chunks += ["", "Generated Step Source:", "\n".join(f"{i:02d}: {line}" for i, line in enumerate(source_lines, start=1))]

        if plan_entry is not None:
            try:
                chunks += ["", "Current Plan Entry:", json.dumps(plan_entry, ensure_ascii=False, indent=2)]
            except Exception:
                chunks += ["", "Current Plan Entry:", repr(plan_entry)]

        chunks += ["", "Python Traceback:", traceback.format_exc()]
        return "\n".join(chunks)

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

    # ---------------- models/entries ----------------
    def _refresh_line_selector(self):
        values = []
        for i, entry in enumerate(self.plan_data.get("entries", []), start=1):
            summary = self._line_summary(entry)
            values.append(f"{i}. {summary}")
        self.selection_combo["values"] = values
        if values:
            self.current_index = max(0, min(self.current_index, len(values) - 1))
            self.selection_var.set(values[self.current_index])

    def _line_summary(self, entry: Dict[str, Any]) -> str:
        etype = entry.get("type", "Line")
        if etype == "Download Model":
            base = f"Download Model - {entry.get('model_name') or '(unset)'}"
        elif etype == "Local Model":
            base = f"Local Model - {Path(entry.get('local_path') or '').name or '(unset)'}"
        elif etype == "Remove Model":
            base = f"Remove Model - {entry.get('model') or '(unset)'}"
        elif etype == "Checkpoint Merge":
            base = f"Checkpoint Merge - {entry.get('output_name') or '(unset)'}"
        elif etype == "LoRA Bake":
            base = f"LoRA Bake - {entry.get('output_name') or '(unset)'}"
        else:
            base = etype
        memo_preview = self._memo_preview_text(entry.get("memo", ""))
        if memo_preview:
            base += f"  ✎ {memo_preview}"
        return base
    
    def _on_line_selection_change(self, _event=None):
        combo = getattr(self, "selection_combo", None)
        if combo is None:
            return
        value = self.selection_var.get()
        values = list(combo["values"])
        if value in values:
            idx = values.index(value)
            self.current_index = idx
            self._update_current_line_indicator()
            self._render_current_line()
            try:
                self._select_model_indices([idx])
            except Exception:
                pass
            self._update_plan_action_buttons()

    def _add_line(self):
        insert_at = self.current_index + 1
        self.plan_data["entries"].insert(insert_at, make_entry("Checkpoint Merge"))
        self.current_index = insert_at
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._save_plan_to_file()

    def _remove_line(self):
        if len(self.plan_data.get("entries", [])) <= 1:
            messagebox.showwarning("Plan", "At least one line must remain.")
            return
        self.plan_data["entries"].pop(self.current_index)
        self.current_index = max(0, self.current_index - 1)
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._save_plan_to_file()

    def _move_line_up(self):
        entries = self.plan_data.get("entries", [])
        if not entries or self.current_index <= 0:
            return
        idx = self.current_index
        entries[idx - 1], entries[idx] = entries[idx], entries[idx - 1]
        self.current_index = idx - 1
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._save_plan_to_file()

    def _move_line_down(self):
        entries = self.plan_data.get("entries", [])
        if not entries or self.current_index >= len(entries) - 1:
            return
        idx = self.current_index
        entries[idx], entries[idx + 1] = entries[idx + 1], entries[idx]
        self.current_index = idx + 1
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._save_plan_to_file()

    def _collect_available_models(self, upto_index: int) -> Dict[str, List[str]]:
        available: Dict[str, Dict[str, str]] = {"Checkpoint": {}, "LoRA": {}, "LyCORIS": {}}
        removed: set[str] = set()
        for entry in self.plan_data.get("entries", [])[:upto_index]:
            etype = entry.get("type")
            if etype == "Remove Model":
                name = (entry.get("model") or "").strip()
                if name:
                    removed.add(name)
                    available["Checkpoint"].pop(name, None)
                    available["LoRA"].pop(name, None)
                    available["LyCORIS"].pop(name, None)
                continue
            if etype == "Download Model":
                name = (entry.get("model_name") or "").strip()
                kind = (entry.get("model_type") or "Checkpoint").strip()
                if name and name not in removed:
                    available.setdefault(kind, {})[name] = kind
            elif etype == "Local Model":
                path = (entry.get("local_path") or "").strip()
                if path:
                    name = Path(path).stem
                    kind = (entry.get("model_type") or "Checkpoint").strip()
                    if name not in removed:
                        available.setdefault(kind, {})[name] = kind
            elif etype in ("Checkpoint Merge", "LoRA Bake"):
                name = (entry.get("output_name") or "").strip()
                if name and name not in removed:
                    available["Checkpoint"][name] = "Checkpoint"
        return {k: sorted(v.keys()) for k, v in available.items()}

    def _candidate_local_model_roots(self) -> List[Path]:
        roots: List[Path] = []
        model_dir = self._normalize_user_path(self.entries.get("model_dir").get().strip() if self.entries.get("model_dir") else "")
        workpath = self._normalize_user_path(self.entries.get("workpath").get().strip() if self.entries.get("workpath") else "")
 
        if model_dir:
            roots.append(Path(model_dir))
        elif workpath:
            wp = Path(workpath)
            preferred = [wp / "tmp" / "models", wp / "models"]
            existing = [p for p in preferred if p.exists() and p.is_dir()]
            if existing:
                roots.extend(existing)
            else:
                roots.append(preferred[0])
 
        unique: List[Path] = []
        seen = set()
        for root in roots:
            root = Path(self._normalize_user_path(root))
            try:
                key = os.path.normcase(str(root.resolve()))
            except Exception:
                key = os.path.normcase(str(root))
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique

    def _scan_local_model_choices(self) -> Dict[str, str]:
        choices: Dict[str, str] = {}
        exts = {".safetensors", ".ckpt", ".pt", ".bin"}
        for root in self._candidate_local_model_roots():
            root = Path(self._normalize_user_path(root))
            if not root.exists() or not root.is_dir():
                continue
            try:
                walker = os.walk(root, topdown=True, followlinks=False)
            except Exception:
                continue
            for dirpath, dirnames, filenames in walker:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for filename in sorted(filenames):
                    if Path(filename).suffix.lower() not in exts:
                        continue
                    full_path = self._normalize_user_path(Path(dirpath) / filename)
                    display = Path(full_path).name
                    choices.setdefault(display, full_path)
        return choices

    def _build_local_selection_row(self, parent, entry: Dict[str, Any]):
        row = Frame(parent)
        row.pack(fill="x", pady=3)
        label_widget = Label(row, text="Local Selection", width=18, anchor="w")
        label_widget.pack(side="left")
 

        choices_map = self._scan_local_model_choices()
        values = list(choices_map.keys())
        current_path = self._normalize_user_path(str(entry.get("local_path", "") or "").strip())
        current_display = self._path_display_name(current_path) if current_path else ""
        if current_display and current_display not in values:
            values = [current_display] + values

        var = tk.StringVar(value=current_display if current_display else (values[0] if values else ""))
        combo = ttk.Combobox(row, textvariable=var, values=values, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)

        def sync(*_args):
            display = var.get().strip()
            selected_path = choices_map.get(display)
            if selected_path:
                entry["local_path"] = selected_path
            elif current_path and self._path_display_name(current_path) == display:
                entry["local_path"] = current_path
            else:
                entry["local_path"] = display
            self._after_entry_change()

        var.trace_add("write", sync)
        self._attach_tooltip(label_widget, self._right_help("Local Selection").get("detail", ""))
        self._add_right_inline_help(parent, "Local Selection")
        return var

    # ---------------- dynamic editor ----------------
    def _render_current_line(self):
        self._hide_autocomplete()
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        entries = self.plan_data.get("entries", [])
        if not entries:
            return
        entry = entries[self.current_index]
        self.current_editor_widgets = []
        body = self.scroll_frame

        header = LabelFrame(body, text="Line Settings", padx=8, pady=8)
        header.pack(fill="x", padx=6, pady=6)
        self._add_right_inline_help(header, "Line Settings", padx=4, wraplength=720)
        self._build_combo_row(header, "Model Merge Type", entry, "type", LINE_TYPES, self._change_line_type)

        if entry["type"] == "Download Model":
            self._render_download_entry(body, entry)
        elif entry["type"] == "Local Model":
            self._render_local_entry(body, entry)
        elif entry["type"] == "Remove Model":
            self._render_remove_entry(body, entry)
        elif entry["type"] == "Checkpoint Merge":
            self._render_checkpoint_merge_entry(body, entry)
        elif entry["type"] == "LoRA Bake":
            self._render_lora_bake_entry(body, entry)

        self.canvas.yview_moveto(0)
        self._refresh_line_selector()
        self.root.after_idle(self._planner_restore_plan_list_focus)

    def _cache_entry_variant(self, entry: Dict[str, Any]):
        entry_id = entry.get("id")
        entry_type = entry.get("type")
        if not entry_id or not entry_type:
            return
        self.line_variant_cache.setdefault(entry_id, {})[entry_type] = copy.deepcopy(entry)

    def _restore_entry_variant(self, current_entry: Dict[str, Any], new_type: str) -> Dict[str, Any]:
        entry_id = current_entry.get("id") or make_entry(new_type).get("id")
        cached = self.line_variant_cache.get(entry_id, {}).get(new_type)
        if cached is not None:
            restored = copy.deepcopy(cached)
            restored["id"] = entry_id
            restored["type"] = new_type
            return restored
        fresh = make_entry(new_type)
        fresh["id"] = entry_id
        return fresh

    def _change_line_type(self, new_type: str):
        self.plan_data["entries"][self.current_index] = make_entry(new_type)
        self._schedule_rerender_current_line()

    def _build_labeled_frame(self, parent, title: str) -> LabelFrame:
        frame = LabelFrame(parent, text=title, padx=8, pady=8)
        frame.pack(fill="x", padx=6, pady=6)
        self._add_right_inline_help(frame, title, padx=4, wraplength=720)
        return frame

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

    def _browse_local_path(self, var: tk.StringVar):
        path = filedialog.askopenfilename(filetypes=[("Model Files", "*.safetensors *.ckpt *.pt *.bin"), ("All", "*.*")])
        path = self._normalize_user_path(path)
        if path:
            var.set(path)

    def _build_combo_row(self, parent, label: str, entry: Dict[str, Any], key: str, values: List[str], callback=None):
        row = Frame(parent)
        row.pack(fill="x", pady=3)
        label_widget = Label(row, text=label, width=18, anchor="w")
        label_widget.pack(side="left")
        var = tk.StringVar(value=str(entry.get(key, values[0] if values else "")))
        combo = ttk.Combobox(row, textvariable=var, values=values, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)

        def sync(*_args):
            entry[key] = var.get()
            if callback:
                callback(var.get())
            else:
                self._after_entry_change()
        var.trace_add("write", sync)
        self._attach_tooltip(label_widget, self._right_help(label).get("detail", ""))
        self._add_right_inline_help(parent, label)
        return var

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

    def _render_download_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Download Model")
        self._build_entry_row(frame, "Model Name", entry, "model_name")
        self._build_entry_row(frame, "Link", entry, "link")
        self._build_combo_row(frame, "Type", entry, "model_type", DOWNLOAD_TYPES)

    def _render_local_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Local Model")
        self._build_local_selection_row(frame, entry)
        self._build_entry_row(frame, "Model Name", entry, "model_name")

        path_row = Frame(frame)
        path_row.pack(fill="x", pady=3)
        Label(path_row, text="Local Path", width=18, anchor="w").pack(side="left")
        path_var = tk.StringVar(value=str(entry.get("local_path", "")))
        path_entry = Entry(path_row, textvariable=path_var, state="readonly")
        path_entry.pack(side="left", fill="x", expand=True, padx=4)

        def sync_path(*_args):
            entry["local_path"] = path_var.get()
            if path_var.get().strip() and not str(entry.get("model_name") or "").strip():
                entry["model_name"] = Path(path_var.get()).stem
            self._after_entry_change()

        path_var.trace_add("write", sync_path)
        ttk.Button(path_row, text="📂", width=3, command=lambda: self._browse_local_path(path_var)).pack(side="left")

        self._build_combo_row(frame, "Type", entry, "model_type", LOCAL_TYPES)
        info = Frame(frame)
        info.pack(fill="x")
        Label(info, text="Local Selection lists models under Model Dir or Workspace Path/tmp/models (fallback: Workspace Path).", fg="#666666", wraplength=620, justify="left").pack(anchor="w", padx=4, pady=(4, 0))
        Label(info, text="Use the folder icon to pick a model outside those locations.", fg="#666666", wraplength=620, justify="left").pack(anchor="w", padx=4, pady=(2, 0))

    def _render_remove_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Remove Model")
        available = self._collect_available_models(self.current_index)
        candidates = available.get("Checkpoint", []) + available.get("LoRA", []) + available.get("LyCORIS", [])
        if entry.get("model") and entry["model"] not in candidates:
            candidates = [entry["model"]] + candidates
        self._build_combo_row(frame, "Model", entry, "model", candidates or [""])

    def _render_checkpoint_merge_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Checkpoint Merge")
        ckpts = self._collect_available_models(self.current_index).get("Checkpoint", [])
        mode_labels = [f"{m['key']} - {m['label']}" for m in self.merge_modes]
        current_mode = entry.get("merge_mode", self.merge_modes[0]["key"])
        current_label = next((f"{m['key']} - {m['label']}" for m in self.merge_modes if m["key"] == current_mode), mode_labels[0])

        row = Frame(frame)
        row.pack(fill="x", pady=3)
        Label(row, text="Merge Mode", width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=current_label)
        combo = ttk.Combobox(row, textvariable=var, values=mode_labels, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)

        def sync_mode(*_args):
            label = var.get()
            key = label.split(" - ", 1)[0]
            entry["merge_mode"] = key
            self._schedule_rerender_current_line()
        var.trace_add("write", sync_mode)

        self._build_combo_row(frame, "Model 0", entry, "model0", ckpts or [""])
        self._build_combo_row(frame, "Model 1", entry, "model1", ckpts or [""])
        mode_info = self.merge_mode_map.get(entry.get("merge_mode"), self.merge_modes[0])
        if mode_info.get("needs_m2"):
            self._build_combo_row(frame, "Model 2", entry, "model2", ckpts or [""])
        if mode_info.get("key") != "CLIPXOR":
            self._build_ratio_section(parent, entry, "alpha", "Alpha")
        if mode_info.get("needs_beta"):
            self._build_ratio_section(parent, entry, "beta", "Beta")
        out_frame = self._build_labeled_frame(parent, "Output")
        self._build_entry_row(out_frame, "Output Name", entry, "output_name")
        self._build_text_row(parent, "Additional Signatures", entry, "raw_signatures", height=5)

    def _render_lora_bake_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "LoRA Bake")
        models = self._collect_available_models(self.current_index)
        ckpts = models.get("Checkpoint", [])
        self._build_combo_row(frame, "Checkpoint", entry, "checkpoint", ckpts or [""])
        self._build_entry_row(frame, "Output Name", entry, "output_name")
        add_lora_btn = ttk.Button(frame, text="+ Add LoRA", command=lambda: self._add_lora_block(entry))
        add_lora_btn.pack(anchor="w", padx=4, pady=4)
        self._attach_tooltip(add_lora_btn, self._right_help("+ Add LoRA").get("detail", ""))
        self._add_right_inline_help(frame, "+ Add LoRA")

        for idx, lora in enumerate(entry.get("loras", [])):
            block = self._build_labeled_frame(parent, f"LoRA {idx + 1}")
            top = Frame(block)
            top.pack(fill="x")
            ttk.Button(top, text="-", width=3, command=lambda i=idx: self._remove_lora_block(entry, i)).pack(side="left", padx=(0, 4))
            lora_names = models.get("LoRA", []) + models.get("LyCORIS", [])
            row = Frame(block)
            row.pack(fill="x", pady=3)
            Label(row, text="LoRA Name", width=18, anchor="w").pack(side="left")
            name_var = tk.StringVar(value=lora.get("name", ""))
            combo = ttk.Combobox(row, textvariable=name_var, values=lora_names or [""], state="readonly")
            self._bind_combobox_mousewheel_passthrough(combo)
            combo.pack(side="left", fill="x", expand=True, padx=4)
            name_var.trace_add("write", lambda *_args, lo=lora, nv=name_var: self._sync_lora_name(lo, nv))
            ratio = lora.setdefault("ratio", default_ratio("Single"))
            ratio_wrap = self._build_collapsible_section(
                block,
                "Ratio",
                key=f"lora_ratio_{entry.get('id', '')}_{idx}",
                default_open=True,
                body_fill="x",
                body_expand=False,
                padx=0,
                pady=4,
            )
            row2 = Frame(ratio_wrap)
            row2.pack(fill="x", pady=3)
            Label(row2, text="Ratio Mode", width=18, anchor="w").pack(side="left")
            ratio_var = tk.StringVar(value=ratio.get("mode", "Single"))
            combo2 = ttk.Combobox(row2, textvariable=ratio_var, values=["Single", "Elemental"], state="readonly")
            self._bind_combobox_mousewheel_passthrough(combo2)
            combo2.pack(side="left", fill="x", expand=True, padx=4)

            def on_ratio_mode(*_args, lo=lora, rv=ratio_var):
                lo["ratio"]["mode"] = rv.get()
                if rv.get() == "Single":
                    lo["ratio"].setdefault("value", "1.0")
                self._schedule_rerender_current_line()
            ratio_var.trace_add("write", on_ratio_mode)
            self._build_ratio_value_widget(ratio_wrap, lora["ratio"], allow_block_weight=False)

        self._build_text_row(parent, "Additional Signatures", entry, "raw_signatures", height=5)

    def _sync_lora_name(self, lora_entry: Dict[str, Any], name_var: tk.StringVar):
        lora_entry["name"] = name_var.get()
        self._after_entry_change()

    def _add_lora_block(self, entry: Dict[str, Any]):
        entry.setdefault("loras", []).append({"name": "", "ratio": default_ratio("Single")})
        self._render_current_line()
        self._refresh_line_selector()

    def _remove_lora_block(self, entry: Dict[str, Any], index: int):
        entry.setdefault("loras", []).pop(index)
        self._render_current_line()
        self._refresh_line_selector()

    def _build_ratio_section(self, parent, entry: Dict[str, Any], key: str, title: str):
        wrapper = self._build_collapsible_section(
            parent,
            title,
            key=f"ratio_section_{entry.get('id', '')}_{key}",
            default_open=True,
            body_fill="x",
            body_expand=False,
        )
        ratio = entry.setdefault(key, default_ratio("Single"))
        row = Frame(wrapper)
        row.pack(fill="x", pady=3)
        label_widget = Label(row, text="Ratio Mode", width=18, anchor="w")
        label_widget.pack(side="left")
        mode_var = tk.StringVar(value=ratio.get("mode", "Single"))
        combo = ttk.Combobox(row, textvariable=mode_var, values=RATIO_MODES, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)

        def sync_mode(*_args):
            ratio["mode"] = mode_var.get()
            if ratio["mode"] == "Single" and not ratio.get("value"):
                ratio["value"] = "0.5"
            elif ratio["mode"] == "Block weight" and not ratio.get("value"):
                ratio["value"] = self._serialize_block_values(
                    [0.0] * len(self._current_block_names()),
                    trailing_comma=False,
                )
            self._schedule_rerender_current_line()
        mode_var.trace_add("write", sync_mode)
        self._attach_tooltip(label_widget, self._right_help("Ratio Mode").get("detail", ""))
        self._add_right_inline_help(wrapper, "Ratio Mode")
        self._build_ratio_value_widget(wrapper, ratio, allow_block_weight=True)

    @staticmethod
    def _clamp_ratio_float(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _coerce_ratio_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return self._clamp_ratio_float(float(str(value).strip()))
        except Exception:
            return self._clamp_ratio_float(default)

    def _format_ratio_float(self, value: Any) -> str:
        v = self._coerce_ratio_float(value)
        text = f"{v:.12f}".rstrip("0").rstrip(".")
        return text or "0"

    def _serialize_block_values(
        self,
        values: List[Any],
        expected_count: int | None = None,
        *,
        trailing_comma: bool = False,
    ) -> str:
        count = expected_count or len(self._current_block_names())
        normalized = [
            self._format_ratio_float(values[i] if i < len(values) else 0.0)
            for i in range(count)
        ]
        text = ",".join(normalized)
        return text + "," if trailing_comma else text

    def _commit_block_ratio_entry(
        self,
        value_var: tk.StringVar,
        scale_var: tk.DoubleVar,
        frame: Frame,
        ratio: Dict[str, Any],
    ):
        value = self._coerce_ratio_float(value_var.get(), default=scale_var.get())
        formatted = self._format_ratio_float(value)
        if value_var.get() != formatted:
            value_var.set(formatted)
        if abs(scale_var.get() - value) > 1e-9:
            scale_var.set(value)
        self._sync_block_ratio(frame, ratio)

    def _sync_block_ratio_scale(
        self,
        value_var: tk.StringVar,
        scale_var: tk.DoubleVar,
        frame: Frame,
        ratio: Dict[str, Any],
    ):
        value = self._coerce_ratio_float(scale_var.get())
        formatted = self._format_ratio_float(value)
        if value_var.get() != formatted:
            value_var.set(formatted)
        self._sync_block_ratio(frame, ratio)

    def _build_ratio_value_widget(self, parent, ratio: Dict[str, Any], allow_block_weight: bool):
        mode = ratio.get("mode", "Single")
        if mode == "Single":
            row = Frame(parent)
            row.pack(fill="x", pady=3)
            label_widget = Label(row, text="Ratio", width=18, anchor="w")
            label_widget.pack(side="left")
            var = tk.StringVar(value=ratio.get("value", "1.0"))
            ent = Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True, padx=4)
            var.trace_add("write", lambda *_args: self._sync_ratio_value(ratio, var.get()))
            self._attach_tooltip(label_widget, self._right_help("Ratio").get("detail", ""))
            self._add_right_inline_help(parent, "Ratio")
            return
        if mode == "Block weight" and allow_block_weight:
            block_names = self._current_block_names()
            vals = self._parse_block_values(ratio.get("value", ""), len(block_names))
            ratio["value"] = self._serialize_block_values(
                vals,
                expected_count=len(block_names),
                trailing_comma=False,
            )
            title = Label(parent, text=f"Ratio ({self.base_model_var.get()} blocks)", fg="#666666")
            title.pack(anchor="w", padx=4, pady=(3, 0))
            self._attach_tooltip(title, self._right_help("Block Weight").get("detail", ""))
            self._add_right_inline_help(parent, "Block Weight", padx=4, wraplength=720)
            block_frame = Frame(parent)
            block_frame.pack(fill="x", pady=2)
            for i, block_name in enumerate(block_names):
                row = Frame(block_frame)
                row.pack(fill="x", pady=(1, 0))
                name_label = Label(row, text=block_name, width=12, anchor="e")
                name_label.pack(side="left")
                value_var = tk.StringVar(value=self._format_ratio_float(vals[i]))
                scale_var = tk.DoubleVar(value=vals[i])
                row._ratio_var = value_var
                ent = Entry(row, textvariable=value_var, width=8, justify="right")
                ent.pack(side="left", padx=4)
                ent.bind(
                    "<FocusOut>",
                    lambda _e, v=value_var, s=scale_var, container=block_frame, r=ratio:
                        self._commit_block_ratio_entry(v, s, container, r),
                )
                ent.bind(
                    "<Return>",
                    lambda _e, v=value_var, s=scale_var, container=block_frame, r=ratio:
                        self._commit_block_ratio_entry(v, s, container, r),
                )
                scale = tk.Scale(
                    row,
                    variable=scale_var,
                    from_=0.0,
                    to=1.0,
                    resolution=0.001,
                    orient="horizontal",
                    showvalue=0,
                    length=220,
                    command=lambda _value, v=value_var, s=scale_var, container=block_frame, r=ratio:
                        self._sync_block_ratio_scale(v, s, container, r),
                )
                scale.pack(side="left", padx=4, fill="x", expand=True)
                desc = Label(block_frame, text=self._block_effect_hint(block_name), fg="#666666", justify="left", wraplength=720, anchor="w")
                desc.pack(fill="x", padx=26, pady=(0, 3))
                self._attach_tooltip([name_label, desc], f"{block_name}: {self._block_effect_hint(block_name)}")
            return
        row = Frame(parent)
        row.pack(fill="both", expand=True, pady=3)
        label_widget = Label(row, text="Ratio", width=18, anchor="nw")
        label_widget.pack(side="left")
        text_widget = Text(row, height=5, font=("Consolas", 12), undo=True, autoseparators=True, maxundo=-1)
        text_widget.pack(side="left", fill="both", expand=True, padx=4)
        self._configure_syntax_tags(text_widget)
        text_widget.insert("1.0", ratio.get("value", ""))
        text_widget.bind("<KeyRelease>", lambda _e: self._sync_ratio_value(ratio, text_widget.get("1.0", "end-1c")))
        text_widget.bind("<Button-1>", lambda _e: text_widget.after_idle(text_widget.focus_set), add="+")
        self._bind_multiline_text_scroll_priority(text_widget)
        text_widget.bind("<Motion>", lambda e, w=text_widget: self._on_elemental_text_motion(w, e), add="+")
        text_widget.bind("<Leave>", self._hide_elemental_alias_hover, add="+")
        text_widget.bind("<FocusOut>", self._hide_elemental_alias_hover, add="+")
        self._attach_tooltip(label_widget, self._right_help("Elemental Ratio").get("detail", ""))
        self._add_right_inline_help(parent, "Elemental Ratio")
        json_help = Label(parent, text=f"Candidate file: {self._elemental_candidate_json_filename()}", fg="#666666", justify="left", anchor="w")
        json_help.pack(anchor="w", padx=4, pady=(0, 3), fill="x")
        self._attach_tooltip(json_help, f"Elemental popup candidates are loaded from {self._elemental_candidate_json_filename()} for the selected base model when that file exists.")
        text_widget._elemental_separator = "," if allow_block_weight else ";"
        self._bind_text_autocomplete(
            text_widget,
            self._elemental_suggestion_items,
            mode="elemental",
            validator=self._refresh_elemental_invalid_highlight,
        )
        # text_widget.bind("<Motion>", lambda e, w=text_widget: self._on_elemental_text_motion(w, e), add="+")
        # text_widget.bind("<Leave>", self._hide_elemental_alias_hover, add="+")
        # text_widget.bind("<FocusOut>", self._hide_elemental_alias_hover, add="+")
        self._refresh_elemental_invalid_highlight(text_widget)
        text_widget.after(10, lambda w=text_widget: w.mark_set("insert", "end-1c"))

    def _parse_block_values(self, value: str, expected_count: int | None = None) -> List[float]:
        count = expected_count or len(self._current_block_names())
        raw = str(value or "").replace("，", ",").replace("\n", ",").strip()
        if not raw:
            return [0.0] * count

        parts = [p.strip() for p in raw.split(",")]
        vals: List[float] = []
        for part in parts:
            if not part:
                continue
            vals.append(self._coerce_ratio_float(part))

        if len(vals) < count:
            vals.extend([0.0] * (count - len(vals)))
        return vals[:count]

    def _sync_block_ratio(self, frame: Frame, ratio: Dict[str, Any]):
        values: List[float] = []
        for child in frame.winfo_children():
            if isinstance(child, Frame):
                var = getattr(child, "_ratio_var", None)
                if var is not None:
                    values.append(self._coerce_ratio_float(var.get()))
        expected = len(self._current_block_names())
        if len(values) < expected:
            values.extend([0.0] * (expected - len(values)))
        ratio["value"] = self._serialize_block_values(
            values[:expected],
            expected_count=expected,
            trailing_comma=False,
        )
        self._after_entry_change()

    def _sync_ratio_value(self, ratio: Dict[str, Any], value: str):
        ratio["value"] = value
        self._after_entry_change()

    def _after_entry_change(self):
        self._refresh_line_selector()
        
    def _build_runner_notebook(self, notebook_path: str) -> str:
        """
        Create a temporary execution notebook whose code cells are transformed
        with the real IPython cell transformer.
        The original generated notebook is left untouched.
        """
        if nbformat is None:
            return notebook_path

        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        shell = InteractiveShell.instance()

        compat_cell = nbformat.v4.new_code_cell(
            """
import os
import subprocess

try:
    from IPython import get_ipython as _real_get_ipython
except Exception:
    _real_get_ipython = None

class _RunnerIPythonShim:
    def system(self, cmd):
        return subprocess.call(
            cmd,
            shell=True,
            executable=os.environ.get("SHELL", "/bin/bash"),
            cwd=os.getcwd(),
        )

    def run_line_magic(self, magic, arg):
        if magic == "cd":
            os.chdir(os.path.expanduser(arg))
            print(os.getcwd())
            return None
        raise RuntimeError(f"Unsupported line magic: %{magic}")

    def run_cell_magic(self, magic, line, cell):
        if magic == "bash":
            script = cell if not line else f"{line}\\n{cell}"
            return subprocess.call(
                script,
                shell=True,
                executable=os.environ.get("SHELL", "/bin/bash"),
                cwd=os.getcwd(),
            )
        raise RuntimeError(f"Unsupported cell magic: %%{magic}")

def get_ipython():
    try:
        ip = _real_get_ipython() if _real_get_ipython else None
    except Exception:
        ip = None
    return ip or _RunnerIPythonShim()
        """.strip()
        )

        transformed_cells = [compat_cell]
        for cell in nb.cells:
            if getattr(cell, "cell_type", None) != "code":
                transformed_cells.append(cell)
                continue

            src = cell.source or ""
            try:
                transformed = shell.transform_cell(src)
            except Exception:
                transformed = src

            cell.source = transformed
            transformed_cells.append(cell)

        nb.cells = transformed_cells

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".ipynb",
            prefix="runner_ipy_",
            dir=str(Path(notebook_path).parent),
            encoding="utf-8",
        )
        tmp_path = tmp.name
        tmp.close()

        with open(tmp_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return tmp_path


    def _build_notebook_run_command(self, notebook_path: str, executed_path: str) -> List[str]:
        papermill_spec = importlib.util.find_spec("papermill")
        if papermill_spec is not None:
            return [
                sys.executable,
                "-m",
                "papermill",
                notebook_path,
                executed_path,
                "--log-output",
                "--kernel",
                "python3",
            ]
        return [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            notebook_path,
            "--output",
            Path(executed_path).name,
            "--output-dir",
            str(Path(executed_path).parent),
            "--ExecutePreprocessor.timeout=-1",
        ]


    def _update_status_from_line(self, line: str):
        stripped = line.strip()
        if not stripped:
            return
        lower = stripped.lower()
        msg = stripped[:140]

        if stripped.startswith("[planner-progress]"):
            progress_msg = stripped[len("[planner-progress]"):].strip() or msg
            self.root.after(0, lambda: self.status_label.config(text=f"Progress: {progress_msg[:100]}"))
            self.console.set_progress(progress_msg)
            return

        if "%|" in stripped or "it/s" in lower or "s/it" in lower:
            self.root.after(0, lambda: self.status_label.config(text=f"Progress: {msg}"))
            self.console.set_progress(msg)
            return

        if stripped.startswith("$"):
            if "merge.py" in lower or "lora_bake.py" in lower:
                self.root.after(0, lambda: self.status_label.config(text=f"Merging: {msg}"))
                self.console.set_progress(msg)
                return
            if any(k in lower for k in ["aria2", "wget", "curl"]):
                self.root.after(0, lambda: self.status_label.config(text=f"Downloading: {msg}"))
                self.console.set_progress(msg)
                return

        if any(k in lower for k in ["download", "aria2", "custom_model"]):
            self.root.after(0, lambda: self.status_label.config(text=f"Downloading: {msg}"))
            self.console.set_progress(msg)
        elif any(k in lower for k in ["merge.py", "merging", "lora_bake.py"]):
            self.root.after(0, lambda: self.status_label.config(text=f"Merging: {msg}"))
            self.console.set_progress(msg)
        elif any(k in lower for k in ["saving", "output", "register"]):
            self.root.after(0, lambda: self.status_label.config(text=f"Saving: {msg}"))
            self.console.set_progress(msg)


    # ---------------- runner ----------------
    def _run_target_notebook(self):
        temp_plan_path = None
        runner_notebook_path = None
        try:
            temp_plan_path = self._write_temp_plan_text()
            notebook_path, executed_path = self._get_notebook_paths()
            params = {key: widget.get() for key, widget in self.entries.items()}
            self.notebook_path_var.set(notebook_path)
            self.executed_notebook_path_var.set(executed_path)
            self._save_current_state_to_config()
            self.console.show()
            self.console.bind_notebook(executed_path)
            self.console.clear_output()
            self.console._stop_requested = False
            self.console.set_state("BUILDING")
            self.console.set_step("Generating notebook from Plan Creator")
            self.console.idle(f"Temporary plan path: {temp_plan_path}")
            self.console.idle(f"Notebook path: {notebook_path}")
            self.progress_indicator.start(3, "Running merge")

            create_plan_ipynb(
                filepath=temp_plan_path,
                workpath=params.get("workpath", ""),
                saveas=notebook_path,
                title=params.get("title", "merge_plan"),
                vae=params.get("vae", ""),
                vae_name=params.get("vae_name", "VAE"),
                bake_vae=bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get()),
                CivitAPI=params.get("CivitAPI", ""),
                HuggingAPI=params.get("HuggingAPI", ""),
                UR=params.get("UR", ""),
                model_dir=params.get("model_dir", ""),
                vae_dir=params.get("vae_dir", ""),
                ignore_install_deps=bool(getattr(self, "ignore_install_deps_var", tk.BooleanVar(value=False)).get()),
                use_online=bool(getattr(self, "use_online_var", tk.BooleanVar(value=False)).get()),
                upload_after_merge=bool(getattr(self, "upload_after_merge_var", tk.BooleanVar(value=False)).get()),
                run_t2i=bool(getattr(self, "run_t2i_var", tk.BooleanVar(value=False)).get()),
                t2i_settings=self._planner_get_t2i_settings() if hasattr(self, "_planner_get_t2i_settings") else None
            )
            self.progress_indicator.update(1, "Notebook generated")
            self.console.idle("Notebook generation completed.")

            runner_notebook_path = self._build_runner_notebook(notebook_path)
            if runner_notebook_path != notebook_path:
                self.console.idle(f"Runner notebook path: {runner_notebook_path}")

            def worker():
                proc = None
                try:
                    self.console.set_state("RUNNING")
                    self.console.set_step("Executing notebook")
                    cmd = self._build_notebook_run_command(runner_notebook_path or notebook_path, executed_path)
                    self.console.idle("Command: " + " ".join(cmd))
                    env = os.environ.copy()
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    popen_kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                    if os.name != "nt":
                        popen_kwargs["start_new_session"] = True
                    proc = subprocess.Popen(cmd, **popen_kwargs)
                    self.root.after(0, lambda p=proc: self.console.attach_process(p))
                    self.root.after(0, lambda: self.progress_indicator.update(2, "Notebook started"))

                    pending: list[str] = []
                    last_flush = time.monotonic()
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip("\n")
                        self._update_status_from_line(raw_line)
                        if self.console._looks_like_progress_line(line):
                            continue
                        pending.append(line)
                        now = time.monotonic()
                        if len(pending) >= 20 or (now - last_flush) >= 0.12:
                            self.console.log("\n".join(pending), "info")
                            pending.clear()
                            last_flush = now
                    if pending:
                        self.console.log("\n".join(pending), "info")

                    return_code = proc.wait()
                    if self.console._stop_requested:
                        self.console.set_state("STOPPED")
                        self.console.set_step("Execution stopped by user")
                        self.console.log("Execution stopped by user.", "error")
                        self.console.set_progress_fraction(None, "Stopped")
                        self.root.after(0, lambda: self.progress_indicator.finish(False))
                        self.root.after(0, lambda: self.status_label.config(text="⏹ Merge stopped"))
                        return
                    if return_code != 0:
                        raise RuntimeError(f"Notebook execution failed with exit code {return_code}")

                    self.root.after(0, lambda: self.progress_indicator.update(3, "Notebook finished"))
                    self.console.set_state("DONE")
                    self.console.set_step("Execution completed")
                    self.console.log("Execution completed successfully.", "success")
                    self.root.after(0, lambda: self.progress_indicator.finish(True))
                    self.root.after(0, lambda: self.status_label.config(text="✅ Merge completed"))
                    self._play_notification_sound()
                except Exception as e:
                    detail = self._format_exception_text("Execution Error", e)
                    self.console.set_state("FAILED")
                    self.console.set_step("Execution failed")
                    self.console.log(detail, "error")
                    self.root.after(0, lambda: self.progress_indicator.finish(False))
                    self.root.after(0, lambda: self.status_label.config(text="❌ Merge failed"))
                    self.root.after(0, lambda: self._show_scrollable_text_dialog("Execution Error", detail))
                finally:
                    if runner_notebook_path and runner_notebook_path != notebook_path and os.path.exists(runner_notebook_path):
                        try:
                            os.remove(runner_notebook_path)
                        except Exception:
                            pass
                    if temp_plan_path and os.path.exists(temp_plan_path):
                        os.remove(temp_plan_path)

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            self.progress_indicator.finish(False)
            if temp_plan_path and os.path.exists(temp_plan_path):
                os.remove(temp_plan_path)
            self._show_detailed_error("Run Error", e, context="Operation: run_target_notebook")

    def _update_status_from_line(self, line: str):
        stripped = line.strip()
        if not stripped:
            return
        lower = stripped.lower()
        msg = stripped[:140]

        def push_progress(text: str, status_prefix: str = "Progress"):
            text = str(text or "").strip()
            if not text:
                return
            fraction = RunnerConsoleWindow._extract_progress_fraction(text)
            self.root.after(0, lambda: self.status_label.config(text=f"{status_prefix}: {text[:100]}"))
            self.console.set_progress(text)
            if fraction is not None:
                self.console.set_progress_fraction(fraction, text)

        if stripped.startswith("[planner-progress]"):
            progress_msg = stripped[len("[planner-progress]"):].strip() or msg
            push_progress(progress_msg, "Progress")
            return

        if stripped.startswith("[#") or ("dl:" in lower and ("eta:" in lower or "%" in stripped)):
            push_progress(msg, "Downloading")
            return

        if "%|" in stripped or "it/s" in lower or "s/it" in lower:
            push_progress(msg, "Progress")
            return

        if stripped.startswith("$"):
            if "merge.py" in lower or "lora_bake.py" in lower:
                push_progress(msg, "Merging")
                return
            if any(k in lower for k in ["aria2", "wget", "curl"]):
                push_progress(msg, "Downloading")
                return

        if any(k in lower for k in ["download", "aria2", "custom_model"]):
            push_progress(msg, "Downloading")
        elif any(k in lower for k in ["merge.py", "merging", "lora_bake.py", "checkpoint merge", "lora bake"]):
            push_progress(msg, "Merging")
        elif any(k in lower for k in ["saving", "output", "register"]):
            push_progress(msg, "Saving")

    # ---------------- upload / misc ----------------
    def _upload_merge_result(self):
        repo_id = self.entries["UR"].get().strip()
        token = self.entries["HuggingAPI"].get().strip()
        model_dir = self.entries["model_dir"].get().strip() or os.path.join(self.entries["workpath"].get().strip() or ".", "tmp", "models")
        if not repo_id or not token:
            messagebox.showwarning("Missing Info", "Please provide both Repo ID and HuggingFace Token")
            return
        if not os.path.exists(model_dir):
            messagebox.showwarning("Model Directory", "Model directory does not exist.")
            return
        safetensors_files = list(Path(model_dir).glob("*.safetensors"))
        if not safetensors_files:
            messagebox.showinfo("No Models", "No .safetensors files found in model directory")
            return
        latest = max(safetensors_files, key=lambda p: p.stat().st_mtime)
        if not messagebox.askyesno("Upload", f"Upload latest model?\n\n{latest.name}\n-> {repo_id}"):
            return

        def worker():
            try:
                self.progress_indicator.start(1, f"Uploading {latest.name}")
                api = HfApi()
                api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
                upload_file(path_or_fileobj=str(latest), path_in_repo=latest.name, repo_id=repo_id, token=token)
                self.progress_indicator.update(1, latest.name)
                self.progress_indicator.finish(True)
                self.root.after(0, lambda: messagebox.showinfo("Upload", f"Uploaded {latest.name} to {repo_id}"))
                self._play_notification_sound()
            except Exception as e:
                self.progress_indicator.finish(False)
                detail = self._format_exception_text("Upload Error", e)
                self.root.after(0, lambda: self.console.log(detail, "error"))
                self.root.after(0, lambda: self._show_scrollable_text_dialog("Upload Error", detail))

        threading.Thread(target=worker, daemon=True).start()

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





    # ---------------- planner interactive extensions ----------------
    _base_init = __init__
    _base_render_current_line = _render_current_line
    _base_load_plan_from_path = _load_plan_from_path
    _base_new_plan = _new_plan
    _base_reset_plan = _reset_plan
    _base_change_line_type = _change_line_type

    def __init__(self, root):
        # Extended planner state is needed before the base layout is built,
        # but some of the later-added state initializers also read self.config.
        # Ensure config exists first so override stacks do not fail when methods
        # with the same name are replaced later in the class body.
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

    def _render_current_line(self):
        self._base_render_current_line()
        self._update_current_line_indicator()
        try:
            entry = self.plan_data.get('entries', [])[self.current_index]
        except Exception:
            self._planner_refresh_plan_meta()
            self._update_current_line_indicator()
            return
        note_frame = LabelFrame(self.scroll_frame, text='Memo', padx=8, pady=8)
        note_frame.pack(fill='both', padx=6, pady=6)
        memo = Text(note_frame, height=4, font=('Consolas', 10), undo=True, autoseparators=True, maxundo=-1)
        memo.pack(fill='both', expand=True)
        memo.insert('1.0', entry.get('memo', ''))
        memo.bind('<KeyRelease>', lambda _e, e=entry, w=memo: (e.__setitem__('memo', w.get('1.0', 'end-1c')), self._after_entry_change()), add='+')
        memo.bind('<<Paste>>', lambda _e, e=entry, w=memo: self.root.after_idle(lambda: (e.__setitem__('memo', w.get('1.0', 'end-1c')), self._after_entry_change())), add='+')
        self._planner_refresh_plan_meta()

    def _after_entry_change(self):
        pending = getattr(self, '_after_entry_change_id', None)
        if pending is not None:
            try:
                self.root.after_cancel(pending)
            except Exception:
                pass
        self._planner_schedule_history_checkpoint()
        self._planner_refresh_plan_meta()
        self._update_current_line_indicator()
        self._after_entry_change_id = self.root.after(120, self._flush_after_entry_change)

    def _flush_after_entry_change(self):
        self._after_entry_change_id = None
        self._refresh_line_selector()
        self._save_plan_to_file()
        self._planner_refresh_plan_meta()
        self._update_current_line_indicator()

    def _load_plan_from_path(self):
        result = self._base_load_plan_from_path()
        self._history_undo.clear()
        self._history_redo.clear()
        self._last_history_snapshot = self._planner_plan_snapshot()
        self._refresh_line_selector()
        self._planner_refresh_plan_meta()
        return result

    def _new_plan(self):
        result = self._base_new_plan()
        self._history_undo.clear()
        self._history_redo.clear()
        self._last_history_snapshot = self._planner_plan_snapshot()
        self._refresh_line_selector()
        self._planner_refresh_plan_meta()
        return result

    def _reset_plan(self):
        self._planner_push_history()
        result = self._base_reset_plan()
        self._refresh_line_selector()
        self._planner_refresh_plan_meta()
        return result

    def _change_line_type(self, new_type: str):
        self._planner_push_history()
        return self._base_change_line_type(new_type)

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


    def _planner_restore_plan_list_focus(self):
        if self.plan_listbox is None:
            return
        try:
            if self.plan_listbox.winfo_exists():
                self.plan_listbox.focus_set()
        except Exception:
            pass


    def _planner_focus_in_plan_scope(self):
        focus = self.root.focus_get()
        if focus is None:
            return False
        return (
            focus is self.plan_listbox
            or self._planner_widget_is_descendant(focus, self.plan_listbox)
            or self._planner_widget_is_descendant(focus, getattr(self, "right_outer", None))
        )


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


    def _planner_is_text_focus(self):
        return self._planner_is_editable_text_widget(self.root.focus_get())



    def _planner_deepcopy_plan(self):
        return copy.deepcopy(self.plan_data)



    def _planner_plan_snapshot(self):
        return json.dumps(self.plan_data, ensure_ascii=False, sort_keys=True)



    def _planner_default_visible_plan(self):
        return {"entries": [make_entry("Checkpoint Merge")]}


    def _planner_init_extra_state(self):
        self.plan_clipboard_entries = []
        self.visible_entry_indices = []
        self.plan_search_var = tk.StringVar(value="")
        self.plan_filter_var = tk.StringVar(value="All")
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

    def _update_current_line_indicator(self):
        var = getattr(self, "current_line_indicator_var", None)
        if var is None:
            return
        entries = self.plan_data.get("entries", []) if isinstance(getattr(self, "plan_data", None), dict) else []
        total = len(entries)
        if total <= 0:
            var.set("Line 0 / 0 : ")
            return
        current_idx = max(0, min(int(getattr(self, "current_index", 0) or 0), total - 1))
        current = current_idx + 1
        summary = ""
        try:
            summary = self._line_summary(entries[current_idx])
        except Exception:
            summary = ""
        var.set(f"Line {current} / {total} : {summary}" if summary else f"Line {current} / {total} : ")


    def _planner_bind_shortcuts(self):
        tag = 'PlannerGlobalShortcuts'
        self._planner_shortcut_bindtag = tag
        bindings = [
            ("<Control-z>", self._shortcut_undo),
            ("<Command-z>", self._shortcut_undo),
            ("<Control-y>", self._shortcut_redo),
            ("<Command-y>", self._shortcut_redo),
            ("<Control-Z>", self._shortcut_redo),
            ("<Command-Z>", self._shortcut_redo),
            ("<Control-c>", self._shortcut_copy_lines),
            ("<Command-c>", self._shortcut_copy_lines),
            ("<Control-v>", self._shortcut_paste_lines),
            ("<Command-v>", self._shortcut_paste_lines),
            ("<Delete>", self._shortcut_remove_line),
            ("<Control-BackSpace>", self._shortcut_remove_line),
            ("<Command-BackSpace>", self._shortcut_remove_line),
            ("<Control-d>", self._shortcut_duplicate_lines),
            ("<Control-D>", self._shortcut_duplicate_lines),
            ("<Command-d>", self._shortcut_duplicate_lines),
            ("<Command-D>", self._shortcut_duplicate_lines),
            ("<Control-plus>", self._shortcut_add_line),
            ("<Control-asterisk>", self._shortcut_add_line),
            ("<Command-plus>", self._shortcut_add_line),
            ("<Command-asterisk>", self._shortcut_add_line),
            ("<Control-V>", self._shortcut_validate),
            ("<Command-V>", self._shortcut_validate),
            ("<Control-Alt-s>", self._shortcut_save_preset),
            ("<Command-Option-s>", self._shortcut_save_preset),
            ("<Control-Alt-l>", self._shortcut_load_preset),
            ("<Command-Option-l>", self._shortcut_load_preset),
            ("<Shift-Up>", self._shortcut_move_lines_up),
            ("<Shift-Down>", self._shortcut_move_lines_down),
        ]
        for sequence, handler in bindings:
            self.root.bind_class(tag, sequence, handler, add='+')
        self.root.bind_all('<FocusIn>', self._planner_register_shortcut_bindtag_on_focus, add='+')
        self._planner_register_shortcut_bindtag_on_tree(self.root)

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


    def _planner_build_fixed_status_bar(self):
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


    def _planner_get_selected_indices(self):
        if self.plan_listbox is None:
            return [self.current_index] if self.plan_data.get('entries') else []
        out = []
        for vis_idx in self.plan_listbox.curselection():
            if 0 <= vis_idx < len(self.visible_entry_indices):
                out.append(self.visible_entry_indices[vis_idx])
        if not out and self.plan_data.get('entries'):
            return [self.current_index]
        return sorted(set(out))


    def _planner_push_history(self):
        snap = self._planner_plan_snapshot()
        if snap == self._last_history_snapshot:
            return
        if self._last_history_snapshot is not None:
            self._history_undo.append((self._last_history_snapshot, self.current_index))
            if len(self._history_undo) > 200:
                self._history_undo = self._history_undo[-200:]
        self._history_redo.clear()
        self._last_history_snapshot = snap


    def _planner_schedule_history_checkpoint(self):
        if self._history_after_id is not None:
            try:
                self.root.after_cancel(self._history_after_id)
            except Exception:
                pass
        self._history_after_id = self.root.after(600, self._planner_commit_history_checkpoint)


    def _planner_commit_history_checkpoint(self):
        self._history_after_id = None
        self._planner_push_history()


    def _planner_restore_snapshot(self, payload, index=0):
        try:
            self.plan_data = self._normalize_plan_preserving_embedded_sources(json.loads(payload))
        except Exception:
            self.plan_data = self._planner_default_visible_plan()
        entries = self.plan_data.get('entries', [])
        self.current_index = max(0, min(index, max(0, len(entries) - 1)))
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._save_plan_to_file()


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


    def _invoke_plan_reorder_button(self, direction: str):
        if not self._planner_focus_allows_line_shortcuts():
            return None
        button = getattr(self, 'plan_move_up_button', None) if direction == 'up' else getattr(self, 'plan_move_down_button', None) if direction == 'down' else None
        if button is not None:
            try:
                if button.winfo_exists():
                    button.focus_set()
                    self.root.update_idletasks()
                    button.invoke()
                    return 'break'
            except Exception:
                pass
        if direction == 'up':
            self._move_line_up()
            return 'break'
        if direction == 'down':
            self._move_line_down()
            return 'break'
        return None


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


    def _planner_dependency_summary(self):
        producers = {}
        missing = 0
        links = 0
        for idx, produces, _consumes in self._compute_aliases():
            for name in produces:
                producers[name] = idx
        for _idx, _produces, consumes in self._compute_aliases():
            for name in consumes:
                links += 1
                if producers.get(name) is None:
                    missing += 1
        return f"Deps: {links} link(s) / {missing} missing"


    def _planner_unused_summary(self):
        aliases = self._compute_aliases()
        produced = []
        consumed = set()
        for _idx, produces, consumes in aliases:
            produced.extend(produces)
            consumed.update(consumes)
        unused = [name for name in produced if name not in consumed]
        return f"Unused: {len(unused)} alias(es)" + (f"  [{', '.join(unused[:3])}{' ...' if len(unused) > 3 else ''}]" if unused else "")


    def _planner_refresh_plan_meta(self):
        if hasattr(self, 'plan_summary_var'):
            count = len(self.plan_data.get('entries', []))
            sel = len(self._planner_get_selected_indices()) if count else 0
            self.plan_summary_var.set(f"{count} lines  |  {sel} selected")
        if hasattr(self, 'plan_deps_var'):
            self.plan_deps_var.set(self._planner_dependency_summary())
        if hasattr(self, 'plan_unused_var'):
            self.plan_unused_var.set(self._planner_unused_summary())


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


    def _set_widget_state_if_present(self, widget, state: str):
        if widget is None:
            return
        try:
            if widget.winfo_exists():
                widget.configure(state=state)
        except Exception:
            pass


    def _update_plan_action_buttons(self):
        entries = self.plan_data.get('entries', [])
        count = len(entries)
        indices = self._planner_get_selected_indices() if count else []
        can_remove = bool(indices) and count > len(indices)
        can_up = bool(indices) and indices[0] > 0
        can_down = bool(indices) and indices[-1] < count - 1
        combo_state = 'readonly' if count else 'disabled'
        combo = getattr(self, 'selection_combo', None)
        if combo is not None:
            try:
                if combo.winfo_exists():
                    combo.configure(state=combo_state)
            except Exception:
                pass
        self._set_widget_state_if_present(getattr(self, 'plan_add_button', None), 'normal')
        self._set_widget_state_if_present(getattr(self, 'plan_remove_button', None), 'normal' if can_remove else 'disabled')
        self._set_widget_state_if_present(getattr(self, 'plan_move_up_button', None), 'normal' if can_up else 'disabled')
        self._set_widget_state_if_present(getattr(self, 'plan_move_down_button', None), 'normal' if can_down else 'disabled')
        self._set_widget_state_if_present(getattr(self, 'plan_reload_button', None), 'normal')


    def _visible_entries_for_filter(self):
        entries = self.plan_data.get('entries', [])
        text = (self.plan_search_var.get().strip().lower() if hasattr(self, 'plan_search_var') else '')
        ftype = (self.plan_filter_var.get().strip() if hasattr(self, 'plan_filter_var') else 'All')
        visible = []
        for idx, entry in enumerate(entries):
            etype = entry.get('type', '')
            summary = self._line_summary(entry)
            hay = json.dumps(entry, ensure_ascii=False).lower() + ' ' + summary.lower()
            if ftype not in ('', 'All') and etype != ftype:
                continue
            if text and text not in hay:
                continue
            visible.append(idx)
        return visible



    def _on_plan_list_select(self, event=None):
        if getattr(self, '_selection_guard', False) or self.plan_listbox is None:
            return
        sel = self.plan_listbox.curselection()
        if not sel:
            return
        vis_idx = sel[-1]
        if 0 <= vis_idx < len(self.visible_entry_indices):
            model_idx = self.visible_entry_indices[vis_idx]
            if model_idx != self.current_index:
                self.current_index = model_idx
                self._render_current_line()
            self._sync_target_line_combo()
            self._planner_refresh_plan_meta()
            self._update_plan_action_buttons()


    def _select_model_indices(self, indices):
        indices = sorted(set(i for i in indices if 0 <= i < len(self.plan_data.get('entries', []))))
        if not indices:
            return
        self.current_index = indices[-1]
        self._refresh_line_selector()
        if self.plan_listbox is None:
            return
        self._selection_guard = True
        try:
            self.plan_listbox.selection_clear(0, 'end')
            for vis_idx, model_idx in enumerate(self.visible_entry_indices):
                if model_idx in indices:
                    self.plan_listbox.selection_set(vis_idx)
            try:
                vis_idx = self.visible_entry_indices.index(self.current_index)
                self.plan_listbox.activate(vis_idx)
                self.plan_listbox.see(vis_idx)
                self._planner_restore_plan_list_focus()
            except ValueError:
                pass
        finally:
            self._selection_guard = False


    def _copy_selected_lines(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return
        self.plan_clipboard_entries = [copy.deepcopy(self.plan_data['entries'][i]) for i in indices]
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(json.dumps(self.plan_clipboard_entries, ensure_ascii=False, indent=2))
        except Exception:
            pass
        self.status_label.config(text=f"Copied {len(self.plan_clipboard_entries)} line(s)")


    def _paste_copied_lines(self):
        payload = self.plan_clipboard_entries
        if not payload:
            try:
                clip = self.root.clipboard_get()
                data = json.loads(clip)
                if isinstance(data, dict):
                    payload = [data]
                elif isinstance(data, list):
                    payload = data
            except Exception:
                payload = []
        if not payload:
            return
        self._planner_push_history()
        insert_at = max(self._planner_get_selected_indices() or [self.current_index]) + 1 if self.plan_data.get('entries') else 0
        pasted = []
        for raw in payload:
            new_entry = self._normalize_entry_preserving_embedded_sources(copy.deepcopy(raw))
            new_entry['id'] = make_entry(new_entry.get('type', 'Download Model')).get('id')
            pasted.append(new_entry)
        self.plan_data['entries'][insert_at:insert_at] = pasted
        self.current_index = insert_at + len(pasted) - 1
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices(list(range(insert_at, insert_at + len(pasted))))
        self.status_label.config(text=f"Pasted {len(pasted)} line(s)")


    def _delete_selected_lines(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return
        entries = self.plan_data.get('entries', [])
        if len(entries) - len(indices) < 1:
            messagebox.showwarning('Plan', 'At least one line must remain.')
            return
        self._planner_push_history()
        for idx in reversed(indices):
            entries.pop(idx)
        self.current_index = max(0, min(indices[0], len(entries) - 1))
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self.status_label.config(text=f"Deleted {len(indices)} line(s)")


    def _duplicate_selected_lines(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return
        self._planner_push_history()
        entries = self.plan_data.get('entries', [])
        insert_at = indices[-1] + 1
        copies = []
        for idx in indices:
            cloned = copy.deepcopy(entries[idx])
            cloned['id'] = make_entry(cloned.get('type', 'Download Model')).get('id')
            copies.append(cloned)
        entries[insert_at:insert_at] = copies
        self.current_index = insert_at + len(copies) - 1
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(list(range(insert_at, insert_at + len(copies))))
        self.status_label.config(text=f"Duplicated {len(copies)} line(s)")


    def _hide_plan_context_menu(self, _event=None):
        popup = getattr(self, "_plan_context_popup", None)
        bind_ids = getattr(self, "_plan_context_root_bind_ids", None)
        if isinstance(bind_ids, dict):
            for sequence, funcid in list(bind_ids.items()):
                if not funcid:
                    continue
                try:
                    self.root.unbind(sequence, funcid)
                except Exception:
                    pass
            bind_ids.clear()
        if popup is not None:
            try:
                popup.destroy()
            except Exception:
                pass
        self._plan_context_popup = None
        self._plan_context_menu = None
        self._planner_restore_plan_list_focus()


    def _plan_list_context_visible_index(self, event=None):
        if self.plan_listbox is None or not self.visible_entry_indices or event is None:
            return None
        try:
            idx = self.plan_listbox.nearest(event.y)
        except Exception:
            return None
        if idx < 0 or idx >= len(self.visible_entry_indices):
            return None
        bbox = self.plan_listbox.bbox(idx)
        if not bbox:
            return None
        bx, by, bw, bh = bbox
        if not (bx <= event.x <= bx + bw and by <= event.y <= by + bh):
            return None
        return idx

    def _plan_list_prepare_context_selection(self, event=None):
        vis_idx = self._plan_list_context_visible_index(event)
        if vis_idx is None:
            return None
        model_idx = self.visible_entry_indices[vis_idx]
        selected = set(self._planner_get_selected_indices())
        if model_idx not in selected:
            self._select_model_indices([model_idx])
        elif self.current_index != model_idx:
            self.current_index = model_idx
            self._render_current_line()
            try:
                self._planner_refresh_plan_meta()
            except Exception:
                pass
        return model_idx

    def _selected_entries_support_remove_context(self):
        supported = {"Checkpoint Merge", "LoRA Bake"}
        try:
            return any(self.plan_data['entries'][idx].get('type') in supported for idx in self._planner_get_selected_indices())
        except Exception:
            return False


    def _insert_remove_model_lines_below_selected(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return
        entries = self.plan_data.get('entries', [])
        insert_groups = []
        for idx in indices:
            if not (0 <= idx < len(entries)):
                continue
            entry = entries[idx]
            etype = entry.get('type')
            names = []
            if etype == 'Checkpoint Merge':
                names = [entry.get('model0'), entry.get('model1'), entry.get('model2')]
            elif etype == 'LoRA Bake':
                names = [entry.get('checkpoint')]
                for lora in entry.get('loras', []) or []:
                    if isinstance(lora, dict):
                        names.append(lora.get('name'))
            else:
                continue
            clean = []
            seen = set()
            for name in names:
                name = str(name or '').strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                clean.append(name)
            if clean:
                insert_groups.append((idx, clean))
        if not insert_groups:
            return
        self._planner_push_history()
        inserted = []
        offset = 0
        for base_idx, names in insert_groups:
            insert_at = base_idx + 1 + offset
            new_items = []
            for name in names:
                item = make_entry('Remove Model')
                item['model'] = name
                new_items.append(item)
            entries[insert_at:insert_at] = new_items
            inserted.extend(range(insert_at, insert_at + len(new_items)))
            offset += len(new_items)
        if inserted:
            self.current_index = inserted[-1]
            self._save_plan_to_file()
            self._refresh_line_selector()
            self._render_current_line()
            self._select_model_indices(inserted)
            self.status_label.config(text=f"Inserted {len(inserted)} Remove Model line(s)")


    def _show_plan_context_menu(self, event=None):
        if self.plan_listbox is None or event is None:
            return 'break'
        model_idx = self._plan_list_prepare_context_selection(event)
        if model_idx is None:
            self._hide_plan_context_menu()
            return 'break'

        self._hide_plan_context_menu()
        self._planner_restore_plan_list_focus()
        colors = self._theme_colors()
        popup = tk.Toplevel(self.root)
        popup.wm_overrideredirect(True)
        popup.attributes('-topmost', True)
        try:
            popup.configure(bg=colors['panel'])
        except Exception:
            pass

        x = int(getattr(event, 'x_root', 0) or 0) + 10
        y = int(getattr(event, 'y_root', 0) or 0) + 8
        popup.wm_geometry(f"+{x}+{y}")

        card = Frame(
            popup,
            bg=colors['surface'],
            highlightthickness=1,
            highlightbackground=colors['border'],
            padx=2,
            pady=2,
        )
        card.pack(fill='both', expand=True)

        header = Label(
            card,
            text='Plan View Actions',
            anchor='w',
            justify='left',
            bg=colors['surface'],
            fg=colors['text'],
            padx=12,
            pady=10,
            font=('MS Gothic', 11, 'bold'),
        )
        header.pack(fill='x')

        button_area = Frame(card, bg=colors['surface'])
        button_area.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        def run_and_close(func):
            def wrapped():
                self._hide_plan_context_menu()
                try:
                    func()
                finally:
                    self._planner_restore_plan_list_focus()
                return 'break'
            return wrapped

        actions = [
            ('Copy', self._copy_selected_lines, True),
            ('Paste Below', self._paste_copied_lines, True),
            ('Duplicate Below', self._duplicate_selected_lines, True),
            ('Insert Remove Model Below', self._insert_remove_model_lines_below_selected, self._selected_entries_support_remove_context()),
            ('Create Preset', self._save_preset_json, True),
            ('Delete', self._delete_selected_lines, True),
        ]

        for label, func, enabled in actions:
            row = Frame(button_area, bg=colors['surface'])
            row.pack(fill='x', pady=2)
            style_name = 'MenuDanger.TButton' if label == 'Delete' else 'Menu.TButton'
            btn = ttk.Button(
                row,
                text=label,
                command=run_and_close(func) if enabled else self._hide_plan_context_menu,
                style=style_name,
                cursor='hand2' if enabled else 'arrow',
                takefocus=False,
            )
            if not enabled:
                btn.state(['disabled'])
            btn.pack(fill='x', expand=True)
            if enabled:
                btn.bind('<ButtonRelease-1>', lambda _e, f=func: run_and_close(f)(), add='+')
                btn.bind('<Return>', lambda _e, f=func: run_and_close(f)(), add='+')
                btn.bind('<space>', lambda _e, f=func: run_and_close(f)(), add='+')

        self._plan_context_popup = popup
        self._plan_context_menu = popup

        def _outside_click(ev=None):
            try:
                widget = getattr(ev, 'widget', None)
                probe = widget
                while probe is not None:
                    if probe is popup:
                        return None
                    probe = getattr(probe, 'master', None)
            except Exception:
                pass
            self._hide_plan_context_menu()
            return None

        bind_ids = {}
        try:
            bind_ids['<Button-1>'] = self.root.bind('<Button-1>', _outside_click, add='+')
            bind_ids['<Button-2>'] = self.root.bind('<Button-2>', _outside_click, add='+')
            bind_ids['<Button-3>'] = self.root.bind('<Button-3>', _outside_click, add='+')
        except Exception:
            bind_ids = {}
        self._plan_context_root_bind_ids = bind_ids
        try:
            popup.update_idletasks()
        except Exception:
            pass
        return 'break'


    def _plan_list_drag_start(self, event):
        if self.plan_listbox is None:
            return
        idx = self.plan_listbox.nearest(event.y)
        self._drag_start_visible = idx
        self._drag_selected_model_indices = self._planner_get_selected_indices()
        state = int(getattr(event, 'state', 0) or 0)
        self._drag_requires_primary = bool(state & 0x0004 or state & 0x0008 or state & 0x0010)


    def _plan_list_drag_end(self, event):
        if self.plan_listbox is None or self._drag_start_visible is None:
            return
        target_vis = self.plan_listbox.nearest(event.y)
        self._drag_start_visible = None
        selected = self._drag_selected_model_indices or self._planner_get_selected_indices()
        requires_primary = bool(getattr(self, '_drag_requires_primary', False))
        self._drag_selected_model_indices = []
        self._drag_requires_primary = False
        if not requires_primary:
            return
        if not selected or not self.visible_entry_indices:
            return
        target_vis = max(0, min(target_vis, len(self.visible_entry_indices) - 1))
        target_model = self.visible_entry_indices[target_vis]
        if target_model in selected:
            return
        self._planner_push_history()
        entries = self.plan_data.get('entries', [])
        block = [entries[i] for i in selected]
        for idx in reversed(selected):
            entries.pop(idx)
        insert_at = target_model
        shift = sum(1 for idx in selected if idx < target_model)
        insert_at -= shift
        if event.y > self.plan_listbox.winfo_height() - 10:
            insert_at += 1
        insert_at = max(0, min(insert_at, len(entries)))
        for offset, item in enumerate(block):
            entries.insert(insert_at + offset, item)
        moved = list(range(insert_at, insert_at + len(block)))
        self.current_index = moved[-1]
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(moved)
        self.status_label.config(text='Reordered selected lines')


    def _compute_aliases(self):
        aliases = []
        for idx, entry in enumerate(self.plan_data.get('entries', [])):
            etype = entry.get('type')
            produces = []
            consumes = []
            if etype == 'Download Model':
                if entry.get('model_name'):
                    produces.append(entry.get('model_name'))
            elif etype == 'Local Model':
                if entry.get('local_path'):
                    produces.append(Path(entry.get('local_path')).stem)
            elif etype == 'Checkpoint Merge':
                consumes.extend([entry.get('model0'), entry.get('model1'), entry.get('model2')])
                if entry.get('output_name'):
                    produces.append(entry.get('output_name'))
            elif etype == 'LoRA Bake':
                consumes.append(entry.get('checkpoint'))
                for lora in entry.get('loras', []):
                    consumes.append(lora.get('name'))
                if entry.get('output_name'):
                    produces.append(entry.get('output_name'))
            elif etype == 'Remove Model':
                consumes.append(entry.get('model'))
            aliases.append((idx, [x for x in produces if x], [x for x in consumes if x]))
        return aliases


    def _show_dependency_view(self):
        lines = ['Dependency View', '']
        producers = {}
        for idx, produces, _consumes in self._compute_aliases():
            for name in produces:
                producers[name] = idx
        for idx, produces, consumes in self._compute_aliases():
            entry = self.plan_data['entries'][idx]
            lines.append(f"[{idx + 1}] {self._line_summary(entry)}")
            if produces:
                lines.append(f"  produces: {', '.join(produces)}")
            if consumes:
                dep = []
                for name in consumes:
                    src = producers.get(name)
                    dep.append(f"{name} <- line {src + 1}" if src is not None else f"{name} <- MISSING")
                lines.append('  consumes: ' + '; '.join(dep))
            lines.append('')
        self._show_scrollable_text_dialog('Dependency View', '\n'.join(lines))


    def _show_unused_models(self):
        alias_info = self._compute_aliases()
        used = set()
        for _idx, _prod, consumes in alias_info:
            used.update(consumes)
        unused = []
        for idx, produces, _consumes in alias_info:
            for name in produces:
                if name not in used:
                    unused.append(f"line {idx + 1}: {name}")
        detail = 'Unused model aliases:\n\n' + ('\n'.join(unused) if unused else 'None')
        self._show_scrollable_text_dialog('Unused Models', detail)


    def _validate_plan_entries(self):
        problems = []
        available = {'Checkpoint': set(), 'LoRA': set(), 'LyCORIS': set()}
        for idx, entry in enumerate(self.plan_data.get('entries', []), start=1):
            etype = entry.get('type')
            prefix = f"line {idx} ({etype})"
            if etype == 'Download Model':
                if not entry.get('model_name'): problems.append(f"{prefix}: model_name is empty")
                if not entry.get('link'): problems.append(f"{prefix}: link is empty")
                name = (entry.get('model_name') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip()
                if name: available.setdefault(kind, set()).add(name)
            elif etype == 'Local Model':
                if not entry.get('local_path'): problems.append(f"{prefix}: local_path is empty")
                path = (entry.get('local_path') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip()
                if path: available.setdefault(kind, set()).add(Path(path).stem)
            elif etype == 'Remove Model':
                if not entry.get('model'): problems.append(f"{prefix}: model is empty")
            elif etype == 'Checkpoint Merge':
                for k in ('model0','model1','output_name'):
                    if not entry.get(k): problems.append(f"{prefix}: {k} is empty")
                for ref in (entry.get('model0'), entry.get('model1'), entry.get('model2')):
                    if ref and ref not in available['Checkpoint']:
                        problems.append(f"{prefix}: checkpoint ref not available -> {ref}")
                if entry.get('output_name'): available['Checkpoint'].add(entry['output_name'])
            elif etype == 'LoRA Bake':
                if not entry.get('checkpoint'): problems.append(f"{prefix}: checkpoint is empty")
                elif entry.get('checkpoint') not in available['Checkpoint']:
                    problems.append(f"{prefix}: checkpoint ref not available -> {entry.get('checkpoint')}")
                if not entry.get('output_name'): problems.append(f"{prefix}: output_name is empty")
                for lora in entry.get('loras', []):
                    name = lora.get('name')
                    if not name:
                        problems.append(f"{prefix}: one LoRA name is empty")
                    elif name not in available['LoRA'] and name not in available['LyCORIS']:
                        problems.append(f"{prefix}: LoRA ref not available -> {name}")
                if entry.get('output_name'): available['Checkpoint'].add(entry['output_name'])
        return problems



    def _show_prevalidation(self):
        issues = self._validate_plan_entries()
        detail = 'Pre-validation results:\n\n' + ('\n'.join(issues) if issues else 'No issues found.')
        self._show_scrollable_text_dialog('Pre-validation', detail)


    def _show_plan_diff_preview(self):
        current_path = (self.entries.get('filepath').get().strip() if self.entries.get('filepath') else '')
        if not current_path or not Path(current_path).exists():
            self._show_scrollable_text_dialog('Plan Diff Preview', 'No saved plan file to compare against.')
            return
        tmp = Path(tempfile.mkstemp(prefix='planner_diff_', suffix='.txt')[1])
        try:
            export_plan_records_txt(str(tmp), self.plan_data)
            current_lines = Path(current_path).read_text(encoding='utf-8', errors='replace').splitlines()
            edited_lines = tmp.read_text(encoding='utf-8', errors='replace').splitlines()
            diff = list(difflib.unified_diff(current_lines, edited_lines, fromfile=current_path, tofile='in-memory-plan', lineterm=''))
            self._show_scrollable_text_dialog('Plan Diff Preview', '\n'.join(diff) if diff else 'No differences.')
        finally:
            try:
                tmp.unlink()
            except Exception:
                pass


    def _save_preset_json(self):
        indices = self._planner_get_selected_indices() or [self.current_index]
        payload = [copy.deepcopy(self.plan_data['entries'][i]) for i in indices]
        path = filedialog.asksaveasfilename(title='Save preset as JSON', defaultextension='.json', filetypes=[('JSON','*.json'),('All files','*.*')])
        if not path:
            return
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        self.status_label.config(text=f"Saved preset: {Path(path).name}")


    def _load_preset_json(self):
        path = filedialog.askopenfilename(
            title='Load preset or plan text',
            filetypes=[('Preset / Plan','*.json *.txt'),('JSON','*.json'),('Plan Text','*.txt'),('All files','*.*')],
        )
        if not path:
            return
        try:
            source_path = Path(path)
            payload = []
            if source_path.suffix.lower() == '.txt':
                raw_plan = normalize_plan(load_plan_records(str(source_path)))
                collapsed = self._collapse_internal_plan_entries(raw_plan)
                for raw in collapsed.get('entries', []) or []:
                    entry = copy.deepcopy(raw)
                    entry['id'] = make_entry(entry.get('type', 'Checkpoint Merge')).get('id')
                    payload.append(entry)
            else:
                data = json.loads(source_path.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    raise ValueError('Preset JSON must be an object or a list of objects')
                for raw in data:
                    entry = self._normalize_entry_preserving_embedded_sources(raw)
                    entry['id'] = make_entry(entry.get('type', 'Checkpoint Merge')).get('id')
                    payload.append(entry)
            if not payload:
                return
            self._planner_push_history()
            insert_at = max(self._planner_get_selected_indices() or [self.current_index]) + 1 if self.plan_data.get('entries') else 0
            self.plan_data['entries'][insert_at:insert_at] = payload
            self.current_index = insert_at + len(payload) - 1
            self._save_plan_to_file()
            self._refresh_line_selector()
            self._render_current_line()
            self._select_model_indices(list(range(insert_at, insert_at + len(payload))))
            status_kind = 'plan text' if source_path.suffix.lower() == '.txt' else 'preset'
            self.status_label.config(text=f"Loaded {status_kind}: {source_path.name}")
        except Exception as e:
            self._show_detailed_error('Load Preset Error', e)




    def _refresh_line_selector(self):
        entries = self.plan_data.get('entries', [])
        self.visible_entry_indices = self._visible_entries_for_filter()
        self._plan_problem_map_cache = self._plan_entry_problem_map()
        if not self.visible_entry_indices and entries:
            self.visible_entry_indices = list(range(len(entries)))
        self._sync_target_line_combo()
        if self.plan_listbox is not None:
            selected_model = set(self._planner_get_selected_indices())
            active_model = self.current_index
            display_items = [f"{idx + 1:02d} • {self._line_summary(entries[idx]):<72} ?" for idx in self.visible_entry_indices]
            render_cache = (tuple(self.visible_entry_indices), tuple(display_items))
            try:
                current_yview = self.plan_listbox.yview()
            except Exception:
                current_yview = None
            if render_cache != getattr(self, '_plan_listbox_render_cache', None):
                self.plan_listbox.delete(0, 'end')
                for item in display_items:
                    self.plan_listbox.insert('end', item)
                self._plan_listbox_render_cache = render_cache
            self._apply_plan_listbox_item_styles()
            self._selection_guard = True
            try:
                self.plan_listbox.selection_clear(0, 'end')
                for vis_idx, model_idx in enumerate(self.visible_entry_indices):
                    if model_idx in selected_model:
                        self.plan_listbox.selection_set(vis_idx)
                if self.visible_entry_indices:
                    try:
                        vis_cur = self.visible_entry_indices.index(active_model)
                    except ValueError:
                        vis_cur = 0
                        self.current_index = self.visible_entry_indices[0]
                        self._sync_target_line_combo()
                    self.plan_listbox.activate(vis_cur)
                    self.plan_listbox.see(vis_cur)
                    if not self.plan_listbox.curselection():
                        self.plan_listbox.selection_set(vis_cur)
                    if current_yview is None:
                        self.plan_listbox.see(vis_cur)
            finally:
                self._selection_guard = False
            if current_yview is not None:
                try:
                    self.plan_listbox.yview_moveto(float(current_yview[0]))
                except Exception:
                    try:
                        self.plan_listbox.see(self.plan_listbox.index('active'))
                    except Exception:
                        pass
        self._planner_refresh_plan_meta()
        self._update_current_line_indicator()
        self._update_plan_action_buttons()


    def _add_line(self):
        self._planner_push_history()
        insert_at = max(self._planner_get_selected_indices() or [self.current_index]) + 1 if self.plan_data.get('entries') else 0
        self.plan_data.setdefault('entries', []).insert(insert_at, make_entry('Checkpoint Merge'))
        self.current_index = insert_at
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices([insert_at])



    def _remove_line(self):
        self._delete_selected_lines()



    def _move_line_up(self):
        indices = self._planner_get_selected_indices()
        if not indices or indices[0] <= 0:
            return
        self._planner_push_history()
        entries = self.plan_data.get('entries', [])
        block = [entries[i] for i in indices]
        for idx in reversed(indices):
            entries.pop(idx)
        insert_at = indices[0] - 1
        for offset, item in enumerate(block):
            entries.insert(insert_at + offset, item)
        moved = list(range(insert_at, insert_at + len(block)))
        self.current_index = moved[-1]
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(moved)



    def _move_line_down(self):
        indices = self._planner_get_selected_indices()
        entries = self.plan_data.get('entries', [])
        if not indices or indices[-1] >= len(entries) - 1:
            return
        self._planner_push_history()
        block = [entries[i] for i in indices]
        for idx in reversed(indices):
            entries.pop(idx)
        insert_at = indices[0] + 1
        for offset, item in enumerate(block):
            entries.insert(insert_at + offset, item)
        moved = list(range(insert_at, insert_at + len(block)))
        self.current_index = moved[-1]
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(moved)




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

    def _planner_visible_line_types(self):
        return ["Checkpoint Merge", "LoRA Bake"]

    def _planner_internal_line_types(self):
        return ["Download Model", "Local Model", "Remove Model"]

    def _slot_sources_dict(self, entry):
        data = entry.get("_slot_sources")
        if not isinstance(data, dict):
            data = {}
            entry["_slot_sources"] = data
        return data

    def _get_entry_slot_source(self, entry, slot_key: str):
        data = entry.get("_slot_sources")
        if isinstance(data, dict):
            value = data.get(slot_key)
            if isinstance(value, dict):
                return value
        return None

    def _set_entry_slot_source(self, entry, slot_key: str, spec):
        data = self._slot_sources_dict(entry)
        if spec:
            data[slot_key] = copy.deepcopy(spec)
        else:
            data.pop(slot_key, None)
        if not data:
            entry.pop("_slot_sources", None)

    def _get_lora_source(self, lora_entry):
        value = lora_entry.get("_source")
        return value if isinstance(value, dict) else None

    def _set_lora_source(self, lora_entry, spec):
        if spec:
            lora_entry["_source"] = copy.deepcopy(spec)
        else:
            lora_entry.pop("_source", None)

    def _source_alias_from_spec(self, spec, fallback: str = "") -> str:
        if not isinstance(spec, dict):
            return str(fallback or "").strip()
        alias = str(spec.get("alias") or "").strip()
        if alias:
            return alias
        if str(spec.get("mode") or "") == "local":
            local_path = str(spec.get("local_path") or "").strip()
            if local_path:
                return Path(local_path).stem
        return str(fallback or "").strip()

    def _normalize_source_kind(self, raw_kind: str, kind_options):
        options = [str(x) for x in (kind_options or []) if str(x).strip()]
        if not options:
            return str(raw_kind or "").strip()
        raw = str(raw_kind or "").strip().lower()
        for option in options:
            if raw == option.lower():
                return option
        alias_map = {
            "lora": "LoRA",
            "lycoris": "LyCORIS",
            "checkpoint": "Checkpoint",
        }
        normalized = alias_map.get(raw)
        if normalized in options:
            return normalized
        return options[0]

    def _guess_alias_from_link(self, link: str, default_value: str = "") -> str:
        text = str(link or "").strip()
        if not text:
            return str(default_value or "").strip()
        tail = re.split(r"[?#]", text)[0].rstrip("/")
        name = Path(tail).name
        stem = Path(name).stem
        return stem or str(default_value or "").strip()

    def _format_source_spec_text(self, spec) -> str:
        if not isinstance(spec, dict):
            return ""
        mode = str(spec.get("mode") or "").strip().title()
        kind = str(spec.get("kind") or "").strip() or "Model"
        alias = self._source_alias_from_spec(spec)
        if str(spec.get("mode") or "") == "download":
            tail = str(spec.get("link") or "").strip()
            tail = tail[:72] + ("…" if len(tail) > 72 else "")
            return f"Embedded source: {mode} {kind} → {alias}  ({tail})"
        if str(spec.get("mode") or "") == "local":
            local_path = str(spec.get("local_path") or "").strip()
            local_name = Path(local_path).name if local_path else alias
            return f"Embedded source: {mode} {kind} → {alias}  ({local_name})"
        return f"Embedded source: {kind} → {alias}"

    def _ask_source_kind(self, title: str, kind_options, default_kind: str = ""):
        options = [str(x) for x in (kind_options or []) if str(x).strip()]
        if not options:
            return ""
        if len(options) == 1:
            return options[0]
        initial = self._normalize_source_kind(default_kind or options[0], options)
        answer = simpledialog.askstring(title, f"Type ({', '.join(options)}):", initialvalue=initial, parent=self.root)
        if answer is None:
            return None
        answer = answer.strip()
        if not answer:
            return initial
        return self._normalize_source_kind(answer, options)

    def _prompt_download_source_dialog(self, label: str, kind_options, current_alias: str = "", current_source=None):
        source = current_source if isinstance(current_source, dict) else {}
        kind = self._ask_source_kind(f"{label} Download", kind_options, str(source.get("kind") or ""))
        if not kind:
            return None
        initial_link = str(source.get("link") or "").strip()
        link = simpledialog.askstring(f"{label} Download", "Model URL:", initialvalue=initial_link, parent=self.root)
        if link is None:
            return None
        link = link.strip()
        if not link:
            return None
        alias_default = str(source.get("alias") or current_alias or self._guess_alias_from_link(link, kind)).strip()
        alias = simpledialog.askstring(f"{label} Download", "Alias / model name:", initialvalue=alias_default, parent=self.root)
        if alias is None:
            return None
        alias = alias.strip() or alias_default
        if not alias:
            return None
        return {
            "mode": "download",
            "kind": kind,
            "alias": alias,
            "link": link,
        }

    def _prompt_local_source_dialog(self, label: str, kind_options, current_source=None):
        source = current_source if isinstance(current_source, dict) else {}
        kind = self._ask_source_kind(f"{label} Local", kind_options, str(source.get("kind") or ""))
        if not kind:
            return None
        path = filedialog.askopenfilename(
            title=f"Select local {kind}",
            filetypes=[("Model Files", "*.safetensors *.ckpt *.pt *.bin"), ("All files", "*.*")],
        )
        path = self._normalize_user_path(path)
        if not path:
            return None
        return {
            "mode": "local",
            "kind": kind,
            "alias": Path(path).stem,
            "local_path": path,
        }

    def _resolve_merge_mode_key(self, raw_mode: str) -> str:
        raw = str(raw_mode or "").strip().lower()
        if not raw:
            return ""
        for item in self.merge_modes:
            key = str(item.get("key") or "").strip()
            if key and raw == key.lower():
                return key
        return ""

    def _strip_mode_signatures(self, raw_text: str) -> str:
        text = str(raw_text or "")
        if not text:
            return ""
        text = re.sub(r"(?<!\S)@(?:m|mode)\s+\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[ 	]+", " ", text)
        text = re.sub(r"[ 	]*\n[ 	]*", "\n", text)
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines).strip()

    def _absorb_mode_signature_from_entry(self, entry):
        if not isinstance(entry, dict) or entry.get("type") != "Checkpoint Merge":
            return entry
        raw = str(entry.get("raw_signatures") or "")
        if not raw:
            return entry
        matches = list(re.finditer(r"(?<!\S)@(?:m|mode)\s+(\S+)", raw, flags=re.IGNORECASE))
        if not matches:
            return entry
        resolved = self._resolve_merge_mode_key(matches[-1].group(1))
        if not resolved:
            return entry
        entry["merge_mode"] = resolved
        entry["raw_signatures"] = self._strip_mode_signatures(raw)
        return entry

    def _normalize_loaded_plan_entries(self, plan_data):
        data = copy.deepcopy(plan_data or {"entries": []})
        entries = data.get("entries") or []
        if not isinstance(entries, list):
            data["entries"] = []
            return data
        for entry in entries:
            self._absorb_mode_signature_from_entry(entry)
        data["entries"] = entries
        return data

    def _restore_embedded_sources_into_entry(self, entry, raw_entry):
        if not isinstance(entry, dict) or not isinstance(raw_entry, dict):
            return entry
        raw_slot_sources = raw_entry.get("_slot_sources")
        if isinstance(raw_slot_sources, dict):
            cleaned_slot_sources = {}
            for key, spec in raw_slot_sources.items():
                if isinstance(spec, dict):
                    cleaned_slot_sources[str(key)] = copy.deepcopy(spec)
            if cleaned_slot_sources:
                entry["_slot_sources"] = cleaned_slot_sources
        raw_loras = raw_entry.get("loras") or []
        loras = entry.get("loras") or []
        if isinstance(raw_loras, list) and isinstance(loras, list):
            for idx, raw_lora in enumerate(raw_loras):
                if idx >= len(loras) or not isinstance(raw_lora, dict) or not isinstance(loras[idx], dict):
                    continue
                spec = raw_lora.get("_source")
                if isinstance(spec, dict):
                    loras[idx]["_source"] = copy.deepcopy(spec)
        return entry

    def _normalize_entry_preserving_embedded_sources(self, raw_entry):
        entry = normalize_plan({'entries': [copy.deepcopy(raw_entry)]})['entries'][0]
        entry = self._restore_embedded_sources_into_entry(entry, raw_entry)
        self._absorb_mode_signature_from_entry(entry)
        return entry

    def _normalize_plan_preserving_embedded_sources(self, raw_plan):
        plan = normalize_plan(copy.deepcopy(raw_plan))
        raw_entries = []
        if isinstance(raw_plan, dict):
            raw_entries = raw_plan.get('entries') or []
        entries = plan.get('entries') or []
        if isinstance(raw_entries, list) and isinstance(entries, list):
            for idx, raw_entry in enumerate(raw_entries):
                if idx >= len(entries):
                    break
                self._restore_embedded_sources_into_entry(entries[idx], raw_entry)
        return self._normalize_loaded_plan_entries(plan)

    def _build_source_backed_ref_row(self, parent, label: str, get_value, set_value, get_source, set_source, candidates, kind_options, allow_local: bool = True):
        wrapper = Frame(parent)
        wrapper.pack(fill="x", pady=3)
        row = Frame(wrapper)
        row.pack(fill="x")
        label_widget = Label(row, text=label, width=18, anchor="w")
        label_widget.pack(side="left")

        current_value = str(get_value() or "").strip()
        values = list(dict.fromkeys([x for x in ([current_value] + list(candidates or [])) if str(x).strip()]))
        var = tk.StringVar(value=current_value)
        combo = ttk.Combobox(row, textvariable=var, values=values or [""], state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)

        detail_wrap = Frame(wrapper)
        detail_wrap.pack(fill="x", padx=22, pady=(2, 0))
        prefix_var = tk.StringVar(value="")
        alias_var = tk.StringVar(value="")
        tail_var = tk.StringVar(value="")
        prefix_lbl = Label(detail_wrap, textvariable=prefix_var, anchor="w", fg="#666666", justify="left")
        prefix_lbl.pack(side="left")
        alias_lbl = Label(detail_wrap, textvariable=alias_var, anchor="w", fg="#1f5fbf", justify="left", cursor="hand2")
        alias_lbl.pack(side="left")
        tail_lbl = Label(detail_wrap, textvariable=tail_var, anchor="w", fg="#1f5fbf", justify="left", cursor="hand2", wraplength=720)
        tail_lbl.pack(side="left", fill="x", expand=True)

        def update_detail(spec=None):
            spec = get_source() if spec is None else spec
            if not isinstance(spec, dict):
                prefix_var.set("")
                alias_var.set("")
                tail_var.set("")
                alias_lbl.configure(fg="#666666", cursor="")
                tail_lbl.configure(fg="#666666", cursor="")
                return
            mode = str(spec.get("mode") or "").strip().lower()
            mode_title = mode.title() if mode else ""
            kind = str(spec.get("kind") or "").strip() or "Model"
            alias = self._source_alias_from_spec(spec, str(get_value() or ""))
            if mode_title:
                prefix_var.set(f"Embedded source: {mode_title} {kind} → ")
            else:
                prefix_var.set(f"Embedded source: {kind} → ")
            alias_var.set(alias)
            if mode == "download":
                tail = str(spec.get("link") or "").strip()
                display = tail[:72] + ("…" if len(tail) > 72 else "")
                tail_var.set(f"  ({display})" if display else "")
                alias_lbl.configure(fg="#1f5fbf", cursor="hand2")
                tail_lbl.configure(fg="#1f5fbf", cursor="hand2")
            elif mode == "local":
                local_path = str(spec.get("local_path") or "").strip()
                local_name = Path(local_path).name if local_path else alias
                tail_var.set(f"  ({local_name})" if local_name else "")
                alias_lbl.configure(fg="#1f5fbf", cursor="hand2")
                tail_lbl.configure(fg="#666666", cursor="")
            else:
                tail_var.set("")
                alias_lbl.configure(fg="#1f5fbf", cursor="hand2")
                tail_lbl.configure(fg="#666666", cursor="")

        def sync(*_args):
            value = var.get().strip()
            set_value(value)
            src = get_source()
            src_alias = self._source_alias_from_spec(src)
            if src and value and value != src_alias:
                set_source(None)
                src = None
            update_detail(src)
            self._after_entry_change()

        var.trace_add("write", sync)

        def edit_alias(_event=None):
            src = get_source()
            if not isinstance(src, dict):
                return "break"
            initial = self._source_alias_from_spec(src, str(get_value() or ""))
            alias = simpledialog.askstring(f"{label} Source", "Alias / model name:", initialvalue=initial, parent=self.root)
            if alias is None:
                return "break"
            alias = alias.strip() or initial
            if not alias:
                return "break"
            updated = copy.deepcopy(src)
            updated["alias"] = alias
            set_source(updated)
            set_value(alias)
            var.set(alias)
            update_detail(updated)
            self._after_entry_change()
            return "break"

        def edit_link(_event=None):
            src = get_source()
            if not isinstance(src, dict) or str(src.get("mode") or "").strip().lower() != "download":
                return "break"
            initial = str(src.get("link") or "").strip()
            link = simpledialog.askstring(f"{label} Download", "Model URL:", initialvalue=initial, parent=self.root)
            if link is None:
                return "break"
            link = link.strip()
            if not link:
                return "break"
            updated = copy.deepcopy(src)
            updated["link"] = link
            set_source(updated)
            update_detail(updated)
            self._after_entry_change()
            return "break"

        alias_lbl.bind("<Double-Button-1>", edit_alias, add="+")
        tail_lbl.bind("<Double-Button-1>", edit_link, add="+")

        def choose_download():
            spec = self._prompt_download_source_dialog(label, kind_options, str(get_value() or ""), get_source())
            if not spec:
                return
            set_source(spec)
            set_value(self._source_alias_from_spec(spec))
            update_detail(spec)
            self._after_entry_change()
            self._schedule_rerender_current_line()

        def choose_local():
            spec = self._prompt_local_source_dialog(label, kind_options, get_source())
            if not spec:
                return
            set_source(spec)
            set_value(self._source_alias_from_spec(spec))
            update_detail(spec)
            self._after_entry_change()
            self._schedule_rerender_current_line()

        def clear_source():
            if get_source():
                set_source(None)
                update_detail(None)
                self._after_entry_change()
                self._schedule_rerender_current_line()

        ttk.Button(row, text="Download...", command=choose_download).pack(side="left", padx=(0, 4))
        if allow_local:
            ttk.Button(row, text="Local...", command=choose_local).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Clear", command=clear_source).pack(side="left")
        update_detail(get_source())
        self._attach_tooltip(label_widget, self._right_help(label).get("detail", ""))
        self._add_right_inline_help(parent, label)
        return var

    def _iter_embedded_sources(self, entry):
        etype = entry.get("type")
        if etype == "Checkpoint Merge":
            for slot in ("model0", "model1", "model2"):
                spec = self._get_entry_slot_source(entry, slot)
                alias = str(entry.get(slot) or self._source_alias_from_spec(spec)).strip()
                if spec and alias:
                    yield slot, alias, str(spec.get("kind") or "Checkpoint"), spec
        elif etype == "LoRA Bake":
            spec = self._get_entry_slot_source(entry, "checkpoint")
            alias = str(entry.get("checkpoint") or self._source_alias_from_spec(spec)).strip()
            if spec and alias:
                yield "checkpoint", alias, str(spec.get("kind") or "Checkpoint"), spec
            for idx, lora in enumerate(entry.get("loras", []) or []):
                spec = self._get_lora_source(lora)
                alias = str(lora.get("name") or self._source_alias_from_spec(spec)).strip()
                if spec and alias:
                    yield f"lora:{idx}", alias, str(spec.get("kind") or "LoRA"), spec

    def _collect_available_models(self, upto_index: int) -> Dict[str, List[str]]:
        available: Dict[str, Dict[str, str]] = {"Checkpoint": {}, "LoRA": {}, "LyCORIS": {}}
        removed = set()
        entries = self.plan_data.get("entries", [])
        for entry in entries[:upto_index]:
            etype = entry.get("type")
            if etype == "Remove Model":
                name = str(entry.get("model") or "").strip()
                if name:
                    removed.add(name)
                    for bucket in available.values():
                        bucket.pop(name, None)
                continue
            if etype == "Download Model":
                name = str(entry.get("model_name") or "").strip()
                kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
                if name and name not in removed:
                    available.setdefault(kind, {})[name] = kind
            elif etype == "Local Model":
                path = str(entry.get("local_path") or "").strip()
                if path:
                    name = Path(path).stem
                    kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
                    if name not in removed:
                        available.setdefault(kind, {})[name] = kind
            else:
                for _slot, alias, kind, _spec in self._iter_embedded_sources(entry):
                    if alias and alias not in removed:
                        available.setdefault(kind, {})[alias] = kind
                if etype in ("Checkpoint Merge", "LoRA Bake"):
                    name = str(entry.get("output_name") or "").strip()
                    if name and name not in removed:
                        available["Checkpoint"][name] = "Checkpoint"
        return {k: sorted(v.keys()) for k, v in available.items()}

    def _plan_entry_problem_map(self) -> Dict[int, List[str]]:
        problems_by_idx: Dict[int, List[str]] = {}
        entries = self.plan_data.get("entries", [])
        for idx, entry in enumerate(entries):
            etype = entry.get("type")
            prefix = f"line {idx + 1} ({etype})"
            problems = []
            available = self._collect_available_models(idx)
            if etype == "Download Model":
                if not entry.get("model_name"):
                    problems.append(f"{prefix}: model_name is empty")
                if not entry.get("link"):
                    problems.append(f"{prefix}: link is empty")
            elif etype == "Local Model":
                if not entry.get("local_path"):
                    problems.append(f"{prefix}: local_path is empty")
            elif etype == "Remove Model":
                if not entry.get("model"):
                    problems.append(f"{prefix}: model is empty")
            elif etype == "Checkpoint Merge":
                for req_key in ("model0", "model1", "output_name"):
                    if not entry.get(req_key):
                        problems.append(f"{prefix}: {req_key} is empty")
                for slot in ("model0", "model1", "model2"):
                    ref = str(entry.get(slot) or "").strip()
                    if ref and ref not in available.get("Checkpoint", []) and not self._get_entry_slot_source(entry, slot):
                        problems.append(f"{prefix}: checkpoint ref not available -> {ref}")
            elif etype == "LoRA Bake":
                checkpoint = str(entry.get("checkpoint") or "").strip()
                if not checkpoint:
                    problems.append(f"{prefix}: checkpoint is empty")
                elif checkpoint not in available.get("Checkpoint", []) and not self._get_entry_slot_source(entry, "checkpoint"):
                    problems.append(f"{prefix}: checkpoint ref not available -> {checkpoint}")
                if not entry.get("output_name"):
                    problems.append(f"{prefix}: output_name is empty")
                for lora in entry.get("loras", []) or []:
                    name = str(lora.get("name") or "").strip()
                    if not name:
                        problems.append(f"{prefix}: one LoRA name is empty")
                    elif (
                        name not in available.get("LoRA", [])
                        and name not in available.get("LyCORIS", [])
                        and not self._get_lora_source(lora)
                    ):
                        problems.append(f"{prefix}: LoRA ref not available -> {name}")
            problems_by_idx[idx] = problems
        return problems_by_idx

    def _entry_consumed_aliases(self, entry: Dict[str, Any]) -> List[str]:
        etype = entry.get("type")
        out = []
        if etype == "Checkpoint Merge":
            out.extend([entry.get("model0"), entry.get("model1"), entry.get("model2")])
        elif etype == "LoRA Bake":
            out.append(entry.get("checkpoint"))
            for lora in entry.get("loras", []) or []:
                out.append(lora.get("name"))
        elif etype == "Remove Model":
            out.append(entry.get("model"))
        return [str(x).strip() for x in out if str(x or "").strip()]

    def _entry_produced_aliases(self, entry: Dict[str, Any]) -> List[str]:
        etype = entry.get("type")
        produced = []
        if etype == "Download Model":
            name = str(entry.get("model_name") or "").strip()
            if name:
                produced.append(name)
        elif etype == "Local Model":
            path = str(entry.get("local_path") or "").strip()
            if path:
                produced.append(Path(path).stem)
        elif etype in ("Checkpoint Merge", "LoRA Bake"):
            name = str(entry.get("output_name") or "").strip()
            if name:
                produced.append(name)
        return produced

    def _planner_analysis(self):
        entries = self.plan_data.get("entries", [])
        produced_by_entry = {idx: self._entry_produced_aliases(entry) for idx, entry in enumerate(entries)}
        consumed_by_entry = {idx: self._entry_consumed_aliases(entry) for idx, entry in enumerate(entries)}
        producer_by_alias = {}
        for idx, aliases in produced_by_entry.items():
            for alias in aliases:
                producer_by_alias[alias] = idx

        final_idx = None
        for idx in range(len(entries) - 1, -1, -1):
            if entries[idx].get("type") != "Remove Model":
                final_idx = idx
                break

        protected_aliases = set()
        if final_idx is not None:
            protected_aliases.update(produced_by_entry.get(final_idx, []))
            protected_aliases.update(consumed_by_entry.get(final_idx, []))

        needed_aliases = set()
        needed_entries = set()
        stack = list(protected_aliases)
        while stack:
            alias = stack.pop()
            if not alias or alias in needed_aliases:
                continue
            needed_aliases.add(alias)
            pidx = producer_by_alias.get(alias)
            if pidx is None:
                continue
            if entries[pidx].get("type") in ("Checkpoint Merge", "LoRA Bake"):
                needed_entries.add(pidx)
            for dep in consumed_by_entry.get(pidx, []):
                if dep and dep not in needed_aliases:
                    stack.append(dep)

        dead_entries = set()
        for idx, entry in enumerate(entries):
            if entry.get("type") not in ("Checkpoint Merge", "LoRA Bake"):
                continue
            produced = produced_by_entry.get(idx, [])
            if idx == final_idx:
                continue
            if produced and not any(alias in needed_aliases for alias in produced):
                dead_entries.add(idx)

        missing_links = []
        for idx, consumes in consumed_by_entry.items():
            for alias in consumes:
                if alias not in producer_by_alias and alias not in self._collect_available_models(idx).get("Checkpoint", []) and alias not in self._collect_available_models(idx).get("LoRA", []) and alias not in self._collect_available_models(idx).get("LyCORIS", []):
                    missing_links.append((idx, alias))

        return {
            "final_index": final_idx,
            "protected_aliases": protected_aliases,
            "needed_aliases": needed_aliases,
            "needed_entries": needed_entries,
            "dead_entries": dead_entries,
            "producer_by_alias": producer_by_alias,
            "produced_by_entry": produced_by_entry,
            "consumed_by_entry": consumed_by_entry,
            "missing_links": missing_links,
        }

    def _planner_dependency_summary(self):
        analysis = self._planner_analysis()
        links = sum(len(v) for v in analysis["consumed_by_entry"].values())
        missing = len(analysis["missing_links"])
        return f"Deps: {links} link(s) / {missing} missing"

    def _planner_unused_summary(self):
        analysis = self._planner_analysis()
        dead_count = len(analysis["dead_entries"])
        dead_lines = sorted(i + 1 for i in analysis["dead_entries"])
        if not dead_lines:
            return "Unused: 0 dead line(s)"
        preview = ", ".join(str(x) for x in dead_lines[:3])
        suffix = " ..." if len(dead_lines) > 3 else ""
        return f"Unused: {dead_count} dead line(s)  [{preview}{suffix}]"

    def _compute_aliases(self):
        aliases = []
        for idx, entry in enumerate(self.plan_data.get("entries", [])):
            produces = []
            consumes = []
            for _slot, alias, _kind, _spec in self._iter_embedded_sources(entry):
                produces.append(alias)
            produces.extend(self._entry_produced_aliases(entry))
            consumes.extend(self._entry_consumed_aliases(entry))
            aliases.append((idx, [x for x in produces if x], [x for x in consumes if x]))
        return aliases

    def _apply_plan_listbox_item_styles(self):
        if self.plan_listbox is None:
            return
        problem_map = getattr(self, "_plan_problem_map_cache", {})
        analysis = self._planner_analysis()
        dead_entries = analysis.get("dead_entries", set())
        for vis_idx, model_idx in enumerate(self.visible_entry_indices):
            entry = self.plan_data.get("entries", [])[model_idx]
            problems = problem_map.get(model_idx, [])
            if problems:
                fg = "#cc2222"
            elif model_idx in dead_entries:
                fg = "#c62828"
            else:
                fg = self._entry_type_color(entry.get("type", ""))
            try:
                self.plan_listbox.itemconfig(vis_idx, fg=fg)
            except Exception:
                try:
                    self.plan_listbox.itemconfigure(vis_idx, foreground=fg)
                except Exception:
                    pass

    def _change_line_type(self, new_type: str):
        self._planner_push_history()
        if new_type not in self._planner_visible_line_types() and new_type not in self._planner_internal_line_types():
            new_type = "Checkpoint Merge"
        return self._base_change_line_type(new_type)

    def _add_line(self):
        self._planner_push_history()
        insert_at = max(self._planner_get_selected_indices() or [self.current_index]) + 1 if self.plan_data.get('entries') else 0
        self.plan_data.setdefault('entries', []).insert(insert_at, make_entry('Checkpoint Merge'))
        self.current_index = insert_at
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices([insert_at])

    def _sanitize_effective_payload(self, value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if str(key).startswith("_"):
                    continue
                out[key] = self._sanitize_effective_payload(item)
            return out
        if isinstance(value, list):
            return [self._sanitize_effective_payload(x) for x in value]
        return copy.deepcopy(value)

    def _make_effective_source_entry(self, spec, alias: str):
        mode = str(spec.get("mode") or "").strip()
        kind = str(spec.get("kind") or "Checkpoint").strip() or "Checkpoint"
        if mode == "download":
            item = make_entry("Download Model")
            item["model_name"] = str(alias or self._source_alias_from_spec(spec)).strip()
            item["model_type"] = kind
            item["link"] = str(spec.get("link") or "").strip()
            return item
        item = make_entry("Local Model")
        item["model_type"] = kind
        item["local_path"] = str(spec.get("local_path") or "").strip()
        return item

    def _planner_effective_plan_data(self):
        effective_entries = []
        for entry in self.plan_data.get("entries", []):
            etype = entry.get("type")
            if etype == "Remove Model":
                continue
            if etype == "Checkpoint Merge":
                for slot in ("model0", "model1", "model2"):
                    spec = self._get_entry_slot_source(entry, slot)
                    alias = str(entry.get(slot) or self._source_alias_from_spec(spec)).strip()
                    if spec and alias:
                        effective_entries.append(self._make_effective_source_entry(spec, alias))
                effective_entries.append(self._sanitize_effective_payload(entry))
            elif etype == "LoRA Bake":
                checkpoint_spec = self._get_entry_slot_source(entry, "checkpoint")
                checkpoint_alias = str(entry.get("checkpoint") or self._source_alias_from_spec(checkpoint_spec)).strip()
                if checkpoint_spec and checkpoint_alias:
                    effective_entries.append(self._make_effective_source_entry(checkpoint_spec, checkpoint_alias))
                for lora in entry.get("loras", []) or []:
                    spec = self._get_lora_source(lora)
                    alias = str(lora.get("name") or self._source_alias_from_spec(spec)).strip()
                    if spec and alias:
                        effective_entries.append(self._make_effective_source_entry(spec, alias))
                effective_entries.append(self._sanitize_effective_payload(entry))
            else:
                effective_entries.append(self._sanitize_effective_payload(entry))
        return self._append_auto_remove_lines({"entries": effective_entries})

    def _append_auto_remove_lines(self, plan_data):
        entries = [copy.deepcopy(x) for x in (plan_data.get("entries", []) or [])]
        if not entries:
            return {"entries": entries}
        final_idx = None
        for idx in range(len(entries) - 1, -1, -1):
            if entries[idx].get("type") != "Remove Model":
                final_idx = idx
                break
        protected = set()
        if final_idx is not None:
            protected.update(self._entry_consumed_aliases(entries[final_idx]))
            protected.update(self._entry_produced_aliases(entries[final_idx]))

        producer_info = {}
        last_use = {}
        for idx, entry in enumerate(entries):
            etype = entry.get("type")
            if etype in ("Download Model", "Checkpoint Merge", "LoRA Bake"):
                for alias in self._entry_produced_aliases(entry):
                    producer_info[alias] = {
                        "index": idx,
                        "type": etype,
                    }
            for alias in self._entry_consumed_aliases(entry):
                last_use[alias] = idx

        inserts = []
        for alias, info in producer_info.items():
            if alias in protected:
                continue
            if info.get("type") == "Local Model":
                continue
            target_idx = max(int(info.get("index", 0)), int(last_use.get(alias, info.get("index", 0))))
            remove_entry = make_entry("Remove Model")
            remove_entry["model"] = alias
            inserts.append((target_idx + 1, remove_entry))

        offset = 0
        for insert_at, item in sorted(inserts, key=lambda x: x[0]):
            entries.insert(max(0, min(len(entries), insert_at + offset)), item)
            offset += 1
        return {"entries": entries}

    def _collapse_internal_plan_entries(self, plan_data):
        entries = [copy.deepcopy(x) for x in (plan_data.get("entries", []) or [])]
        out = []
        i = 0
        while i < len(entries):
            entry = entries[i]
            etype = entry.get("type")
            if etype == "Remove Model":
                i += 1
                continue
            if etype not in ("Download Model", "Local Model"):
                out.append(entry)
                i += 1
                continue
            block = []
            j = i
            while j < len(entries) and entries[j].get("type") in ("Download Model", "Local Model"):
                block.append(entries[j])
                j += 1
            if j >= len(entries) or entries[j].get("type") not in ("Checkpoint Merge", "LoRA Bake"):
                out.extend(block)
                i = j
                continue
            target = copy.deepcopy(entries[j])
            remaining = []
            for src in block:
                src_type = src.get("type")
                if src_type == "Download Model":
                    alias = str(src.get("model_name") or "").strip()
                    spec = {
                        "mode": "download",
                        "kind": str(src.get("model_type") or "Checkpoint").strip() or "Checkpoint",
                        "alias": alias,
                        "link": str(src.get("link") or "").strip(),
                    }
                else:
                    local_path = str(src.get("local_path") or "").strip()
                    alias = Path(local_path).stem if local_path else ""
                    spec = {
                        "mode": "local",
                        "kind": str(src.get("model_type") or "Checkpoint").strip() or "Checkpoint",
                        "alias": alias,
                        "local_path": local_path,
                    }
                attached = False
                if target.get("type") == "Checkpoint Merge":
                    for slot in ("model0", "model1", "model2"):
                        if str(target.get(slot) or "").strip() == alias:
                            self._set_entry_slot_source(target, slot, spec)
                            attached = True
                elif target.get("type") == "LoRA Bake":
                    if str(target.get("checkpoint") or "").strip() == alias:
                        self._set_entry_slot_source(target, "checkpoint", spec)
                        attached = True
                    for lora in target.get("loras", []) or []:
                        if str(lora.get("name") or "").strip() == alias:
                            self._set_lora_source(lora, spec)
                            attached = True
                if not attached:
                    remaining.append(src)
            out.extend(remaining)
            out.append(target)
            i = j + 1
        return self._normalize_loaded_plan_entries({"entries": out})

    def _save_plan_to_file(self):
        path = self._ensure_plan_path()
        export_plan_records_txt(path, self._planner_effective_plan_data())
        self._schedule_config_save()

    def _write_temp_plan_text(self) -> str:
        fd, temp_path = tempfile.mkstemp(prefix="planner_", suffix=".txt")
        os.close(fd)
        export_plan_records_txt(temp_path, self._planner_effective_plan_data())
        return temp_path

    def _restore_session_state(self):
        filepath = self.entries["filepath"].get().strip()
        if filepath and os.path.exists(filepath):
            try:
                raw = normalize_plan(load_plan_records(filepath))
                self.plan_data = self._collapse_internal_plan_entries(raw)
            except Exception:
                self.plan_data = self._planner_default_visible_plan()
        else:
            self.plan_data = self._planner_default_visible_plan()

    def _load_plan_from_path(self):
        path = self._normalize_user_path(self.entries["filepath"].get().strip())
        if not path:
            messagebox.showwarning("Plan Path", "Plan Text Path is empty.")
            return
        try:
            raw = normalize_plan(load_plan_records(path))
            self.plan_data = self._collapse_internal_plan_entries(raw)
            self.current_index = 0
            self._refresh_line_selector()
            self._render_current_line()
            self.status_label.config(text=f"Loaded plan: {self._path_display_name(path)}")
        except Exception as e:
            self._show_detailed_error("Load Error", e)

    def _render_checkpoint_merge_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Checkpoint Merge")
        ckpts = self._collect_available_models(self.current_index).get("Checkpoint", [])
        mode_labels = [f"{m['key']} - {m['label']}" for m in self.merge_modes]
        current_mode = entry.get("merge_mode", self.merge_modes[0]["key"])
        current_label = next((f"{m['key']} - {m['label']}" for m in self.merge_modes if m["key"] == current_mode), mode_labels[0])

        row = Frame(frame)
        row.pack(fill="x", pady=3)
        Label(row, text="Merge Mode", width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=current_label)
        combo = ttk.Combobox(row, textvariable=var, values=mode_labels, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)

        def sync_mode(*_args):
            label = var.get()
            key = label.split(" - ", 1)[0]
            entry["merge_mode"] = key
            self._schedule_rerender_current_line()
        var.trace_add("write", sync_mode)

        self._build_source_backed_ref_row(
            frame,
            "Model 0",
            lambda e=entry: e.get("model0", ""),
            lambda value, e=entry: e.__setitem__("model0", value),
            lambda e=entry: self._get_entry_slot_source(e, "model0"),
            lambda spec, e=entry: self._set_entry_slot_source(e, "model0", spec),
            ckpts or [""],
            ["Checkpoint"],
            allow_local=True,
        )
        self._build_source_backed_ref_row(
            frame,
            "Model 1",
            lambda e=entry: e.get("model1", ""),
            lambda value, e=entry: e.__setitem__("model1", value),
            lambda e=entry: self._get_entry_slot_source(e, "model1"),
            lambda spec, e=entry: self._set_entry_slot_source(e, "model1", spec),
            ckpts or [""],
            ["Checkpoint"],
            allow_local=True,
        )
        mode_info = self.merge_mode_map.get(entry.get("merge_mode"), self.merge_modes[0])
        if mode_info.get("needs_m2"):
            self._build_source_backed_ref_row(
                frame,
                "Model 2",
                lambda e=entry: e.get("model2", ""),
                lambda value, e=entry: e.__setitem__("model2", value),
                lambda e=entry: self._get_entry_slot_source(e, "model2"),
                lambda spec, e=entry: self._set_entry_slot_source(e, "model2", spec),
                ckpts or [""],
                ["Checkpoint"],
                allow_local=True,
            )
        if mode_info.get("key") != "CLIPXOR":
            self._build_ratio_section(parent, entry, "alpha", "Alpha")
        if mode_info.get("needs_beta"):
            self._build_ratio_section(parent, entry, "beta", "Beta")
        out_frame = self._build_labeled_frame(parent, "Output")
        self._build_entry_row(out_frame, "Output Name", entry, "output_name")
        self._build_text_row(parent, "Additional Signatures", entry, "raw_signatures", height=5)

    def _render_lora_bake_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "LoRA Bake")
        models = self._collect_available_models(self.current_index)
        ckpts = models.get("Checkpoint", [])
        self._build_source_backed_ref_row(
            frame,
            "Checkpoint",
            lambda e=entry: e.get("checkpoint", ""),
            lambda value, e=entry: e.__setitem__("checkpoint", value),
            lambda e=entry: self._get_entry_slot_source(e, "checkpoint"),
            lambda spec, e=entry: self._set_entry_slot_source(e, "checkpoint", spec),
            ckpts or [""],
            ["Checkpoint"],
            allow_local=True,
        )
        self._build_entry_row(frame, "Output Name", entry, "output_name")
        add_lora_btn = ttk.Button(frame, text="+ Add LoRA", command=lambda: self._add_lora_block(entry))
        add_lora_btn.pack(anchor="w", padx=4, pady=4)
        self._attach_tooltip(add_lora_btn, self._right_help("+ Add LoRA").get("detail", ""))
        self._add_right_inline_help(frame, "+ Add LoRA")

        for idx, lora in enumerate(entry.get("loras", []) or []):
            block = self._build_labeled_frame(parent, f"LoRA {idx + 1}")
            top = Frame(block)
            top.pack(fill="x")
            ttk.Button(top, text="-", width=3, command=lambda i=idx: self._remove_lora_block(entry, i)).pack(side="left", padx=(0, 4))
            lora_names = models.get("LoRA", []) + models.get("LyCORIS", [])
            self._build_source_backed_ref_row(
                block,
                "LoRA Name",
                lambda lo=lora: lo.get("name", ""),
                lambda value, lo=lora: lo.__setitem__("name", value),
                lambda lo=lora: self._get_lora_source(lo),
                lambda spec, lo=lora: self._set_lora_source(lo, spec),
                lora_names or [""],
                ["LoRA", "LyCORIS"],
                allow_local=True,
            )
            ratio = lora.setdefault("ratio", default_ratio("Single"))
            ratio_wrap = self._build_collapsible_section(
                block,
                "Ratio",
                key=f"lora_ratio_{entry.get('id', '')}_{idx}",
                default_open=True,
                body_fill="x",
                body_expand=False,
                padx=0,
                pady=4,
            )
            row2 = Frame(ratio_wrap)
            row2.pack(fill="x", pady=3)
            Label(row2, text="Ratio Mode", width=18, anchor="w").pack(side="left")
            ratio_var = tk.StringVar(value=ratio.get("mode", "Single"))
            combo2 = ttk.Combobox(row2, textvariable=ratio_var, values=["Single", "Elemental"], state="readonly")
            self._bind_combobox_mousewheel_passthrough(combo2)
            combo2.pack(side="left", fill="x", expand=True, padx=4)

            def on_ratio_mode(*_args, lo=lora, rv=ratio_var):
                lo["ratio"]["mode"] = rv.get()
                if rv.get() == "Single":
                    lo["ratio"].setdefault("value", "1.0")
                self._schedule_rerender_current_line()
            ratio_var.trace_add("write", on_ratio_mode)
            self._build_ratio_value_widget(ratio_wrap, lora["ratio"], allow_block_weight=False)

        self._build_text_row(parent, "Additional Signatures", entry, "raw_signatures", height=5)

    def _ratio_single_float(self, ratio, default_value: float = 0.5):
        if not isinstance(ratio, dict):
            return float(default_value)
        if str(ratio.get("mode") or "Single") != "Single":
            raise ValueError("Only Single ratios can be collapsed")
        try:
            return float(str(ratio.get("value") or default_value).strip())
        except Exception as exc:
            raise ValueError(f"Invalid Single ratio: {ratio!r}") from exc

    def _selected_entries_support_ws_collapse(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return False
        entries = self.plan_data.get("entries", [])
        selected = [entries[idx] for idx in indices if 0 <= idx < len(entries)]
        return bool(selected) and all(entry.get("type") == "Checkpoint Merge" for entry in selected)

    def _gather_selected_source_specs_by_alias(self, indices):
        mapping = {}
        for idx in indices:
            try:
                entry = self.plan_data["entries"][idx]
            except Exception:
                continue
            for _slot, alias, _kind, spec in self._iter_embedded_sources(entry):
                if alias and isinstance(spec, dict) and alias not in mapping:
                    mapping[alias] = copy.deepcopy(spec)
        return mapping

    def _collapse_selected_ws_chain(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return
        entries = self.plan_data.get("entries", [])
        selected = [idx for idx in indices if 0 <= idx < len(entries)]
        if not selected:
            return
        producer_by_alias = {}
        for idx in selected:
            entry = entries[idx]
            if entry.get("type") != "Checkpoint Merge":
                messagebox.showwarning("Simplify", "Only Checkpoint Merge lines can be collapsed.")
                return
            if str(entry.get("merge_mode") or "WS") != "WS":
                messagebox.showwarning("Simplify", "Collapse currently supports WS chains only.")
                return
            output_name = str(entry.get("output_name") or "").strip()
            if not output_name:
                messagebox.showwarning("Simplify", "Every selected merge must have an output name.")
                return
            producer_by_alias[output_name] = idx

        sink_candidates = []
        for idx in selected:
            output_name = str(entries[idx].get("output_name") or "").strip()
            used_inside = False
            for other in selected:
                if other == idx:
                    continue
                if output_name in self._entry_consumed_aliases(entries[other]):
                    used_inside = True
                    break
            if not used_inside:
                sink_candidates.append(idx)
        if len(sink_candidates) != 1:
            messagebox.showwarning("Simplify", "Select a single collapsible WS chain or tree with one sink output.")
            return
        sink_idx = sink_candidates[0]
        sink_output = str(entries[sink_idx].get("output_name") or "").strip()

        def expand_alias(alias, trail=None):
            trail = list(trail or [])
            if alias in trail:
                raise ValueError("Cycle detected while collapsing WS chain")
            if alias not in producer_by_alias:
                return {alias: 1.0}
            idx = producer_by_alias[alias]
            entry = entries[idx]
            alpha = self._ratio_single_float(entry.get("alpha"), 0.5)
            left = expand_alias(str(entry.get("model0") or "").strip(), trail + [alias])
            right = expand_alias(str(entry.get("model1") or "").strip(), trail + [alias])
            out = {}
            for key, value in left.items():
                out[key] = out.get(key, 0.0) + (1.0 - alpha) * float(value)
            for key, value in right.items():
                out[key] = out.get(key, 0.0) + alpha * float(value)
            return out

        try:
            weights = expand_alias(sink_output)
        except Exception as exc:
            messagebox.showwarning("Simplify", f"Could not collapse selected chain.\n\n{exc}")
            return

        leaves = [(alias, weight) for alias, weight in weights.items() if alias and abs(weight) > 1e-12]
        if len(leaves) not in (2, 3):
            messagebox.showwarning("Simplify", "Collapse currently supports chains that reduce to 2 or 3 source models.")
            return

        total = sum(weight for _alias, weight in leaves)
        if abs(total) <= 1e-12:
            messagebox.showwarning("Simplify", "Collapsed weights sum to zero.")
            return
        leaves = [(alias, weight / total) for alias, weight in leaves]

        new_entry = make_entry("Checkpoint Merge")
        new_entry["output_name"] = sink_output
        new_entry["raw_signatures"] = str(entries[sink_idx].get("raw_signatures") or "")
        merged_memo = []
        for idx in selected:
            memo = str(entries[idx].get("memo") or "").strip()
            if memo:
                merged_memo.append(f"[{idx + 1}] {memo}")
        summary = ", ".join(f"{alias}:{weight:.6g}" for alias, weight in leaves)
        memo_text = f"Collapsed WS chain from lines {', '.join(str(i + 1) for i in selected)} -> {summary}"
        if merged_memo:
            memo_text += "\n" + "\n".join(merged_memo)
        new_entry["memo"] = memo_text

        source_specs = self._gather_selected_source_specs_by_alias(selected)
        if len(leaves) == 2:
            (a0, w0), (a1, w1) = leaves
            new_entry["merge_mode"] = "WS"
            new_entry["model0"] = a0
            new_entry["model1"] = a1
            new_entry["alpha"] = default_ratio("Single")
            new_entry["alpha"]["value"] = self._format_ratio_float(w1)
            for slot, alias in (("model0", a0), ("model1", a1)):
                spec = source_specs.get(alias)
                if spec:
                    self._set_entry_slot_source(new_entry, slot, spec)
        else:
            (a0, w0), (a1, w1), (a2, w2) = leaves
            new_entry["merge_mode"] = "TRS"
            new_entry["model0"] = a0
            new_entry["model1"] = a1
            new_entry["model2"] = a2
            new_entry["alpha"] = default_ratio("Single")
            new_entry["alpha"]["value"] = self._format_ratio_float(w1)
            new_entry["beta"] = default_ratio("Single")
            new_entry["beta"]["value"] = self._format_ratio_float(w2)
            for slot, alias in (("model0", a0), ("model1", a1), ("model2", a2)):
                spec = source_specs.get(alias)
                if spec:
                    self._set_entry_slot_source(new_entry, slot, spec)

        self._planner_push_history()
        block = [entries[idx] for idx in selected]
        sink_original = sink_idx
        for idx in reversed(selected):
            entries.pop(idx)
        insert_at = sink_original - sum(1 for idx in selected if idx < sink_original)
        entries.insert(insert_at, new_entry)
        self.current_index = insert_at
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices([insert_at])
        self.status_label.config(text=f"Collapsed {len(block)} WS merge line(s)")

    def _remove_dead_lines(self):
        dead = sorted(self._planner_analysis().get("dead_entries", set()))
        if not dead:
            self.status_label.config(text="No dead merge lines to remove")
            return
        self._planner_push_history()
        entries = self.plan_data.get("entries", [])
        for idx in reversed(dead):
            if 0 <= idx < len(entries):
                entries.pop(idx)
        self.current_index = max(0, min(self.current_index, max(0, len(entries) - 1)))
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self.status_label.config(text=f"Removed {len(dead)} dead line(s)")

    def _show_dependency_view(self):
        analysis = self._planner_analysis()
        lines = ["Dependency View", ""]
        producer_by_alias = analysis["producer_by_alias"]
        for idx, entry in enumerate(self.plan_data.get("entries", [])):
            lines.append(f"[{idx + 1}] {self._line_summary(entry)}")
            produced = self._entry_produced_aliases(entry)
            embedded = [alias for _slot, alias, _kind, _spec in self._iter_embedded_sources(entry)]
            consumes = self._entry_consumed_aliases(entry)
            if embedded:
                lines.append("  embedded sources: " + ", ".join(embedded))
            if produced:
                lines.append("  produces: " + ", ".join(produced))
            if consumes:
                detail = []
                for alias in consumes:
                    src = producer_by_alias.get(alias)
                    detail.append(f"{alias} <- line {src + 1}" if src is not None else f"{alias} <- external")
                lines.append("  consumes: " + "; ".join(detail))
            if idx in analysis.get("dead_entries", set()):
                lines.append("  status: DEAD / not required by final line")
            lines.append("")
        self._show_scrollable_text_dialog("Dependency View", "\n".join(lines))

    def _show_unused_models(self):
        analysis = self._planner_analysis()
        dead = sorted(analysis.get("dead_entries", set()))
        detail_lines = ["Dead / unused merge lines:", ""]
        if not dead:
            detail_lines.append("None")
        else:
            for idx in dead:
                detail_lines.append(f"line {idx + 1}: {self._line_summary(self.plan_data['entries'][idx])}")
        self._show_scrollable_text_dialog("Unused Models", "\n".join(detail_lines))

    def _show_plan_context_menu(self, event=None):
        if self.plan_listbox is None or event is None:
            return 'break'
        model_idx = self._plan_list_prepare_context_selection(event)
        if model_idx is None:
            self._hide_plan_context_menu()
            return 'break'

        self._hide_plan_context_menu()
        self._planner_restore_plan_list_focus()
        colors = self._theme_colors()
        popup = tk.Toplevel(self.root)
        popup.wm_overrideredirect(True)
        popup.attributes('-topmost', True)
        try:
            popup.configure(bg=colors['panel'])
        except Exception:
            pass

        x = int(getattr(event, 'x_root', 0) or 0) + 10
        y = int(getattr(event, 'y_root', 0) or 0) + 8
        popup.wm_geometry(f"+{x}+{y}")

        card = Frame(
            popup,
            bg=colors['surface'],
            highlightthickness=1,
            highlightbackground=colors['border'],
            padx=2,
            pady=2,
        )
        card.pack(fill='both', expand=True)

        header = Label(
            card,
            text='Plan View Actions',
            anchor='w',
            justify='left',
            bg=colors['surface'],
            fg=colors['text'],
            padx=12,
            pady=10,
            font=('MS Gothic', 11, 'bold'),
        )
        header.pack(fill='x')

        button_area = Frame(card, bg=colors['surface'])
        button_area.pack(fill='both', expand=True, padx=6, pady=(0, 6))

        def run_and_close(func):
            def wrapped():
                self._hide_plan_context_menu()
                try:
                    func()
                finally:
                    self._planner_restore_plan_list_focus()
                return 'break'
            return wrapped

        actions = [
            ('Copy', self._copy_selected_lines, True),
            ('Paste Below', self._paste_copied_lines, True),
            ('Duplicate Below', self._duplicate_selected_lines, True),
            ('Optimize Plan', self._planner_optimize_plan_full, True),
            ('Remove Dead Lines', self._remove_dead_lines, bool(self._planner_analysis().get('dead_entries'))),
            ('Create Preset', self._save_preset_json, True),
            ('Delete', self._delete_selected_lines, True),
        ]

        for label, func, enabled in actions:
            row = Frame(button_area, bg=colors['surface'])
            row.pack(fill='x', pady=2)
            style_name = 'MenuDanger.TButton' if label in {'Delete', 'Remove Dead Lines'} else 'Menu.TButton'
            btn = ttk.Button(
                row,
                text=label,
                command=run_and_close(func) if enabled else self._hide_plan_context_menu,
                style=style_name,
                cursor='hand2' if enabled else 'arrow',
                takefocus=False,
            )
            if not enabled:
                btn.state(['disabled'])
            btn.pack(fill='x', expand=True)
            if enabled:
                btn.bind('<ButtonRelease-1>', lambda _e, f=func: run_and_close(f)(), add='+')
                btn.bind('<Return>', lambda _e, f=func: run_and_close(f)(), add='+')
                btn.bind('<space>', lambda _e, f=func: run_and_close(f)(), add='+')

        self._plan_context_popup = popup
        self._plan_context_menu = popup

        def _outside_click(ev=None):
            try:
                widget = getattr(ev, 'widget', None)
                probe = widget
                while probe is not None:
                    if probe is popup:
                        return None
                    probe = getattr(probe, 'master', None)
            except Exception:
                pass
            self._hide_plan_context_menu()
            return None

        bind_ids = {}
        try:
            bind_ids['<Button-1>'] = self.root.bind('<Button-1>', _outside_click, add='+')
            bind_ids['<Button-2>'] = self.root.bind('<Button-2>', _outside_click, add='+')
            bind_ids['<Button-3>'] = self.root.bind('<Button-3>', _outside_click, add='+')
        except Exception:
            bind_ids = {}
        self._plan_context_root_bind_ids = bind_ids
        try:
            popup.update_idletasks()
        except Exception:
            pass
        return 'break'



    def _planner_init_extra_state(self):
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

    def _planner_apply_entry_defaults(self):
        for entry in self.plan_data.get('entries', []) or []:
            if not isinstance(entry, dict):
                continue
            entry.setdefault('memo', '')
            if '_locked' not in entry:
                entry['_locked'] = False
            if '_disabled' not in entry:
                entry['_disabled'] = False
            if '_row_color' not in entry:
                entry['_row_color'] = ''

    def _planner_meta_sidecar_path(self, plan_path: str | None = None) -> Path | None:
        raw = plan_path
        if not raw:
            try:
                raw = self._normalize_user_path(self.entries['filepath'].get().strip())
            except Exception:
                raw = ''
        raw = str(raw or '').strip()
        if not raw:
            return None
        return Path(raw + '.planner_meta.json')

    def _planner_meta_payload(self):
        entries = []
        for entry in self.plan_data.get('entries', []) or []:
            if not isinstance(entry, dict):
                continue
            entries.append({
                'memo': str(entry.get('memo') or ''),
                '_locked': bool(entry.get('_locked')),
                '_disabled': bool(entry.get('_disabled')),
                '_row_color': str(entry.get('_row_color') or ''),
            })
        return {'version': 1, 'entries': entries}

    def _planner_save_meta_to_disk(self, plan_path: str | None = None):
        sidecar = self._planner_meta_sidecar_path(plan_path)
        if sidecar is None:
            return
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(self._planner_meta_payload(), ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

    def _planner_apply_loaded_meta(self, payload):
        self._planner_apply_entry_defaults()
        entries = self.plan_data.get('entries', []) or []
        meta_entries = payload.get('entries') if isinstance(payload, dict) else None
        if not isinstance(meta_entries, list):
            return
        for idx, entry in enumerate(entries):
            if idx >= len(meta_entries):
                break
            meta = meta_entries[idx]
            if not isinstance(meta, dict):
                continue
            if 'memo' in meta:
                entry['memo'] = str(meta.get('memo') or '')
            entry['_locked'] = bool(meta.get('_locked'))
            entry['_disabled'] = bool(meta.get('_disabled'))
            entry['_row_color'] = str(meta.get('_row_color') or '')

    def _planner_load_meta_from_disk(self, plan_path: str | None = None):
        self._planner_apply_entry_defaults()
        sidecar = self._planner_meta_sidecar_path(plan_path)
        if sidecar is None or not sidecar.exists():
            return
        try:
            payload = json.loads(sidecar.read_text(encoding='utf-8'))
        except Exception:
            return
        self._planner_apply_loaded_meta(payload)

    def _planner_backup_root(self, plan_path: str | None = None) -> Path:
        raw = plan_path
        if not raw:
            try:
                raw = self._normalize_user_path(self.entries['filepath'].get().strip())
            except Exception:
                raw = ''
        raw = str(raw or '').strip()
        if raw:
            base = Path(raw)
            name = base.stem or 'plan'
            return base.parent / '.planner_backups' / name
        return Path(os.getcwd()) / '.planner_backups' / 'plan'

    def _planner_maybe_create_backup(self, plan_path: str | None = None, *, reason: str = 'autosave', force: bool = False):
        try:
            payload = {
                'version': 1,
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'reason': str(reason or 'autosave'),
                'plan_data': copy.deepcopy(self.plan_data),
            }
            snapshot = json.dumps(payload['plan_data'], ensure_ascii=False, sort_keys=True)
            now = time.time()
            if not force:
                if snapshot == getattr(self, '_last_backup_snapshot', None):
                    return
                if now - float(getattr(self, '_last_backup_time', 0.0) or 0.0) < 15.0:
                    return
            root = self._planner_backup_root(plan_path)
            root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            safe_reason = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(reason or 'autosave')).strip('_') or 'autosave'
            out = root / f'{stamp}_{safe_reason}.planbundle.json'
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            bundles = sorted(root.glob('*.planbundle.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            keep = max(3, int(getattr(self, 'backup_keep_generations', 20) or 20))
            for stale in bundles[keep:]:
                try:
                    stale.unlink()
                except Exception:
                    pass
            self._last_backup_snapshot = snapshot
            self._last_backup_time = now
        except Exception:
            pass

    def _planner_restore_visible_plan(self, raw_plan):
        entries = []
        raw_entries = []
        if isinstance(raw_plan, dict):
            raw_entries = raw_plan.get('entries') or []
        elif isinstance(raw_plan, list):
            raw_entries = raw_plan
        if not isinstance(raw_entries, list):
            raw_entries = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            entry = self._normalize_entry_preserving_embedded_sources(copy.deepcopy(raw_entry))
            entry['memo'] = str(raw_entry.get('memo') or '')
            entry['_locked'] = bool(raw_entry.get('_locked'))
            entry['_disabled'] = bool(raw_entry.get('_disabled'))
            entry['_row_color'] = str(raw_entry.get('_row_color') or '')
            entries.append(entry)
        self.plan_data = {'entries': entries or [make_entry('Checkpoint Merge')]}
        self._planner_apply_entry_defaults()

    def _show_backup_manager(self):
        root_dir = self._planner_backup_root(None)
        win = tk.Toplevel(self.root)
        win.title('Backup Manager')
        win.geometry('980x620+110+110')
        outer = Frame(win, padx=8, pady=8)
        outer.pack(fill='both', expand=True)
        info = Label(outer, text=f'Backup folder: {root_dir}', anchor='w', justify='left')
        info.pack(fill='x', pady=(0, 6))
        main = Frame(outer)
        main.pack(fill='both', expand=True)
        listbox = tk.Listbox(main, exportselection=False, activestyle='none')
        scroll = Scrollbar(main, orient='vertical', command=listbox.yview)
        listbox.configure(yscrollcommand=scroll.set)
        listbox.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        detail = Text(outer, height=14, wrap='word', font=('Consolas', 10))
        detail.pack(fill='both', expand=False, pady=(8, 0))
        bundles = []

        def refresh():
            nonlocal bundles
            listbox.delete(0, 'end')
            detail.delete('1.0', 'end')
            if root_dir.exists():
                bundles = sorted(root_dir.glob('*.planbundle.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            else:
                bundles = []
            for path in bundles:
                try:
                    payload = json.loads(path.read_text(encoding='utf-8'))
                    lines = len((payload.get('plan_data') or {}).get('entries') or []) if isinstance(payload.get('plan_data'), dict) else len(payload.get('plan_data') or [])
                    saved_at = str(payload.get('saved_at') or '')
                    reason = str(payload.get('reason') or '')
                    label = f'{path.name}  |  {saved_at}  |  {reason}  |  {lines} line(s)'
                except Exception:
                    label = path.name
                listbox.insert('end', label)
            if bundles:
                listbox.selection_set(0)
                show_detail()

        def current_path():
            sel = listbox.curselection()
            if not sel:
                return None
            idx = sel[-1]
            if 0 <= idx < len(bundles):
                return bundles[idx]
            return None

        def show_detail(_event=None):
            path = current_path()
            detail.delete('1.0', 'end')
            if path is None:
                return
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                text_payload = json.dumps(payload, ensure_ascii=False, indent=2)
            except Exception as exc:
                text_payload = f'Failed to read backup.\n\n{type(exc).__name__}: {exc}'
            detail.insert('1.0', text_payload)

        def restore_selected():
            path = current_path()
            if path is None:
                return
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                raw_plan = payload.get('plan_data') if isinstance(payload, dict) else payload
                if raw_plan is None:
                    raise ValueError('Backup bundle does not contain plan_data')
                self._planner_push_history()
                self._planner_restore_visible_plan(raw_plan)
                self.current_index = max(0, min(self.current_index, len(self.plan_data.get('entries', [])) - 1))
                self._save_plan_to_file()
                self._refresh_line_selector()
                self._render_current_line()
                self.status_label.config(text=f'Restored backup: {path.name}')
            except Exception as exc:
                self._show_detailed_error('Backup Restore Error', exc)

        def delete_selected():
            path = current_path()
            if path is None:
                return
            if not messagebox.askyesno('Delete Backup', f'Delete backup\n\n{path.name}\n\nfrom disk?'):
                return
            try:
                path.unlink()
            except Exception as exc:
                self._show_detailed_error('Backup Delete Error', exc)
                return
            refresh()

        button_bar = Frame(outer)
        button_bar.pack(fill='x', pady=(8, 0))
        ttk.Button(button_bar, text='Refresh', command=refresh).pack(side='left')
        ttk.Button(button_bar, text='Force Backup Now', command=lambda: (self._planner_maybe_create_backup(None, reason='manual', force=True), refresh())).pack(side='left', padx=(6, 0))
        ttk.Button(button_bar, text='Restore Selected', command=restore_selected).pack(side='left', padx=(6, 0))
        ttk.Button(button_bar, text='Delete Selected', command=delete_selected).pack(side='left', padx=(6, 0))
        ttk.Button(button_bar, text='Close', command=win.destroy).pack(side='right')
        listbox.bind('<<ListboxSelect>>', show_detail, add='+')
        listbox.bind('<Double-Button-1>', lambda _e: restore_selected(), add='+')
        refresh()
        try:
            colors = self._theme_colors()
            win.configure(bg=colors['bg'])
            self._apply_theme_to_children(win, colors)
        except Exception:
            pass

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

    def _entry_is_locked(self, entry: Dict[str, Any]) -> bool:
        return bool(entry.get('_locked')) if isinstance(entry, dict) else False

    def _entry_is_disabled(self, entry: Dict[str, Any]) -> bool:
        return bool(entry.get('_disabled')) if isinstance(entry, dict) else False

    def _entry_row_color(self, entry: Dict[str, Any]) -> str:
        return str(entry.get('_row_color') or '') if isinstance(entry, dict) else ''

    def _entry_consumed_aliases(self, entry: Dict[str, Any]) -> List[str]:
        if self._entry_is_disabled(entry):
            return []
        etype = entry.get('type')
        out = []
        if etype == 'Checkpoint Merge':
            out.extend([entry.get('model0'), entry.get('model1'), entry.get('model2')])
        elif etype == 'LoRA Bake':
            out.append(entry.get('checkpoint'))
            for lora in entry.get('loras', []) or []:
                out.append(lora.get('name'))
        elif etype == 'Remove Model':
            out.append(entry.get('model'))
        return [str(x).strip() for x in out if str(x or '').strip()]

    def _entry_produced_aliases(self, entry: Dict[str, Any]) -> List[str]:
        if self._entry_is_disabled(entry):
            return []
        etype = entry.get('type')
        produced = []
        if etype == 'Download Model':
            name = str(entry.get('model_name') or '').strip()
            if name:
                produced.append(name)
        elif etype == 'Local Model':
            path = str(entry.get('local_path') or '').strip()
            if path:
                produced.append(Path(path).stem)
        elif etype in ('Checkpoint Merge', 'LoRA Bake'):
            name = str(entry.get('output_name') or '').strip()
            if name:
                produced.append(name)
        return produced

    def _planner_locked_indices(self, indices=None):
        entries = self.plan_data.get('entries', [])
        indices = list(indices if indices is not None else self._planner_get_selected_indices())
        return [idx for idx in indices if 0 <= idx < len(entries) and self._entry_is_locked(entries[idx])]

    def _planner_guard_unlocked(self, indices, action_label: str) -> bool:
        locked = self._planner_locked_indices(indices)
        if not locked:
            return True
        lines = ', '.join(str(idx + 1) for idx in locked)
        messagebox.showwarning('Locked Lines', f'Cannot {action_label} because line(s) {lines} are locked.')
        self._select_model_indices(locked)
        return False

    def _planner_problem_severity(self, problems: List[str]) -> str:
        problems = [str(p) for p in (problems or []) if str(p or '').strip()]
        if not problems:
            return 'ok'
        error_lines = [p for p in problems if p.startswith('ERROR:')]
        warning_lines = [p for p in problems if p.startswith('WARN:')]
        if error_lines:
            lowered = [p.lower() for p in error_lines]
            draft_only = all(' is empty' in p for p in lowered)
            if draft_only:
                return 'draft'
            return 'error'
        if warning_lines:
            return 'warning'
        return 'ok'

    def _planner_analysis(self):
        entries = self.plan_data.get('entries', [])
        produced_by_entry = {idx: self._entry_produced_aliases(entry) for idx, entry in enumerate(entries)}
        consumed_by_entry = {idx: self._entry_consumed_aliases(entry) for idx, entry in enumerate(entries)}
        producer_lines = {}
        producer_by_alias = {}
        for idx, aliases in produced_by_entry.items():
            for alias in aliases:
                producer_lines.setdefault(alias, []).append(idx)
                producer_by_alias[alias] = idx

        final_idx = None
        for idx in range(len(entries) - 1, -1, -1):
            if self._entry_is_disabled(entries[idx]):
                continue
            if entries[idx].get('type') != 'Remove Model':
                final_idx = idx
                break

        protected_aliases = set()
        if final_idx is not None:
            protected_aliases.update(produced_by_entry.get(final_idx, []))
            protected_aliases.update(consumed_by_entry.get(final_idx, []))

        needed_aliases = set()
        needed_entries = set()
        stack = list(protected_aliases)
        while stack:
            alias = stack.pop()
            if not alias or alias in needed_aliases:
                continue
            needed_aliases.add(alias)
            pidx = producer_by_alias.get(alias)
            if pidx is None:
                continue
            if entries[pidx].get('type') in ('Checkpoint Merge', 'LoRA Bake'):
                needed_entries.add(pidx)
            for dep in consumed_by_entry.get(pidx, []):
                if dep and dep not in needed_aliases:
                    stack.append(dep)

        dead_entries = set()
        consumed_global = set()
        for vals in consumed_by_entry.values():
            consumed_global.update(vals)
        unreferenced_aliases = {}
        for alias, lines in producer_lines.items():
            if alias not in consumed_global and alias not in protected_aliases:
                unreferenced_aliases[alias] = list(lines)
        for idx, entry in enumerate(entries):
            if self._entry_is_disabled(entry):
                continue
            if entry.get('type') not in ('Checkpoint Merge', 'LoRA Bake'):
                continue
            produced = produced_by_entry.get(idx, [])
            if idx == final_idx:
                continue
            if produced and not any(alias in needed_aliases for alias in produced):
                dead_entries.add(idx)

        missing_links = []
        for idx, consumes in consumed_by_entry.items():
            available = self._collect_available_models(idx)
            for alias in consumes:
                if alias not in producer_by_alias and alias not in available.get('Checkpoint', []) and alias not in available.get('LoRA', []) and alias not in available.get('LyCORIS', []):
                    missing_links.append((idx, alias))

        duplicate_aliases = {alias: list(lines) for alias, lines in producer_lines.items() if len(lines) > 1}
        return {
            'final_index': final_idx,
            'protected_aliases': protected_aliases,
            'needed_aliases': needed_aliases,
            'needed_entries': needed_entries,
            'dead_entries': dead_entries,
            'producer_by_alias': producer_by_alias,
            'producer_lines': producer_lines,
            'produced_by_entry': produced_by_entry,
            'consumed_by_entry': consumed_by_entry,
            'missing_links': missing_links,
            'duplicate_aliases': duplicate_aliases,
            'unreferenced_aliases': unreferenced_aliases,
        }

    def _planner_dependency_summary(self):
        analysis = self._planner_analysis()
        links = sum(len(v) for v in analysis['consumed_by_entry'].values())
        missing = len(analysis['missing_links'])
        dup = len(analysis['duplicate_aliases'])
        return f'Deps: {links} link(s) / {missing} missing / {dup} duplicate alias(es)'

    def _planner_unused_summary(self):
        analysis = self._planner_analysis()
        dead_count = len(analysis['dead_entries'])
        extra = len(analysis['unreferenced_aliases'])
        if dead_count <= 0 and extra <= 0:
            return 'Unused: 0 dead line(s)'
        return f'Unused: {dead_count} dead line(s) / {extra} unreferenced alias(es)'

    def _plan_entry_problem_map(self) -> Dict[int, List[str]]:
        problems_by_idx: Dict[int, List[str]] = {}
        available = {'Checkpoint': set(), 'LoRA': set(), 'LyCORIS': set()}
        analysis = self._planner_analysis()
        duplicate_aliases = analysis.get('duplicate_aliases', {})
        unref_aliases = analysis.get('unreferenced_aliases', {})
        dead_entries = set(analysis.get('dead_entries', set()))
        for idx, entry in enumerate(self.plan_data.get('entries', []), start=1):
            etype = entry.get('type')
            prefix = f'line {idx} ({etype})'
            problems: List[str] = []
            if self._entry_is_disabled(entry):
                problems.append('WARN: entry is disabled and excluded from export/runtime')
            if self._entry_is_locked(entry):
                problems.append('WARN: entry is locked against accidental edits')
            if etype == 'Download Model':
                if not entry.get('model_name'):
                    problems.append(f'ERROR: {prefix}: model_name is empty')
                if not entry.get('link'):
                    problems.append(f'ERROR: {prefix}: link is empty')
                name = (entry.get('model_name') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip()
                if name and not self._entry_is_disabled(entry):
                    available.setdefault(kind, set()).add(name)
            elif etype == 'Local Model':
                if not entry.get('local_path'):
                    problems.append(f'ERROR: {prefix}: local_path is empty')
                path = (entry.get('local_path') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip()
                if path and not self._entry_is_disabled(entry):
                    available.setdefault(kind, set()).add(Path(path).stem)
            elif etype == 'Remove Model':
                if not entry.get('model'):
                    problems.append(f'ERROR: {prefix}: model is empty')
            elif etype == 'Checkpoint Merge':
                for req_key in ('model0', 'model1', 'output_name'):
                    if not entry.get(req_key):
                        problems.append(f'ERROR: {prefix}: {req_key} is empty')
                for ref in (entry.get('model0'), entry.get('model1'), entry.get('model2')):
                    if ref and ref not in available['Checkpoint']:
                        problems.append(f'ERROR: {prefix}: checkpoint ref not available -> {ref}')
                if entry.get('output_name') and not self._entry_is_disabled(entry):
                    available['Checkpoint'].add(entry['output_name'])
            elif etype == 'LoRA Bake':
                if not entry.get('checkpoint'):
                    problems.append(f'ERROR: {prefix}: checkpoint is empty')
                elif entry.get('checkpoint') not in available['Checkpoint']:
                    problems.append(f'ERROR: {prefix}: checkpoint ref not available -> {entry.get("checkpoint")}')
                if not entry.get('output_name'):
                    problems.append(f'ERROR: {prefix}: output_name is empty')
                for lora in entry.get('loras', []):
                    name = lora.get('name')
                    if not name:
                        problems.append(f'ERROR: {prefix}: one LoRA name is empty')
                    elif name not in available['LoRA'] and name not in available['LyCORIS']:
                        problems.append(f'ERROR: {prefix}: LoRA ref not available -> {name}')
                if entry.get('output_name') and not self._entry_is_disabled(entry):
                    available['Checkpoint'].add(entry['output_name'])

            for alias in self._entry_produced_aliases(entry):
                lines = duplicate_aliases.get(alias, [])
                if len(lines) > 1:
                    labels = ', '.join(str(i + 1) for i in lines)
                    problems.append(f'WARN: duplicate produced alias -> {alias} (lines {labels})')
                if alias in unref_aliases:
                    problems.append(f'WARN: produced alias is currently unreferenced -> {alias}')
            if idx - 1 in dead_entries:
                problems.append('WARN: merge output is not required by the final active line')
            problems_by_idx[idx - 1] = problems
        return problems_by_idx

    def _build_plan_item_tooltip_text(self, model_idx: int, entry: Dict[str, Any], problems: List[str] | None = None) -> str:
        problems = problems or []
        lines = [f'Line {model_idx + 1}', self._line_summary(entry)]
        status_parts = []
        if self._entry_is_locked(entry):
            status_parts.append('locked')
        if self._entry_is_disabled(entry):
            status_parts.append('disabled')
        row_color = self._entry_row_color(entry)
        if row_color:
            status_parts.append(f'color={row_color}')
        if status_parts:
            lines += ['', 'Status: ' + ', '.join(status_parts)]
        if problems:
            lines += ['', 'Issues:'] + [f'• {problem}' for problem in problems]
        memo = str(entry.get('memo') or '').strip()
        if memo:
            lines += ['', 'Memo:', memo]
        return '\n'.join(lines)

    def _line_summary(self, entry: Dict[str, Any]) -> str:
        etype = entry.get('type', 'Line')
        if etype == 'Download Model':
            base = f"Download Model - {entry.get('model_name') or '(unset)'}"
        elif etype == 'Local Model':
            base = f"Local Model - {Path(entry.get('local_path') or '').name or '(unset)'}"
        elif etype == 'Remove Model':
            base = f"Remove Model - {entry.get('model') or '(unset)'}"
        elif etype == 'Checkpoint Merge':
            base = f"Checkpoint Merge - {entry.get('output_name') or '(unset)'}"
        elif etype == 'LoRA Bake':
            base = f"LoRA Bake - {entry.get('output_name') or '(unset)'}"
        else:
            base = etype
        flags = []
        if self._entry_is_locked(entry):
            flags.append('🔒')
        if self._entry_is_disabled(entry):
            flags.append('⏸')
        if self._entry_row_color(entry):
            flags.append('■')
        if flags:
            base = ' '.join(flags) + ' ' + base
        memo_preview = self._memo_preview_text(entry.get('memo', ''))
        if memo_preview:
            base += f'  ✎ {memo_preview}'
        return base

    def _apply_plan_listbox_item_styles(self):
        if self.plan_listbox is None:
            return

        def _hex_to_rgb(value: str):
            raw = str(value or '').strip().lstrip('#')
            if len(raw) == 3:
                raw = ''.join(ch * 2 for ch in raw)
            if len(raw) != 6:
                return None
            try:
                return tuple(int(raw[i:i + 2], 16) for i in (0, 2, 4))
            except Exception:
                return None

        def _mix(color_a: str, color_b: str, ratio: float) -> str:
            rgb_a = _hex_to_rgb(color_a)
            rgb_b = _hex_to_rgb(color_b)
            if rgb_a is None or rgb_b is None:
                return color_a or color_b
            ratio = max(0.0, min(1.0, float(ratio)))
            mixed = tuple(int(round((a * (1.0 - ratio)) + (b * ratio))) for a, b in zip(rgb_a, rgb_b))
            return '#%02x%02x%02x' % mixed

        def _contrast_text(bg_color: str, prefer_dark: str, prefer_light: str) -> str:
            rgb = _hex_to_rgb(bg_color)
            if rgb is None:
                return prefer_light
            luminance = (0.2126 * rgb[0]) + (0.7152 * rgb[1]) + (0.0722 * rgb[2])
            return prefer_dark if luminance >= 150 else prefer_light

        problem_map = getattr(self, '_plan_problem_map_cache', {})
        colors = self._theme_colors()
        palette = self._planner_row_color_presets()
        is_light = str(getattr(self, 'theme_mode', 'dark') or 'dark').lower() == 'light'
        dark_text = '#17203a'
        light_text = '#f3f6ff'
        error_fg = '#c62828' if is_light else '#ff8f8f'
        warning_fg = '#b45309' if is_light else '#ffbf69'
        draft_fg = '#8a5a00' if is_light else '#ffd166'
        error_bg = '#fdecec' if is_light else '#30151c'
        warning_bg = '#fff4e5' if is_light else '#35240f'
        draft_bg = '#f6f0de' if is_light else '#2c2414'

        for vis_idx, model_idx in enumerate(self.visible_entry_indices):
            entry = self.plan_data.get('entries', [])[model_idx]
            problems = problem_map.get(model_idx, [])
            sev = self._planner_problem_severity(problems)
            row_color = palette.get(self._entry_row_color(entry))
            type_color = self._entry_type_color(entry.get('type', ''))

            bg = row_color or colors['entry_bg']
            if row_color:
                fg = _contrast_text(bg, dark_text, light_text)
            else:
                fg = type_color or colors['entry_fg']

            if self._entry_is_disabled(entry):
                fg = colors['muted']
                bg = colors['subtle']
            elif sev == 'error':
                fg = error_fg
                bg = _mix(bg, error_bg, 0.72 if row_color else 1.0)
            elif sev == 'warning':
                fg = warning_fg
                bg = _mix(bg, warning_bg, 0.62 if row_color else 1.0)
            elif sev == 'draft':
                fg = type_color if not row_color else _contrast_text(bg, dark_text, light_text)
                bg = _mix(bg, draft_bg, 0.35 if row_color else 1.0)
            elif self._entry_is_locked(entry):
                fg = '#1f5fbf' if is_light else '#9db7ff'

            try:
                self.plan_listbox.itemconfig(
                    vis_idx,
                    foreground=fg,
                    background=bg,
                    selectforeground=colors['select_fg'],
                    selectbackground=colors['select_bg'],
                )
            except Exception:
                try:
                    self.plan_listbox.itemconfigure(vis_idx, fg=fg, bg=bg)
                except Exception:
                    pass

    def _collect_available_models(self, upto_index: int) -> Dict[str, List[str]]:
        available: Dict[str, Dict[str, str]] = {'Checkpoint': {}, 'LoRA': {}, 'LyCORIS': {}}
        removed: set[str] = set()
        for entry in self.plan_data.get('entries', [])[:upto_index]:
            if self._entry_is_disabled(entry):
                continue
            etype = entry.get('type')
            if etype == 'Remove Model':
                name = (entry.get('model') or '').strip()
                if name:
                    removed.add(name)
                    available['Checkpoint'].pop(name, None)
                    available['LoRA'].pop(name, None)
                    available['LyCORIS'].pop(name, None)
                continue
            if etype == 'Download Model':
                name = (entry.get('model_name') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip()
                if name and name not in removed:
                    available.setdefault(kind, {})[name] = kind
            elif etype == 'Local Model':
                path = (entry.get('local_path') or '').strip()
                if path:
                    name = Path(path).stem
                    kind = (entry.get('model_type') or 'Checkpoint').strip()
                    if name not in removed:
                        available.setdefault(kind, {})[name] = kind
            elif etype in ('Checkpoint Merge', 'LoRA Bake'):
                name = (entry.get('output_name') or '').strip()
                if name and name not in removed:
                    available['Checkpoint'][name] = 'Checkpoint'
        return {k: sorted(v.keys()) for k, v in available.items()}

    def _planner_effective_plan_data(self):
        effective_entries = []
        for entry in self.plan_data.get('entries', []):
            if self._entry_is_disabled(entry):
                continue
            etype = entry.get('type')
            if etype == 'Remove Model':
                continue
            if etype == 'Checkpoint Merge':
                for slot in ('model0', 'model1', 'model2'):
                    spec = self._get_entry_slot_source(entry, slot)
                    alias = str(entry.get(slot) or self._source_alias_from_spec(spec)).strip()
                    if spec and alias:
                        effective_entries.append(self._make_effective_source_entry(spec, alias))
                effective_entries.append(self._sanitize_effective_payload(entry))
            elif etype == 'LoRA Bake':
                checkpoint_spec = self._get_entry_slot_source(entry, 'checkpoint')
                checkpoint_alias = str(entry.get('checkpoint') or self._source_alias_from_spec(checkpoint_spec)).strip()
                if checkpoint_spec and checkpoint_alias:
                    effective_entries.append(self._make_effective_source_entry(checkpoint_spec, checkpoint_alias))
                for lora in entry.get('loras', []) or []:
                    spec = self._get_lora_source(lora)
                    alias = str(lora.get('name') or self._source_alias_from_spec(spec)).strip()
                    if spec and alias:
                        effective_entries.append(self._make_effective_source_entry(spec, alias))
                effective_entries.append(self._sanitize_effective_payload(entry))
            else:
                effective_entries.append(self._sanitize_effective_payload(entry))
        return self._append_auto_remove_lines({'entries': effective_entries})

    def _save_plan_to_file(self):
        path = self._ensure_plan_path()
        self._planner_apply_entry_defaults()
        export_plan_records_txt(path, self._planner_effective_plan_data())
        self._planner_save_meta_to_disk(path)
        self._planner_maybe_create_backup(path, reason='autosave')
        self._schedule_config_save()

    def _restore_session_state(self):
        filepath = self.entries['filepath'].get().strip()
        if filepath and os.path.exists(filepath):
            try:
                raw = normalize_plan(load_plan_records(filepath))
                collapse = getattr(self, '_collapse_internal_plan_entries', None)
                self.plan_data = collapse(raw) if callable(collapse) else raw
            except Exception:
                self.plan_data = self._planner_default_visible_plan()
        else:
            self.plan_data = self._planner_default_visible_plan()
        load_meta = getattr(self, '_planner_load_meta_from_disk', None)
        if callable(load_meta):
            load_meta(filepath)
        apply_defaults = getattr(self, '_planner_apply_entry_defaults', None)
        if callable(apply_defaults):
            apply_defaults()

    def _load_plan_from_path(self):
        path = self._normalize_user_path(self.entries['filepath'].get().strip())
        if not path:
            messagebox.showwarning('Plan Path', 'Plan Text Path is empty.')
            return
        try:
            raw = normalize_plan(load_plan_records(path))
            self.plan_data = self._collapse_internal_plan_entries(raw)
            self._planner_load_meta_from_disk(path)
            self._planner_apply_entry_defaults()
            self.current_index = 0
            self._refresh_line_selector()
            self._render_current_line()
            self.status_label.config(text=f'Loaded plan: {self._path_display_name(path)}')
        except Exception as e:
            self._show_detailed_error('Load Error', e)

    def _planner_convert_entry_type_preserving_common_fields(self, entry: Dict[str, Any], new_type: str) -> Dict[str, Any]:
        new_entry = make_entry(new_type)
        common_name = str(entry.get('output_name') or entry.get('model_name') or entry.get('model') or Path(entry.get('local_path') or '').stem or '').strip()
        for key in ('memo', '_locked', '_disabled', '_row_color'):
            if key in entry:
                new_entry[key] = copy.deepcopy(entry.get(key))
        if new_type == 'Download Model':
            new_entry['model_name'] = common_name
            new_entry['model_type'] = str(entry.get('model_type') or 'Checkpoint')
            new_entry['link'] = str(entry.get('link') or '')
        elif new_type == 'Local Model':
            new_entry['model_type'] = str(entry.get('model_type') or 'Checkpoint')
            new_entry['local_path'] = str(entry.get('local_path') or '')
        elif new_type == 'Remove Model':
            new_entry['model'] = common_name
        elif new_type == 'Checkpoint Merge':
            new_entry['model0'] = str(entry.get('model0') or entry.get('checkpoint') or '')
            new_entry['model1'] = str(entry.get('model1') or '')
            new_entry['model2'] = str(entry.get('model2') or '')
            new_entry['output_name'] = common_name
        elif new_type == 'LoRA Bake':
            new_entry['checkpoint'] = str(entry.get('checkpoint') or entry.get('model0') or '')
            new_entry['output_name'] = common_name
            if entry.get('loras'):
                new_entry['loras'] = copy.deepcopy(entry.get('loras') or [])
        return new_entry

    def _planner_replace_in_values(self, value, find_text: str, replace_text: str):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if str(key).startswith('_') and key not in {'_slot_sources'}:
                    out[key] = copy.deepcopy(item)
                else:
                    out[key] = self._planner_replace_in_values(item, find_text, replace_text)
            return out
        if isinstance(value, list):
            return [self._planner_replace_in_values(x, find_text, replace_text) for x in value]
        if isinstance(value, str):
            return value.replace(find_text, replace_text)
        return copy.deepcopy(value)

    def _show_bulk_edit_dialog(self):
        indices = self._planner_get_selected_indices() or ([self.current_index] if self.plan_data.get('entries') else [])
        if not indices:
            messagebox.showwarning('Bulk Edit', 'No lines are selected.')
            return
        win = tk.Toplevel(self.root)
        win.title('Bulk Edit Selected Lines')
        win.geometry('620x300+140+140')
        outer = Frame(win, padx=10, pady=10)
        outer.pack(fill='both', expand=True)
        Label(outer, text=f'Selected lines: {", ".join(str(i + 1) for i in indices)}', anchor='w').pack(fill='x', pady=(0, 8))

        row1 = Frame(outer)
        row1.pack(fill='x', pady=4)
        Label(row1, text='Change type to', width=16, anchor='w').pack(side='left')
        type_var = tk.StringVar(value='(keep)')
        type_combo = ttk.Combobox(row1, textvariable=type_var, values=['(keep)'] + self._planner_visible_line_types() + self._planner_internal_line_types(), state='readonly')
        type_combo.pack(side='left', fill='x', expand=True)

        row2 = Frame(outer)
        row2.pack(fill='x', pady=4)
        Label(row2, text='Find text', width=16, anchor='w').pack(side='left')
        find_var = tk.StringVar(value='')
        Entry(row2, textvariable=find_var).pack(side='left', fill='x', expand=True)

        row3 = Frame(outer)
        row3.pack(fill='x', pady=4)
        Label(row3, text='Replace with', width=16, anchor='w').pack(side='left')
        replace_var = tk.StringVar(value='')
        Entry(row3, textvariable=replace_var).pack(side='left', fill='x', expand=True)

        info = Label(outer, text='Type change converts each selected entry using a safe default mapping. String replace walks all visible entry fields except internal metadata keys.', justify='left', anchor='w', fg='#666666')
        info.pack(fill='x', pady=(8, 0))

        def apply_changes():
            selected = self._planner_get_selected_indices() or indices
            if not selected:
                return
            entries = self.plan_data.get('entries', [])
            locked = self._planner_locked_indices(selected)
            editable = [idx for idx in selected if idx not in locked]
            if not editable:
                messagebox.showwarning('Bulk Edit', 'All selected lines are locked.')
                return
            self._planner_push_history()
            chosen_type = type_var.get().strip()
            find_text = find_var.get()
            replace_text = replace_var.get()
            changed = 0
            for idx in editable:
                entry = copy.deepcopy(entries[idx])
                if chosen_type and chosen_type != '(keep)':
                    entry = self._planner_convert_entry_type_preserving_common_fields(entry, chosen_type)
                if find_text:
                    entry = self._planner_replace_in_values(entry, find_text, replace_text)
                entries[idx] = entry
                changed += 1
            self._save_plan_to_file()
            self._refresh_line_selector()
            self._render_current_line()
            self._select_model_indices(selected)
            msg = f'Bulk edit applied to {changed} line(s)'
            if locked:
                msg += f' (skipped locked lines: {", ".join(str(i + 1) for i in locked)})'
            self.status_label.config(text=msg)
            win.destroy()

        button_bar = Frame(outer)
        button_bar.pack(fill='x', pady=(12, 0))
        ttk.Button(button_bar, text='Apply', command=apply_changes).pack(side='left')
        ttk.Button(button_bar, text='Close', command=win.destroy).pack(side='right')
        try:
            colors = self._theme_colors()
            win.configure(bg=colors['bg'])
            self._apply_theme_to_children(win, colors)
        except Exception:
            pass

    def _show_history_view(self):
        win = tk.Toplevel(self.root)
        win.title('Undo / Redo History')
        win.geometry('980x620+130+130')
        outer = Frame(win, padx=8, pady=8)
        outer.pack(fill='both', expand=True)
        Label(outer, text=f'Undo entries: {len(self._history_undo)}    Redo entries: {len(self._history_redo)}', anchor='w').pack(fill='x', pady=(0, 6))
        main = Frame(outer)
        main.pack(fill='both', expand=True)
        listbox = tk.Listbox(main, exportselection=False, activestyle='none')
        scroll = Scrollbar(main, orient='vertical', command=listbox.yview)
        listbox.configure(yscrollcommand=scroll.set)
        listbox.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        detail = Text(outer, height=14, wrap='word', font=('Consolas', 10))
        detail.pack(fill='both', expand=False, pady=(8, 0))
        records = []

        def unpack(item):
            if isinstance(item, dict):
                return str(item.get('payload') or ''), int(item.get('index', 0) or 0)
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                return str(item[0]), int(item[1] or 0)
            return '', 0

        def describe(payload, index, prefix):
            try:
                plan = json.loads(payload)
                entries = plan.get('entries', []) if isinstance(plan, dict) else []
                count = len(entries)
                summary = ''
                if entries:
                    index = max(0, min(int(index or 0), len(entries) - 1))
                    summary = self._line_summary(entries[index])
                return f'[{prefix}] line {index + 1} / {count}  •  {summary}'
            except Exception:
                return f'[{prefix}] snapshot'

        current_payload = self._planner_plan_snapshot()
        records.append(('CURRENT', current_payload, self.current_index))
        for item in reversed(self._history_undo[-40:]):
            payload, index = unpack(item)
            records.append(('UNDO', payload, index))
        for item in reversed(self._history_redo[-40:]):
            payload, index = unpack(item)
            records.append(('REDO', payload, index))
        for kind, payload, index in records:
            listbox.insert('end', describe(payload, index, kind))

        def update_detail(_event=None):
            detail.delete('1.0', 'end')
            sel = listbox.curselection()
            if not sel:
                return
            kind, payload, index = records[sel[-1]]
            try:
                current = json.loads(current_payload)
                other = json.loads(payload)
                left = json.dumps(current, ensure_ascii=False, indent=2).splitlines()
                right = json.dumps(other, ensure_ascii=False, indent=2).splitlines()
                diff = '\n'.join(difflib.unified_diff(left, right, fromfile='current', tofile=f'{kind.lower()}_snapshot', lineterm=''))
                detail.insert('1.0', diff or json.dumps(other, ensure_ascii=False, indent=2))
            except Exception:
                detail.insert('1.0', payload)

        def restore_selected():
            sel = listbox.curselection()
            if not sel:
                return
            kind, payload, index = records[sel[-1]]
            if kind == 'CURRENT':
                return
            self._planner_push_history()
            self._planner_restore_snapshot(payload, index)
            self.status_label.config(text=f'History snapshot restored from {kind.lower()} stack')
            win.destroy()

        listbox.bind('<<ListboxSelect>>', update_detail, add='+')
        listbox.bind('<Double-Button-1>', lambda _e: restore_selected(), add='+')
        if records:
            listbox.selection_set(0)
            update_detail()
        button_bar = Frame(outer)
        button_bar.pack(fill='x', pady=(8, 0))
        ttk.Button(button_bar, text='Restore Selected Snapshot', command=restore_selected).pack(side='left')
        ttk.Button(button_bar, text='Close', command=win.destroy).pack(side='right')
        try:
            colors = self._theme_colors()
            win.configure(bg=colors['bg'])
            self._apply_theme_to_children(win, colors)
        except Exception:
            pass

    def _show_row_color_dialog(self):
        indices = self._planner_get_selected_indices() or ([self.current_index] if self.plan_data.get('entries') else [])
        if not indices:
            return
        win = tk.Toplevel(self.root)
        win.title('Row Color Preset')
        win.geometry('420x140+180+180')
        outer = Frame(win, padx=10, pady=10)
        outer.pack(fill='both', expand=True)
        Label(outer, text=f'Apply to lines: {", ".join(str(i + 1) for i in indices)}', anchor='w').pack(fill='x', pady=(0, 8))
        presets = list(self._planner_row_color_presets().keys())
        current_name = self._entry_row_color(self.plan_data['entries'][indices[-1]]) or 'Default'
        var = tk.StringVar(value=current_name if current_name in presets else 'Default')
        combo = ttk.Combobox(outer, textvariable=var, values=presets, state='readonly')
        combo.pack(fill='x')

        def apply_color():
            self._planner_push_history()
            for idx in indices:
                if 0 <= idx < len(self.plan_data.get('entries', [])):
                    self.plan_data['entries'][idx]['_row_color'] = '' if var.get() == 'Default' else var.get()
            self._save_plan_to_file()
            self._refresh_line_selector()
            self._render_current_line()
            self._select_model_indices(indices)
            self.status_label.config(text=f'Applied row color: {var.get()}')
            win.destroy()

        button_bar = Frame(outer)
        button_bar.pack(fill='x', pady=(10, 0))
        ttk.Button(button_bar, text='Apply', command=apply_color).pack(side='left')
        ttk.Button(button_bar, text='Close', command=win.destroy).pack(side='right')
        try:
            colors = self._theme_colors()
            win.configure(bg=colors['bg'])
            self._apply_theme_to_children(win, colors)
        except Exception:
            pass

    def _toggle_selected_locked(self):
        indices = self._planner_get_selected_indices() or ([self.current_index] if self.plan_data.get('entries') else [])
        if not indices:
            return
        entries = self.plan_data.get('entries', [])
        target = not all(self._entry_is_locked(entries[idx]) for idx in indices if 0 <= idx < len(entries))
        self._planner_push_history()
        for idx in indices:
            if 0 <= idx < len(entries):
                entries[idx]['_locked'] = target
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices(indices)
        self.status_label.config(text=('Locked' if target else 'Unlocked') + f' {len(indices)} line(s)')

    def _toggle_selected_disabled(self):
        indices = self._planner_get_selected_indices() or ([self.current_index] if self.plan_data.get('entries') else [])
        if not indices:
            return
        entries = self.plan_data.get('entries', [])
        target = not all(self._entry_is_disabled(entries[idx]) for idx in indices if 0 <= idx < len(entries))
        self._planner_push_history()
        for idx in indices:
            if 0 <= idx < len(entries):
                entries[idx]['_disabled'] = target
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices(indices)
        self.status_label.config(text=('Disabled' if target else 'Enabled') + f' {len(indices)} line(s)')

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

    def _render_current_line(self):
        self._base_render_current_line()
        self._update_current_line_indicator()
        try:
            entry = self.plan_data.get('entries', [])[self.current_index]
        except Exception:
            self._planner_refresh_plan_meta()
            self._update_current_line_indicator()
            return

        meta_frame = LabelFrame(self.scroll_frame, text='Line Meta', padx=8, pady=8)
        meta_frame.pack(fill='x', padx=6, pady=6)
        state_text = []
        if self._entry_is_locked(entry):
            state_text.append('Locked')
        if self._entry_is_disabled(entry):
            state_text.append('Disabled')
        if self._entry_row_color(entry):
            state_text.append(f"Color={self._entry_row_color(entry)}")
        Label(meta_frame, text=('Status: ' + ', '.join(state_text)) if state_text else 'Status: normal', anchor='w', justify='left').pack(fill='x', pady=(0, 6))
        button_row = Frame(meta_frame)
        button_row.pack(fill='x')
        ttk.Button(button_row, text='Unlock' if self._entry_is_locked(entry) else 'Lock', command=self._toggle_selected_locked).pack(side='left')
        ttk.Button(button_row, text='Enable' if self._entry_is_disabled(entry) else 'Disable', command=self._toggle_selected_disabled).pack(side='left', padx=(6, 0))
        ttk.Button(button_row, text='Row Color', command=self._show_row_color_dialog).pack(side='left', padx=(6, 0))
        ttk.Button(button_row, text='Bulk Edit', command=self._show_bulk_edit_dialog).pack(side='left', padx=(6, 0))
        ttk.Button(button_row, text='History', command=self._show_history_view).pack(side='left', padx=(6, 0))
        ttk.Button(button_row, text='Graph', command=self._show_dependency_view).pack(side='left', padx=(6, 0))
        ttk.Button(button_row, text='Backups', command=self._show_backup_manager).pack(side='left', padx=(6, 0))

        note_frame = LabelFrame(self.scroll_frame, text='Memo', padx=8, pady=8)
        note_frame.pack(fill='both', padx=6, pady=6)
        memo = Text(note_frame, height=4, font=('Consolas', 10), undo=True, autoseparators=True, maxundo=-1)
        memo.pack(fill='both', expand=True)
        memo.insert('1.0', entry.get('memo', ''))
        memo.bind('<KeyRelease>', lambda _e, e=entry, w=memo: (e.__setitem__('memo', w.get('1.0', 'end-1c')), self._after_entry_change()), add='+')
        memo.bind('<<Paste>>', lambda _e, e=entry, w=memo: self.root.after_idle(lambda: (e.__setitem__('memo', w.get('1.0', 'end-1c')), self._after_entry_change())), add='+')

        if self._entry_is_locked(entry):
            self._planner_disable_widget_subtree(self.scroll_frame, exempt_roots={meta_frame})
        self._planner_refresh_plan_meta()

    def _delete_selected_lines(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return
        if not self._planner_guard_unlocked(indices, 'delete the selected lines'):
            return
        entries = self.plan_data.get('entries', [])
        if len(entries) - len(indices) < 1:
            messagebox.showwarning('Plan', 'At least one line must remain.')
            return
        self._planner_push_history()
        for idx in reversed(indices):
            entries.pop(idx)
        self.current_index = max(0, min(indices[0], len(entries) - 1))
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self.status_label.config(text=f'Deleted {len(indices)} line(s)')

    def _duplicate_selected_lines(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return
        self._planner_push_history()
        entries = self.plan_data.get('entries', [])
        insert_at = indices[-1] + 1
        copies = []
        for idx in indices:
            cloned = copy.deepcopy(entries[idx])
            cloned['id'] = make_entry(cloned.get('type', 'Download Model')).get('id')
            copies.append(cloned)
        entries[insert_at:insert_at] = copies
        self.current_index = insert_at + len(copies) - 1
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(list(range(insert_at, insert_at + len(copies))))
        self.status_label.config(text=f'Duplicated {len(copies)} line(s)')

    def _paste_copied_lines(self):
        payload = self.plan_clipboard_entries
        if not payload:
            try:
                clip = self.root.clipboard_get()
                data = json.loads(clip)
                if isinstance(data, dict):
                    payload = [data]
                elif isinstance(data, list):
                    payload = data
            except Exception:
                payload = []
        if not payload:
            return
        self._planner_push_history()
        insert_at = max(self._planner_get_selected_indices() or [self.current_index]) + 1 if self.plan_data.get('entries') else 0
        pasted = []
        for raw in payload:
            new_entry = self._normalize_entry_preserving_embedded_sources(copy.deepcopy(raw))
            new_entry['id'] = make_entry(new_entry.get('type', 'Download Model')).get('id')
            new_entry['memo'] = str(raw.get('memo') or new_entry.get('memo') or '') if isinstance(raw, dict) else str(new_entry.get('memo') or '')
            new_entry['_locked'] = bool(raw.get('_locked')) if isinstance(raw, dict) else False
            new_entry['_disabled'] = bool(raw.get('_disabled')) if isinstance(raw, dict) else False
            new_entry['_row_color'] = str(raw.get('_row_color') or '') if isinstance(raw, dict) else ''
            pasted.append(new_entry)
        self.plan_data['entries'][insert_at:insert_at] = pasted
        self.current_index = insert_at + len(pasted) - 1
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices(list(range(insert_at, insert_at + len(pasted))))
        self.status_label.config(text=f'Pasted {len(pasted)} line(s)')

    def _change_line_type(self, new_type: str):
        entries = self.plan_data.get('entries', [])
        if 0 <= self.current_index < len(entries) and self._entry_is_locked(entries[self.current_index]):
            messagebox.showwarning('Locked Line', f'Line {self.current_index + 1} is locked and cannot change type.')
            return
        self._planner_push_history()
        if new_type not in self._planner_visible_line_types() and new_type not in self._planner_internal_line_types():
            new_type = 'Checkpoint Merge'
        return self._base_change_line_type(new_type)

    def _move_line_up(self):
        indices = self._planner_get_selected_indices()
        if not indices or indices[0] <= 0:
            return
        if not self._planner_guard_unlocked(indices, 'move the selected lines'):
            return
        entries = self.plan_data.get('entries', [])
        if self._entry_is_locked(entries[indices[0] - 1]):
            messagebox.showwarning('Locked Line', f'Cannot move above locked line {indices[0]}.')
            return
        self._planner_push_history()
        block = [entries[i] for i in indices]
        for idx in reversed(indices):
            entries.pop(idx)
        insert_at = indices[0] - 1
        for offset, item in enumerate(block):
            entries.insert(insert_at + offset, item)
        moved = list(range(insert_at, insert_at + len(block)))
        self.current_index = moved[-1]
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(moved)

    def _move_line_down(self):
        indices = self._planner_get_selected_indices()
        entries = self.plan_data.get('entries', [])
        if not indices or indices[-1] >= len(entries) - 1:
            return
        if not self._planner_guard_unlocked(indices, 'move the selected lines'):
            return
        if self._entry_is_locked(entries[indices[-1] + 1]):
            messagebox.showwarning('Locked Line', f'Cannot move below locked line {indices[-1] + 2}.')
            return
        self._planner_push_history()
        block = [entries[i] for i in indices]
        after_index = indices[-1] + 1
        for idx in reversed(indices):
            entries.pop(idx)
        insert_at = after_index - len(indices) + 1
        for offset, item in enumerate(block):
            entries.insert(insert_at + offset, item)
        moved = list(range(insert_at, insert_at + len(block)))
        self.current_index = moved[-1]
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(moved)

    def _hide_plan_drag_guide(self):
        guide = getattr(self, '_plan_drag_guide', None)
        if guide is not None:
            try:
                guide.destroy()
            except Exception:
                pass
        self._plan_drag_guide = None
        self._plan_drag_guide_target = None

    def _show_plan_drag_guide(self, vis_idx: int, after: bool = False):
        if self.plan_listbox is None or not self.visible_entry_indices:
            self._hide_plan_drag_guide()
            return
        vis_idx = max(0, min(int(vis_idx), len(self.visible_entry_indices) - 1))
        bbox = self.plan_listbox.bbox(vis_idx)
        if not bbox:
            self._hide_plan_drag_guide()
            return
        bx, by, bw, bh = bbox
        x = self.plan_listbox.winfo_rootx() + 2
        y = self.plan_listbox.winfo_rooty() + by + (bh if after else 0) - 1
        w = max(40, self.plan_listbox.winfo_width() - 4)
        guide = getattr(self, '_plan_drag_guide', None)
        if guide is None or not guide.winfo_exists():
            guide = tk.Toplevel(self.root)
            guide.wm_overrideredirect(True)
            guide.attributes('-topmost', True)
            try:
                guide.configure(bg='#3567d6')
            except Exception:
                pass
            self._plan_drag_guide = guide
        try:
            guide.geometry(f'{w}x3+{x}+{y}')
        except Exception:
            pass
        self._plan_drag_guide_target = (vis_idx, bool(after))

    def _plan_list_drag_start(self, event):
        if self.plan_listbox is None:
            return
        idx = self.plan_listbox.nearest(event.y)
        self._drag_start_visible = idx
        self._drag_selected_model_indices = self._planner_get_selected_indices()
        state = int(getattr(event, 'state', 0) or 0)
        self._drag_requires_primary = bool(state & 0x0004 or state & 0x0008 or state & 0x0010)
        if self._drag_requires_primary:
            self._show_plan_drag_guide(idx, after=False)

    def _on_plan_list_motion(self, event):
        if self._drag_start_visible is not None and getattr(self, '_drag_requires_primary', False):
            if self.plan_listbox is None or not self.visible_entry_indices:
                self._hide_plan_drag_guide()
                return
            idx = self.plan_listbox.nearest(event.y)
            bbox = self.plan_listbox.bbox(max(0, min(idx, len(self.visible_entry_indices) - 1)))
            if bbox:
                _bx, by, _bw, bh = bbox
                after = bool(event.y > by + (bh / 2.0))
                self._show_plan_drag_guide(idx, after=after)
            return
        if self.plan_listbox is None or not self.visible_entry_indices:
            self._hide_plan_item_hover()
            return
        idx = self.plan_listbox.nearest(event.y)
        if idx < 0 or idx >= len(self.visible_entry_indices):
            self._hide_plan_item_hover()
            return
        bbox = self.plan_listbox.bbox(idx)
        if not bbox:
            self._hide_plan_item_hover()
            return
        bx, by, bw, bh = bbox
        if not (by <= event.y <= by + bh):
            self._hide_plan_item_hover()
            return
        model_idx = self.visible_entry_indices[idx]
        try:
            font = tkfont.Font(font=self.plan_listbox.cget('font'))
            hover_width = max(18, font.measure(f'{model_idx + 1:02d}') + 8)
        except Exception:
            hover_width = 28
        if not (bx <= event.x <= min(bx + bw, bx + hover_width)):
            self._hide_plan_item_hover()
            return
        entry = self.plan_data.get('entries', [])[model_idx]
        problems = getattr(self, '_plan_problem_map_cache', {}).get(model_idx, [])
        text = self._build_plan_item_tooltip_text(model_idx, entry, problems)
        if idx == getattr(self, '_plan_hover_index', None) and text == getattr(self, '_plan_hover_text', ''):
            return
        self._plan_hover_index = idx
        self._plan_hover_text = text
        self._show_plan_item_hover(event, text)

    def _plan_list_drag_end(self, event):
        self._hide_plan_drag_guide()
        if self.plan_listbox is None or self._drag_start_visible is None:
            return
        target_vis = self.plan_listbox.nearest(event.y)
        self._drag_start_visible = None
        selected = self._drag_selected_model_indices or self._planner_get_selected_indices()
        requires_primary = bool(getattr(self, '_drag_requires_primary', False))
        self._drag_selected_model_indices = []
        self._drag_requires_primary = False
        if not requires_primary:
            return
        if not selected or not self.visible_entry_indices:
            return
        if not self._planner_guard_unlocked(selected, 'reorder the selected lines'):
            return
        target_vis = max(0, min(target_vis, len(self.visible_entry_indices) - 1))
        target_model = self.visible_entry_indices[target_vis]
        if target_model in selected:
            return
        entries = self.plan_data.get('entries', [])
        if 0 <= target_model < len(entries) and self._entry_is_locked(entries[target_model]):
            messagebox.showwarning('Locked Line', f'Cannot drop onto locked line {target_model + 1}.')
            return
        self._planner_push_history()
        block = [entries[i] for i in selected]
        for idx in reversed(selected):
            entries.pop(idx)
        insert_at = target_model
        shift = sum(1 for idx in selected if idx < target_model)
        insert_at -= shift
        bbox = self.plan_listbox.bbox(target_vis)
        if bbox:
            _bx, by, _bw, bh = bbox
            if event.y > by + (bh / 2.0):
                insert_at += 1
        insert_at = max(0, min(insert_at, len(entries)))
        if insert_at < len(entries) and self._entry_is_locked(entries[insert_at]):
            messagebox.showwarning('Locked Line', f'Cannot insert before locked line {insert_at + 1}.')
            return
        for offset, item in enumerate(block):
            entries.insert(insert_at + offset, item)
        moved = list(range(insert_at, insert_at + len(block)))
        self.current_index = moved[-1]
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._planner_restore_plan_list_focus()
        self._select_model_indices(moved)
        self.status_label.config(text='Reordered selected lines')

    def _show_dependency_view(self):
        analysis = self._planner_analysis()
        lines = ['Dependency View', '', 'Adjacency:', '']
        producer_by_alias = analysis['producer_by_alias']
        for idx, entry in enumerate(self.plan_data.get('entries', [])):
            tag = []
            if self._entry_is_locked(entry):
                tag.append('locked')
            if self._entry_is_disabled(entry):
                tag.append('disabled')
            lines.append(f'[{idx + 1}] {self._line_summary(entry)}' + (f'  [{", ".join(tag)}]' if tag else ''))
            produced = self._entry_produced_aliases(entry)
            embedded = [alias for _slot, alias, _kind, _spec in self._iter_embedded_sources(entry)]
            consumes = self._entry_consumed_aliases(entry)
            if embedded:
                lines.append('  embedded sources: ' + ', '.join(embedded))
            if produced:
                lines.append('  produces: ' + ', '.join(produced))
            if consumes:
                detail = []
                for alias in consumes:
                    src = producer_by_alias.get(alias)
                    detail.append(f'{alias} <- line {src + 1}' if src is not None else f'{alias} <- external')
                lines.append('  consumes: ' + '; '.join(detail))
            if idx in analysis.get('dead_entries', set()):
                lines.append('  status: DEAD / not required by final active line')
            lines.append('')
        lines += ['', 'Mermaid-like graph:', 'graph TD']
        for idx, entry in enumerate(self.plan_data.get('entries', [])):
            label = self._line_summary(entry).replace('"', "'")
            lines.append(f'  L{idx + 1}["{idx + 1}: {label}"]')
        for idx, consumes in analysis.get('consumed_by_entry', {}).items():
            for alias in consumes:
                src = producer_by_alias.get(alias)
                if src is None:
                    lines.append(f'  EXT_{re.sub(r"[^A-Za-z0-9_]", "_", alias)}(["external:{alias}"]) -->|{alias}| L{idx + 1}')
                else:
                    lines.append(f'  L{src + 1} -->|{alias}| L{idx + 1}')
        self._show_scrollable_text_dialog('Dependency View', '\n'.join(lines))

    def _show_prevalidation(self):
        problem_map = self._plan_entry_problem_map()
        issues = []
        for idx in sorted(problem_map):
            for problem in problem_map.get(idx, []) or []:
                issues.append((idx, problem))
        if not issues:
            self._show_scrollable_text_dialog('Pre-validation', 'No issues found.')
            return
        win = tk.Toplevel(self.root)
        win.title('Pre-validation')
        win.geometry('980x620+120+120')
        outer = Frame(win, padx=8, pady=8)
        outer.pack(fill='both', expand=True)
        Label(outer, text='Double-click an issue to jump to that line.', anchor='w').pack(fill='x', pady=(0, 6))
        list_row = Frame(outer)
        list_row.pack(fill='both', expand=True)
        listbox = tk.Listbox(list_row, exportselection=False, activestyle='none')
        scroll = Scrollbar(list_row, orient='vertical', command=listbox.yview)
        listbox.configure(yscrollcommand=scroll.set)
        listbox.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        for idx, problem in issues:
            listbox.insert('end', f'Line {idx + 1}: {problem}')
        for row_idx, (_idx, problem) in enumerate(issues):
            sev = 'error' if str(problem).startswith('ERROR:') else 'warning'
            try:
                listbox.itemconfig(row_idx, foreground='#cc2222' if sev == 'error' else '#d97706')
            except Exception:
                pass
        def jump_selected():
            sel = listbox.curselection()
            if not sel:
                return
            idx, _problem = issues[sel[-1]]
            self.current_index = idx
            self._refresh_line_selector()
            self._render_current_line()
            self._select_model_indices([idx])
            self.status_label.config(text=f'Jumped to line {idx + 1} from validation results')
        listbox.bind('<Double-Button-1>', lambda _e: jump_selected(), add='+')
        button_bar = Frame(outer)
        button_bar.pack(fill='x', pady=(8, 0))
        ttk.Button(button_bar, text='Jump to Selected Line', command=jump_selected).pack(side='left')
        ttk.Button(button_bar, text='Copy All', command=lambda: (win.clipboard_clear(), win.clipboard_append('\n'.join(f'Line {idx + 1}: {problem}' for idx, problem in issues)))).pack(side='left', padx=(6, 0))
        ttk.Button(button_bar, text='Close', command=win.destroy).pack(side='right')
        try:
            colors = self._theme_colors()
            win.configure(bg=colors['bg'])
            self._apply_theme_to_children(win, colors)
        except Exception:
            pass

    def _show_plan_context_menu(self, event=None):
        if self.plan_listbox is None or event is None:
            return 'break'
        model_idx = self._plan_list_prepare_context_selection(event)
        if model_idx is None:
            self._hide_plan_context_menu()
            return 'break'
        self._hide_plan_context_menu()
        self._planner_restore_plan_list_focus()
        colors = self._theme_colors()
        popup = tk.Toplevel(self.root)
        popup.wm_overrideredirect(True)
        popup.attributes('-topmost', True)
        try:
            popup.configure(bg=colors['panel'])
        except Exception:
            pass
        x = int(getattr(event, 'x_root', 0) or 0) + 10
        y = int(getattr(event, 'y_root', 0) or 0) + 8
        popup.wm_geometry(f'+{x}+{y}')
        card = Frame(popup, bg=colors['surface'], highlightthickness=1, highlightbackground=colors['border'], padx=2, pady=2)
        card.pack(fill='both', expand=True)
        header = Label(card, text='Plan View Actions', anchor='w', justify='left', bg=colors['surface'], fg=colors['text'], padx=12, pady=10, font=('MS Gothic', 11, 'bold'))
        header.pack(fill='x')
        button_area = Frame(card, bg=colors['surface'])
        button_area.pack(fill='both', expand=True, padx=6, pady=(0, 6))
        selected = self._planner_get_selected_indices() or [model_idx]
        entries = self.plan_data.get('entries', [])
        all_locked = bool(selected) and all(0 <= i < len(entries) and self._entry_is_locked(entries[i]) for i in selected)
        any_disabled = any(0 <= i < len(entries) and self._entry_is_disabled(entries[i]) for i in selected)

        def run_and_close(func):
            def wrapped():
                self._hide_plan_context_menu()
                try:
                    func()
                finally:
                    self._planner_restore_plan_list_focus()
                return 'break'
            return wrapped

        actions = [
            ('Copy', self._copy_selected_lines, True),
            ('Paste Below', self._paste_copied_lines, True),
            ('Duplicate Below', self._duplicate_selected_lines, True),
            ('Lock Selected' if not all_locked else 'Unlock Selected', self._toggle_selected_locked, True),
            ('Disable Selected' if not any_disabled else 'Enable Selected', self._toggle_selected_disabled, True),
            ('Set Row Color…', self._show_row_color_dialog, True),
            ('Bulk Edit…', self._show_bulk_edit_dialog, True),
            ('History View…', self._show_history_view, True),
            ('Dependency Graph…', self._show_dependency_view, True),
            ('Backup Manager…', self._show_backup_manager, True),
            ('Optimize Plan', self._planner_optimize_plan_full, True),
            ('Remove Dead Lines', self._remove_dead_lines, bool(self._planner_analysis().get('dead_entries'))),
            ('Create Preset', self._save_preset_json, True),
            ('Delete', self._delete_selected_lines, True),
        ]

        for label, func, enabled in actions:
            row = Frame(button_area, bg=colors['surface'])
            row.pack(fill='x', pady=2)
            style_name = 'MenuDanger.TButton' if label in {'Delete', 'Remove Dead Lines'} else 'Menu.TButton'
            btn = ttk.Button(row, text=label, command=run_and_close(func) if enabled else self._hide_plan_context_menu, style=style_name, cursor='hand2' if enabled else 'arrow', takefocus=False)
            if not enabled:
                btn.state(['disabled'])
            btn.pack(fill='x', expand=True)
            if enabled:
                btn.bind('<ButtonRelease-1>', lambda _e, f=func: run_and_close(f)(), add='+')
                btn.bind('<Return>', lambda _e, f=func: run_and_close(f)(), add='+')
                btn.bind('<space>', lambda _e, f=func: run_and_close(f)(), add='+')
        self._plan_context_popup = popup
        self._plan_context_menu = popup
        def _outside_click(ev=None):
            try:
                widget = getattr(ev, 'widget', None)
                probe = widget
                while probe is not None:
                    if probe is popup:
                        return None
                    probe = getattr(probe, 'master', None)
            except Exception:
                pass
            self._hide_plan_context_menu()
            return None
        bind_ids = {}
        try:
            bind_ids['<Button-1>'] = self.root.bind('<Button-1>', _outside_click, add='+')
            bind_ids['<Button-2>'] = self.root.bind('<Button-2>', _outside_click, add='+')
            bind_ids['<Button-3>'] = self.root.bind('<Button-3>', _outside_click, add='+')
        except Exception:
            bind_ids = {}
        self._plan_context_root_bind_ids = bind_ids
        try:
            popup.update_idletasks()
        except Exception:
            pass
        return 'break'



# Enhanced plan error-cause viewer for validation results.
try:
    import json as _planner_json_mod
    import re as _planner_re_mod

    def _planner_problem_explanation(self, model_idx: int, problem: str) -> str:
        entries = list((self.plan_data or {}).get('entries', []) or [])
        entry = entries[model_idx] if 0 <= model_idx < len(entries) else {}
        available_before = {}
        try:
            available_before = self._collect_available_models(model_idx)
        except Exception:
            available_before = {'Checkpoint': [], 'LoRA': [], 'LyCORIS': []}
        analysis = {}
        try:
            analysis = self._planner_analysis()
        except Exception:
            analysis = {}

        lines = [f'Line {model_idx + 1}', self._line_summary(entry) if entry else '(unknown entry)', '', f'Issue: {problem}']
        severity = 'Error' if str(problem).startswith('ERROR:') else ('Warning' if str(problem).startswith('WARN:') else 'Info')
        lines.append(f'Severity: {severity}')

        problem_text = str(problem or '').strip()
        plain = _planner_re_mod.sub(r'^(ERROR|WARN):\s*', '', problem_text)
        plain_tail = plain.split(': ', 1)[1] if ': ' in plain else plain
        cause_title = 'Cause'
        cause_lines = []
        fix_lines = []

        m_empty = _planner_re_mod.search(r':\s*([A-Za-z0-9_ ]+) is empty\s*$', problem_text)
        m_missing = _planner_re_mod.search(r':\s*(checkpoint|LoRA) ref not available\s*->\s*(.+?)\s*$', problem_text, _planner_re_mod.IGNORECASE)
        m_dup = _planner_re_mod.search(r'duplicate produced alias\s*->\s*(.+?)\s*\(lines\s+(.+?)\)\s*$', problem_text, _planner_re_mod.IGNORECASE)
        m_unref = _planner_re_mod.search(r'unreferenced\s*->\s*(.+?)\s*$', problem_text, _planner_re_mod.IGNORECASE)

        if m_empty:
            field_name = m_empty.group(1).strip()
            cause_lines.append(f'This line is missing the required field "{field_name}".')
            if entry:
                relevant_value = entry.get(field_name)
                if relevant_value not in (None, '', [], {}):
                    cause_lines.append(f'Current stored value: {relevant_value!r}')
            fix_lines.append(f'Fill in "{field_name}" on this line before export or notebook run.')
        elif m_missing:
            ref_kind = m_missing.group(1).strip()
            missing_name = m_missing.group(2).strip()
            cause_lines.append(f'This line refers to a {ref_kind} alias named "{missing_name}", but that alias is not available before this line executes.')
            available_list = available_before.get('Checkpoint' if ref_kind.lower() == 'checkpoint' else 'LoRA', [])
            if ref_kind.lower() == 'lora':
                available_list = sorted(set((available_before.get('LoRA', []) or []) + (available_before.get('LyCORIS', []) or [])))
            if available_list:
                cause_lines.append(f'Available {ref_kind} aliases before this line: ' + ', '.join(available_list))
            else:
                cause_lines.append(f'No {ref_kind} aliases are available before this line.')
            if entry.get('type') == 'Checkpoint Merge' and ref_kind.lower() == 'checkpoint':
                fix_lines.append('Place the producing Download/Local/Merge line above this merge, or change the selected checkpoint alias.')
            elif entry.get('type') == 'LoRA Bake' and ref_kind.lower() == 'checkpoint':
                fix_lines.append('Choose a checkpoint that exists earlier in the plan, or move its producer above this bake line.')
            else:
                fix_lines.append('Make sure the referenced alias is produced by an earlier active line, and that it was not removed or disabled.')
        elif m_dup:
            alias = m_dup.group(1).strip()
            dup_lines = m_dup.group(2).strip()
            cause_lines.append(f'The output alias "{alias}" is produced by multiple active lines ({dup_lines}).')
            cause_lines.append('When later lines refer to that alias, it may be unclear which producer you intended.')
            fix_lines.append('Rename one of the outputs so every produced alias is unique.')
        elif 'merge output is not required by the final active line' in problem_text:
            cause_lines.append('This merge line currently does not contribute to the final active result chain.')
            needed = sorted(int(x) + 1 for x in (analysis.get('needed_entries') or set()))
            if needed:
                cause_lines.append('Lines currently required by the final active result: ' + ', '.join(map(str, needed)))
            fix_lines.append('Either connect this output into a later line, or remove/disable it if it is intentionally unused.')
        elif 'produced alias is currently unreferenced' in problem_text:
            alias = problem_text.rsplit('->', 1)[-1].strip() if '->' in problem_text else ''
            cause_lines.append(f'The produced alias {alias or "(unknown)"} is not consumed by any later active line.')
            fix_lines.append('Use this alias in a later merge/bake line, or remove/disable the producer if it is not needed.')
        elif 'entry is disabled' in problem_text:
            cause_lines.append('This line is intentionally disabled, so it is skipped from export/runtime and can make later references unavailable.')
            fix_lines.append('Re-enable the line if later steps are supposed to use it.')
        elif 'entry is locked' in problem_text:
            cause_lines.append('This line is locked to prevent edits. Locking itself is not a runtime error.')
            fix_lines.append('Unlock the line only if you need to edit it.')
        else:
            cause_lines.append(plain_tail or problem_text)
            fix_lines.append('Review the referenced fields and aliases on this line.')

        lines += ['', cause_title + ':'] + [f'- {line}' for line in cause_lines]
        if fix_lines:
            lines += ['', 'Suggested fix:'] + [f'- {line}' for line in fix_lines]

        if entry:
            try:
                lines += ['', 'Current line payload:', _planner_json_mod.dumps(entry, ensure_ascii=False, indent=2)]
            except Exception:
                lines += ['', 'Current line payload:', repr(entry)]

        if available_before:
            lines += ['', 'Aliases available before this line:']
            for key in ('Checkpoint', 'LoRA', 'LyCORIS'):
                values = list(available_before.get(key, []) or [])
                lines.append(f'- {key}: ' + (', '.join(values) if values else '(none)'))

        return '\n'.join(lines)

    def _planner_collect_issue_rows(self):
        issue_rows = []
        problem_map = self._plan_entry_problem_map()
        for idx in sorted(problem_map):
            for problem in problem_map.get(idx, []) or []:
                issue_rows.append((idx, problem))
        return issue_rows

    def _show_prevalidation_with_causes(self):
        issues = self._planner_collect_issue_rows()
        if not issues:
            self._show_scrollable_text_dialog('Pre-validation', 'No issues found.')
            return
        win = tk.Toplevel(self.root)
        win.title('Pre-validation')
        win.geometry('1180x720+110+110')
        outer = Frame(win, padx=8, pady=8)
        outer.pack(fill='both', expand=True)
        Label(outer, text='Select an issue to see the cause and suggested fix. Double-click still jumps to the line.', anchor='w').pack(fill='x', pady=(0, 6))

        body = ttk.PanedWindow(outer, orient='horizontal')
        body.pack(fill='both', expand=True)
        left = Frame(body)
        right = Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        list_row = Frame(left)
        list_row.pack(fill='both', expand=True)
        listbox = tk.Listbox(list_row, exportselection=False, activestyle='none')
        scroll = Scrollbar(list_row, orient='vertical', command=listbox.yview)
        listbox.configure(yscrollcommand=scroll.set)
        listbox.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')

        for idx, problem in issues:
            listbox.insert('end', f'Line {idx + 1}: {problem}')
        for row_idx, (_idx, problem) in enumerate(issues):
            sev = 'error' if str(problem).startswith('ERROR:') else 'warning'
            try:
                listbox.itemconfig(row_idx, foreground='#cc2222' if sev == 'error' else '#d97706')
            except Exception:
                pass

        Label(right, text='Cause / Suggested fix', anchor='w').pack(fill='x', pady=(0, 4))
        detail_frame = Frame(right)
        detail_frame.pack(fill='both', expand=True)
        yscroll = Scrollbar(detail_frame, orient='vertical')
        xscroll = Scrollbar(detail_frame, orient='horizontal')
        detail = Text(detail_frame, wrap='none', font=('Consolas', 10), yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.config(command=detail.yview)
        xscroll.config(command=detail.xview)
        detail.grid(row=0, column=0, sticky='nsew')
        yscroll.grid(row=0, column=1, sticky='ns')
        xscroll.grid(row=1, column=0, sticky='ew')
        detail_frame.grid_rowconfigure(0, weight=1)
        detail_frame.grid_columnconfigure(0, weight=1)

        def update_detail(*_args):
            sel = listbox.curselection()
            if not sel:
                return
            idx, problem = issues[sel[-1]]
            rendered = self._planner_problem_explanation(idx, problem)
            detail.configure(state='normal')
            detail.delete('1.0', 'end')
            detail.insert('1.0', rendered)
            detail.configure(state='disabled')

        def jump_selected():
            sel = listbox.curselection()
            if not sel:
                return
            idx, _problem = issues[sel[-1]]
            self.current_index = idx
            self._refresh_line_selector()
            self._render_current_line()
            try:
                self._select_model_indices([idx])
            except Exception:
                pass
            try:
                self._planner_restore_plan_list_focus()
            except Exception:
                pass
            self.status_label.config(text=f'Jumped to line {idx + 1} from validation results')

        listbox.bind('<<ListboxSelect>>', update_detail, add='+')
        listbox.bind('<Double-Button-1>', lambda _e: jump_selected(), add='+')

        button_bar = Frame(outer)
        button_bar.pack(fill='x', pady=(8, 0))
        ttk.Button(button_bar, text='Jump to Selected Line', command=jump_selected).pack(side='left')
        ttk.Button(button_bar, text='Copy Selected Detail', command=lambda: (win.clipboard_clear(), win.clipboard_append(detail.get('1.0', 'end-1c')))).pack(side='left', padx=(6, 0))
        ttk.Button(button_bar, text='Copy All', command=lambda: (win.clipboard_clear(), win.clipboard_append('\n\n'.join(self._planner_problem_explanation(idx, problem) for idx, problem in issues)))).pack(side='left', padx=(6, 0))
        ttk.Button(button_bar, text='Close', command=win.destroy).pack(side='right')

        if issues:
            listbox.selection_set(0)
            listbox.activate(0)
            update_detail()
        try:
            colors = self._theme_colors()
            win.configure(bg=colors['bg'])
            self._apply_theme_to_children(win, colors)
        except Exception:
            pass

    def _build_plan_item_tooltip_text_with_causes(self, model_idx: int, entry: Dict[str, Any], problems: List[str] | None = None) -> str:
        problems = problems or []
        base = _planner_prev_build_tooltip(self, model_idx, entry, problems) if _planner_prev_build_tooltip is not None else ''
        if not problems:
            return base
        first_problem = problems[0]
        explanation = self._planner_problem_explanation(model_idx, first_problem)
        preview_lines = []
        hit = False
        for raw_line in explanation.splitlines():
            line = raw_line.strip()
            if line == 'Cause:':
                hit = True
                continue
            if hit:
                if not line or line.endswith(':'):
                    break
                preview_lines.append(line.lstrip('- ').strip())
            if len(preview_lines) >= 2:
                break
        if not preview_lines:
            return base
        addon = '\n'.join(['', 'Cause preview:'] + [f'• {line}' for line in preview_lines])
        return (base + addon).strip()

    _planner_prev_build_tooltip = getattr(ModelPlannerApp, '_build_plan_item_tooltip_text', None)
    ModelPlannerApp._planner_problem_explanation = _planner_problem_explanation
    ModelPlannerApp._planner_collect_issue_rows = _planner_collect_issue_rows
    ModelPlannerApp._show_prevalidation = _show_prevalidation_with_causes
    ModelPlannerApp._build_plan_item_tooltip_text = _build_plan_item_tooltip_text_with_causes
except Exception:
    pass

# Final compatibility bootstrap for patched planner builds with layered same-name overrides.
# Keep the latest rich implementations, but make their initialization assumptions safe.
try:
    _planner_final_init = ModelPlannerApp.__dict__.get('__init__')
    if _planner_final_init is not None:
        _planner_prev_init = getattr(ModelPlannerApp, '_base_init', None)
        def _planner_safe_init(self, root, _impl=_planner_final_init, _prev=_planner_prev_init):
            self.root = root
            if not hasattr(self, 'config') or not isinstance(getattr(self, 'config', None), dict):
                try:
                    self.config = load_config_from_disk()
                except Exception:
                    self.config = INIT_CONFIG.copy()
            return _impl(self, root)
        ModelPlannerApp.__init__ = _planner_safe_init
except Exception:
    pass



# Final validation/message normalization patch:
# - Do not show checkpoint-reference errors for download lines.
# - Only show download-link errors when the link is actually empty, malformed, or confirmed 404/410.
# - Normalize validation messages into title-cased sentences.
try:
    from urllib.parse import urlsplit as _planner_urlsplit
    from urllib.request import Request as _planner_Request, urlopen as _planner_urlopen
    from urllib.error import HTTPError as _planner_HTTPError, URLError as _planner_URLError
    import re as _planner_patch_re

    def _planner_labelize_token(token: str) -> str:
        raw = str(token or '').strip()
        if not raw:
            return 'Value'
        alias_map = {
            'model0': 'Model 0',
            'model1': 'Model 1',
            'model2': 'Model 2',
            'output_name': 'Output Name',
            'model_name': 'Model Name',
            'local_path': 'Local Path',
            'checkpoint': 'Checkpoint',
            'lora': 'LoRA',
            'lycoris': 'LyCORIS',
            'model_type': 'Model Type',
            'link': 'Link',
            'rank': 'Rank',
            'alpha': 'Alpha',
            'beta': 'Beta',
        }
        lower = raw.lower()
        if lower in {k.lower() for k in alias_map}:
            for k, v in alias_map.items():
                if lower == k.lower():
                    return v
        parts = raw.replace('_', ' ').split()
        out = []
        for p in parts:
            if p.lower() in ('lora', 'lycoris', 'vae', 'api', 'url', 'hf'):
                out.append(p.upper() if p.lower() != 'lycoris' else 'LyCORIS')
            else:
                out.append(p[:1].upper() + p[1:])
        return ' '.join(out) if out else raw

    def _planner_make_problem(level: str, line_idx: int, etype: str, message: str) -> str:
        level = str(level or 'ERROR').upper()
        etype = str(etype or 'Line').strip() or 'Line'
        msg = str(message or '').strip()
        return f'{level}: Line {line_idx} ({etype}): {msg}'

    def _planner_download_link_issue(self, link: str, *, eager: bool = False):
        text = str(link or '').strip()
        if not text:
            return 'Link Is Empty'
        parsed = _planner_urlsplit(text)
        if parsed.scheme.lower() not in ('http', 'https') or not parsed.netloc:
            return f'Download Link Format Is Invalid -> {text}'
        cache = getattr(self, '_planner_download_link_validation_cache', None)
        if not isinstance(cache, dict):
            cache = {}
            self._planner_download_link_validation_cache = cache
        if text in cache:
            return cache[text]
        if not eager:
            return None
        result = None
        try:
            req = _planner_Request(text, method='HEAD', headers={'User-Agent': 'ModelPlanner/1.0'})
            with _planner_urlopen(req, timeout=3.5) as resp:
                status = getattr(resp, 'status', None) or resp.getcode()
                if status in (404, 410):
                    result = f'Download Link Not Available -> {text}'
        except _planner_HTTPError as e:
            if e.code in (404, 410):
                result = f'Download Link Not Available -> {text}'
            elif e.code == 405:
                try:
                    req = _planner_Request(text, method='GET', headers={'User-Agent': 'ModelPlanner/1.0'})
                    with _planner_urlopen(req, timeout=3.5) as resp:
                        status = getattr(resp, 'status', None) or resp.getcode()
                        if status in (404, 410):
                            result = f'Download Link Not Available -> {text}'
                except _planner_HTTPError as e2:
                    if e2.code in (404, 410):
                        result = f'Download Link Not Available -> {text}'
                except Exception:
                    result = None
        except (_planner_URLError, ValueError):
            result = None
        except Exception:
            result = None
        cache[text] = result
        return result

    def _plan_entry_problem_map_titleized(self) -> Dict[int, List[str]]:
        problems_by_idx: Dict[int, List[str]] = {}
        available = {'Checkpoint': set(), 'LoRA': set(), 'LyCORIS': set()}
        analysis = self._planner_analysis()
        duplicate_aliases = analysis.get('duplicate_aliases', {})
        unref_aliases = analysis.get('unreferenced_aliases', {})
        dead_entries = set(analysis.get('dead_entries', set()))
        eager_link_validation = bool(getattr(self, '_planner_eager_link_validation', False))

        def slot_label(slot_name: str, *, lora_index: int | None = None) -> str:
            slot = str(slot_name or '').strip().lower()
            mapping = {
                'model0': 'Model 0',
                'model1': 'Model 1',
                'model2': 'Model 2',
                'checkpoint': 'Checkpoint',
            }
            if slot in mapping:
                return mapping[slot]
            if slot == 'lora':
                if lora_index is not None and lora_index >= 0:
                    return f'LoRA {lora_index + 1}'
                return 'LoRA'
            return _planner_labelize_token(slot_name)

        def embedded_source_issue(spec, slot_name: str, *, lora_index: int | None = None):
            if not isinstance(spec, dict):
                return None
            mode = str(spec.get('mode') or '').strip().lower()
            label = slot_label(slot_name, lora_index=lora_index)
            if mode == 'download':
                link_issue = _planner_download_link_issue(self, str(spec.get('link') or '').strip(), eager=eager_link_validation)
                if link_issue:
                    return f'{label} {link_issue}'
                return None
            if mode == 'local':
                local_path = str(spec.get('local_path') or '').strip()
                if not local_path:
                    return f'{label} Local Path Is Empty'
                return None
            return None

        for idx, entry in enumerate(self.plan_data.get('entries', []), start=1):
            etype = entry.get('type')
            problems: List[str] = []
            if self._entry_is_disabled(entry):
                problems.append('WARN: Entry Is Disabled And Excluded From Export/Runtime')
            if self._entry_is_locked(entry):
                problems.append('WARN: Entry Is Locked Against Accidental Edits')

            if etype == 'Download Model':
                if not entry.get('model_name'):
                    problems.append(_planner_make_problem('ERROR', idx, etype, 'Model Name Is Empty'))
                link = str(entry.get('link') or '').strip()
                if not link:
                    problems.append(_planner_make_problem('ERROR', idx, etype, 'Link Is Empty'))
                else:
                    link_issue = _planner_download_link_issue(self, link, eager=eager_link_validation)
                    if link_issue:
                        problems.append(_planner_make_problem('ERROR', idx, etype, link_issue))
                name = (entry.get('model_name') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if name and not self._entry_is_disabled(entry):
                    available.setdefault(kind, set()).add(name)

            elif etype == 'Local Model':
                if not entry.get('local_path'):
                    problems.append(_planner_make_problem('ERROR', idx, etype, 'Local Path Is Empty'))
                path_value = (entry.get('local_path') or '').strip()
                kind = (entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if path_value and not self._entry_is_disabled(entry):
                    available.setdefault(kind, set()).add(Path(path_value).stem)

            elif etype == 'Remove Model':
                if not entry.get('model'):
                    problems.append(_planner_make_problem('ERROR', idx, etype, 'Model Is Empty'))

            elif etype == 'Checkpoint Merge':
                for req_key in ('model0', 'model1', 'output_name'):
                    if not entry.get(req_key):
                        problems.append(_planner_make_problem('ERROR', idx, etype, f'{_planner_labelize_token(req_key)} Is Empty'))
                for slot in ('model0', 'model1', 'model2'):
                    ref = str(entry.get(slot) or '').strip()
                    if not ref:
                        continue
                    spec = self._get_entry_slot_source(entry, slot)
                    issue = embedded_source_issue(spec, slot)
                    if issue:
                        problems.append(_planner_make_problem('ERROR', idx, etype, issue))
                    elif not isinstance(spec, dict) and ref not in available['Checkpoint']:
                        problems.append(_planner_make_problem('ERROR', idx, etype, f'Checkpoint Reference Not Available -> {ref}'))
                if entry.get('output_name') and not self._entry_is_disabled(entry):
                    available['Checkpoint'].add(entry['output_name'])

            elif etype == 'LoRA Bake':
                checkpoint = str(entry.get('checkpoint') or '').strip()
                checkpoint_spec = self._get_entry_slot_source(entry, 'checkpoint')
                if not checkpoint:
                    problems.append(_planner_make_problem('ERROR', idx, etype, 'Checkpoint Is Empty'))
                else:
                    issue = embedded_source_issue(checkpoint_spec, 'checkpoint')
                    if issue:
                        problems.append(_planner_make_problem('ERROR', idx, etype, issue))
                    elif not isinstance(checkpoint_spec, dict) and checkpoint not in available['Checkpoint']:
                        problems.append(_planner_make_problem('ERROR', idx, etype, f'Checkpoint Reference Not Available -> {checkpoint}'))
                if not entry.get('output_name'):
                    problems.append(_planner_make_problem('ERROR', idx, etype, 'Output Name Is Empty'))
                for lora_idx, lora in enumerate(entry.get('loras', []) or []):
                    name = str(lora.get('name') or '').strip()
                    spec = self._get_lora_source(lora)
                    slot_name = slot_label('lora', lora_index=lora_idx)
                    if not name:
                        problems.append(_planner_make_problem('ERROR', idx, etype, f'{slot_name} Name Is Empty'))
                        continue
                    issue = embedded_source_issue(spec, 'lora', lora_index=lora_idx)
                    if issue:
                        problems.append(_planner_make_problem('ERROR', idx, etype, issue))
                    elif not isinstance(spec, dict) and name not in available['LoRA'] and name not in available['LyCORIS']:
                        problems.append(_planner_make_problem('ERROR', idx, etype, f'LoRA Reference Not Available -> {name}'))
                if entry.get('output_name') and not self._entry_is_disabled(entry):
                    available['Checkpoint'].add(entry['output_name'])

            for alias in self._entry_produced_aliases(entry):
                lines = duplicate_aliases.get(alias, [])
                if len(lines) > 1:
                    labels = ', '.join(str(i + 1) for i in lines)
                    problems.append(f'WARN: Duplicate Produced Alias -> {alias} (Lines {labels})')
                if alias in unref_aliases:
                    problems.append(f'WARN: Produced Alias Is Currently Unreferenced -> {alias}')
            if idx - 1 in dead_entries:
                problems.append('WARN: Merge Output Is Not Required By The Final Active Line')
            problems_by_idx[idx - 1] = problems
        return problems_by_idx

    def _planner_problem_explanation_titleized(self, model_idx: int, problem: str) -> str:
        entry = {}
        try:
            entry = self.plan_data.get('entries', [])[model_idx]
        except Exception:
            entry = {}
        problem_text = str(problem or '').strip()
        plain = _planner_patch_re.sub(r'^(ERROR|WARN):\s*', '', problem_text, flags=_planner_patch_re.IGNORECASE)
        cause_title = 'Cause'
        cause_lines = []
        fix_lines = []
        available_before = self._collect_available_models(model_idx)

        m_empty = _planner_patch_re.search(r':\s*([A-Za-z0-9_ ]+) Is Empty\s*$', problem_text, _planner_patch_re.IGNORECASE)
        m_missing = _planner_patch_re.search(r':\s*(checkpoint|lora)\s+(?:reference|ref)\s+not\s+available\s*->\s*(.+?)\s*$', problem_text, _planner_patch_re.IGNORECASE)
        m_slot_link_missing = _planner_patch_re.search(r':\s*([A-Za-z0-9 ]+) Download Link Not Available\s*->\s*(.+?)\s*$', problem_text, _planner_patch_re.IGNORECASE)
        m_slot_link_format = _planner_patch_re.search(r':\s*([A-Za-z0-9 ]+) Download Link Format Is Invalid\s*->\s*(.+?)\s*$', problem_text, _planner_patch_re.IGNORECASE)
        m_slot_link_empty = _planner_patch_re.search(r':\s*([A-Za-z0-9 ]+) Link Is Empty\s*$', problem_text, _planner_patch_re.IGNORECASE)
        m_slot_local_empty = _planner_patch_re.search(r':\s*([A-Za-z0-9 ]+) Local Path Is Empty\s*$', problem_text, _planner_patch_re.IGNORECASE)
        m_dup = _planner_patch_re.search(r'Duplicate Produced Alias\s*->\s*(.+?)\s*\(Lines\s+(.+?)\)\s*$', problem_text, _planner_patch_re.IGNORECASE)
        m_unref = _planner_patch_re.search(r'Produced Alias Is Currently Unreferenced\s*->\s*(.+?)\s*$', problem_text, _planner_patch_re.IGNORECASE)

        if m_slot_link_empty:
            field_label = m_slot_link_empty.group(1).strip()
            if field_label.lower() == 'link':
                cause_lines.append('This line is configured to download a model, but its URL field is empty.')
                fix_lines.append('Enter a full download URL.')
            else:
                cause_lines.append(f'The {field_label} source is set to Download, but its URL field is empty.')
                fix_lines.append(f'Open the {field_label} source selector and enter a full download URL.')
        elif m_slot_link_format:
            slot_label = m_slot_link_format.group(1).strip()
            link = m_slot_link_format.group(2).strip()
            cause_lines.append(f'The {slot_label} source is set to Download, but the URL is not a valid HTTP or HTTPS link.')
            cause_lines.append(f'Current link: {link}')
            fix_lines.append(f'Enter a valid direct download URL for {slot_label}.')
        elif m_slot_link_missing:
            slot_label = m_slot_link_missing.group(1).strip()
            link = m_slot_link_missing.group(2).strip()
            cause_lines.append(f'The {slot_label} source is set to Download, and the URL was checked but returned a not-found response.')
            cause_lines.append(f'Checked link: {link}')
            fix_lines.append(f'Replace the URL for {slot_label} with a working model link.')
        elif m_slot_local_empty:
            slot_label = m_slot_local_empty.group(1).strip()
            cause_lines.append(f'The {slot_label} source is set to Local, but no local file path is stored.')
            fix_lines.append(f'Select a local model file again for {slot_label}.')
        elif m_empty:
            field_label = m_empty.group(1).strip()
            cause_lines.append(f'This line is missing the required field "{field_label}".')
            fix_lines.append(f'Fill in "{field_label}" on this line before export or notebook run.')
        elif m_missing:
            ref_kind = m_missing.group(1).strip().title()
            if ref_kind.lower() == 'lora':
                ref_kind = 'LoRA'
            missing_name = m_missing.group(2).strip()
            cause_lines.append(f'This line refers to a {ref_kind} alias named "{missing_name}", but that alias is not available before this line executes.')
            if ref_kind == 'Checkpoint':
                available_list = available_before.get('Checkpoint', [])
            else:
                available_list = sorted(set((available_before.get('LoRA', []) or []) + (available_before.get('LyCORIS', []) or [])))
            if available_list:
                cause_lines.append(f'Available {ref_kind} aliases before this line: ' + ', '.join(available_list))
            else:
                cause_lines.append(f'No {ref_kind} aliases are available before this line.')
            fix_lines.append('Make sure the referenced alias is produced by an earlier active line, and that it was not removed or disabled.')
        elif m_dup:
            alias = m_dup.group(1).strip()
            dup_lines = m_dup.group(2).strip()
            cause_lines.append(f'The output alias "{alias}" is produced by multiple active lines ({dup_lines}).')
            cause_lines.append('Later references to that alias may resolve ambiguously.')
            fix_lines.append('Rename one of the outputs so every produced alias is unique.')
        elif m_unref:
            alias = m_unref.group(1).strip()
            cause_lines.append(f'The produced alias "{alias}" is not used by any later active line.')
            fix_lines.append('Connect this alias into a later merge/bake step, or remove the unused producer if it is no longer needed.')
        elif 'Merge Output Is Not Required By The Final Active Line'.lower() in problem_text.lower():
            cause_lines.append('This line does not contribute to the final active result chain.')
            fix_lines.append('Either use this result in a later active line, or disable/remove the line if it is no longer needed.')
        elif 'Entry Is Disabled And Excluded From Export/Runtime'.lower() in problem_text.lower():
            cause_lines.append('This line is disabled, so it will not be included in export or notebook execution.')
            fix_lines.append('Re-enable the line if you want it to produce or provide references again.')
        elif 'Entry Is Locked Against Accidental Edits'.lower() in problem_text.lower():
            cause_lines.append('This line is locked to avoid accidental edits.')
            fix_lines.append('Unlock the line before trying to reorder or modify it.')
        else:
            cause_lines.append(plain)

        sections = [f'Line {model_idx + 1}: {self._line_summary(entry)}', '', 'Problem:', f'- {plain}', '', f'{cause_title}:']
        sections.extend(f'- {line}' for line in cause_lines) if cause_lines else sections.append('- No additional cause details are available.')
        if fix_lines:
            sections.extend(['', 'Suggested Fix:'])
            sections.extend(f'- {line}' for line in fix_lines)
        memo = str(entry.get('memo') or '').strip()
        if memo:
            sections.extend(['', 'Memo:', memo])
        return '\n'.join(sections).strip()
    _planner_prev_show_prevalidation_for_download_patch = getattr(ModelPlannerApp, '_show_prevalidation', None)
    def _show_prevalidation_with_download_checks(self):
        prev_flag = bool(getattr(self, '_planner_eager_link_validation', False))
        self._planner_eager_link_validation = True
        try:
            if _planner_prev_show_prevalidation_for_download_patch is not None:
                return _planner_prev_show_prevalidation_for_download_patch(self)
            problem_map = self._plan_entry_problem_map()
            issues = [(idx, p) for idx, items in problem_map.items() for p in items]
            if not issues:
                return self._show_scrollable_text_dialog('Pre-validation', 'No issues found.')
            detail = 'Pre-validation results:\n\n' + '\n'.join(f'Line {idx + 1}: {p}' for idx, p in issues)
            return self._show_scrollable_text_dialog('Pre-validation', detail)
        finally:
            self._planner_eager_link_validation = prev_flag
            try:
                self._plan_problem_map_cache = self._plan_entry_problem_map()
                self._apply_plan_listbox_item_styles()
            except Exception:
                pass

    ModelPlannerApp._plan_entry_problem_map = _plan_entry_problem_map_titleized
    ModelPlannerApp._planner_problem_explanation = _planner_problem_explanation_titleized
    ModelPlannerApp._show_prevalidation = _show_prevalidation_with_download_checks
except Exception:
    pass


# Final patch: retained local-source candidates, source-aware dependency updates, and LoRA reordering.
try:
    def _planner_norm_path_key(self, path: str) -> str:
        try:
            return os.path.normcase(os.path.abspath(os.path.expanduser(str(path or ''))))
        except Exception:
            return os.path.normcase(str(path or ''))

    def _planner_add_local_choice(self, choices: Dict[str, str], path: str, preferred_label: str = ''):
        path = self._normalize_user_path(str(path or '').strip())
        if not path:
            return
        key = _planner_norm_path_key(self, path)
        for existing in list(choices.values()):
            if _planner_norm_path_key(self, existing) == key:
                return
        base = str(preferred_label or '').strip() or Path(path).name or path
        label = base
        if label in choices and _planner_norm_path_key(self, choices[label]) != key:
            parent = Path(path).parent.name or 'local'
            label = f'{Path(path).name} [{parent}]'
        suffix = 2
        while label in choices and _planner_norm_path_key(self, choices[label]) != key:
            label = f'{Path(path).name} [local #{suffix}]'
            suffix += 1
        choices[label] = path

    _planner_prev_scan_local_model_choices = getattr(ModelPlannerApp, '_scan_local_model_choices', None)
    def _scan_local_model_choices_retained(self) -> Dict[str, str]:
        choices: Dict[str, str] = {}
        if _planner_prev_scan_local_model_choices is not None:
            try:
                for label, path in (_planner_prev_scan_local_model_choices(self) or {}).items():
                    _planner_add_local_choice(self, choices, path, str(label or ''))
            except Exception:
                pass
        for entry in (self.plan_data or {}).get('entries', []) or []:
            if not isinstance(entry, dict):
                continue
            if entry.get('type') == 'Local Model':
                _planner_add_local_choice(self, choices, entry.get('local_path') or '')
            try:
                for _slot, alias, _kind, spec in self._iter_embedded_sources(entry):
                    if isinstance(spec, dict) and str(spec.get('mode') or '').strip().lower() == 'local':
                        path = str(spec.get('local_path') or '').strip()
                        label = f'{alias} ({Path(path).name})' if alias and path else ''
                        _planner_add_local_choice(self, choices, path, label)
            except Exception:
                pass
        return choices

    ModelPlannerApp._scan_local_model_choices = _scan_local_model_choices_retained

    def _planner_display_for_local_path(self, choices: Dict[str, str], path: str) -> str:
        key = _planner_norm_path_key(self, path)
        for label, candidate in (choices or {}).items():
            if _planner_norm_path_key(self, candidate) == key:
                return label
        return Path(str(path or '')).name if str(path or '').strip() else ''

    def _build_local_selection_row_retained(self, parent, entry: Dict[str, Any]):
        row = Frame(parent)
        row.pack(fill='x', pady=3)
        label_widget = Label(row, text='Local Selection', width=18, anchor='w')
        label_widget.pack(side='left')
        choices_map = self._scan_local_model_choices()
        current_path = self._normalize_user_path(str(entry.get('local_path', '') or '').strip())
        current_display = _planner_display_for_local_path(self, choices_map, current_path) if current_path else ''
        if current_path and current_display and current_display not in choices_map:
            _planner_add_local_choice(self, choices_map, current_path, current_display)
        values = list(choices_map.keys())
        var = tk.StringVar(value=current_display if current_display else (values[0] if values else ''))
        combo = ttk.Combobox(row, textvariable=var, values=values, state='readonly')
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side='left', fill='x', expand=True, padx=4)
        def sync(*_args):
            display = var.get().strip()
            selected_path = choices_map.get(display)
            if selected_path:
                entry['local_path'] = selected_path
                if not str(entry.get('model_name') or '').strip():
                    entry['model_name'] = Path(selected_path).stem
            elif current_path and display == current_display:
                entry['local_path'] = current_path
            elif display:
                entry['local_path'] = display
            self._after_entry_change()
        var.trace_add('write', sync)
        self._attach_tooltip(label_widget, self._right_help('Local Selection').get('detail', ''))
        self._add_right_inline_help(parent, 'Local Selection')
        return var

    ModelPlannerApp._build_local_selection_row = _build_local_selection_row_retained

    def _prompt_local_source_dialog_retained(self, label: str, kind_options, current_source=None):
        source = current_source if isinstance(current_source, dict) else {}
        options = [str(x) for x in (kind_options or []) if str(x).strip()] or ['Checkpoint']
        initial_kind = self._normalize_source_kind(str(source.get('kind') or options[0]), options)
        choices_map = self._scan_local_model_choices()
        current_path = self._normalize_user_path(str(source.get('local_path') or '').strip())
        current_display = _planner_display_for_local_path(self, choices_map, current_path) if current_path else ''
        if current_path and current_display and current_display not in choices_map:
            _planner_add_local_choice(self, choices_map, current_path, current_display)
        values = list(choices_map.keys())
        result = {'ok': False, 'spec': None}
        win = tk.Toplevel(self.root)
        win.title(f'{label} Local')
        win.geometry('760x260+160+160')
        win.transient(self.root)
        outer = Frame(win, padx=10, pady=10)
        outer.pack(fill='both', expand=True)
        Label(outer, text='Select a previously used local model, or browse for a new file.', anchor='w').pack(fill='x', pady=(0, 8))
        row_kind = Frame(outer); row_kind.pack(fill='x', pady=3)
        Label(row_kind, text='Type', width=16, anchor='w').pack(side='left')
        kind_var = tk.StringVar(value=initial_kind)
        kind_combo = ttk.Combobox(row_kind, textvariable=kind_var, values=options, state='readonly')
        kind_combo.pack(side='left', fill='x', expand=True, padx=4)
        row_choice = Frame(outer); row_choice.pack(fill='x', pady=3)
        Label(row_choice, text='Known Local', width=16, anchor='w').pack(side='left')
        choice_var = tk.StringVar(value=current_display if current_display else (values[0] if values else ''))
        choice_combo = ttk.Combobox(row_choice, textvariable=choice_var, values=values, state='readonly')
        choice_combo.pack(side='left', fill='x', expand=True, padx=4)
        row_path = Frame(outer); row_path.pack(fill='x', pady=3)
        Label(row_path, text='Local Path', width=16, anchor='w').pack(side='left')
        path_var = tk.StringVar(value=current_path or choices_map.get(choice_var.get(), ''))
        path_entry = Entry(row_path, textvariable=path_var)
        path_entry.pack(side='left', fill='x', expand=True, padx=4)
        def browse():
            path = filedialog.askopenfilename(
                title=f'Select local {kind_var.get() or "model"}',
                filetypes=[('Model Files', '*.safetensors *.ckpt *.pt *.bin'), ('All files', '*.*')],
            )
            path = self._normalize_user_path(path)
            if path:
                path_var.set(path)
                alias_var.set(Path(path).stem)
        ttk.Button(row_path, text='📂', width=3, command=browse).pack(side='left')
        row_alias = Frame(outer); row_alias.pack(fill='x', pady=3)
        Label(row_alias, text='Alias', width=16, anchor='w').pack(side='left')
        alias_default = str(source.get('alias') or '').strip() or (Path(current_path).stem if current_path else '')
        alias_var = tk.StringVar(value=alias_default)
        alias_entry = Entry(row_alias, textvariable=alias_var)
        alias_entry.pack(side='left', fill='x', expand=True, padx=4)
        def on_choice(*_args):
            path = choices_map.get(choice_var.get().strip())
            if path:
                path_var.set(path)
                if not alias_var.get().strip():
                    alias_var.set(Path(path).stem)
        choice_var.trace_add('write', on_choice)
        def ok():
            path = self._normalize_user_path(path_var.get().strip())
            if not path:
                messagebox.showwarning(f'{label} Local', 'Local path is empty.', parent=win)
                return
            alias = alias_var.get().strip() or Path(path).stem
            result['ok'] = True
            result['spec'] = {'mode': 'local', 'kind': self._normalize_source_kind(kind_var.get(), options), 'alias': alias, 'local_path': path}
            win.destroy()
        def cancel():
            win.destroy()
        btns = Frame(outer); btns.pack(fill='x', pady=(10, 0))
        ttk.Button(btns, text='OK', command=ok).pack(side='left')
        ttk.Button(btns, text='Cancel', command=cancel).pack(side='left', padx=(6, 0))
        try:
            colors = self._theme_colors(); win.configure(bg=colors['bg']); self._apply_theme_to_children(win, colors)
        except Exception:
            pass
        try:
            win.grab_set()
            alias_entry.focus_set()
            self.root.wait_window(win)
        except Exception:
            pass
        return result['spec'] if result.get('ok') else None

    ModelPlannerApp._prompt_local_source_dialog = _prompt_local_source_dialog_retained

    def _entry_produced_aliases_source_aware(self, entry: Dict[str, Any]) -> List[str]:
        if self._entry_is_disabled(entry):
            return []
        etype = entry.get('type')
        produced = []
        if etype == 'Download Model':
            name = str(entry.get('model_name') or '').strip()
            if name:
                produced.append(name)
        elif etype == 'Local Model':
            path = str(entry.get('local_path') or '').strip()
            name = str(entry.get('model_name') or '').strip() or (Path(path).stem if path else '')
            if name:
                produced.append(name)
        elif etype in ('Checkpoint Merge', 'LoRA Bake'):
            name = str(entry.get('output_name') or '').strip()
            if name:
                produced.append(name)
        return produced

    ModelPlannerApp._entry_produced_aliases = _entry_produced_aliases_source_aware

    def _collect_available_models_source_aware(self, upto_index: int) -> Dict[str, List[str]]:
        available: Dict[str, Dict[str, str]] = {'Checkpoint': {}, 'LoRA': {}, 'LyCORIS': {}}
        for entry in (self.plan_data or {}).get('entries', [])[:upto_index]:
            if self._entry_is_disabled(entry):
                continue
            etype = entry.get('type')
            if etype == 'Remove Model':
                name = str(entry.get('model') or '').strip()
                if name:
                    for bucket in available.values():
                        bucket.pop(name, None)
                continue
            try:
                for _slot, alias, kind, _spec in self._iter_embedded_sources(entry):
                    alias = str(alias or '').strip()
                    kind = str(kind or 'Checkpoint').strip() or 'Checkpoint'
                    if alias:
                        available.setdefault(kind, {})[alias] = kind
            except Exception:
                pass
            if etype == 'Download Model':
                name = str(entry.get('model_name') or '').strip()
                kind = str(entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if name:
                    available.setdefault(kind, {})[name] = kind
            elif etype == 'Local Model':
                path = str(entry.get('local_path') or '').strip()
                name = str(entry.get('model_name') or '').strip() or (Path(path).stem if path else '')
                kind = str(entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if name:
                    available.setdefault(kind, {})[name] = kind
            elif etype in ('Checkpoint Merge', 'LoRA Bake'):
                name = str(entry.get('output_name') or '').strip()
                if name:
                    available['Checkpoint'][name] = 'Checkpoint'
        return {k: sorted(v.keys()) for k, v in available.items()}

    ModelPlannerApp._collect_available_models = _collect_available_models_source_aware

    def _planner_source_signature(self, mode: str, value: str) -> str:
        mode = str(mode or '').strip().lower()
        value = str(value or '').strip()
        if mode == 'local':
            return 'local:' + _planner_norm_path_key(self, value)
        if mode == 'download':
            return 'download:' + re.sub(r'#.*$', '', value)
        return mode + ':' + value

    def _planner_source_records(self):
        records = []
        for idx, entry in enumerate((self.plan_data or {}).get('entries', []) or []):
            if not isinstance(entry, dict) or self._entry_is_disabled(entry):
                continue
            etype = entry.get('type')
            if etype == 'Download Model':
                alias = str(entry.get('model_name') or '').strip()
                link = str(entry.get('link') or '').strip()
                kind = str(entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if alias and link:
                    records.append({'idx': idx, 'alias': alias, 'kind': kind, 'mode': 'download', 'signature': _planner_source_signature(self, 'download', link), 'detail': link, 'embedded': False})
            elif etype == 'Local Model':
                path = str(entry.get('local_path') or '').strip()
                alias = str(entry.get('model_name') or '').strip() or (Path(path).stem if path else '')
                kind = str(entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if alias and path:
                    records.append({'idx': idx, 'alias': alias, 'kind': kind, 'mode': 'local', 'signature': _planner_source_signature(self, 'local', path), 'detail': path, 'embedded': False})
            try:
                for slot, alias, kind, spec in self._iter_embedded_sources(entry):
                    if not isinstance(spec, dict):
                        continue
                    mode = str(spec.get('mode') or '').strip().lower()
                    if mode == 'local':
                        detail = str(spec.get('local_path') or '').strip()
                    elif mode == 'download':
                        detail = str(spec.get('link') or '').strip()
                    else:
                        detail = ''
                    alias = str(alias or '').strip()
                    if alias and detail:
                        records.append({'idx': idx, 'alias': alias, 'kind': str(kind or spec.get('kind') or 'Checkpoint'), 'mode': mode, 'signature': _planner_source_signature(self, mode, detail), 'detail': detail, 'slot': slot, 'embedded': True})
            except Exception:
                pass
        return records

    ModelPlannerApp._planner_source_records = _planner_source_records

    def _planner_analysis_source_aware(self):
        entries = (self.plan_data or {}).get('entries', []) or []
        produced_by_entry = {idx: self._entry_produced_aliases(entry) for idx, entry in enumerate(entries)}
        consumed_by_entry = {idx: self._entry_consumed_aliases(entry) for idx, entry in enumerate(entries)}
        source_records = self._planner_source_records()
        embedded_by_idx = {}
        for rec in source_records:
            if rec.get('embedded'):
                embedded_by_idx.setdefault(rec['idx'], []).append(rec)
        producer_lines = {}
        producer_by_alias = {}
        edge_by_entry = {}
        missing_links = []
        active = {}
        for idx, entry in enumerate(entries):
            if self._entry_is_disabled(entry):
                continue
            etype = entry.get('type')
            if etype == 'Remove Model':
                name = str(entry.get('model') or '').strip()
                if name:
                    active.pop(name, None)
                continue
            for rec in embedded_by_idx.get(idx, []):
                alias = rec.get('alias')
                if alias:
                    active[alias] = idx
                    producer_lines.setdefault(alias, []).append(idx)
            edge_by_entry[idx] = {}
            for alias in consumed_by_entry.get(idx, []):
                src = active.get(alias)
                edge_by_entry[idx][alias] = src
                if src is None:
                    missing_links.append((idx, alias))
            for alias in produced_by_entry.get(idx, []):
                active[alias] = idx
                producer_lines.setdefault(alias, []).append(idx)
        producer_by_alias.update(active)
        final_idx = None
        for idx in range(len(entries) - 1, -1, -1):
            if self._entry_is_disabled(entries[idx]):
                continue
            if entries[idx].get('type') != 'Remove Model':
                final_idx = idx
                break
        protected_aliases = set()
        needed_aliases = set()
        needed_entries = set()
        visited_entries = set()
        def visit_entry(idx):
            if idx is None or idx in visited_entries or not (0 <= idx < len(entries)):
                return
            visited_entries.add(idx)
            entry = entries[idx]
            if entry.get('type') in ('Checkpoint Merge', 'LoRA Bake'):
                needed_entries.add(idx)
            for alias in produced_by_entry.get(idx, []):
                if alias:
                    needed_aliases.add(alias)
            for alias in consumed_by_entry.get(idx, []):
                if not alias:
                    continue
                needed_aliases.add(alias)
                src = edge_by_entry.get(idx, {}).get(alias)
                if src is not None and src != idx:
                    visit_entry(src)
        if final_idx is not None:
            protected_aliases.update(produced_by_entry.get(final_idx, []))
            protected_aliases.update(consumed_by_entry.get(final_idx, []))
            visit_entry(final_idx)
        dead_entries = set()
        for idx, entry in enumerate(entries):
            if self._entry_is_disabled(entry):
                continue
            if entry.get('type') in ('Checkpoint Merge', 'LoRA Bake') and idx != final_idx and idx not in needed_entries:
                dead_entries.add(idx)
        consumed_global = set()
        for vals in consumed_by_entry.values():
            consumed_global.update(vals)
        unreferenced_aliases = {}
        for alias, lines in producer_lines.items():
            if alias not in consumed_global and alias not in protected_aliases:
                unreferenced_aliases[alias] = list(dict.fromkeys(lines))
        by_signature = {}
        by_alias = {}
        for rec in source_records:
            if rec.get('signature'):
                by_signature.setdefault(rec['signature'], []).append(rec)
            if rec.get('alias'):
                by_alias.setdefault(rec['alias'], []).append(rec)
        same_source_aliases = {}
        for sig, recs in by_signature.items():
            aliases = sorted({r['alias'] for r in recs if r.get('alias')})
            if len(aliases) > 1:
                same_source_aliases[sig] = recs
        alias_updates = {}
        for alias, recs in by_alias.items():
            sigs = {r.get('signature') for r in recs if r.get('signature')}
            if len(sigs) > 1:
                alias_updates[alias] = recs
        return {
            'final_index': final_idx,
            'protected_aliases': protected_aliases,
            'needed_aliases': needed_aliases,
            'needed_entries': needed_entries,
            'dead_entries': dead_entries,
            'producer_by_alias': producer_by_alias,
            'producer_lines': producer_lines,
            'produced_by_entry': produced_by_entry,
            'consumed_by_entry': consumed_by_entry,
            'edge_by_entry': edge_by_entry,
            'missing_links': missing_links,
            'duplicate_aliases': {},
            'unreferenced_aliases': unreferenced_aliases,
            'source_records': source_records,
            'same_source_aliases': same_source_aliases,
            'alias_updates': alias_updates,
        }

    ModelPlannerApp._planner_analysis = _planner_analysis_source_aware

    def _show_dependency_view_source_aware(self):
        analysis = self._planner_analysis()
        lines = ['Dependency View', '']
        alias_updates = analysis.get('alias_updates') or {}
        if alias_updates:
            lines += ['Alias updates (same external name, different internal source):']
            for alias, recs in sorted(alias_updates.items()):
                parts = []
                for rec in recs:
                    detail = rec.get('detail') or ''
                    short = Path(detail).name if rec.get('mode') == 'local' else detail[:64] + ('…' if len(detail) > 64 else '')
                    parts.append(f"line {rec['idx'] + 1} [{rec.get('mode')}] {short}")
                lines.append(f'  🔁 {alias}: ' + '  ->  '.join(parts))
            lines.append('  Later lines only see the newest registration for that alias.')
            lines.append('')
        same_source = analysis.get('same_source_aliases') or {}
        if same_source:
            lines += ['Same source with different aliases:']
            for _sig, recs in sorted(same_source.items()):
                aliases = []
                for rec in recs:
                    aliases.append(f"{rec.get('alias')} (line {rec['idx'] + 1})")
                detail = recs[0].get('detail') or ''
                detail_short = detail if len(detail) <= 120 else detail[:117] + '…'
                lines.append('  ⚠ ' + ', '.join(dict.fromkeys(aliases)))
                lines.append(f'    source: {detail_short}')
            lines.append('')
        lines += ['Adjacency:', '']
        edge_by_entry = analysis.get('edge_by_entry') or {}
        for idx, entry in enumerate((self.plan_data or {}).get('entries', []) or []):
            tag = []
            if self._entry_is_locked(entry):
                tag.append('locked')
            if self._entry_is_disabled(entry):
                tag.append('disabled')
            lines.append(f'[{idx + 1}] {self._line_summary(entry)}' + (f'  [{", ".join(tag)}]' if tag else ''))
            embedded = [alias for _slot, alias, _kind, _spec in self._iter_embedded_sources(entry)]
            produced = self._entry_produced_aliases(entry)
            consumes = self._entry_consumed_aliases(entry)
            if embedded:
                lines.append('  embedded sources: ' + ', '.join(embedded))
            if produced:
                lines.append('  produces/updates: ' + ', '.join(produced))
            if consumes:
                detail = []
                for alias in consumes:
                    src = edge_by_entry.get(idx, {}).get(alias)
                    if src is None:
                        detail.append(f'{alias} <- external/MISSING')
                    elif src == idx:
                        detail.append(f'{alias} <- embedded source on this line')
                    else:
                        detail.append(f'{alias} <- line {src + 1}')
                lines.append('  consumes: ' + '; '.join(detail))
            if idx in analysis.get('dead_entries', set()):
                lines.append('  status: DEAD / not required by final active line')
            lines.append('')
        lines += ['', 'Mermaid-like graph:', 'graph TD']
        for idx, entry in enumerate((self.plan_data or {}).get('entries', []) or []):
            label = self._line_summary(entry).replace('"', "'")
            lines.append(f'  L{idx + 1}["{idx + 1}: {label}"]')
        for idx, consumes in (analysis.get('consumed_by_entry') or {}).items():
            for alias in consumes:
                src = edge_by_entry.get(idx, {}).get(alias)
                safe = re.sub(r'[^A-Za-z0-9_]', '_', alias)
                if src is None:
                    lines.append(f'  EXT_{safe}(["external:{alias}"]) -->|{alias}| L{idx + 1}')
                elif src == idx:
                    lines.append(f'  EMB_{idx + 1}_{safe}(["embedded:{alias}"]) -->|{alias}| L{idx + 1}')
                else:
                    lines.append(f'  L{src + 1} -->|{alias}| L{idx + 1}')
        self._show_scrollable_text_dialog('Dependency View', '\n'.join(lines))

    ModelPlannerApp._show_dependency_view = _show_dependency_view_source_aware

    def _make_effective_source_entry_alias_preserving(self, spec, alias: str):
        mode = str(spec.get('mode') or '').strip()
        kind = str(spec.get('kind') or 'Checkpoint').strip() or 'Checkpoint'
        alias = str(alias or self._source_alias_from_spec(spec)).strip()
        if mode == 'download':
            item = make_entry('Download Model')
            item['model_name'] = alias
            item['model_type'] = kind
            item['link'] = str(spec.get('link') or '').strip()
            return item
        item = make_entry('Local Model')
        item['model_type'] = kind
        item['model_name'] = alias
        item['local_path'] = str(spec.get('local_path') or '').strip()
        return item

    ModelPlannerApp._make_effective_source_entry = _make_effective_source_entry_alias_preserving

    def _planner_rerender_preserve_scroll(self):
        canvas = getattr(self, 'canvas', None)
        yview = None
        try:
            if canvas is not None and canvas.winfo_exists():
                yview = canvas.yview()
        except Exception:
            yview = None
        self._render_current_line()
        self._refresh_line_selector()
        if yview is not None:
            def _restore(c=canvas, view=yview):
                try:
                    if c is not None and c.winfo_exists():
                        c.update_idletasks()
                        c.yview_moveto(float(view[0]))
                except Exception:
                    pass
            self.root.after_idle(_restore)

    def _add_lora_block_preserve_scroll(self, entry: Dict[str, Any]):
        entry.setdefault('loras', []).append({'name': '', 'ratio': default_ratio('Single')})
        self._after_entry_change()
        _planner_rerender_preserve_scroll(self)

    def _remove_lora_block_preserve_scroll(self, entry: Dict[str, Any], index: int):
        loras = entry.setdefault('loras', [])
        if 0 <= index < len(loras):
            loras.pop(index)
            self._after_entry_change()
            _planner_rerender_preserve_scroll(self)

    def _move_lora_block(self, entry: Dict[str, Any], index: int, delta: int):
        loras = entry.setdefault('loras', [])
        new_index = index + delta
        if not (0 <= index < len(loras) and 0 <= new_index < len(loras)):
            return
        loras[index], loras[new_index] = loras[new_index], loras[index]
        self._after_entry_change()
        _planner_rerender_preserve_scroll(self)

    ModelPlannerApp._add_lora_block = _add_lora_block_preserve_scroll
    ModelPlannerApp._remove_lora_block = _remove_lora_block_preserve_scroll
    ModelPlannerApp._move_lora_block = _move_lora_block

    def _render_lora_bake_entry_reorderable(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, 'LoRA Bake')
        models = self._collect_available_models(self.current_index)
        ckpts = models.get('Checkpoint', [])
        self._build_source_backed_ref_row(
            frame,
            'Checkpoint',
            lambda e=entry: e.get('checkpoint', ''),
            lambda value, e=entry: e.__setitem__('checkpoint', value),
            lambda e=entry: self._get_entry_slot_source(e, 'checkpoint'),
            lambda spec, e=entry: self._set_entry_slot_source(e, 'checkpoint', spec),
            ckpts or [''],
            ['Checkpoint'],
            allow_local=True,
        )
        self._build_entry_row(frame, 'Output Name', entry, 'output_name')
        add_lora_btn = ttk.Button(frame, text='+ Add LoRA', command=lambda: self._add_lora_block(entry))
        add_lora_btn.pack(anchor='w', padx=4, pady=4)
        self._attach_tooltip(add_lora_btn, self._right_help('+ Add LoRA').get('detail', ''))
        self._add_right_inline_help(frame, '+ Add LoRA')
        lora_names = models.get('LoRA', []) + models.get('LyCORIS', [])
        loras = entry.get('loras', []) or []
        for idx, lora in enumerate(loras):
            block = self._build_labeled_frame(parent, f'LoRA {idx + 1}')
            top = Frame(block)
            top.pack(fill='x', pady=(0, 4))
            ttk.Button(top, text='-', width=3, command=lambda i=idx: self._remove_lora_block(entry, i)).pack(side='left', padx=(0, 4))
            up_btn = ttk.Button(top, text='↑ Up', width=7, command=lambda i=idx: self._move_lora_block(entry, i, -1))
            down_btn = ttk.Button(top, text='↓ Down', width=8, command=lambda i=idx: self._move_lora_block(entry, i, 1))
            if idx <= 0:
                up_btn.state(['disabled'])
            if idx >= len(loras) - 1:
                down_btn.state(['disabled'])
            up_btn.pack(side='left', padx=(0, 4))
            down_btn.pack(side='left', padx=(0, 8))
            Label(top, text=f'LoRA {idx + 1} order controls', anchor='w', fg='#666666').pack(side='left')
            self._build_source_backed_ref_row(
                block,
                'LoRA Name',
                lambda lo=lora: lo.get('name', ''),
                lambda value, lo=lora: lo.__setitem__('name', value),
                lambda lo=lora: self._get_lora_source(lo),
                lambda spec, lo=lora: self._set_lora_source(lo, spec),
                lora_names or [''],
                ['LoRA', 'LyCORIS'],
                allow_local=True,
            )
            ratio = lora.setdefault('ratio', default_ratio('Single'))
            ratio_wrap = self._build_collapsible_section(
                block,
                'Ratio',
                key=f"lora_ratio_{entry.get('id', '')}_{idx}",
                default_open=True,
                body_fill='x',
                body_expand=False,
                padx=0,
                pady=4,
            )
            row2 = Frame(ratio_wrap)
            row2.pack(fill='x', pady=3)
            Label(row2, text='Ratio Mode', width=18, anchor='w').pack(side='left')
            ratio_var = tk.StringVar(value=ratio.get('mode', 'Single'))
            combo2 = ttk.Combobox(row2, textvariable=ratio_var, values=['Single', 'Elemental'], state='readonly')
            self._bind_combobox_mousewheel_passthrough(combo2)
            combo2.pack(side='left', fill='x', expand=True, padx=4)
            def on_ratio_mode(*_args, lo=lora, rv=ratio_var):
                lo['ratio']['mode'] = rv.get()
                if rv.get() == 'Single':
                    lo['ratio'].setdefault('value', '1.0')
                self._schedule_rerender_current_line()
            ratio_var.trace_add('write', on_ratio_mode)
            self._build_ratio_value_widget(ratio_wrap, lora['ratio'], allow_block_weight=False)
        self._build_text_row(parent, 'Additional Signatures', entry, 'raw_signatures', height=5)

    ModelPlannerApp._render_lora_bake_entry = _render_lora_bake_entry_reorderable


    def _plan_entry_problem_map_source_aware(self) -> Dict[int, List[str]]:
        problems_by_idx: Dict[int, List[str]] = {}
        entries = (self.plan_data or {}).get('entries', []) or []
        analysis = self._planner_analysis()
        dead_entries = set(analysis.get('dead_entries', set()))
        unref_aliases = analysis.get('unreferenced_aliases', {}) or {}
        same_source_by_idx = {}
        for _sig, recs in (analysis.get('same_source_aliases') or {}).items():
            aliases = sorted({r.get('alias') for r in recs if r.get('alias')})
            if len(aliases) <= 1:
                continue
            text = 'WARN: Same Source Registered With Different Aliases -> ' + ', '.join(aliases)
            for rec in recs:
                same_source_by_idx.setdefault(rec.get('idx'), set()).add(text)
        eager_link_validation = bool(getattr(self, '_planner_eager_link_validation', False))
        def make_problem(level: str, line_idx: int, etype: str, message: str) -> str:
            return f'{str(level or "ERROR").upper()}: Line {line_idx} ({etype}): {message}'
        def link_issue(link: str):
            fn = globals().get('_planner_download_link_issue')
            if callable(fn):
                try:
                    return fn(self, str(link or '').strip(), eager=eager_link_validation)
                except Exception:
                    pass
            text = str(link or '').strip()
            if not text:
                return 'Link Is Empty'
            if not re.match(r'^https?://', text, flags=re.IGNORECASE):
                return f'Download Link Format Is Invalid -> {text}'
            return None
        def embedded_issue(spec, label: str):
            if not isinstance(spec, dict):
                return None
            mode = str(spec.get('mode') or '').strip().lower()
            if mode == 'download':
                issue = link_issue(str(spec.get('link') or '').strip())
                return f'{label} {issue}' if issue else None
            if mode == 'local':
                if not str(spec.get('local_path') or '').strip():
                    return f'{label} Local Path Is Empty'
            return None
        for zero_idx, entry in enumerate(entries):
            idx = zero_idx + 1
            etype = entry.get('type')
            problems: List[str] = []
            if self._entry_is_disabled(entry):
                problems.append('WARN: Entry Is Disabled And Excluded From Export/Runtime')
            if self._entry_is_locked(entry):
                problems.append('WARN: Entry Is Locked Against Accidental Edits')
            available = self._collect_available_models(zero_idx)
            if etype == 'Download Model':
                if not entry.get('model_name'):
                    problems.append(make_problem('ERROR', idx, etype, 'Model Name Is Empty'))
                issue = link_issue(str(entry.get('link') or '').strip())
                if issue:
                    problems.append(make_problem('ERROR', idx, etype, issue))
            elif etype == 'Local Model':
                if not entry.get('local_path'):
                    problems.append(make_problem('ERROR', idx, etype, 'Local Path Is Empty'))
            elif etype == 'Remove Model':
                if not entry.get('model'):
                    problems.append(make_problem('ERROR', idx, etype, 'Model Is Empty'))
            elif etype == 'Checkpoint Merge':
                for req_key, label in (('model0', 'Model 0'), ('model1', 'Model 1'), ('output_name', 'Output Name')):
                    if not entry.get(req_key):
                        problems.append(make_problem('ERROR', idx, etype, f'{label} Is Empty'))
                for slot, label in (('model0', 'Model 0'), ('model1', 'Model 1'), ('model2', 'Model 2')):
                    ref = str(entry.get(slot) or '').strip()
                    if not ref:
                        continue
                    spec = self._get_entry_slot_source(entry, slot)
                    issue = embedded_issue(spec, label)
                    if issue:
                        problems.append(make_problem('ERROR', idx, etype, issue))
                    elif not isinstance(spec, dict) and ref not in available.get('Checkpoint', []):
                        problems.append(make_problem('ERROR', idx, etype, f'Checkpoint Reference Not Available -> {ref}'))
            elif etype == 'LoRA Bake':
                checkpoint = str(entry.get('checkpoint') or '').strip()
                checkpoint_spec = self._get_entry_slot_source(entry, 'checkpoint')
                if not checkpoint:
                    problems.append(make_problem('ERROR', idx, etype, 'Checkpoint Is Empty'))
                else:
                    issue = embedded_issue(checkpoint_spec, 'Checkpoint')
                    if issue:
                        problems.append(make_problem('ERROR', idx, etype, issue))
                    elif not isinstance(checkpoint_spec, dict) and checkpoint not in available.get('Checkpoint', []):
                        problems.append(make_problem('ERROR', idx, etype, f'Checkpoint Reference Not Available -> {checkpoint}'))
                if not entry.get('output_name'):
                    problems.append(make_problem('ERROR', idx, etype, 'Output Name Is Empty'))
                for lora_idx, lora in enumerate(entry.get('loras', []) or []):
                    name = str(lora.get('name') or '').strip()
                    slot_label = f'LoRA {lora_idx + 1}'
                    if not name:
                        problems.append(make_problem('ERROR', idx, etype, f'{slot_label} Name Is Empty'))
                        continue
                    spec = self._get_lora_source(lora)
                    issue = embedded_issue(spec, slot_label)
                    if issue:
                        problems.append(make_problem('ERROR', idx, etype, issue))
                    elif not isinstance(spec, dict) and name not in available.get('LoRA', []) and name not in available.get('LyCORIS', []):
                        problems.append(make_problem('ERROR', idx, etype, f'LoRA Reference Not Available -> {name}'))
            for alias in self._entry_produced_aliases(entry):
                if alias in unref_aliases:
                    problems.append(f'WARN: Produced Alias Is Currently Unreferenced -> {alias}')
            if zero_idx in same_source_by_idx:
                problems.extend(sorted(same_source_by_idx.get(zero_idx) or []))
            if zero_idx in dead_entries:
                problems.append('WARN: Merge Output Is Not Required By The Final Active Line')
            problems_by_idx[zero_idx] = problems
        return problems_by_idx

    ModelPlannerApp._plan_entry_problem_map = _plan_entry_problem_map_source_aware
except Exception:
    pass

# ---------------- Developer / Experiment Mode extensions ----------------
try:
    _DEV_DEFAULT_T2I_SETTINGS = {
        "prompts": [
            {"name": "default", "prompt": "masterpiece, best quality, scenery", "negative": "lowres, bad anatomy, watermark"}
        ],
        "seed": -1,
        "steps": 20,
        "width": 768,
        "height": 1152,
        "cfg": 4.5,
        "num_gen": 1,
    }

    _DEV_DEFAULT_SHORTCUTS = {
        "undo": ["<Control-z>", "<Command-z>"],
        "redo": ["<Control-y>", "<Command-y>", "<Control-Z>", "<Command-Z>"],
        "copy_lines": ["<Control-c>", "<Command-c>"],
        "paste_lines": ["<Control-v>", "<Command-v>"],
        "delete_lines": ["<Delete>", "<Control-BackSpace>", "<Command-BackSpace>"],
        "duplicate_lines": ["<Control-d>", "<Command-d>"],
        "add_line": ["<Control-plus>", "<Command-plus>", "<Control-asterisk>", "<Command-asterisk>"],
        "move_up": ["<Shift-Up>"],
        "move_down": ["<Shift-Down>"],
        "validate": ["<Control-V>", "<Command-V>"],
        "save_preset": ["<Control-Alt-s>", "<Command-Option-s>"],
        "load_preset": ["<Control-Alt-l>", "<Command-Option-l>"],
        "run_notebook": ["<Control-Return>", "<Command-Return>"],
        "save_plan": ["<Control-s>", "<Command-s>"],
        "export_notebook": ["<Control-Shift-E>", "<Command-Shift-E>"],
        "diff": ["<Control-Shift-colon>", "<Command-Shift-colon>"],
        "dependency_graph": ["<Control-Shift-G>", "<Command-Shift-G>"],
        "optimize_plan": ["<Control-Shift-O>", "<Command-Shift-O>"],
        "t2i_settings": ["<Control-Shift-T>", "<Command-Shift-T>"],
        "ratio_sweep": ["<Control-Shift-S>", "<Command-Shift-S>"],
        "shortcut_settings": ["<Control-Shift-K>", "<Command-Shift-K>"],
    }

    def _planner_dev_ensure_config(self):
        cfg = getattr(self, "config", None)
        if not isinstance(cfg, dict):
            return
        cfg.setdefault("dev_mode", False)
        cfg.setdefault("t2i_settings", copy.deepcopy(_DEV_DEFAULT_T2I_SETTINGS))
        cfg.setdefault("shortcut_bindings", copy.deepcopy(_DEV_DEFAULT_SHORTCUTS))
        cfg.setdefault("alpha_presets", {})
        cfg.setdefault("experiment_versions", [])
        cfg.setdefault("final_memo", "")

    def _planner_dev_mode_enabled(self) -> bool:
        var = getattr(self, "dev_mode_var", None)
        try:
            return bool(var.get())
        except Exception:
            return bool(getattr(self, "config", {}).get("dev_mode", False))

    def _planner_dev_only(self, func):
        if not self._planner_dev_mode_enabled():
            messagebox.showinfo("Developer Mode", "Enable Developer (Experiment) Mode first.")
            return None
        return func()

    def _planner_get_t2i_settings(self):
        self._planner_dev_ensure_config()
        settings = copy.deepcopy(getattr(self, "config", {}).get("t2i_settings") or {})
        if not isinstance(settings, dict):
            settings = {}
        merged = copy.deepcopy(_DEV_DEFAULT_T2I_SETTINGS)
        merged.update(settings)
        prompts = merged.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            prompts = copy.deepcopy(_DEV_DEFAULT_T2I_SETTINGS["prompts"])
        merged["prompts"] = [p for p in prompts if isinstance(p, dict)] or copy.deepcopy(_DEV_DEFAULT_T2I_SETTINGS["prompts"])
        return merged

    def _planner_set_t2i_settings(self, settings):
        if not isinstance(settings, dict):
            return
        self.config["t2i_settings"] = copy.deepcopy(settings)
        self._schedule_config_save()

    def _planner_json_dump_current_plan(self):
        payload = copy.deepcopy(self.plan_data if isinstance(getattr(self, "plan_data", None), dict) else {"entries": []})
        payload["final_memo"] = str(getattr(self, "config", {}).get("final_memo", "") or "")
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _planner_save_version_snapshot(self, label: str = ""):
        self._planner_dev_ensure_config()
        versions = self.config.setdefault("experiment_versions", [])
        if not isinstance(versions, list):
            versions = []
            self.config["experiment_versions"] = versions
        payload = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": str(label or "snapshot").strip() or "snapshot",
            "base_model": self.base_model_var.get() if hasattr(self, "base_model_var") else "",
            "final_memo": str(self.config.get("final_memo", "") or ""),
            "plan_data": copy.deepcopy(self.plan_data),
        }
        try:
            import hashlib
            payload["plan_hash"] = hashlib.sha256(json.dumps(payload["plan_data"], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        except Exception:
            payload["plan_hash"] = ""
        versions.append(payload)
        if len(versions) > 200:
            del versions[:-200]
        self._schedule_config_save()
        return payload

    def _planner_show_versions(self):
        self._planner_dev_ensure_config()
        win = tk.Toplevel(self.root)
        win.title("Plan Versions / Experiment History")
        win.geometry("980x620+160+120")
        colors = self._theme_colors() if hasattr(self, "_theme_colors") else {"surface":"#ffffff", "text":"#111111"}
        top = Frame(win, padx=8, pady=8)
        top.pack(fill="x")
        Label(top, text="Final Memo", font=("MS Gothic", 11, "bold")).pack(anchor="w")
        memo = Text(top, height=4, wrap="word")
        memo.pack(fill="x", expand=False, pady=(2, 6))
        memo.insert("1.0", str(self.config.get("final_memo", "") or ""))
        row = Frame(top)
        row.pack(fill="x")
        Label(row, text="Snapshot Label", width=14, anchor="w").pack(side="left")
        label_var = tk.StringVar(value=time.strftime("snapshot_%Y%m%d_%H%M%S"))
        Entry(row, textvariable=label_var).pack(side="left", fill="x", expand=True, padx=(0, 8))

        body = Frame(win, padx=8, pady=8)
        body.pack(fill="both", expand=True)
        listbox = tk.Listbox(body, width=34, exportselection=False)
        listbox.pack(side="left", fill="y")
        detail = Text(body, wrap="none", font=("Consolas", 10))
        detail.pack(side="left", fill="both", expand=True, padx=(8, 0))

        def refresh():
            self.config["final_memo"] = memo.get("1.0", "end-1c")
            listbox.delete(0, "end")
            for idx, item in enumerate(self.config.get("experiment_versions", []) or []):
                listbox.insert("end", f"{idx+1:03d} {item.get('created_at','')}  {item.get('label','')}")
            self._schedule_config_save()

        def show_selected(_event=None):
            detail.delete("1.0", "end")
            sel = listbox.curselection()
            if not sel:
                return
            item = (self.config.get("experiment_versions", []) or [])[sel[0]]
            preview = copy.deepcopy(item)
            if isinstance(preview.get("plan_data"), dict):
                preview["plan_data_summary"] = f"{len(preview['plan_data'].get('entries', []) or [])} entries"
                preview.pop("plan_data", None)
            detail.insert("1.0", json.dumps(preview, ensure_ascii=False, indent=2))

        def add_snapshot():
            self.config["final_memo"] = memo.get("1.0", "end-1c")
            snap = self._planner_save_version_snapshot(label_var.get())
            refresh()
            try:
                listbox.selection_clear(0, "end")
                listbox.selection_set("end")
                listbox.see("end")
            except Exception:
                pass
            detail.delete("1.0", "end")
            detail.insert("1.0", json.dumps({k:v for k,v in snap.items() if k != "plan_data"}, ensure_ascii=False, indent=2))

        def restore_selected():
            sel = listbox.curselection()
            if not sel:
                return
            item = (self.config.get("experiment_versions", []) or [])[sel[0]]
            data = copy.deepcopy(item.get("plan_data") or {})
            if not isinstance(data, dict):
                return
            self._planner_push_history()
            self.plan_data = self._normalize_plan_preserving_embedded_sources(data) if hasattr(self, "_normalize_plan_preserving_embedded_sources") else normalize_plan(data)
            self.config["final_memo"] = str(item.get("final_memo", "") or "")
            self.current_index = 0
            self._refresh_line_selector()
            self._render_current_line()
            self._save_plan_to_file()
            self.status_label.config(text="Restored selected plan version")

        listbox.bind("<<ListboxSelect>>", show_selected, add="+")
        btns = Frame(top)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Save Snapshot", command=add_snapshot).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="Restore Selected", command=restore_selected).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right")
        refresh()

    def _planner_open_t2i_settings(self):
        settings = self._planner_get_t2i_settings()
        win = tk.Toplevel(self.root)
        win.title("Developer T2I Settings")
        win.geometry("980x680+180+100")
        outer = Frame(win, padx=10, pady=10)
        outer.pack(fill="both", expand=True)
        scalar = Frame(outer)
        scalar.pack(fill="x")
        vars_map = {}
        specs = [("seed", "Seed (-1 random)", 12), ("steps", "Steps", 8), ("width", "Width", 8), ("height", "Height", 8), ("cfg", "CFG", 8), ("num_gen", "Num Gen", 8)]
        for key, label, width in specs:
            cell = Frame(scalar)
            cell.pack(side="left", padx=(0, 8))
            Label(cell, text=label).pack(anchor="w")
            v = tk.StringVar(value=str(settings.get(key, _DEV_DEFAULT_T2I_SETTINGS.get(key, ""))))
            Entry(cell, textvariable=v, width=width).pack(anchor="w")
            vars_map[key] = v

        list_frame = LabelFrame(outer, text="Prompts", padx=8, pady=8)
        list_frame.pack(fill="both", expand=True, pady=(10, 0))
        canvas = Canvas(list_frame, highlightthickness=0)
        scroll = Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner = Frame(canvas)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.create_window((0, 0), window=inner, anchor="nw", tags=("inner",))
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure("inner", width=e.width))
        prompt_rows = []

        def rebuild():
            for child in list(inner.winfo_children()):
                child.destroy()
            for idx, row in enumerate(prompt_rows):
                box = LabelFrame(inner, text=f"Prompt {idx+1}", padx=6, pady=6)
                box.pack(fill="x", expand=True, pady=4)
                top = Frame(box)
                top.pack(fill="x")
                Label(top, text="Name", width=10, anchor="w").pack(side="left")
                Entry(top, textvariable=row["name"], width=24).pack(side="left", padx=(0, 8))
                ttk.Button(top, text="↑", width=3, command=lambda i=idx: move_prompt(i, -1)).pack(side="left", padx=(0, 2))
                ttk.Button(top, text="↓", width=3, command=lambda i=idx: move_prompt(i, 1)).pack(side="left", padx=(0, 2))
                ttk.Button(top, text="-", width=3, command=lambda i=idx: remove_prompt(i)).pack(side="left")
                Label(box, text="Prompt", anchor="w").pack(anchor="w")
                t = Text(box, height=4, wrap="word")
                t.pack(fill="x", expand=True)
                t.insert("1.0", row.get("prompt_text", ""))
                row["prompt_widget"] = t
                Label(box, text="Negative", anchor="w").pack(anchor="w", pady=(4, 0))
                n = Text(box, height=3, wrap="word")
                n.pack(fill="x", expand=True)
                n.insert("1.0", row.get("negative_text", ""))
                row["negative_widget"] = n

        def sync_widgets():
            for row in prompt_rows:
                if row.get("prompt_widget") is not None:
                    row["prompt_text"] = row["prompt_widget"].get("1.0", "end-1c")
                if row.get("negative_widget") is not None:
                    row["negative_text"] = row["negative_widget"].get("1.0", "end-1c")

        def add_prompt(item=None):
            sync_widgets()
            item = item or {}
            prompt_rows.append({
                "name": tk.StringVar(value=str(item.get("name") or f"prompt_{len(prompt_rows)+1}")),
                "prompt_text": str(item.get("prompt") or ""),
                "negative_text": str(item.get("negative") or item.get("neg") or ""),
                "prompt_widget": None,
                "negative_widget": None,
            })
            rebuild()

        def remove_prompt(i):
            sync_widgets()
            if 0 <= i < len(prompt_rows) and len(prompt_rows) > 1:
                prompt_rows.pop(i)
                rebuild()

        def move_prompt(i, delta):
            sync_widgets()
            j = i + delta
            if 0 <= i < len(prompt_rows) and 0 <= j < len(prompt_rows):
                prompt_rows[i], prompt_rows[j] = prompt_rows[j], prompt_rows[i]
                rebuild()

        def save():
            sync_widgets()
            out = {}
            for key, var in vars_map.items():
                raw = var.get().strip()
                try:
                    out[key] = float(raw) if key == "cfg" else int(raw)
                except Exception:
                    out[key] = _DEV_DEFAULT_T2I_SETTINGS.get(key)
            out["prompts"] = []
            for row in prompt_rows:
                out["prompts"].append({
                    "name": row["name"].get().strip() or f"prompt_{len(out['prompts'])+1}",
                    "prompt": row.get("prompt_text", ""),
                    "negative": row.get("negative_text", ""),
                })
            self._planner_set_t2i_settings(out)
            self.status_label.config(text="Saved Developer T2I settings")
            win.destroy()

        for item in settings.get("prompts") or _DEV_DEFAULT_T2I_SETTINGS["prompts"]:
            add_prompt(item)
        btns = Frame(outer)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="+ Add Prompt", command=lambda: add_prompt()).pack(side="left")
        ttk.Button(btns, text="Save", command=save).pack(side="right", padx=(4, 0))
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right")

    def _parse_sweep_values(self, raw: str) -> List[str]:
        raw = str(raw or "").strip()
        if not raw:
            return []
        # range form: start:end:step
        if re.fullmatch(r"[-+]?\d*\.?\d+\s*:\s*[-+]?\d*\.?\d+\s*:\s*[-+]?\d*\.?\d+", raw):
            a, b, s = [float(x.strip()) for x in raw.split(":")]
            if s == 0:
                return []
            out = []
            x = a
            guard = 0
            if s > 0:
                while x <= b + 1e-9 and guard < 1000:
                    out.append(f"{x:.6g}"); x += s; guard += 1
            else:
                while x >= b - 1e-9 and guard < 1000:
                    out.append(f"{x:.6g}"); x += s; guard += 1
            return out
        return [x.strip() for x in re.split(r"[,\n]+", raw) if x.strip()]

    def _planner_open_ratio_sweep(self):
        entries = self.plan_data.get("entries", [])
        if not entries:
            return
        entry = entries[self.current_index]
        if entry.get("type") != "Checkpoint Merge":
            messagebox.showinfo("Ratio Sweep", "Select a Checkpoint Merge line first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Ratio Sweep / Grid Search")
        win.geometry("640x420+220+160")
        box = Frame(win, padx=10, pady=10)
        box.pack(fill="both", expand=True)
        Label(box, text="Values may be comma/newline separated, or range as start:end:step", anchor="w").pack(fill="x")
        alpha_text = Text(box, height=4)
        beta_text = Text(box, height=4)
        out_prefix = tk.StringVar(value=str(entry.get("output_name") or "sweep"))
        Label(box, text="Alpha values", anchor="w").pack(fill="x", pady=(8, 0))
        alpha_text.pack(fill="x")
        alpha_text.insert("1.0", "0.1,0.15,0.2")
        Label(box, text="Beta values (blank = keep current)", anchor="w").pack(fill="x", pady=(8, 0))
        beta_text.pack(fill="x")
        row = Frame(box)
        row.pack(fill="x", pady=(8, 0))
        Label(row, text="Output prefix", width=14, anchor="w").pack(side="left")
        Entry(row, textvariable=out_prefix).pack(side="left", fill="x", expand=True)

        def apply():
            alphas = self._parse_sweep_values(alpha_text.get("1.0", "end-1c"))
            betas = self._parse_sweep_values(beta_text.get("1.0", "end-1c")) or [None]
            if not alphas:
                messagebox.showwarning("Ratio Sweep", "No alpha values were parsed.")
                return
            if len(alphas) * len(betas) > 200 and not messagebox.askyesno("Ratio Sweep", "This will create more than 200 lines. Continue?"):
                return
            self._planner_push_history()
            insert_at = self.current_index + 1
            new_entries = []
            for a in alphas:
                for b in betas:
                    cp = copy.deepcopy(entry)
                    cp["id"] = make_entry("Checkpoint Merge").get("id")
                    cp.setdefault("alpha", default_ratio("Single"))["mode"] = "Single"
                    cp["alpha"]["value"] = str(a)
                    suffix = f"a{str(a).replace('.', 'p').replace('-', 'm')}"
                    if b is not None:
                        cp.setdefault("beta", default_ratio("Single"))["mode"] = "Single"
                        cp["beta"]["value"] = str(b)
                        suffix += f"_b{str(b).replace('.', 'p').replace('-', 'm')}"
                    cp["output_name"] = f"{out_prefix.get().strip() or 'sweep'}_{suffix}"
                    cp["memo"] = f"Ratio Sweep generated from line {self.current_index + 1}. alpha={a}" + (f", beta={b}" if b is not None else "")
                    new_entries.append(cp)
            self.plan_data.setdefault("entries", [])[insert_at:insert_at] = new_entries
            self.current_index = insert_at
            self._save_plan_to_file()
            self._refresh_line_selector()
            self._render_current_line()
            self.status_label.config(text=f"Added {len(new_entries)} sweep line(s)")
            win.destroy()
        ttk.Button(box, text="Create Sweep Lines", command=apply).pack(side="right", pady=(10, 0))

    def _planner_ratio_to_lines(self, ratio):
        if not isinstance(ratio, dict):
            return [str(ratio)]
        mode = str(ratio.get("mode") or "Single")
        value = str(ratio.get("value") or "")
        if mode == "Elemental":
            return [x.strip() for x in re.split(r"[,\n]+", value) if x.strip()]
        if mode == "Block weight":
            blocks = self.block_sets.get(self.base_model_var.get(), SDXL_BLOCKS) if hasattr(self, "block_sets") else SDXL_BLOCKS
            vals = [x.strip() for x in value.split(",")]
            return [f"{blocks[i] if i < len(blocks) else i}:{v}" for i, v in enumerate(vals)]
        return [f"Single:{value}"]

    def _planner_show_ratio_diff(self):
        indices = self._planner_get_selected_indices() if hasattr(self, "_planner_get_selected_indices") else [self.current_index]
        entries = self.plan_data.get("entries", [])
        if len(indices) < 2:
            messagebox.showinfo("Ratio Diff", "Select two plan lines in Plan View first.")
            return
        a, b = entries[indices[0]], entries[indices[1]]
        def collect(entry):
            out = {}
            if entry.get("type") == "Checkpoint Merge":
                for key in ("alpha", "beta"):
                    for line in self._planner_ratio_to_lines(entry.get(key)):
                        k = f"{key}:{line.split(':',1)[0]}" if ':' in line else f"{key}:{line}"
                        out[k] = line.split(":", 1)[1] if ":" in line else line
            elif entry.get("type") == "LoRA Bake":
                for i, lora in enumerate(entry.get("loras", []) or []):
                    for line in self._planner_ratio_to_lines(lora.get("ratio")):
                        k = f"lora{i+1}:{line.split(':',1)[0]}" if ':' in line else f"lora{i+1}:{line}"
                        out[k] = line.split(":", 1)[1] if ":" in line else line
            return out
        da, db = collect(a), collect(b)
        keys = sorted(set(da) | set(db))
        lines = [f"Ratio Diff: line {indices[0]+1} vs line {indices[1]+1}", ""]
        for k in keys:
            va, vb = da.get(k, "<missing>"), db.get(k, "<missing>")
            if va != vb:
                lines.append(f"{k}: {va}  ->  {vb}")
        self._show_scrollable_text_dialog("Ratio Diff", "\n".join(lines) if len(lines) > 2 else "No ratio differences found.")

    def _planner_save_alpha_preset(self):
        entries = self.plan_data.get("entries", [])
        if not entries:
            return
        entry = entries[self.current_index]
        if entry.get("type") != "Checkpoint Merge":
            messagebox.showinfo("Alpha Preset", "Select a Checkpoint Merge line first.")
            return
        name = simpledialog.askstring("Save Alpha Preset", "Preset name:", parent=self.root)
        if not name:
            return
        self._planner_dev_ensure_config()
        self.config.setdefault("alpha_presets", {})[name.strip()] = copy.deepcopy(entry.get("alpha", default_ratio("Single")))
        self._schedule_config_save()
        self.status_label.config(text=f"Saved alpha preset: {name.strip()}")

    def _planner_load_alpha_preset(self):
        self._planner_dev_ensure_config()
        presets = self.config.get("alpha_presets") or {}
        if not presets:
            messagebox.showinfo("Alpha Preset", "No alpha presets saved yet.")
            return
        name = simpledialog.askstring("Load Alpha Preset", "Preset name:\n" + "\n".join(sorted(presets)), parent=self.root)
        if not name or name.strip() not in presets:
            return
        entries = self.plan_data.get("entries", [])
        if not entries or entries[self.current_index].get("type") != "Checkpoint Merge":
            messagebox.showinfo("Alpha Preset", "Select a Checkpoint Merge line first.")
            return
        self._planner_push_history()
        entries[self.current_index]["alpha"] = copy.deepcopy(presets[name.strip()])
        self._save_plan_to_file()
        self._render_current_line()
        self.status_label.config(text=f"Loaded alpha preset: {name.strip()}")

    def _planner_model_anatomy_scan(self):
        path = filedialog.askopenfilename(title="Select model / LoRA for anatomy scan", filetypes=[("Model files", "*.safetensors *.ckpt *.pt *.bin"), ("All files", "*.*")])
        if not path:
            return
        lines = [f"Model Anatomy Scanner", f"File: {path}", ""]
        try:
            keys = []
            if path.lower().endswith(".safetensors"):
                try:
                    from safetensors import safe_open
                    with safe_open(path, framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                except Exception as e:
                    raise RuntimeError("safetensors is required to scan .safetensors files") from e
            else:
                try:
                    import torch
                    obj = torch.load(path, map_location="cpu")
                    if isinstance(obj, dict):
                        sd = obj.get("state_dict") if isinstance(obj.get("state_dict"), dict) else obj
                        keys = list(sd.keys())
                except Exception as e:
                    raise RuntimeError("torch.load failed for this file") from e
            prefixes = {}
            modules = {"unet/diffusion":0, "clip/text":0, "vae/first_stage":0, "lora":0}
            for k in keys:
                prefixes[k.split('.',1)[0]] = prefixes.get(k.split('.',1)[0], 0) + 1
                lk = k.lower()
                if "lora" in lk: modules["lora"] += 1
                if "first_stage" in lk or ".vae" in lk or lk.startswith("vae") or "decoder" in lk: modules["vae/first_stage"] += 1
                if "conditioner" in lk or "text" in lk or "clip" in lk: modules["clip/text"] += 1
                if "diffusion_model" in lk or "model.diffusion" in lk or "unet" in lk: modules["unet/diffusion"] += 1
            lines.append(f"Total keys: {len(keys)}")
            lines.append("")
            lines.append("Module hints:")
            for k, v in modules.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Top prefixes:")
            for k, v in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:40]:
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Sample keys:")
            lines.extend("  " + k for k in keys[:120])
        except Exception as e:
            lines.append(f"ERROR: {type(e).__name__}: {e}")
        self._show_scrollable_text_dialog("Model Anatomy Scanner", "\n".join(lines))

    def _planner_lora_compatibility_check(self):
        entries = self.plan_data.get("entries", [])
        if not entries:
            return
        entry = entries[self.current_index]
        ckpt_path = ""
        lora_paths = []
        if entry.get("type") == "LoRA Bake":
            spec = self._get_entry_slot_source(entry, "checkpoint") if hasattr(self, "_get_entry_slot_source") else None
            if isinstance(spec, dict) and spec.get("mode") == "local":
                ckpt_path = spec.get("local_path") or ""
            for lora in entry.get("loras", []) or []:
                spec = self._get_lora_source(lora) if hasattr(self, "_get_lora_source") else None
                if isinstance(spec, dict) and spec.get("mode") == "local":
                    lora_paths.append(spec.get("local_path") or "")
        if not ckpt_path:
            ckpt_path = filedialog.askopenfilename(title="Select checkpoint", filetypes=[("Checkpoint", "*.safetensors *.ckpt *.pt"), ("All files", "*.*")])
        if not lora_paths:
            sel = filedialog.askopenfilenames(title="Select LoRA / LyCORIS", filetypes=[("LoRA", "*.safetensors *.pt *.bin"), ("All files", "*.*")])
            lora_paths = list(sel)
        if not ckpt_path or not lora_paths:
            return
        def keys_for(path):
            if path.lower().endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(path, framework="pt", device="cpu") as f:
                    return list(f.keys())
            import torch
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                sd = obj.get("state_dict") if isinstance(obj.get("state_dict"), dict) else obj
                return list(sd.keys())
            return []
        def normalize_lora_target(k):
            s = re.sub(r"\.lora_(?:up|down|mid|alpha)\.weight$", "", k)
            s = s.replace("lora_unet_", "").replace("lora_te_", "")
            s = s.replace("_", ".")
            return s.lower()
        try:
            ck = keys_for(ckpt_path)
            ck_l = [x.lower() for x in ck]
            lines = ["LoRA Bake Compatibility", f"Checkpoint: {ckpt_path}", f"Checkpoint keys: {len(ck)}", ""]
            for lp in lora_paths:
                lk = keys_for(lp)
                targets = sorted({normalize_lora_target(k) for k in lk if "lora_down" in k or "lora_up" in k})
                resolved = 0
                unresolved = []
                for t in targets:
                    parts = [p for p in t.split('.') if p]
                    tail = ".".join(parts[-3:]) if len(parts) >= 3 else t
                    ok = any(tail in c or t in c for c in ck_l)
                    if ok: resolved += 1
                    else: unresolved.append(t)
                total = max(1, len(targets))
                lines.append(f"LoRA: {lp}")
                lines.append(f"  targets: {len(targets)}")
                lines.append(f"  resolved approx: {resolved}/{len(targets)} ({resolved/total*100:.1f}%)")
                if unresolved:
                    lines.append("  unresolved samples:")
                    lines.extend("    - " + u for u in unresolved[:40])
                lines.append("")
        except Exception as e:
            lines = [f"Compatibility check failed: {type(e).__name__}: {e}"]
        self._show_scrollable_text_dialog("LoRA Compatibility", "\n".join(lines))

    def _planner_estimate_disk_runtime(self):
        entries = self.plan_data.get("entries", [])
        output_count = sum(1 for e in entries if e.get("type") in ("Checkpoint Merge", "LoRA Bake") and str(e.get("output_name") or "").strip())
        download_count = 0
        local_paths = []
        for e in entries:
            if e.get("type") == "Download Model" or any(isinstance(s, dict) and s.get("mode") == "download" for *_x, s in self._iter_embedded_sources(e)):
                download_count += 1
            for *_x, spec in self._iter_embedded_sources(e):
                if isinstance(spec, dict) and spec.get("mode") == "local" and spec.get("local_path"):
                    local_paths.append(spec.get("local_path"))
            if e.get("type") == "Local Model" and e.get("local_path"):
                local_paths.append(e.get("local_path"))
        local_bytes = 0
        for pth in set(local_paths):
            try:
                local_bytes += Path(pth).expanduser().stat().st_size
            except Exception:
                pass
        typical_gb = 6.5 if str(self.base_model_var.get()).upper() == "SDXL" else 12.0 if self.base_model_var.get() in ("Flux", "Anima") else 4.0
        generated_gb = output_count * typical_gb
        try:
            work = self.entries.get("workpath").get() if self.entries.get("workpath") else os.getcwd()
            usage = shutil.disk_usage(work)
            free_gb = usage.free / (1024**3)
        except Exception:
            free_gb = 0.0
        estimated_total = generated_gb + local_bytes/(1024**3)
        risk = "LOW" if free_gb > estimated_total * 1.4 else "MEDIUM" if free_gb > estimated_total * 0.9 else "HIGH"
        lines = [
            "Disk / Runtime Estimator", "",
            f"Base model: {self.base_model_var.get() if hasattr(self, 'base_model_var') else ''}",
            f"Merge/Bake outputs: {output_count}",
            f"Download source count: {download_count}",
            f"Known local source size: {local_bytes/(1024**3):.2f} GB",
            f"Estimated generated checkpoints: {generated_gb:.2f} GB ({typical_gb:.1f} GB each heuristic)",
            f"Current free disk: {free_gb:.2f} GB",
            f"Risk: {risk}", "",
            "Suggestions:",
            "- Use Optimize Plan before large sweep runs.",
            "- Insert Remove Model after last use for long chains.",
            "- Avoid enabling large Ratio Sweep without enough disk headroom.",
        ]
        self._show_scrollable_text_dialog("Disk / Runtime Estimator", "\n".join(lines))

    def _planner_show_image_compare(self):
        base = ""
        try:
            base = os.path.join(self.entries.get("workpath").get(), "working", "t2i_images")
        except Exception:
            base = os.getcwd()
        folder = filedialog.askdirectory(title="Select T2I image folder", initialdir=base if os.path.isdir(base) else os.getcwd())
        if not folder:
            return
        imgs = sorted([p for p in Path(folder).glob("*.png")])[:120]
        if not imgs:
            messagebox.showinfo("Image Compare", "No PNG images found.")
            return
        win = tk.Toplevel(self.root)
        win.title("T2I Image Compare")
        win.geometry("1180x760+120+80")
        canvas = Canvas(win, bg="#111111", highlightthickness=0)
        scroll = Scrollbar(win, orient="vertical", command=canvas.yview)
        inner = Frame(canvas, bg="#111111")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.create_window((0,0), window=inner, anchor="nw", tags=("inner",))
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure("inner", width=e.width))
        refs = []
        for idx, imgp in enumerate(imgs):
            try:
                pil = Image.open(imgp)
                pil.thumbnail((260, 260), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                refs.append(photo)
                cell = Frame(inner, bg="#111111", padx=8, pady=8)
                cell.grid(row=idx//4, column=idx%4, sticky="n")
                Label(cell, image=photo, bg="#111111").pack()
                Label(cell, text=imgp.name, bg="#111111", fg="#eeeeee", wraplength=240).pack(fill="x")
            except Exception:
                pass
        win._image_refs = refs

    def _planner_show_dependency_graph(self):
        analysis = self._planner_analysis()
        entries = self.plan_data.get("entries", [])
        win = tk.Toplevel(self.root)
        win.title("Interactive Dependency Graph")
        win.geometry("1280x760+80+80")
        canvas = Canvas(win, bg="#0f1320", scrollregion=(0, 0, 1800, max(900, len(entries)*110)))
        xscroll = Scrollbar(win, orient="horizontal", command=canvas.xview)
        yscroll = Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        win.grid_rowconfigure(0, weight=1); win.grid_columnconfigure(0, weight=1)
        colors = {"Checkpoint Merge":"#3b82f6", "LoRA Bake":"#8b5cf6", "Download Model":"#22c55e", "Local Model":"#14b8a6", "Remove Model":"#f97316", "dead":"#ef4444", "missing":"#f43f5e"}
        positions = {}
        for idx, entry in enumerate(entries):
            x = 90 + (idx % 3) * 520
            y = 70 + (idx // 3) * 150
            positions[idx] = (x, y)
        for idx, entry in enumerate(entries):
            x, y = positions[idx]
            dead = idx in analysis.get("dead_entries", set())
            fill = colors["dead"] if dead else colors.get(entry.get("type"), "#64748b")
            rect = canvas.create_rectangle(x, y, x+390, y+82, fill=fill, outline="#e5e7eb", width=2, tags=(f"node{idx}",))
            summary = f"{idx+1}. {self._line_summary(entry)}"
            canvas.create_text(x+12, y+12, text=summary[:64], anchor="nw", fill="white", font=("Consolas", 10, "bold"), width=360, tags=(f"node{idx}",))
            produced = ", ".join(self._entry_produced_aliases(entry)) or "-"
            canvas.create_text(x+12, y+48, text=f"produces: {produced[:58]}", anchor="nw", fill="#e5e7eb", font=("Consolas", 9), width=360, tags=(f"node{idx}",))
            canvas.tag_bind(f"node{idx}", "<Button-1>", lambda _e, i=idx: (setattr(self, "current_index", i), self._refresh_line_selector(), self._render_current_line(), win.lift()))
        producer_by_alias = analysis.get("producer_by_alias", {})
        for idx, aliases in analysis.get("consumed_by_entry", {}).items():
            sx, sy = positions.get(idx, (0,0))
            for alias in aliases:
                src = producer_by_alias.get(alias)
                if src is None or src not in positions:
                    continue
                x1, y1 = positions[src]
                x2, y2 = positions[idx]
                canvas.create_line(x1+390, y1+41, x2, y2+41, fill="#cbd5e1", arrow="last", width=2, smooth=True)
                mx, my = (x1+390+x2)/2, (y1+41+y2+41)/2
                canvas.create_text(mx, my-8, text=alias[:32], fill="#fde68a", font=("Consolas", 8))
        legend = "Blue=Checkpoint Merge  Purple=LoRA Bake  Green=Download  Teal=Local  Orange=Remove  Red=Dead"
        canvas.create_text(12, 12, text=legend, fill="#ffffff", anchor="nw", font=("Consolas", 10, "bold"))

    def _planner_optimize_plan_full(self):
        analysis = self._planner_analysis()
        entries = self.plan_data.get("entries", [])
        dead = sorted(analysis.get("dead_entries", set()))
        lines = ["Optimize Plan Preview", ""]
        if dead:
            lines.append("Remove dead lines:")
            for idx in dead:
                lines.append(f"  line {idx+1}: {self._line_summary(entries[idx])}")
        else:
            lines.append("No dead lines detected.")
        lines.append("")
        lines.append("This optimizer removes dead merge/bake lines. Existing source-aware cleanup remains active in notebook runtime.")
        if not messagebox.askyesno("Optimize Plan", "\n".join(lines[:20]) + "\n\nApply optimization?"):
            return
        self._planner_push_history()
        for idx in reversed(dead):
            if 0 <= idx < len(entries):
                entries.pop(idx)
        self.current_index = max(0, min(self.current_index, max(0, len(entries)-1)))
        # Preserve existing WS-chain collapse as an extra pass when the selection supports it.
        try:
            if self._selected_entries_support_ws_collapse():
                self._collapse_selected_ws_chain()
                return
        except Exception:
            pass
        self._save_plan_to_file(); self._refresh_line_selector(); self._render_current_line()
        self.status_label.config(text=f"Optimized plan: removed {len(dead)} dead line(s)")

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

    _dev_base_save_current_state = ModelPlannerApp._save_current_state_to_config
    def _save_current_state_to_config_dev(self):
        self._planner_dev_ensure_config()
        if hasattr(self, "dev_mode_var"):
            try: self.config["dev_mode"] = bool(self.dev_mode_var.get())
            except Exception: pass
        return _dev_base_save_current_state(self)

    _dev_base_build_left_panel = ModelPlannerApp._build_left_panel
    def _build_left_panel_dev(self, parent):
        self._planner_dev_ensure_config()
        result = _dev_base_build_left_panel(self, parent)
        try:
            section = self._build_collapsible_section(parent, "Developer (Experiment) Mode", key="left_developer_mode", default_open=bool(self.config.get("dev_mode", False)), body_fill="x", body_expand=False)
        except Exception:
            section = LabelFrame(parent, text="Developer (Experiment) Mode", padx=8, pady=8); section.pack(fill="x", pady=6)
        self.dev_mode_var = tk.BooleanVar(value=bool(self.config.get("dev_mode", False)))
        button_refs = []
        def refresh_dev_buttons():
            enabled = self._planner_dev_mode_enabled()
            self.config["dev_mode"] = enabled
            for btn, dev_only in button_refs:
                try: btn.configure(state=("normal" if enabled or not dev_only else "disabled"))
                except Exception: pass
            self._schedule_config_save()
        chk = ttk.Checkbutton(section, text="Enable Developer (Experiment) Mode", variable=self.dev_mode_var, command=refresh_dev_buttons)
        chk.pack(anchor="w", pady=(0, 6))
        grid = Frame(section)
        grid.pack(fill="x")
        for c in range(3): grid.grid_columnconfigure(c, weight=1)
        actions = [
            ("T2I Settings", lambda: self._planner_dev_only(self._planner_open_t2i_settings), True),
            ("Ratio Sweep", lambda: self._planner_dev_only(self._planner_open_ratio_sweep), True),
            ("Compare Images", lambda: self._planner_dev_only(self._planner_show_image_compare), True),
            ("Ratio Diff", lambda: self._planner_dev_only(self._planner_show_ratio_diff), True),
            ("Model Anatomy", lambda: self._planner_dev_only(self._planner_model_anatomy_scan), True),
            ("LoRA Check", lambda: self._planner_dev_only(self._planner_lora_compatibility_check), True),
            ("Save Alpha Preset", self._planner_save_alpha_preset, False),
            ("Load Alpha Preset", self._planner_load_alpha_preset, False),
            ("Dependency Graph", self._planner_show_dependency_graph, False),
            ("Optimize Plan", self._planner_optimize_plan_full, False),
            ("Disk Estimate", self._planner_estimate_disk_runtime, False),
            ("Plan Versions", self._planner_show_versions, False),
            ("Shortcuts", self._planner_show_shortcut_settings, False),
        ]
        for idx, (label, cmd, dev_only) in enumerate(actions):
            btn = ttk.Button(grid, text=label, command=cmd)
            btn.grid(row=idx//3, column=idx%3, sticky="ew", padx=3, pady=3)
            button_refs.append((btn, dev_only))
        self._developer_mode_buttons = button_refs
        refresh_dev_buttons()
        return result

    _dev_base_line_summary = ModelPlannerApp._line_summary
    def _line_summary_with_help_mark(self, entry: Dict[str, Any]) -> str:
        base = _dev_base_line_summary(self, entry)
        # Right-edge-style help indicator for Plan View list rows. Kept textual so Listbox remains lightweight.
        return base if base.rstrip().endswith("?") else base + "  ?"

    _dev_base_show_plan_context_menu = ModelPlannerApp._show_plan_context_menu
    def _show_plan_context_menu_dev(self, event=None):
        # Keep the existing lightweight menu, but expose the integrated optimizer through the existing Optimize Plan label.
        return _dev_base_show_plan_context_menu(self, event)

    _dev_base_meta_payload = ModelPlannerApp._planner_meta_payload
    def _planner_meta_payload_dev(self):
        payload = _dev_base_meta_payload(self)
        payload["final_memo"] = str(self.config.get("final_memo", "") or "")
        payload["experiment_versions"] = copy.deepcopy(self.config.get("experiment_versions", []) or [])
        return payload

    _dev_base_apply_loaded_meta = ModelPlannerApp._planner_apply_loaded_meta
    def _planner_apply_loaded_meta_dev(self, payload):
        _dev_base_apply_loaded_meta(self, payload)
        if isinstance(payload, dict):
            if "final_memo" in payload:
                self.config["final_memo"] = str(payload.get("final_memo") or "")
            if isinstance(payload.get("experiment_versions"), list):
                self.config["experiment_versions"] = payload.get("experiment_versions")

    # Replace / extend selected methods.
    ModelPlannerApp._planner_dev_ensure_config = _planner_dev_ensure_config
    ModelPlannerApp._planner_dev_mode_enabled = _planner_dev_mode_enabled
    ModelPlannerApp._planner_dev_only = _planner_dev_only
    ModelPlannerApp._planner_get_t2i_settings = _planner_get_t2i_settings
    ModelPlannerApp._planner_set_t2i_settings = _planner_set_t2i_settings
    ModelPlannerApp._planner_save_version_snapshot = _planner_save_version_snapshot
    ModelPlannerApp._planner_show_versions = _planner_show_versions
    ModelPlannerApp._planner_open_t2i_settings = _planner_open_t2i_settings
    ModelPlannerApp._parse_sweep_values = _parse_sweep_values
    ModelPlannerApp._planner_open_ratio_sweep = _planner_open_ratio_sweep
    ModelPlannerApp._planner_ratio_to_lines = _planner_ratio_to_lines
    ModelPlannerApp._planner_show_ratio_diff = _planner_show_ratio_diff
    ModelPlannerApp._planner_save_alpha_preset = _planner_save_alpha_preset
    ModelPlannerApp._planner_load_alpha_preset = _planner_load_alpha_preset
    ModelPlannerApp._planner_model_anatomy_scan = _planner_model_anatomy_scan
    ModelPlannerApp._planner_lora_compatibility_check = _planner_lora_compatibility_check
    ModelPlannerApp._planner_estimate_disk_runtime = _planner_estimate_disk_runtime
    ModelPlannerApp._planner_show_image_compare = _planner_show_image_compare
    ModelPlannerApp._planner_show_dependency_graph = _planner_show_dependency_graph
    ModelPlannerApp._planner_optimize_plan_full = _planner_optimize_plan_full
    ModelPlannerApp._planner_shortcut_registry = _planner_shortcut_registry
    ModelPlannerApp._planner_bind_shortcuts = _planner_bind_shortcuts_configurable
    ModelPlannerApp._planner_show_shortcut_settings = _planner_show_shortcut_settings
    ModelPlannerApp._save_current_state_to_config = _save_current_state_to_config_dev
    # Old Developer Mode left-panel renderer is kept only as a reference;
    # the final renderer is installed in the compact UI patch layer below.
    # Old textual help marker renderer is kept only as a reference;
    # the final Plan View help behavior is installed in the compact UI patch layer below.
    ModelPlannerApp._planner_meta_payload = _planner_meta_payload_dev
    ModelPlannerApp._planner_apply_loaded_meta = _planner_apply_loaded_meta_dev
except Exception:
    pass


# -----------------------------------------------------------------------------
# Merge CLI options + dedicated LoRA Merge line type
# -----------------------------------------------------------------------------
try:
    if "LoRA Merge" not in LINE_TYPES:
        LINE_TYPES.append("LoRA Merge")

    RIGHT_PANEL_FIELD_HELP.update({
        "LoRA Merge": {
            "short": "Merge multiple LoRA/LyCORIS models into one LoRA without baking into a checkpoint.",
            "detail": "This is separate from LoRA Bake. It emits lora_bake.py with --merge_loras and registers the output as a LoRA so it can be baked or merged later.",
        },
        "Checkpoint Merge Options": {
            "short": "Structured UI for merge.py CLI options.",
            "detail": "These controls expose merge.py flags such as use_dif, rand_alpha/beta, cosine routing, turbo/deturbo, rebasin, fine/fine_sat, CFG sensitivity, saturation boost, VAE saturation, metadata, and overwrite options. Values are compiled together with Additional Signatures.",
        },
        "LoRA Bake Options": {
            "short": "Structured UI for lora_bake.py bake options.",
            "detail": "These controls expose bake-only options such as DARE, text encoder scaling, UNet-only baking, normalization, rank cap, clamp, delta guard, fp32 accumulation, budget report, metadata, and memo.",
        },
        "LoRA Merge Options": {
            "short": "Structured UI for lora_bake.py --merge_loras options.",
            "detail": "These controls expose LoRA-to-LoRA merge settings such as rank cap, architecture mapping, normalization, global scale, UNet-only merge, metadata, and memo.",
        },
    })

    _LORA_MERGE_CLI_SPECS = {
        "Checkpoint Merge": {
            "groups": [
                ("Model routing", [
                    ("m0_name", "Model 0 Name", "text", ""),
                    ("m1_name", "Model 1 Name", "text", ""),
                    ("m2_name", "Model 2 Name", "text", ""),
                    ("use_dif_10", "Use Difference 1-0", "bool", False),
                    ("use_dif_20", "Use Difference 2-0", "bool", False),
                    ("use_dif_21", "Use Difference 2-1", "bool", False),
                ]),
                ("Random / cosine / conversion", [
                    ("rand_alpha", "Random Alpha", "text", ""),
                    ("rand_beta", "Random Beta", "text", ""),
                    ("cosine0", "Cosine 0", "bool", False),
                    ("cosine1", "Cosine 1", "bool", False),
                    ("cosine2", "Cosine 2", "bool", False),
                    ("turbo", "Turbo Convert", "bool", False),
                    ("deturbo", "De-turbo Convert", "bool", False),
                    ("seed", "Seed", "text", ""),
                    ("rebasin", "ReBasin Iterations", "text", ""),
                ]),
                ("Fine / CFG / color", [
                    ("fine", "Fine", "text", ""),
                    ("fine_sat", "Fine Saturation", "text", ""),
                    ("cfg_sens", "CFG Sensitivity", "text", ""),
                    ("cfg_sens_targets", "CFG Targets", "text", ""),
                    ("sat_boost", "Saturation Boost", "text", ""),
                    ("sat_boost_side", "Sat Boost Side", "choice", "alpha", ["alpha", "beta", "both"]),
                    ("sat_boost_tags", "Sat Boost Tags", "text", ""),
                    ("sat_profile", "Sat Profile", "choice", "legacy", ["legacy", "safe_attn2_out"]),
                    ("sat_delta_cap_pct", "Sat Delta Cap %", "text", ""),
                    ("sat_boost_mix", "Sat Boost Mix", "text", ""),
                    ("boost_clamp", "Boost Clamp", "choice", "auto", ["auto", "clamp01", "none"]),
                    ("vae_sat", "VAE Saturation", "text", ""),
                ]),
                ("Save / metadata", [
                    ("keep_ema", "Keep EMA", "bool", False),
                    ("delete_source", "Delete Source", "bool", False),
                    ("no_metadata", "No Metadata", "bool", False),
                    ("force", "Force Overwrite", "bool", False),
                    ("memo", "Memo Metadata", "text", ""),
                ]),
            ]
        },
        "LoRA Bake": {
            "groups": [
                ("Bake method", [
                    ("dare", "Use DARE", "bool", False),
                    ("bake_clip_scale", "Text Encoder Scale", "text", ""),
                    ("bake_unet_only", "UNet / DiT Only", "bool", False),
                    ("bake_norm", "Bake Norm", "choice", "sqrt", ["none", "sqrt", "mean"]),
                    ("bake_scale", "Bake Scale", "text", ""),
                    ("bake_rank_cap", "Rank Cap", "text", ""),
                ]),
                ("Bake guard", [
                    ("bake_clamp_q", "Clamp Quantile", "text", ""),
                    ("bake_delta_cap", "Delta Cap", "text", ""),
                    ("bake_fp32", "Bake FP32", "bool", False),
                    ("bake_guard", "Guard Mode", "choice", "auto", ["none", "auto", "cap"]),
                    ("bake_guard_cap", "Guard Cap", "text", ""),
                    ("bake_guard_skip", "Guard Skip", "text", ""),
                    ("bake_budget_report", "Budget Report", "bool", False),
                ]),
                ("Save / metadata", [
                    ("keep_ema", "Keep EMA", "bool", False),
                    ("no_metadata", "No Metadata", "bool", False),
                    ("memo", "Memo Metadata", "text", ""),
                ]),
            ]
        },
        "LoRA Merge": {
            "groups": [
                ("LoRA merge", [
                    ("merge_rank", "Merge Rank", "text", "64"),
                    ("merge_arch", "Merge Arch", "choice", "auto", ["auto", "sd", "sdxl", "flux", "zi", "am"]),
                    ("merge_norm", "Merge Norm", "choice", "none", ["none", "sqrt", "mean"]),
                    ("merge_scale", "Merge Scale", "text", ""),
                    ("merge_unet_only", "UNet / DiT Only", "bool", False),
                    ("merge_clamp_q", "Clamp Quantile", "text", ""),
                    ("merge_intermediate_mult", "Intermediate Rank Mult", "text", ""),
                ]),
                ("Save / metadata", [
                    ("no_metadata", "No Metadata", "bool", False),
                    ("memo", "Memo Metadata", "text", ""),
                ]),
            ]
        },
    }

    _LORA_MERGE_DEFAULTS = {
        "Checkpoint Merge": {
            "m0_name": "", "m1_name": "", "m2_name": "",
            "use_dif_10": False, "use_dif_20": False, "use_dif_21": False,
            "rand_alpha": "", "rand_beta": "",
            "cosine0": False, "cosine1": False, "cosine2": False,
            "keep_ema": False, "delete_source": False, "no_metadata": False, "force": False,
            "turbo": False, "deturbo": False, "seed": "", "rebasin": "", "memo": "", "fine": "", "fine_sat": "",
            "cfg_sens": "", "cfg_sens_targets": "", "sat_boost": "", "sat_boost_side": "alpha", "sat_boost_tags": "",
            "sat_profile": "legacy", "sat_delta_cap_pct": "", "sat_boost_mix": "", "boost_clamp": "auto", "vae_sat": "",
        },
        "LoRA Bake": {
            "dare": False, "keep_ema": False, "no_metadata": False, "memo": "",
            "bake_clip_scale": "", "bake_unet_only": False, "bake_norm": "sqrt", "bake_scale": "", "bake_rank_cap": "",
            "bake_clamp_q": "", "bake_delta_cap": "", "bake_fp32": False,
            "bake_guard": "auto", "bake_guard_cap": "", "bake_guard_skip": "", "bake_budget_report": False,
        },
        "LoRA Merge": {
            "merge_rank": "64", "merge_arch": "auto", "merge_norm": "none", "merge_scale": "",
            "merge_unet_only": False, "merge_clamp_q": "", "merge_intermediate_mult": "", "no_metadata": False, "memo": "",
        },
    }

    def _lm_ensure_cli_options(entry: Dict[str, Any]) -> Dict[str, Any]:
        etype = str(entry.get("type") or "")
        defaults = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(etype, {}))
        raw = entry.get("cli_options")
        if isinstance(raw, dict):
            defaults.update({k: v for k, v in raw.items() if k in defaults})
        entry["cli_options"] = defaults
        return defaults

    def _lm_option_changed(self, entry: Dict[str, Any], key: str, value, default):
        opts = _lm_ensure_cli_options(entry)
        if isinstance(default, bool):
            opts[key] = bool(value)
        else:
            opts[key] = str(value or "")
        self._after_entry_change()

    def _lm_build_cli_options_panel(self, parent, entry: Dict[str, Any], entry_type: str):
        specs = _LORA_MERGE_CLI_SPECS.get(entry_type, {}).get("groups", [])
        if not specs:
            return
        opts = _lm_ensure_cli_options(entry)
        try:
            wrapper = self._build_collapsible_section(
                parent,
                f"{entry_type} Options",
                key=f"cli_options_{entry_type}_{entry.get('id','')}",
                default_open=False,
                body_fill="x",
                body_expand=False,
                padx=6,
                pady=6,
            )
        except Exception:
            wrapper = self._build_labeled_frame(parent, f"{entry_type} Options")
        help_text = self._right_help(f"{entry_type} Options").get("detail", "")
        if help_text:
            Label(wrapper, text=help_text, anchor="w", justify="left", wraplength=820, fg="#64748b").pack(fill="x", pady=(0, 6))
        for group_name, fields in specs:
            gf = LabelFrame(wrapper, text=group_name, padx=6, pady=6)
            gf.pack(fill="x", pady=4)
            for idx, field in enumerate(fields):
                key, label, kind, default = field[:4]
                row = Frame(gf)
                row.grid(row=idx // 2, column=(idx % 2) * 2, columnspan=2, sticky="ew", padx=3, pady=2)
                gf.grid_columnconfigure(1, weight=1)
                gf.grid_columnconfigure(3, weight=1)
                Label(row, text=label, width=18, anchor="w").pack(side="left")
                if kind == "bool":
                    var = tk.BooleanVar(value=bool(opts.get(key, default)))
                    chk = ttk.Checkbutton(row, variable=var, command=lambda v=var, k=key, d=default, e=entry: _lm_option_changed(self, e, k, v.get(), d))
                    chk.pack(side="left")
                elif kind == "choice":
                    choices = field[4] if len(field) > 4 else []
                    var = tk.StringVar(value=str(opts.get(key, default) or default))
                    combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=16)
                    self._bind_combobox_mousewheel_passthrough(combo)
                    combo.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _lm_option_changed(self, e, k, v.get(), d))
                else:
                    var = tk.StringVar(value=str(opts.get(key, default) or ""))
                    ent = Entry(row, textvariable=var, width=28)
                    ent.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _lm_option_changed(self, e, k, v.get(), d))
        reset_row = Frame(wrapper)
        reset_row.pack(fill="x", pady=(4, 0))
        def reset_options(e=entry, et=entry_type):
            e["cli_options"] = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(et, {}))
            self._after_entry_change()
            self._schedule_rerender_current_line()
        ttk.Button(reset_row, text="Reset Options", command=reset_options).pack(side="right")

    _lm_orig_render_checkpoint = getattr(ModelPlannerApp, "_render_checkpoint_merge_entry", None)
    def _render_checkpoint_merge_entry_with_cli_options(self, parent, entry: Dict[str, Any]):
        if _lm_orig_render_checkpoint:
            _lm_orig_render_checkpoint(self, parent, entry)
        _lm_build_cli_options_panel(self, parent, entry, "Checkpoint Merge")

    _lm_orig_render_lora_bake = getattr(ModelPlannerApp, "_render_lora_bake_entry", None)
    def _render_lora_bake_entry_with_cli_options(self, parent, entry: Dict[str, Any]):
        if _lm_orig_render_lora_bake:
            _lm_orig_render_lora_bake(self, parent, entry)
        _lm_build_cli_options_panel(self, parent, entry, "LoRA Bake")

    def _render_lora_merge_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "LoRA Merge")
        models = self._collect_available_models(self.current_index)
        self._build_entry_row(frame, "Output Name", entry, "output_name")
        add_lora_btn = ttk.Button(frame, text="+ Add LoRA", command=lambda: self._add_lora_block(entry))
        add_lora_btn.pack(anchor="w", padx=4, pady=4)
        self._attach_tooltip(add_lora_btn, self._right_help("+ Add LoRA").get("detail", ""))
        self._add_right_inline_help(frame, "+ Add LoRA")
        entry.setdefault("loras", [])
        for idx, lora in enumerate(entry.get("loras", []) or []):
            block = self._build_labeled_frame(parent, f"LoRA {idx + 1}")
            top = Frame(block)
            top.pack(fill="x")
            ttk.Button(top, text="-", width=3, command=lambda i=idx: self._remove_lora_block(entry, i)).pack(side="left", padx=(0, 4))
            lora_names = models.get("LoRA", []) + models.get("LyCORIS", [])
            self._build_source_backed_ref_row(
                block,
                "LoRA Name",
                lambda lo=lora: lo.get("name", ""),
                lambda value, lo=lora: lo.__setitem__("name", value),
                lambda lo=lora: self._get_lora_source(lo),
                lambda spec, lo=lora: self._set_lora_source(lo, spec),
                lora_names or [""],
                ["LoRA", "LyCORIS"],
                allow_local=True,
            )
            ratio = lora.setdefault("ratio", default_ratio("Single"))
            ratio_wrap = self._build_collapsible_section(
                block,
                "Ratio",
                key=f"lora_merge_ratio_{entry.get('id', '')}_{idx}",
                default_open=True,
                body_fill="x",
                body_expand=False,
                padx=0,
                pady=4,
            )
            row2 = Frame(ratio_wrap)
            row2.pack(fill="x", pady=3)
            Label(row2, text="Ratio Mode", width=18, anchor="w").pack(side="left")
            ratio_var = tk.StringVar(value=ratio.get("mode", "Single"))
            combo2 = ttk.Combobox(row2, textvariable=ratio_var, values=["Single", "Elemental"], state="readonly")
            self._bind_combobox_mousewheel_passthrough(combo2)
            combo2.pack(side="left", fill="x", expand=True, padx=4)
            def on_ratio_mode(*_args, lo=lora, rv=ratio_var):
                lo["ratio"]["mode"] = rv.get()
                if rv.get() == "Single":
                    lo["ratio"].setdefault("value", "1.0")
                self._schedule_rerender_current_line()
            ratio_var.trace_add("write", on_ratio_mode)
            self._build_ratio_value_widget(ratio_wrap, lora["ratio"], allow_block_weight=False)
        self._build_text_row(parent, "Additional Signatures", entry, "raw_signatures", height=5)
        _lm_build_cli_options_panel(self, parent, entry, "LoRA Merge")

    def _base_render_current_line_with_lora_merge(self):
        self._hide_autocomplete()
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        entries = self.plan_data.get("entries", [])
        if not entries:
            return
        entry = entries[self.current_index]
        self.current_editor_widgets = []
        body = self.scroll_frame
        header = LabelFrame(body, text="Line Settings", padx=8, pady=8)
        header.pack(fill="x", padx=6, pady=6)
        self._add_right_inline_help(header, "Line Settings", padx=4, wraplength=720)
        self._build_combo_row(header, "Model Merge Type", entry, "type", LINE_TYPES, self._change_line_type)
        etype = entry.get("type")
        if etype == "Download Model":
            self._render_download_entry(body, entry)
        elif etype == "Local Model":
            self._render_local_entry(body, entry)
        elif etype == "Remove Model":
            self._render_remove_entry(body, entry)
        elif etype == "Checkpoint Merge":
            self._render_checkpoint_merge_entry(body, entry)
        elif etype == "LoRA Bake":
            self._render_lora_bake_entry(body, entry)
        elif etype == "LoRA Merge":
            self._render_lora_merge_entry(body, entry)
        try:
            self.canvas.yview_moveto(0)
        except Exception:
            pass
        self._refresh_line_selector()
        self.root.after_idle(self._planner_restore_plan_list_focus)

    _lm_orig_visible_line_types = getattr(ModelPlannerApp, "_planner_visible_line_types", None)
    def _planner_visible_line_types_with_lora_merge(self):
        vals = list(_lm_orig_visible_line_types(self) if _lm_orig_visible_line_types else LINE_TYPES)
        if "LoRA Merge" not in vals:
            vals.append("LoRA Merge")
        return vals

    _lm_orig_iter_embedded_sources = getattr(ModelPlannerApp, "_iter_embedded_sources", None)
    def _iter_embedded_sources_with_lora_merge(self, entry):
        if entry.get("type") == "LoRA Merge":
            for idx, lora in enumerate(entry.get("loras", []) or []):
                spec = self._get_lora_source(lora)
                alias = str(lora.get("name") or self._source_alias_from_spec(spec)).strip()
                if spec and alias:
                    yield f"lora:{idx}", alias, str(spec.get("kind") or "LoRA"), spec
            return
        if _lm_orig_iter_embedded_sources:
            yield from _lm_orig_iter_embedded_sources(self, entry)

    _lm_orig_collect_models = getattr(ModelPlannerApp, "_collect_available_models", None)
    def _collect_available_models_with_lora_merge(self, upto_index: int) -> Dict[str, List[str]]:
        available: Dict[str, Dict[str, str]] = {"Checkpoint": {}, "LoRA": {}, "LyCORIS": {}}
        removed = set()
        entries = self.plan_data.get("entries", [])
        for entry in entries[:upto_index]:
            etype = entry.get("type")
            if etype == "Remove Model":
                name = str(entry.get("model") or "").strip()
                if name:
                    removed.add(name)
                    for bucket in available.values():
                        bucket.pop(name, None)
                continue
            if etype == "Download Model":
                name = str(entry.get("model_name") or "").strip()
                kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
                if name and name not in removed:
                    available.setdefault(kind, {})[name] = kind
            elif etype == "Local Model":
                path = str(entry.get("local_path") or "").strip()
                alias = str(entry.get("model_name") or "").strip()
                if path:
                    name = alias or Path(path).stem
                    kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
                    if name not in removed:
                        available.setdefault(kind, {})[name] = kind
            else:
                for _slot, alias, kind, _spec in self._iter_embedded_sources(entry):
                    if alias and alias not in removed:
                        available.setdefault(kind, {})[alias] = kind
                if etype in ("Checkpoint Merge", "LoRA Bake"):
                    name = str(entry.get("output_name") or "").strip()
                    if name and name not in removed:
                        available["Checkpoint"][name] = "Checkpoint"
                elif etype == "LoRA Merge":
                    name = str(entry.get("output_name") or "").strip()
                    if name and name not in removed:
                        available["LoRA"][name] = "LoRA"
        return {k: sorted(v.keys()) for k, v in available.items()}

    def _plan_entry_problem_map_with_lora_merge(self) -> Dict[int, List[str]]:
        problems_by_idx: Dict[int, List[str]] = {}
        entries = self.plan_data.get("entries", [])
        for idx, entry in enumerate(entries):
            etype = entry.get("type")
            prefix = f"line {idx + 1} ({etype})"
            problems = []
            available = self._collect_available_models(idx)
            if etype == "Download Model":
                if not entry.get("model_name"):
                    problems.append(f"{prefix}: model_name is empty")
                if not entry.get("link"):
                    problems.append(f"{prefix}: link is empty")
            elif etype == "Local Model":
                if not entry.get("local_path"):
                    problems.append(f"{prefix}: local_path is empty")
            elif etype == "Remove Model":
                if not entry.get("model"):
                    problems.append(f"{prefix}: model is empty")
            elif etype == "Checkpoint Merge":
                for req_key in ("model0", "model1", "output_name"):
                    if not entry.get(req_key):
                        problems.append(f"{prefix}: {req_key} is empty")
                for slot in ("model0", "model1", "model2"):
                    ref = str(entry.get(slot) or "").strip()
                    if ref and ref not in available.get("Checkpoint", []) and not self._get_entry_slot_source(entry, slot):
                        problems.append(f"{prefix}: checkpoint ref not available -> {ref}")
            elif etype == "LoRA Bake":
                checkpoint = str(entry.get("checkpoint") or "").strip()
                if not checkpoint:
                    problems.append(f"{prefix}: checkpoint is empty")
                elif checkpoint not in available.get("Checkpoint", []) and not self._get_entry_slot_source(entry, "checkpoint"):
                    problems.append(f"{prefix}: checkpoint ref not available -> {checkpoint}")
                if not entry.get("output_name"):
                    problems.append(f"{prefix}: output_name is empty")
                for lora in entry.get("loras", []) or []:
                    name = str(lora.get("name") or "").strip()
                    if not name:
                        problems.append(f"{prefix}: one LoRA name is empty")
                    elif name not in available.get("LoRA", []) and name not in available.get("LyCORIS", []) and not self._get_lora_source(lora):
                        problems.append(f"{prefix}: LoRA ref not available -> {name}")
            elif etype == "LoRA Merge":
                if not entry.get("output_name"):
                    problems.append(f"{prefix}: output_name is empty")
                loras = entry.get("loras", []) or []
                if not loras:
                    problems.append(f"{prefix}: no LoRAs selected")
                for lora in loras:
                    name = str(lora.get("name") or "").strip()
                    if not name:
                        problems.append(f"{prefix}: one LoRA name is empty")
                    elif name not in available.get("LoRA", []) and name not in available.get("LyCORIS", []) and not self._get_lora_source(lora):
                        problems.append(f"{prefix}: LoRA ref not available -> {name}")
            problems_by_idx[idx] = problems
        return problems_by_idx

    _lm_orig_entry_consumed = getattr(ModelPlannerApp, "_entry_consumed_aliases", None)
    def _entry_consumed_aliases_with_lora_merge(self, entry: Dict[str, Any]) -> List[str]:
        if hasattr(self, "_entry_is_disabled") and self._entry_is_disabled(entry):
            return []
        if entry.get("type") == "LoRA Merge":
            return [str(lora.get("name") or "").strip() for lora in entry.get("loras", []) or [] if str(lora.get("name") or "").strip()]
        return _lm_orig_entry_consumed(self, entry) if _lm_orig_entry_consumed else []

    _lm_orig_entry_produced = getattr(ModelPlannerApp, "_entry_produced_aliases", None)
    def _entry_produced_aliases_with_lora_merge(self, entry: Dict[str, Any]) -> List[str]:
        if hasattr(self, "_entry_is_disabled") and self._entry_is_disabled(entry):
            return []
        if entry.get("type") == "LoRA Merge":
            name = str(entry.get("output_name") or "").strip()
            return [name] if name else []
        return _lm_orig_entry_produced(self, entry) if _lm_orig_entry_produced else []

    _lm_orig_entry_type_color = getattr(ModelPlannerApp, "_entry_type_color", None)
    def _entry_type_color_with_lora_merge(self, etype: str) -> str:
        if str(etype or "") == "LoRA Merge":
            return "#0ea5e9"
        return _lm_orig_entry_type_color(self, etype) if _lm_orig_entry_type_color else "#333333"

    _lm_orig_line_summary = getattr(ModelPlannerApp, "_line_summary", None)
    def _line_summary_with_lora_merge(self, entry: Dict[str, Any]) -> str:
        if entry.get("type") == "LoRA Merge":
            n = len(entry.get("loras", []) or [])
            base = f"LoRA Merge - {entry.get('output_name') or '(unset)'} ({n} LoRA{'s' if n != 1 else ''})"
            return base + "  ?"
        return _lm_orig_line_summary(self, entry) if _lm_orig_line_summary else str(entry.get("type") or "")

    _lm_orig_tooltip = getattr(ModelPlannerApp, "_build_plan_item_tooltip_text", None)
    def _build_plan_item_tooltip_text_with_lora_merge(self, model_idx: int, entry: Dict[str, Any], problems: List[str] | None = None) -> str:
        if entry.get("type") == "LoRA Merge":
            lines = [f"Line {model_idx + 1}: LoRA Merge", f"Output: {entry.get('output_name') or '(unset)'}"]
            loras = [str(l.get("name") or "(unset)") for l in entry.get("loras", []) or []]
            lines.append("LoRAs: " + (", ".join(loras) if loras else "(none)"))
            if problems:
                lines.append("")
                lines.append("Problems:")
                lines.extend([f"- {p}" for p in problems])
            return "\n".join(lines)
        return _lm_orig_tooltip(self, model_idx, entry, problems) if _lm_orig_tooltip else str(entry)

    _lm_orig_convert_entry = getattr(ModelPlannerApp, "_planner_convert_entry_type_preserving_common_fields", None)
    def _planner_convert_entry_type_preserving_common_fields_with_lora_merge(self, entry: Dict[str, Any], new_type: str) -> Dict[str, Any]:
        if new_type == "LoRA Merge":
            new_entry = make_entry("LoRA Merge")
            common_name = str(entry.get("output_name") or entry.get("model_name") or entry.get("model") or Path(entry.get("local_path") or "").stem or "").strip()
            new_entry["output_name"] = common_name
            if entry.get("loras"):
                new_entry["loras"] = copy.deepcopy(entry.get("loras") or [])
            for key in ("memo", "_locked", "_disabled", "_row_color"):
                if key in entry:
                    new_entry[key] = copy.deepcopy(entry.get(key))
            return new_entry
        return _lm_orig_convert_entry(self, entry, new_type) if _lm_orig_convert_entry else make_entry(new_type)

    _lm_orig_apply_defaults = getattr(ModelPlannerApp, "_planner_apply_entry_defaults", None)
    def _planner_apply_entry_defaults_with_lora_merge(self):
        if _lm_orig_apply_defaults:
            _lm_orig_apply_defaults(self)
        for entry in self.plan_data.get("entries", []) or []:
            if isinstance(entry, dict) and entry.get("type") in _LORA_MERGE_DEFAULTS:
                _lm_ensure_cli_options(entry)
                if entry.get("type") == "LoRA Merge":
                    entry.setdefault("loras", [])

    _lm_orig_estimate = getattr(ModelPlannerApp, "_planner_estimate_disk_runtime", None)
    def _planner_estimate_disk_runtime_with_lora_merge(self):
        return _lm_orig_estimate(self) if _lm_orig_estimate else None

    # Apply monkey patches after all previous extension layers.
    ModelPlannerApp._render_checkpoint_merge_entry = _render_checkpoint_merge_entry_with_cli_options
    ModelPlannerApp._render_lora_bake_entry = _render_lora_bake_entry_with_cli_options
    ModelPlannerApp._render_lora_merge_entry = _render_lora_merge_entry
    ModelPlannerApp._base_render_current_line = _base_render_current_line_with_lora_merge
    ModelPlannerApp._planner_visible_line_types = _planner_visible_line_types_with_lora_merge
    ModelPlannerApp._iter_embedded_sources = _iter_embedded_sources_with_lora_merge
    ModelPlannerApp._collect_available_models = _collect_available_models_with_lora_merge
    ModelPlannerApp._plan_entry_problem_map = _plan_entry_problem_map_with_lora_merge
    ModelPlannerApp._entry_consumed_aliases = _entry_consumed_aliases_with_lora_merge
    ModelPlannerApp._entry_produced_aliases = _entry_produced_aliases_with_lora_merge
    ModelPlannerApp._entry_type_color = _entry_type_color_with_lora_merge
    ModelPlannerApp._line_summary = _line_summary_with_lora_merge
    ModelPlannerApp._build_plan_item_tooltip_text = _build_plan_item_tooltip_text_with_lora_merge
    ModelPlannerApp._planner_convert_entry_type_preserving_common_fields = _planner_convert_entry_type_preserving_common_fields_with_lora_merge
    ModelPlannerApp._planner_apply_entry_defaults = _planner_apply_entry_defaults_with_lora_merge
except Exception:
    pass


# -----------------------------------------------------------------------------
# Additional Signatures candidate expansion for merge.py / lora_bake.py options
# -----------------------------------------------------------------------------
try:
    _sig_orig_candidates = getattr(ModelPlannerApp, "_signature_autocomplete_candidates", None)
    def _signature_autocomplete_candidates_extended(self):
        base = list(_sig_orig_candidates(self) if _sig_orig_candidates else [])
        extra = [
            # merge.py
            "--m0_name NAME", "--m1_name NAME", "--m2_name NAME",
            "--use_dif_10", "--use_dif_20", "--use_dif_21",
            "--rand_alpha 0.1,0.3", "--rand_beta 0.1,0.3",
            "--cosine0", "--cosine1", "--cosine2", "--keep_ema", "--delete_source", "--no_metadata", "--force",
            "--turbo", "--deturbo", "--seed 1234", "--rebasin 8", "--memo text", "--fine key_spec", "--fine_sat 0.85",
            "--cfg_sens 1.10", "--cfg_sens_targets kv,out",
            "--sat_boost 1.5", "--sat_boost_side alpha", "--sat_boost_side beta", "--sat_boost_side both",
            "--sat_boost_tags IN00,OUT03", "--sat_profile legacy", "--sat_profile safe_attn2_out",
            "--sat_delta_cap_pct 99.5", "--sat_boost_mix 0.5", "--boost_clamp auto", "--boost_clamp clamp01", "--boost_clamp none", "--vae_sat 0.9",
            # lora_bake.py bake
            "--dare", "--bake_clip_scale 0.8", "--bake_unet_only", "--bake_norm sqrt", "--bake_norm mean", "--bake_norm none",
            "--bake_scale 1.0", "--bake_rank_cap 64", "--bake_clamp_q 0.999", "--bake_delta_cap 0.05",
            "--bake_fp32", "--bake_guard auto", "--bake_guard cap", "--bake_guard none", "--bake_guard_cap 0.05", "--bake_guard_skip 0.25", "--bake_budget_report",
            # lora_bake.py --merge_loras
            "--merge_loras", "--merge_rank 64", "--merge_arch auto", "--merge_arch sdxl", "--merge_arch flux", "--merge_arch zi", "--merge_arch am",
            "--merge_norm none", "--merge_norm sqrt", "--merge_norm mean", "--merge_scale 1.0", "--merge_unet_only",
            "--merge_clamp_q 0.99", "--merge_intermediate_mult 4",
        ]
        return list(dict.fromkeys(base + extra))
    ModelPlannerApp._signature_autocomplete_candidates = _signature_autocomplete_candidates_extended
except Exception:
    pass


# -----------------------------------------------------------------------------
# Signature-less options UI / layout refinements
# -----------------------------------------------------------------------------
try:
    if "Randomize" not in RATIO_MODES:
        RATIO_MODES.append("Randomize")

    RIGHT_PANEL_FIELD_HELP.update({
        "Unmapped Legacy Signatures": {
            "short": "Legacy @/-- tokens that could not be represented by structured options.",
            "detail": "Additional Signatures are no longer shown as a separate editor. When old plans are imported, supported tokens are converted into Option controls; unknown or unsupported tokens are preserved here and appended during export/run.",
        },
        "Architecture": {
            "short": "Architecture hint imported from @arch.",
            "detail": "Legacy @arch values are loaded here. When exporting to legacy txt, this is written back as @arch.",
        },
        "Precision": {
            "short": "Save precision imported from @p/@precision.",
            "detail": "This controls save precision and is written to legacy txt as @p when it is not the default half precision.",
        },
    })

    _SIGLESS_PRECISION_CHOICES = ["half", "bhalf", "quarter", "fp32"]
    _SIGLESS_ARCH_CHOICES = ["", "auto", "sd", "sdxl", "flux", "zi", "am", "anima", "zimage"]
    _SIGLESS_SEED_MODES = {"DARE", "XDARE"}

    def _sigless_ensure_cli_options(entry: Dict[str, Any]) -> Dict[str, Any]:
        etype = str(entry.get("type") or "")
        defaults = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(etype, {})) if '_LORA_MERGE_DEFAULTS' in globals() else {}
        # rand_alpha/rand_beta moved into Ratio Mode.
        defaults.pop("rand_alpha", None)
        defaults.pop("rand_beta", None)
        raw = entry.get("cli_options")
        if isinstance(raw, dict):
            defaults.update({k: v for k, v in raw.items() if k in defaults})
        entry["cli_options"] = defaults
        entry.setdefault("architecture", "")
        entry.setdefault("precision", "")
        entry.setdefault("unmapped_signatures", "")
        return defaults

    def _sigless_mode_uses_seed(entry: Dict[str, Any]) -> bool:
        return str(entry.get("merge_mode") or "").upper() in _SIGLESS_SEED_MODES

    def _sigless_model_count(self, entry: Dict[str, Any], mode_info: Dict[str, Any]) -> int:
        key = str(mode_info.get("key") or entry.get("merge_mode") or "").upper()
        opts = _sigless_ensure_cli_options(entry)
        if key in {"NOIN", "RM"}:
            return 1
        if mode_info.get("needs_m2") or bool(opts.get("turbo")) or bool(opts.get("deturbo")):
            return 3
        return 2

    def _sigless_option_changed(self, entry: Dict[str, Any], key: str, value, default):
        opts = _sigless_ensure_cli_options(entry)
        if isinstance(default, bool):
            opts[key] = bool(value)
        else:
            opts[key] = str(value or "")
        self._after_entry_change()

    def _sigless_special_option_row(self, parent, entry: Dict[str, Any], key: str, label: str, choices: List[str]):
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        Label(row, text=label, width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=str(entry.get(key) or ""))
        combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=18)
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        var.trace_add("write", lambda *_args, e=entry, k=key, v=var: (e.__setitem__(k, v.get()), self._after_entry_change()))
        self._attach_tooltip(row, self._right_help(label).get("detail", ""))

    def _sigless_build_unmapped_editor(self, parent, entry: Dict[str, Any]):
        box = LabelFrame(parent, text="Unmapped Legacy Signatures", padx=6, pady=6)
        box.pack(fill="x", pady=4)
        msg = Label(box, text="Unsupported legacy @/-- tokens are preserved here.", anchor="w", justify="left", fg="#64748b")
        msg.pack(fill="x", pady=(0, 4))
        txt = Text(box, height=3, font=("Consolas", 10), undo=True, maxundo=-1)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", str(entry.get("unmapped_signatures") or ""))
        def sync(_event=None, e=entry, w=txt):
            e["unmapped_signatures"] = w.get("1.0", "end-1c").strip()
            self._after_entry_change()
        txt.bind("<KeyRelease>", sync)
        self._bind_multiline_text_scroll_priority(txt)
        self._attach_tooltip(box, self._right_help("Unmapped Legacy Signatures").get("detail", ""))

    _SIGLESS_CLI_SPECS = copy.deepcopy(_LORA_MERGE_CLI_SPECS) if '_LORA_MERGE_CLI_SPECS' in globals() else {}
    try:
        # Drop random alpha/beta from Checkpoint Options.
        new_groups = []
        for group_name, fields in _SIGLESS_CLI_SPECS.get("Checkpoint Merge", {}).get("groups", []):
            filtered = [f for f in fields if f[0] not in {"rand_alpha", "rand_beta", "seed"}]
            if filtered:
                new_groups.append((group_name, filtered))
        _SIGLESS_CLI_SPECS.setdefault("Checkpoint Merge", {})["groups"] = new_groups
    except Exception:
        pass

    def _sigless_build_cli_options_panel(self, parent, entry: Dict[str, Any], entry_type: str):
        specs = copy.deepcopy(_SIGLESS_CLI_SPECS.get(entry_type, {}).get("groups", []))
        if entry_type == "Checkpoint Merge" and _sigless_mode_uses_seed(entry):
            specs.insert(1, ("Seed", [("seed", "Seed", "text", "")]))
        if not specs and entry_type not in ("Checkpoint Merge", "LoRA Bake"):
            return
        opts = _sigless_ensure_cli_options(entry)
        try:
            wrapper = self._build_collapsible_section(
                parent,
                f"{entry_type} Options",
                key=f"sigless_cli_options_{entry_type}_{entry.get('id','')}",
                default_open=False,
                body_fill="x",
                body_expand=False,
                padx=6,
                pady=6,
            )
        except Exception:
            wrapper = self._build_labeled_frame(parent, f"{entry_type} Options")

        if entry_type in ("Checkpoint Merge", "LoRA Bake"):
            basics = LabelFrame(wrapper, text="Basic", padx=6, pady=6)
            basics.pack(fill="x", pady=4)
            _sigless_special_option_row(self, basics, entry, "architecture", "Architecture", _SIGLESS_ARCH_CHOICES)
            _sigless_special_option_row(self, basics, entry, "precision", "Precision", _SIGLESS_PRECISION_CHOICES)

        for group_name, fields in specs:
            gf = LabelFrame(wrapper, text=group_name, padx=6, pady=6)
            gf.pack(fill="x", pady=4)
            for idx, field in enumerate(fields):
                key, label, kind, default = field[:4]
                row = Frame(gf)
                row.grid(row=idx // 2, column=(idx % 2) * 2, columnspan=2, sticky="ew", padx=3, pady=2)
                gf.grid_columnconfigure(1, weight=1)
                gf.grid_columnconfigure(3, weight=1)
                Label(row, text=label, width=18, anchor="w").pack(side="left")
                if kind == "bool":
                    var = tk.BooleanVar(value=bool(opts.get(key, default)))
                    chk = ttk.Checkbutton(row, variable=var, command=lambda v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                    chk.pack(side="left")
                elif kind == "choice":
                    choices = field[4] if len(field) > 4 else []
                    var = tk.StringVar(value=str(opts.get(key, default) or default))
                    combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=16)
                    self._bind_combobox_mousewheel_passthrough(combo)
                    combo.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                else:
                    var = tk.StringVar(value=str(opts.get(key, default) or ""))
                    ent = Entry(row, textvariable=var, width=28)
                    ent.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
        _sigless_build_unmapped_editor(self, wrapper, entry)
        reset_row = Frame(wrapper)
        reset_row.pack(fill="x", pady=(4, 0))
        def reset_options(e=entry, et=entry_type):
            e["cli_options"] = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(et, {})) if '_LORA_MERGE_DEFAULTS' in globals() else {}
            e["cli_options"].pop("rand_alpha", None)
            e["cli_options"].pop("rand_beta", None)
            self._after_entry_change()
            self._schedule_rerender_current_line()
        ttk.Button(reset_row, text="Reset Options", command=reset_options).pack(side="right")

    def _sigless_ratio_presets(self) -> Dict[str, Any]:
        self._planner_dev_ensure_config() if hasattr(self, "_planner_dev_ensure_config") else None
        presets = self.config.setdefault("ratio_presets", {})
        if not isinstance(presets, dict):
            presets = {}
            self.config["ratio_presets"] = presets
        return presets

    def _planner_save_ratio_preset(self, ratio: Dict[str, Any]):
        name = simpledialog.askstring("Save Ratio Preset", "Preset name:", parent=self.root)
        if not name:
            return
        _sigless_ratio_presets(self)[name.strip()] = copy.deepcopy(ratio)
        self._schedule_config_save()
        messagebox.showinfo("Save Ratio Preset", f"Saved ratio preset: {name.strip()}")

    def _planner_load_ratio_preset(self, ratio: Dict[str, Any]):
        presets = _sigless_ratio_presets(self)
        if not presets:
            messagebox.showinfo("Load Ratio Preset", "No ratio presets saved yet.")
            return
        name = simpledialog.askstring("Load Ratio Preset", "Preset name:\n" + "\n".join(sorted(presets)), parent=self.root)
        if not name:
            return
        key = name.strip()
        if key not in presets:
            messagebox.showwarning("Load Ratio Preset", f"Preset not found: {key}")
            return
        ratio.clear()
        ratio.update(copy.deepcopy(presets[key]))
        self._after_entry_change()
        self._schedule_rerender_current_line()

    def _planner_build_ratio_preset_buttons(self, parent, ratio: Dict[str, Any]):
        row = Frame(parent)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Save Ratio Preset", command=lambda r=ratio: self._planner_save_ratio_preset(r)).pack(side="left", padx=(0, 4))
        ttk.Button(row, text="Load Ratio Preset", command=lambda r=ratio: self._planner_load_ratio_preset(r)).pack(side="left")

    _sigless_orig_build_ratio_value_widget = ModelPlannerApp._build_ratio_value_widget
    def _build_ratio_value_widget_sigless(self, parent, ratio: Dict[str, Any], allow_block_weight: bool):
        if str(ratio.get("mode") or "") == "Randomize":
            row = Frame(parent)
            row.pack(fill="x", pady=3)
            Label(row, text="Random Range", width=18, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(ratio.get("value") or "0.0,1.0"))
            ent = Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True, padx=4)
            var.trace_add("write", lambda *_args, r=ratio, v=var: self._sync_ratio_value(r, v.get()))
            Label(parent, text="Example: 0.05,0.20 or block/random syntax accepted by backend rand_ratio().", fg="#64748b", anchor="w", justify="left", wraplength=720).pack(fill="x", padx=4)
            return
        return _sigless_orig_build_ratio_value_widget(self, parent, ratio, allow_block_weight)

    def _build_ratio_section_sigless(self, parent, entry: Dict[str, Any], key: str, title: str):
        wrapper = self._build_collapsible_section(
            parent,
            title,
            key=f"ratio_section_{entry.get('id', '')}_{key}",
            default_open=True,
            body_fill="x",
            body_expand=False,
        )
        ratio = entry.setdefault(key, default_ratio("Single"))
        row = Frame(wrapper)
        row.pack(fill="x", pady=3)
        label_widget = Label(row, text="Ratio Mode", width=18, anchor="w")
        label_widget.pack(side="left")
        mode_var = tk.StringVar(value=ratio.get("mode", "Single"))
        combo = ttk.Combobox(row, textvariable=mode_var, values=RATIO_MODES, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def sync_mode(*_args):
            ratio["mode"] = mode_var.get()
            if ratio["mode"] == "Single" and not ratio.get("value"):
                ratio["value"] = "0.5"
            elif ratio["mode"] == "Randomize" and not ratio.get("value"):
                ratio["value"] = "0.0,1.0"
            elif ratio["mode"] == "Block weight" and not ratio.get("value"):
                ratio["value"] = self._serialize_block_values([0.0] * len(self._current_block_names()), trailing_comma=False)
            self._schedule_rerender_current_line()
        mode_var.trace_add("write", sync_mode)
        self._attach_tooltip(label_widget, self._right_help("Ratio Mode").get("detail", ""))
        self._add_right_inline_help(wrapper, "Ratio Mode")
        self._build_ratio_value_widget(wrapper, ratio, allow_block_weight=True)
        self._planner_build_ratio_preset_buttons(wrapper, ratio)

    def _sigless_lora_ratio_mode_combo(self, row, ratio: Dict[str, Any], on_ratio_mode):
        ratio_var = tk.StringVar(value=ratio.get("mode", "Single"))
        combo = ttk.Combobox(row, textvariable=ratio_var, values=["Single", "Elemental"], state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        ratio_var.trace_add("write", on_ratio_mode)
        return combo, ratio_var

    def _render_checkpoint_merge_entry_sigless(self, parent, entry: Dict[str, Any]):
        _sigless_ensure_cli_options(entry)
        frame = self._build_labeled_frame(parent, "Checkpoint Merge")
        ckpts = self._collect_available_models(self.current_index).get("Checkpoint", [])
        mode_labels = [f"{m['key']} - {m['label']}" for m in self.merge_modes]
        current_mode = entry.get("merge_mode", self.merge_modes[0]["key"])
        current_label = next((f"{m['key']} - {m['label']}" for m in self.merge_modes if m["key"] == current_mode), mode_labels[0])
        row = Frame(frame)
        row.pack(fill="x", pady=3)
        Label(row, text="Merge Mode", width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=current_label)
        combo = ttk.Combobox(row, textvariable=var, values=mode_labels, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def sync_mode(*_args):
            label = var.get()
            key = label.split(" - ", 1)[0]
            entry["merge_mode"] = key
            self._schedule_rerender_current_line()
        var.trace_add("write", sync_mode)
        mode_info = self.merge_mode_map.get(entry.get("merge_mode"), self.merge_modes[0])
        model_count = _sigless_model_count(self, entry, mode_info)
        if model_count >= 1:
            self._build_combo_row(frame, "Model 0", entry, "model0", ckpts or [""])
        if model_count >= 2:
            self._build_combo_row(frame, "Model 1", entry, "model1", ckpts or [""])
        if model_count >= 3:
            self._build_combo_row(frame, "Model 2", entry, "model2", ckpts or [""])
        if mode_info.get("key") != "CLIPXOR":
            self._build_ratio_section(parent, entry, "alpha", "Alpha")
        if mode_info.get("needs_beta"):
            self._build_ratio_section(parent, entry, "beta", "Beta")
        out_frame = self._build_labeled_frame(parent, "Output")
        self._build_entry_row(out_frame, "Output Name", entry, "output_name")
        _sigless_build_cli_options_panel(self, parent, entry, "Checkpoint Merge")

    def _render_lora_bake_entry_sigless(self, parent, entry: Dict[str, Any]):
        _sigless_ensure_cli_options(entry)
        frame = self._build_labeled_frame(parent, "LoRA Bake")
        models = self._collect_available_models(self.current_index)
        ckpts = models.get("Checkpoint", [])
        if hasattr(self, "_build_source_backed_ref_row"):
            self._build_source_backed_ref_row(
                frame, "Checkpoint",
                lambda e=entry: e.get("checkpoint", ""),
                lambda value, e=entry: e.__setitem__("checkpoint", value),
                lambda e=entry: self._get_entry_slot_source(e, "checkpoint"),
                lambda spec, e=entry: self._set_entry_slot_source(e, "checkpoint", spec),
                ckpts or [""], ["Checkpoint"], allow_local=True,
            )
        else:
            self._build_combo_row(frame, "Checkpoint", entry, "checkpoint", ckpts or [""])
        self._build_entry_row(frame, "Output Name", entry, "output_name")
        add_lora_btn = ttk.Button(frame, text="+ Add LoRA", command=lambda: self._add_lora_block(entry))
        add_lora_btn.pack(anchor="w", padx=4, pady=4)
        lora_names = models.get("LoRA", []) + models.get("LyCORIS", [])
        loras = entry.get("loras", []) or []
        for idx, lora in enumerate(loras):
            block = self._build_labeled_frame(parent, f"LoRA {idx + 1}")
            top = Frame(block); top.pack(fill="x", pady=(0, 4))
            ttk.Button(top, text="-", width=3, command=lambda i=idx: self._remove_lora_block(entry, i)).pack(side="left", padx=(0, 4))
            if hasattr(self, "_move_lora_block"):
                up_btn = ttk.Button(top, text="↑ Up", width=7, command=lambda i=idx: self._move_lora_block(entry, i, -1))
                down_btn = ttk.Button(top, text="↓ Down", width=8, command=lambda i=idx: self._move_lora_block(entry, i, 1))
                if idx <= 0: up_btn.state(["disabled"])
                if idx >= len(loras) - 1: down_btn.state(["disabled"])
                up_btn.pack(side="left", padx=(0, 4)); down_btn.pack(side="left", padx=(0, 8))
            if hasattr(self, "_build_source_backed_ref_row"):
                self._build_source_backed_ref_row(block, "LoRA Name", lambda lo=lora: lo.get("name", ""), lambda value, lo=lora: lo.__setitem__("name", value), lambda lo=lora: self._get_lora_source(lo), lambda spec, lo=lora: self._set_lora_source(lo, spec), lora_names or [""], ["LoRA", "LyCORIS"], allow_local=True)
            else:
                self._build_combo_row(block, "LoRA Name", lora, "name", lora_names or [""])
            ratio = lora.setdefault("ratio", default_ratio("Single"))
            ratio_wrap = self._build_collapsible_section(block, "Ratio", key=f"lora_ratio_{entry.get('id', '')}_{idx}", default_open=True, body_fill="x", body_expand=False, padx=0, pady=4)
            row2 = Frame(ratio_wrap); row2.pack(fill="x", pady=3)
            Label(row2, text="Ratio Mode", width=18, anchor="w").pack(side="left")
            ratio_var = tk.StringVar(value=ratio.get("mode", "Single"))
            combo2 = ttk.Combobox(row2, textvariable=ratio_var, values=["Single", "Elemental"], state="readonly")
            self._bind_combobox_mousewheel_passthrough(combo2)
            combo2.pack(side="left", fill="x", expand=True, padx=4)
            def on_ratio_mode(*_args, lo=lora, rv=ratio_var):
                lo["ratio"]["mode"] = rv.get()
                if rv.get() == "Single": lo["ratio"].setdefault("value", "1.0")
                self._schedule_rerender_current_line()
            ratio_var.trace_add("write", on_ratio_mode)
            self._build_ratio_value_widget(ratio_wrap, lora["ratio"], allow_block_weight=False)
            self._planner_build_ratio_preset_buttons(ratio_wrap, lora["ratio"])
        _sigless_build_cli_options_panel(self, parent, entry, "LoRA Bake")

    def _render_lora_merge_entry_sigless(self, parent, entry: Dict[str, Any]):
        _sigless_ensure_cli_options(entry)
        frame = self._build_labeled_frame(parent, "LoRA Merge")
        models = self._collect_available_models(self.current_index)
        self._build_entry_row(frame, "Output Name", entry, "output_name")
        add_lora_btn = ttk.Button(frame, text="+ Add LoRA", command=lambda: self._add_lora_block(entry))
        add_lora_btn.pack(anchor="w", padx=4, pady=4)
        lora_names = models.get("LoRA", []) + models.get("LyCORIS", [])
        entry.setdefault("loras", [])
        for idx, lora in enumerate(entry.get("loras", []) or []):
            block = self._build_labeled_frame(parent, f"LoRA {idx + 1}")
            top = Frame(block); top.pack(fill="x")
            ttk.Button(top, text="-", width=3, command=lambda i=idx: self._remove_lora_block(entry, i)).pack(side="left", padx=(0, 4))
            if hasattr(self, "_build_source_backed_ref_row"):
                self._build_source_backed_ref_row(block, "LoRA Name", lambda lo=lora: lo.get("name", ""), lambda value, lo=lora: lo.__setitem__("name", value), lambda lo=lora: self._get_lora_source(lo), lambda spec, lo=lora: self._set_lora_source(lo, spec), lora_names or [""], ["LoRA", "LyCORIS"], allow_local=True)
            else:
                self._build_combo_row(block, "LoRA Name", lora, "name", lora_names or [""])
            ratio = lora.setdefault("ratio", default_ratio("Single"))
            ratio_wrap = self._build_collapsible_section(block, "Ratio", key=f"lora_merge_ratio_{entry.get('id', '')}_{idx}", default_open=True, body_fill="x", body_expand=False, padx=0, pady=4)
            row2 = Frame(ratio_wrap); row2.pack(fill="x", pady=3)
            Label(row2, text="Ratio Mode", width=18, anchor="w").pack(side="left")
            ratio_var = tk.StringVar(value=ratio.get("mode", "Single"))
            combo2 = ttk.Combobox(row2, textvariable=ratio_var, values=["Single", "Elemental"], state="readonly")
            self._bind_combobox_mousewheel_passthrough(combo2)
            combo2.pack(side="left", fill="x", expand=True, padx=4)
            def on_ratio_mode(*_args, lo=lora, rv=ratio_var):
                lo["ratio"]["mode"] = rv.get()
                if rv.get() == "Single": lo["ratio"].setdefault("value", "1.0")
                self._schedule_rerender_current_line()
            ratio_var.trace_add("write", on_ratio_mode)
            self._build_ratio_value_widget(ratio_wrap, lora["ratio"], allow_block_weight=False)
            self._planner_build_ratio_preset_buttons(ratio_wrap, lora["ratio"])
        _sigless_build_cli_options_panel(self, parent, entry, "LoRA Merge")

    def _on_plan_help_click_sigless(self, event):
        lb = getattr(self, "plan_listbox", None)
        if lb is None or not getattr(self, "visible_entry_indices", None):
            return
        idx = lb.nearest(event.y)
        if idx < 0 or idx >= len(self.visible_entry_indices):
            return
        bbox = lb.bbox(idx)
        if not bbox:
            return
        bx, by, bw, bh = bbox
        if not (by <= event.y <= by + bh):
            return
        near_right = event.x >= max(bx + bw - 36, lb.winfo_width() - 48)
        if not near_right:
            return
        model_idx = self.visible_entry_indices[idx]
        entry = self.plan_data.get("entries", [])[model_idx]
        problems = getattr(self, "_plan_problem_map_cache", {}).get(model_idx, [])
        text = self._build_plan_item_tooltip_text(model_idx, entry, problems)
        messagebox.showinfo(f"Line {model_idx + 1} Help", text)
        return "break"

    _sigless_orig_on_plan_motion = getattr(ModelPlannerApp, "_on_plan_list_motion", None)
    def _on_plan_list_motion_sigless(self, event):
        lb = getattr(self, "plan_listbox", None)
        if lb is not None and getattr(self, "visible_entry_indices", None):
            idx = lb.nearest(event.y)
            if 0 <= idx < len(self.visible_entry_indices):
                bbox = lb.bbox(idx)
                if bbox:
                    bx, by, bw, bh = bbox
                    near_right = event.x >= max(bx + bw - 36, lb.winfo_width() - 48)
                    if near_right and by <= event.y <= by + bh:
                        model_idx = self.visible_entry_indices[idx]
                        entry = self.plan_data.get("entries", [])[model_idx]
                        problems = getattr(self, "_plan_problem_map_cache", {}).get(model_idx, [])
                        text = self._build_plan_item_tooltip_text(model_idx, entry, problems)
                        if idx != getattr(self, "_plan_hover_index", None) or text != getattr(self, "_plan_hover_text", ""):
                            self._plan_hover_index = idx; self._plan_hover_text = text
                            self._show_plan_item_hover(event, text)
                        return
        if _sigless_orig_on_plan_motion:
            return _sigless_orig_on_plan_motion(self, event)

    def _build_left_panel_sigless(self, parent):
        self._planner_dev_ensure_config()
        base_builder = globals().get("_dev_base_build_left_panel")
        result = base_builder(self, parent) if callable(base_builder) else None
        try:
            section = self._build_collapsible_section(parent, "Developer (Experiment) Mode", key="left_developer_mode", default_open=bool(self.config.get("dev_mode", False)), body_fill="x", body_expand=False)
        except Exception:
            section = LabelFrame(parent, text="Developer (Experiment) Mode", padx=8, pady=8); section.pack(fill="x", pady=6)
        self.dev_mode_var = tk.BooleanVar(value=bool(self.config.get("dev_mode", False)))
        button_refs = []
        def refresh_dev_buttons():
            enabled = self._planner_dev_mode_enabled()
            self.config["dev_mode"] = enabled
            for btn in button_refs:
                try: btn.configure(state=("normal" if enabled else "disabled"))
                except Exception: pass
            self._schedule_config_save()
        top = Frame(section); top.pack(fill="x", pady=(0, 6))
        chk = ttk.Checkbutton(top, variable=self.dev_mode_var, command=refresh_dev_buttons)
        chk.pack(side="left", padx=(0, 4))
        Label(top, text="Enable Dev Mode", anchor="w").pack(side="left")
        grid = Frame(section); grid.pack(fill="x")
        for c in range(3): grid.grid_columnconfigure(c, weight=1)
        actions = [
            ("T2I Settings", lambda: self._planner_dev_only(self._planner_open_t2i_settings)),
            ("Ratio Sweep", lambda: self._planner_dev_only(self._planner_open_ratio_sweep)),
            ("Compare Images", lambda: self._planner_dev_only(self._planner_show_image_compare)),
            ("Ratio Diff", lambda: self._planner_dev_only(self._planner_show_ratio_diff)),
            ("Model Anatomy", lambda: self._planner_dev_only(self._planner_model_anatomy_scan)),
            ("LoRA Check", lambda: self._planner_dev_only(self._planner_lora_compatibility_check)),
        ]
        for idx, (label, cmd) in enumerate(actions):
            btn = ttk.Button(grid, text=label, command=cmd)
            btn.grid(row=idx//3, column=idx%3, sticky="ew", padx=3, pady=3)
            button_refs.append(btn)
        self._developer_mode_buttons = [(b, True) for b in button_refs]
        refresh_dev_buttons()
        return result

    # Replace methods after previous extension layers.
    ModelPlannerApp._build_ratio_value_widget = _build_ratio_value_widget_sigless
    ModelPlannerApp._build_ratio_section = _build_ratio_section_sigless
    ModelPlannerApp._planner_save_ratio_preset = _planner_save_ratio_preset
    ModelPlannerApp._planner_load_ratio_preset = _planner_load_ratio_preset
    ModelPlannerApp._planner_build_ratio_preset_buttons = _planner_build_ratio_preset_buttons
    ModelPlannerApp._render_checkpoint_merge_entry = _render_checkpoint_merge_entry_sigless
    ModelPlannerApp._render_lora_bake_entry = _render_lora_bake_entry_sigless
    ModelPlannerApp._render_lora_merge_entry = _render_lora_merge_entry_sigless
    ModelPlannerApp._on_plan_help_click = _on_plan_help_click_sigless
    ModelPlannerApp._on_plan_list_motion = _on_plan_list_motion_sigless
    ModelPlannerApp._build_left_panel = _build_left_panel_sigless
except Exception:
    pass


# -----------------------------------------------------------------------------
# Final UI polish: robust disk estimator, display-only arch/precision names,
# and clearly disabled Developer Mode controls.
# -----------------------------------------------------------------------------
try:
    _DISPLAY_PRECISION_CHOICES = ["", "FP16", "BF16", "FP8", "FP32"]
    _DISPLAY_ARCH_CHOICES = ["", "Auto", "SD1.5", "SDXL", "Flux", "ZImage", "Anima"]

    _PRECISION_DISPLAY_ALIASES = {
        "": "", "half": "FP16", "fp16": "FP16", "float16": "FP16", "16": "FP16",
        "bhalf": "BF16", "bf16": "BF16", "bfloat16": "BF16",
        "quarter": "FP8", "fp8": "FP8", "float8": "FP8", "8": "FP8",
        "fp32": "FP32", "float32": "FP32", "full": "FP32", "32": "FP32",
    }
    _ARCH_DISPLAY_ALIASES = {
        "": "", "auto": "Auto",
        "sd": "SD1.5", "sd15": "SD1.5", "sd1.5": "SD1.5", "sd-1.5": "SD1.5", "sd_1.5": "SD1.5",
        "xl": "SDXL", "sdxl": "SDXL", "sd-xl": "SDXL", "sd_xl": "SDXL",
        "flux": "Flux",
        "zi": "ZImage", "zimage": "ZImage", "z-image": "ZImage", "z_image": "ZImage",
        "am": "Anima", "anima": "Anima",
    }

    def _planner_display_precision_name(value):
        raw = str(value or "").strip()
        key = raw.lower().replace(" ", "")
        return _PRECISION_DISPLAY_ALIASES.get(key, raw.upper() if raw else "")

    def _planner_display_arch_name(value):
        raw = str(value or "").strip()
        key = raw.lower().replace(" ", "").replace("_", "-")
        return _ARCH_DISPLAY_ALIASES.get(key, raw[:1].upper() + raw[1:] if raw else "")

    # Replace global choice lists used by the sigless options renderer.
    _SIGLESS_PRECISION_CHOICES = list(_DISPLAY_PRECISION_CHOICES)
    _SIGLESS_ARCH_CHOICES = list(_DISPLAY_ARCH_CHOICES)

    # Normalize LoRA Merge architecture choices without duplicates in the UI.
    try:
        for idx, group in enumerate(_SIGLESS_CLI_SPECS.get("LoRA Merge", {}).get("groups", [])):
            gname, fields = group
            patched = []
            for field in fields:
                if len(field) >= 5 and field[0] == "merge_arch":
                    patched.append((field[0], field[1], field[2], "Auto", list(_DISPLAY_ARCH_CHOICES[1:])))
                else:
                    patched.append(field)
            _SIGLESS_CLI_SPECS["LoRA Merge"]["groups"][idx] = (gname, patched)
        if "_LORA_MERGE_DEFAULTS" in globals():
            _LORA_MERGE_DEFAULTS.setdefault("LoRA Merge", {})["merge_arch"] = "Auto"
    except Exception:
        pass

    def _sigless_special_option_row_display(self, parent, entry: Dict[str, Any], key: str, label: str, choices: List[str]):
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        Label(row, text=label, width=18, anchor="w").pack(side="left")
        if key == "precision":
            value = _planner_display_precision_name(entry.get(key))
            choices = _DISPLAY_PRECISION_CHOICES
        elif key == "architecture":
            value = _planner_display_arch_name(entry.get(key))
            choices = _DISPLAY_ARCH_CHOICES
        else:
            value = str(entry.get(key) or "")
        var = tk.StringVar(value=value)
        combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=18)
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def _sync(*_args, e=entry, k=key, v=var):
            e[k] = v.get()
            self._after_entry_change()
        var.trace_add("write", _sync)
        self._attach_tooltip(row, self._right_help(label).get("detail", ""))

    _sigless_special_option_row = _sigless_special_option_row_display

    _old_sigless_ensure_cli_options = _sigless_ensure_cli_options
    def _sigless_ensure_cli_options_display(entry: Dict[str, Any]) -> Dict[str, Any]:
        opts = _old_sigless_ensure_cli_options(entry)
        if entry.get("type") == "LoRA Merge" and "merge_arch" in opts:
            opts["merge_arch"] = _planner_display_arch_name(opts.get("merge_arch") or "Auto")
            entry["cli_options"] = opts
        if entry.get("architecture"):
            entry["architecture"] = _planner_display_arch_name(entry.get("architecture"))
        if entry.get("precision"):
            entry["precision"] = _planner_display_precision_name(entry.get("precision"))
        return opts

    _sigless_ensure_cli_options = _sigless_ensure_cli_options_display

    _old_sigless_option_changed = _sigless_option_changed
    def _sigless_option_changed_display(self, entry: Dict[str, Any], key: str, value, default):
        if key == "merge_arch":
            value = _planner_display_arch_name(value)
        return _old_sigless_option_changed(self, entry, key, value, default)

    _sigless_option_changed = _sigless_option_changed_display

    def _planner_nearest_existing_path(path_value):
        import os as _os
        from pathlib import Path as _Path
        raw = str(path_value or "").strip().strip('"')
        candidates = []
        if raw:
            try:
                p = _Path(raw).expanduser()
                candidates.append(p)
                candidates.extend(p.parents)
            except Exception:
                pass
        candidates.extend([_Path.cwd(), _Path.home(), _Path(_os.getenv("TMPDIR") or _os.getenv("TEMP") or _os.getenv("TMP") or _os.getcwd()), _Path(_os.getcwd())])
        seen = set()
        for cand in candidates:
            try:
                key = str(cand)
                if key in seen:
                    continue
                seen.add(key)
                if cand.exists():
                    return str(cand)
            except Exception:
                continue
        return _os.getcwd()

    def _planner_disk_usage_for_path(path_value):
        import shutil as _shutil
        p = _planner_nearest_existing_path(path_value)
        total, used, free = _shutil.disk_usage(p)
        return {"path": p, "total": total, "used": used, "free": free}

    def _planner_unique_disk_key(path_value):
        import os as _os
        p = _planner_nearest_existing_path(path_value)
        try:
            stat = _os.stat(p)
            return (getattr(stat, "st_dev", None), _os.path.splitdrive(_os.path.abspath(p))[0].lower())
        except Exception:
            return (_os.path.splitdrive(_os.path.abspath(p))[0].lower(), p)

    def _planner_collect_disk_usages(self):
        import os as _os, tempfile as _tempfile, string as _string
        targets = []
        try:
            if self.entries.get("workpath"):
                targets.append(("Workspace", self.entries.get("workpath").get()))
        except Exception:
            pass
        try:
            if self.entries.get("model_dir"):
                targets.append(("Model Dir", self.entries.get("model_dir").get()))
        except Exception:
            pass
        try:
            if self.entries.get("vae_dir"):
                targets.append(("VAE Dir", self.entries.get("vae_dir").get()))
        except Exception:
            pass
        try:
            fp = self.entries.get("filepath").get() if self.entries.get("filepath") else ""
            if fp:
                targets.append(("Plan File", str(Path(fp).expanduser().parent)))
        except Exception:
            pass
        targets.extend([("Current Directory", _os.getcwd()), ("Temp", _tempfile.gettempdir()), ("Home", str(Path.home()))])
        if _os.name == "nt":
            for letter in _string.ascii_uppercase:
                drive = f"{letter}:\\"
                if _os.path.exists(drive):
                    targets.append((f"Drive {letter}:", drive))
        else:
            for p in ("/", "/mnt", "/mnt/data", "/tmp"):
                if _os.path.exists(p):
                    targets.append((p, p))
        out = []
        seen = set()
        for label, p in targets:
            try:
                info = _planner_disk_usage_for_path(p)
                key = _planner_unique_disk_key(info["path"])
                if key in seen:
                    continue
                seen.add(key)
                info["label"] = label
                out.append(info)
            except Exception:
                continue
        return out

    def _planner_estimate_disk_runtime_robust(self):
        entries = self.plan_data.get("entries", []) if isinstance(getattr(self, "plan_data", None), dict) else []

        # FP16 / pruned checkpoint-size heuristics requested for peak estimation.
        checkpoint_size_by_base = {
            "SD1.5": 1.99,
            "SDXL": 6.46,
            "Flux": 22.17,
            "ZImage": 11.46,
            "Anima": 3.90,
        }
        lora_size_gb = 0.25

        def _gb_from_bytes(value) -> float:
            return float(value or 0) / (1024 ** 3)

        def _fmt_gb(value) -> str:
            return f"{float(value or 0.0):.2f} GB"

        def _display_base_name(value: str) -> str:
            raw = str(value or "").strip()
            if not raw:
                return "SDXL"
            try:
                if callable(globals().get("_planner_display_arch_name")):
                    shown = globals()["_planner_display_arch_name"](raw)
                    if shown:
                        return shown
            except Exception:
                pass
            key = raw.lower().replace(" ", "").replace("_", "").replace("-", "")
            aliases = {
                "sd": "SD1.5",
                "sd15": "SD1.5",
                "sd1.5": "SD1.5",
                "sdxl": "SDXL",
                "xl": "SDXL",
                "flux": "Flux",
                "zi": "ZImage",
                "zimage": "ZImage",
                "am": "Anima",
                "anima": "Anima",
            }
            return aliases.get(key, raw[:1].upper() + raw[1:])

        base = _display_base_name(self.base_model_var.get() if hasattr(self, "base_model_var") else "")
        checkpoint_size_gb = float(checkpoint_size_by_base.get(base, checkpoint_size_by_base.get("SDXL", 6.46)))

        def _kind_size_gb(kind: str) -> float:
            k = str(kind or "").strip().lower()
            if k in ("lora", "lycoris"):
                return lora_size_gb
            return checkpoint_size_gb

        def _path_size_gb(path_value: str, kind: str = "Checkpoint") -> float:
            raw = str(path_value or "").strip()
            if raw:
                try:
                    p = Path(raw).expanduser()
                    if p.exists() and p.is_file():
                        return _gb_from_bytes(p.stat().st_size)
                except Exception:
                    pass
            return _kind_size_gb(kind)

        def _alias_from_source_spec(spec, fallback: str = "") -> str:
            try:
                alias = self._source_alias_from_spec(spec, fallback)
                if alias:
                    return str(alias).strip()
            except Exception:
                pass
            if isinstance(spec, dict):
                alias = str(spec.get("alias") or "").strip()
                if alias:
                    return alias
                local_path = str(spec.get("local_path") or "").strip()
                if local_path:
                    return Path(local_path).stem
            return str(fallback or "").strip()

        def _source_size_gb(spec, kind: str = "Checkpoint") -> float:
            if isinstance(spec, dict):
                source_kind = str(spec.get("kind") or kind or "Checkpoint").strip()
                if str(spec.get("mode") or "") == "local":
                    return _path_size_gb(spec.get("local_path") or "", source_kind)
                return _kind_size_gb(source_kind)
            return _kind_size_gb(kind)

        def _active_total(active: Dict[str, Dict[str, Any]]) -> float:
            return sum(float(item.get("size_gb", 0.0) or 0.0) for item in active.values())

        def _active_lookup_size(active: Dict[str, Dict[str, Any]], alias: str, kind: str = "Checkpoint") -> float:
            name = str(alias or "").strip()
            if name and name in active:
                return float(active[name].get("size_gb", 0.0) or 0.0)
            return _kind_size_gb(kind)

        def _register_active(active: Dict[str, Dict[str, Any]], alias: str, kind: str, size_gb: float, source: str = ""):
            name = str(alias or "").strip()
            if not name:
                return
            active[name] = {
                "kind": str(kind or "Checkpoint").strip() or "Checkpoint",
                "size_gb": max(0.0, float(size_gb or 0.0)),
                "source": str(source or "").strip(),
            }

        def _entry_cli_options(entry: Dict[str, Any]) -> Dict[str, Any]:
            try:
                opts = self._sigless_ensure_cli_options(entry)
                return opts if isinstance(opts, dict) else {}
            except Exception:
                value = entry.get("cli_options")
                return value if isinstance(value, dict) else {}

        def _entry_used_models(entry: Dict[str, Any]) -> List[tuple[str, str, Any]]:
            etype = str(entry.get("type") or "")
            out: List[tuple[str, str, Any]] = []
            if etype == "Checkpoint Merge":
                mode = str(entry.get("merge_mode") or "WS").strip() or "WS"
                mode_info = self.merge_mode_map.get(mode, {}) if hasattr(self, "merge_mode_map") else {}
                opts = _entry_cli_options(entry)
                use_model2 = bool(mode_info.get("needs_m2")) or bool(opts.get("turbo")) or bool(opts.get("deturbo"))
                for slot in ("model0", "model1"):
                    alias = str(entry.get(slot) or "").strip()
                    spec = None
                    try:
                        spec = self._get_entry_slot_source(entry, slot)
                    except Exception:
                        spec = None
                    alias = alias or _alias_from_source_spec(spec)
                    if alias or spec:
                        out.append((alias, "Checkpoint", spec))
                if use_model2:
                    spec = None
                    try:
                        spec = self._get_entry_slot_source(entry, "model2")
                    except Exception:
                        spec = None
                    alias = str(entry.get("model2") or "").strip() or _alias_from_source_spec(spec)
                    if alias or spec:
                        out.append((alias, "Checkpoint", spec))
            elif etype == "LoRA Bake":
                spec = None
                try:
                    spec = self._get_entry_slot_source(entry, "checkpoint")
                except Exception:
                    spec = None
                alias = str(entry.get("checkpoint") or "").strip() or _alias_from_source_spec(spec)
                if alias or spec:
                    out.append((alias, "Checkpoint", spec))
                for lora in entry.get("loras", []) or []:
                    spec = None
                    try:
                        spec = self._get_lora_source(lora)
                    except Exception:
                        spec = None
                    kind = "LoRA"
                    if isinstance(spec, dict):
                        kind = str(spec.get("kind") or kind)
                    alias = str(lora.get("name") or "").strip() or _alias_from_source_spec(spec)
                    if alias or spec:
                        out.append((alias, kind, spec))
            elif etype == "LoRA Merge":
                for lora in entry.get("loras", []) or []:
                    spec = None
                    try:
                        spec = self._get_lora_source(lora)
                    except Exception:
                        spec = None
                    kind = "LoRA"
                    if isinstance(spec, dict):
                        kind = str(spec.get("kind") or kind)
                    alias = str(lora.get("name") or "").strip() or _alias_from_source_spec(spec)
                    if alias or spec:
                        out.append((alias, kind, spec))
            return out

        output_count = sum(1 for e in entries if e.get("type") in ("Checkpoint Merge", "LoRA Bake") and str(e.get("output_name") or "").strip())
        lora_output_count = sum(1 for e in entries if e.get("type") == "LoRA Merge" and str(e.get("output_name") or "").strip())

        download_count = 0
        local_paths = []
        for e in entries:
            try:
                if e.get("type") == "Download Model":
                    download_count += 1
                embedded_sources = list(self._iter_embedded_sources(e))
                download_count += sum(1 for *_x, s in embedded_sources if isinstance(s, dict) and s.get("mode") == "download")
                for *_x, spec in embedded_sources:
                    if isinstance(spec, dict) and spec.get("mode") == "local" and spec.get("local_path"):
                        local_paths.append(spec.get("local_path"))
            except Exception:
                pass
            if e.get("type") == "Local Model" and e.get("local_path"):
                local_paths.append(e.get("local_path"))

        local_bytes = 0
        for pth in set(local_paths):
            try:
                local_bytes += Path(pth).expanduser().stat().st_size
            except Exception:
                pass

        generated_gb = output_count * checkpoint_size_gb + lora_output_count * lora_size_gb
        estimated_total = generated_gb + local_bytes / (1024**3)

        # Peak scratch-space estimate. At each merge-like line:
        #   active remaining models + models used by this line + one new output model.
        # This intentionally double-counts inputs that are also active, matching the
        # requested macOS-style conservative working-space formula.
        active_models: Dict[str, Dict[str, Any]] = {}
        most_output_size_gb = 0.0
        most_output_line = ""
        most_output_breakdown = (0.0, 0.0, 0.0)

        for idx, entry in enumerate(entries, start=1):
            etype = str(entry.get("type") or "")
            if etype == "Remove Model":
                alias = str(entry.get("model") or "").strip()
                if alias:
                    active_models.pop(alias, None)
                continue

            if etype == "Download Model":
                alias = str(entry.get("model_name") or "").strip()
                kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
                _register_active(active_models, alias, kind, _kind_size_gb(kind), "download")
                continue

            if etype == "Local Model":
                local_path = str(entry.get("local_path") or "").strip()
                alias = str(entry.get("model_name") or "").strip() or (Path(local_path).stem if local_path else "")
                kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
                _register_active(active_models, alias, kind, _path_size_gb(local_path, kind), local_path)
                continue

            if etype in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
                remaining_gb = _active_total(active_models)
                used_gb = 0.0
                for alias, kind, spec in _entry_used_models(entry):
                    if isinstance(spec, dict):
                        source_size = _source_size_gb(spec, kind)
                        if alias and alias in active_models:
                            used_gb += _active_lookup_size(active_models, alias, kind)
                        else:
                            used_gb += source_size
                    else:
                        used_gb += _active_lookup_size(active_models, alias, kind)

                output_alias = str(entry.get("output_name") or "").strip()
                output_kind = "LoRA" if etype == "LoRA Merge" else "Checkpoint"
                one_output_gb = _kind_size_gb(output_kind)
                peak_gb = remaining_gb + used_gb + one_output_gb
                if peak_gb > most_output_size_gb:
                    most_output_size_gb = peak_gb
                    most_output_line = f"Line {idx}: {etype}" + (f" -> {output_alias}" if output_alias else "")
                    most_output_breakdown = (remaining_gb, used_gb, one_output_gb)

                if output_alias:
                    _register_active(active_models, output_alias, output_kind, one_output_gb, f"line {idx}")
                continue

        if most_output_size_gb <= 0.0:
            most_output_size_gb = _active_total(active_models)
            most_output_line = "No merge-like line detected"
            most_output_breakdown = (most_output_size_gb, 0.0, 0.0)

        disk_infos = _planner_collect_disk_usages(self)
        try:
            work_target = self.entries.get("workpath").get() if self.entries.get("workpath") else os.getcwd()
            target_disk = _planner_disk_usage_for_path(work_target)
        except Exception:
            target_disk = disk_infos[0] if disk_infos else {"path": os.getcwd(), "free": 0, "total": 0, "used": 0}
        free_gb = float(target_disk.get("free", 0)) / (1024**3)
        risk_basis = max(float(estimated_total or 0.0), float(most_output_size_gb or 0.0))
        risk = "LOW" if free_gb > risk_basis * 1.4 else "MEDIUM" if free_gb > risk_basis * 0.9 else "HIGH"

        remaining_peak_gb, used_peak_gb, output_peak_gb = most_output_breakdown
        lines = [
            "Disk / Runtime Estimator", "",
            f"Base model: {base}",
            f"Reference checkpoint size: {_fmt_gb(checkpoint_size_gb)} (FP16, pruned heuristic)",
            f"Merge/Bake outputs: {output_count}",
            f"LoRA Merge outputs: {lora_output_count}",
            f"Download source count: {download_count}",
            f"Known local source size: {local_bytes/(1024**3):.2f} GB",
            f"Estimated generated outputs: {generated_gb:.2f} GB ({checkpoint_size_gb:.2f} GB/checkpoint heuristic, {lora_size_gb:.2f} GB/LoRA heuristic)",
            f"Most output size: {most_output_size_gb:.2f} GB",
            f"  Peak line: {most_output_line}",
            f"  Formula: remaining models {remaining_peak_gb:.2f} GB + current input models {used_peak_gb:.2f} GB + one output model {output_peak_gb:.2f} GB",
            f"Current free disk: {free_gb:.2f} GB  ({target_disk.get('path', '')})",
            f"Risk: {risk}", "",
            "Detected disks / mount points:",
        ]
        if disk_infos:
            for info in disk_infos:
                lines.append(f"- {info.get('label','Disk')}: {info.get('path','')} | free {_gb_from_bytes(info.get('free',0)):.2f} GB / total {_gb_from_bytes(info.get('total',0)):.2f} GB")
        else:
            lines.append("- No disk usage information could be collected.")
        lines.extend([
            "",
            "Suggestions:",
            "- Use Optimize Plan before large sweep runs.",
            "- Insert Remove Model after last use for long chains.",
            "- Keep workspace/model/output paths on a disk with enough free space.",
        ])
        self._show_scrollable_text_dialog("Disk / Runtime Estimator", "\n".join(lines))

    ModelPlannerApp._planner_estimate_disk_runtime = _planner_estimate_disk_runtime_robust

    def _build_left_panel_final_dev(self, parent):
        self._planner_dev_ensure_config()
        base_builder = globals().get("_dev_base_build_left_panel")
        result = base_builder(self, parent) if callable(base_builder) else None
        try:
            section = self._build_collapsible_section(parent, "Developer (Experiment) Mode", key="left_developer_mode", default_open=bool(self.config.get("dev_mode", False)), body_fill="x", body_expand=False)
        except Exception:
            section = LabelFrame(parent, text="Developer (Experiment) Mode", padx=8, pady=8)
            section.pack(fill="x", pady=6)
        self.dev_mode_var = tk.BooleanVar(value=bool(self.config.get("dev_mode", False)))
        button_refs = []
        colors = self._theme_colors() if hasattr(self, "_theme_colors") else {"surface": "#ffffff", "text": "#111111", "muted": "#777777", "border": "#cccccc"}
        def _button_colors(enabled):
            if enabled:
                return {"state": "normal", "fg": colors.get("text", "#111111"), "disabledforeground": colors.get("muted", "#777777"), "relief": "raised"}
            return {"state": "disabled", "fg": colors.get("muted", "#777777"), "disabledforeground": colors.get("muted", "#777777"), "relief": "groove"}
        def refresh_dev_buttons():
            enabled = self._planner_dev_mode_enabled()
            self.config["dev_mode"] = enabled
            for btn in button_refs:
                try:
                    btn.configure(**_button_colors(enabled))
                except Exception:
                    try:
                        btn.configure(state=("normal" if enabled else "disabled"))
                    except Exception:
                        pass
            self._schedule_config_save()
        top = Frame(section)
        top.pack(fill="x", pady=(0, 6))
        chk = ttk.Checkbutton(top, variable=self.dev_mode_var, command=refresh_dev_buttons)
        chk.pack(side="left", padx=(0, 4))
        header_btn = tk.Button(top, text="Developer Mode", padx=8, pady=2, relief="groove", state="disabled", disabledforeground=colors.get("muted", "#777777"))
        header_btn.pack(side="left", fill="x", expand=True)
        grid = Frame(section)
        grid.pack(fill="x")
        for c in range(3):
            grid.grid_columnconfigure(c, weight=1)
        actions = [
            ("T2I Settings", lambda: self._planner_dev_only(self._planner_open_t2i_settings)),
            ("Ratio Sweep", lambda: self._planner_dev_only(self._planner_open_ratio_sweep)),
            ("Compare Images", lambda: self._planner_dev_only(self._planner_show_image_compare)),
            ("Ratio Diff", lambda: self._planner_dev_only(self._planner_show_ratio_diff)),
            ("Model Anatomy", lambda: self._planner_dev_only(self._planner_model_anatomy_scan)),
            ("LoRA Check", lambda: self._planner_dev_only(self._planner_lora_compatibility_check)),
        ]
        for idx, (label, cmd) in enumerate(actions):
            btn = tk.Button(grid, text=label, command=cmd, padx=6, pady=3, disabledforeground=colors.get("muted", "#777777"), fg=colors.get("text", "#111111"))
            btn.grid(row=idx//3, column=idx%3, sticky="ew", padx=3, pady=3)
            button_refs.append(btn)
        self._developer_mode_buttons = [(b, True) for b in button_refs]
        refresh_dev_buttons()
        return result

    ModelPlannerApp._build_left_panel = _build_left_panel_final_dev
except Exception:
    pass


# -----------------------------------------------------------------------------
# Hover help expansion for Options, choices, merge modes, and model selectors.
# -----------------------------------------------------------------------------
try:
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

    _TOOLTIP_CATALOG = PLANNER_TOOLTIP_CATALOG
    _OPTION_HELP_TEXT = dict(_TOOLTIP_CATALOG.get("options") or {})
    _CHOICE_HELP_TEXT = dict(_TOOLTIP_CATALOG.get("choices") or {})
    _MODE_HELP_TEXT = dict(_TOOLTIP_CATALOG.get("modes") or {})

    def _planner_mode_key_from_label(raw: str) -> str:
        text = str(raw or "").strip()
        if " - " in text:
            text = text.split(" - ", 1)[0]
        return text.strip()

    def _planner_mode_catalog_entry(mode_key: str):
        key = str(mode_key or "").strip()
        if not key:
            return None
        if key in _MODE_HELP_TEXT:
            return _MODE_HELP_TEXT.get(key)
        key_lower = key.lower()
        for candidate_key, value in _MODE_HELP_TEXT.items():
            if str(candidate_key).lower() == key_lower:
                return value
        return None

    def _planner_mode_label_from_info(mode_key: str, mode_info=None) -> str:
        entry = _planner_mode_catalog_entry(mode_key)
        if isinstance(entry, dict):
            label = str(entry.get("label") or entry.get("name") or "").strip()
            if label:
                return label
        if isinstance(mode_info, dict):
            label = str(mode_info.get("label") or mode_info.get("name") or "").strip()
            if label:
                return label
        return str(mode_key or "").strip()

    def _planner_mode_detail_text(mode_key: str, mode_info=None, *, fallback: str = "Backend merge mode.") -> str:
        entry = _planner_mode_catalog_entry(mode_key)
        if isinstance(entry, str):
            return entry.strip() or fallback
        if isinstance(entry, dict):
            parts = []
            short = str(entry.get("short") or "").strip()
            detail = str(entry.get("detail") or entry.get("description") or entry.get("help") or "").strip()
            if short and detail and short != detail and short not in detail:
                parts.append(short)
            if detail:
                parts.append(detail)
            req = []
            needs_m2 = bool(entry.get("needs_m2") or (isinstance(mode_info, dict) and mode_info.get("needs_m2")))
            uses_beta = bool(entry.get("uses_beta") or entry.get("needs_beta") or (isinstance(mode_info, dict) and mode_info.get("needs_beta")))
            if needs_m2:
                req.append("Model 2")
            if uses_beta:
                req.append("Beta")
            if req and not any("Required inputs:" in p or "Inputs:" in p for p in parts):
                parts.append("Required inputs: " + ", ".join(req))
            if parts:
                return "\n\n".join(parts).strip()
        return fallback

    def _planner_attach_dynamic_tooltip(self, widgets, provider, *, delay=700, wraplength=860):
        if not isinstance(widgets, (list, tuple, set)):
            widgets = [widgets]
        for widget in widgets:
            if widget is not None:
                _PlannerDynamicTooltip(widget, provider, delay=delay, wraplength=wraplength)

    def _planner_mode_choice_tooltip(self, current_label: str = "") -> str:
        current_key = str(current_label or "").split(" - ", 1)[0].strip()
        lines = ["Merge Mode", ""]
        if current_key:
            lines.append(f"Selected: {current_key}")
            lines.append(_planner_mode_detail_text(current_key, None, fallback="Backend merge mode. Check merge.py for exact tensor behavior."))
            lines.append("")
        lines.append("Available modes:")
        try:
            modes = list(getattr(self, "merge_modes", []) or [])
        except Exception:
            modes = []
        seen = set()
        for m in modes:
            key = str(m.get("key") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            label = str(m.get("label") or key)
            detail = _planner_mode_detail_text(key, m, fallback="Backend-provided merge mode.")
            req = []
            if m.get("needs_m2"):
                req.append("Model 2")
            if m.get("needs_beta"):
                req.append("Beta")
            suffix = f" Requires: {', '.join(req)}." if req else ""
            lines.append(f"- {key} ({label}): {detail}{suffix}")
        return "\n".join(lines)

    def _planner_choice_help_lines(label_or_key: str, choices=None) -> list[str]:
        label = str(label_or_key or "")
        table = _CHOICE_HELP_TEXT.get(label) or _CHOICE_HELP_TEXT.get(label.replace(" ", "_")) or _CHOICE_HELP_TEXT.get(label.lower())
        out = []
        if table:
            for choice in choices or table.keys():
                c = str(choice)
                if c == "":
                    continue
                out.append(f"- {c}: {table.get(c, table.get(c.lower(), '')) or 'Available choice.'}")
        elif choices:
            for choice in choices:
                c = str(choice)
                if c:
                    out.append(f"- {c}")
        return out

    def _planner_option_tooltip_text(self, entry_type: str, key: str, label: str, kind: str = "", choices=None, current=None) -> str:
        key = str(key or "")
        label = str(label or key)
        lines = [label]
        if entry_type:
            lines.append(f"Scope: {entry_type}")
        detail = _OPTION_HELP_TEXT.get(key) or self._right_help(label).get("detail", "") if hasattr(self, "_right_help") else ""
        if detail:
            lines.extend(["", detail])
        if current is not None and str(current) != "":
            lines.extend(["", f"Current: {current}"])
        if choices:
            lines.extend(["", "Choices:"])
            lines.extend(_planner_choice_help_lines(key, choices) or _planner_choice_help_lines(label, choices))
        return "\n".join(lines).strip()

    def _planner_model_choice_tooltip_text(self, label: str, current: str = "", values=None) -> str:
        current = str(current or "").strip()
        label = str(label or "Model")
        lines = [label, ""]
        if current:
            lines.append(f"Selected: {current}")
        records = []
        try:
            records = list(self._planner_source_records()) if hasattr(self, "_planner_source_records") else []
        except Exception:
            records = []
        recs = [r for r in records if str(r.get("alias") or "") == current]
        if recs:
            lines.append("Source / previous registration:")
            for r in recs[-4:]:
                detail = str(r.get("detail") or "")
                if len(detail) > 160:
                    detail = detail[:157] + "…"
                embedded = "embedded " if r.get("embedded") else ""
                lines.append(f"- line {int(r.get('idx', -1)) + 1}: {embedded}{r.get('mode', '')} {r.get('kind', '')} → {detail}")
        else:
            try:
                analysis = self._planner_analysis() if hasattr(self, "_planner_analysis") else {}
                lines_for_alias = (analysis.get("producer_lines") or {}).get(current, [])
                if lines_for_alias:
                    lines.append("Produced / registered by:")
                    for i in lines_for_alias[-4:]:
                        try:
                            entry = (self.plan_data or {}).get("entries", [])[i]
                            summary = self._line_summary(entry) if hasattr(self, "_line_summary") else str(entry.get("type"))
                            lines.append(f"- line {i + 1}: {summary}")
                        except Exception:
                            lines.append(f"- line {i + 1}")
            except Exception:
                pass
        if values:
            vals = [str(v) for v in values if str(v).strip()]
            if vals:
                lines.extend(["", "Available choices:"])
                for v in vals[:24]:
                    marker = "*" if v == current else "-"
                    extra = ""
                    for r in records:
                        if str(r.get("alias") or "") == v:
                            mode = str(r.get("mode") or "")
                            kind = str(r.get("kind") or "")
                            detail = str(r.get("detail") or "")
                            short = Path(detail).name if mode == "local" else (detail[:48] + ("…" if len(detail) > 48 else ""))
                            extra = f" [{mode} {kind}: {short}]"
                    lines.append(f"{marker} {v}{extra}")
                if len(vals) > 24:
                    lines.append(f"... {len(vals) - 24} more choices")
        return "\n".join(lines).strip()

    # Patch normal combo rows so model-like selectors show selected source / previous use.
    _hover_prev_build_combo_row = ModelPlannerApp._build_combo_row
    def _build_combo_row_hover(self, parent, label: str, entry: Dict[str, Any], key: str, values: List[str], callback=None):
        before = set(parent.winfo_children())
        var = _hover_prev_build_combo_row(self, parent, label, entry, key, values, callback)
        try:
            new_children = [w for w in parent.winfo_children() if w not in before]
            target_row = new_children[-1] if new_children else None
            combos = []
            labels = []
            if target_row is not None:
                for child in target_row.winfo_children():
                    if isinstance(child, ttk.Combobox) or child.winfo_class() == "TCombobox":
                        combos.append(child)
                    elif child.winfo_class() == "Label":
                        labels.append(child)
            is_modelish = any(t in str(label).lower() for t in ("model", "checkpoint", "lora", "lycoris", "local selection"))
            if is_modelish:
                provider = lambda v=var, vals=list(values or []), lab=label: self._planner_model_choice_tooltip_text(lab, v.get(), vals)
            else:
                provider = lambda v=var, vals=list(values or []), lab=label: self._planner_option_tooltip_text("", key, lab, "choice", vals, v.get())
            self._planner_attach_dynamic_tooltip([target_row] + combos + labels, provider)
        except Exception:
            pass
        return var
    ModelPlannerApp._build_combo_row = _build_combo_row_hover

    # Patch embedded source rows in the same style.
    _hover_prev_source_row = getattr(ModelPlannerApp, "_build_source_backed_ref_row", None)
    if callable(_hover_prev_source_row):
        def _build_source_backed_ref_row_hover(self, parent, label: str, get_value, set_value, get_source, set_source, candidates, kind_options, allow_local: bool = True):
            before = set(parent.winfo_children())
            var = _hover_prev_source_row(self, parent, label, get_value, set_value, get_source, set_source, candidates, kind_options, allow_local)
            try:
                new_children = [w for w in parent.winfo_children() if w not in before]
                target = new_children[-1] if new_children else None
                widgets = [target]
                if target is not None:
                    stack = list(target.winfo_children())
                    while stack:
                        w = stack.pop(0)
                        widgets.append(w)
                        try:
                            stack.extend(w.winfo_children())
                        except Exception:
                            pass
                provider = lambda v=var, vals=list(candidates or []), lab=label: self._planner_model_choice_tooltip_text(lab, v.get(), vals)
                self._planner_attach_dynamic_tooltip(widgets, provider)
            except Exception:
                pass
            return var
        ModelPlannerApp._build_source_backed_ref_row = _build_source_backed_ref_row_hover

    def _sigless_special_option_row_hover(self, parent, entry: Dict[str, Any], key: str, label: str, choices: List[str]):
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        lab = Label(row, text=label, width=18, anchor="w")
        lab.pack(side="left")
        if key == "precision":
            value = _planner_display_precision_name(entry.get(key)) if "_planner_display_precision_name" in globals() else str(entry.get(key) or "")
            choices = _DISPLAY_PRECISION_CHOICES if "_DISPLAY_PRECISION_CHOICES" in globals() else choices
            help_key = "Precision"
        elif key == "architecture":
            value = _planner_display_arch_name(entry.get(key)) if "_planner_display_arch_name" in globals() else str(entry.get(key) or "")
            choices = _DISPLAY_ARCH_CHOICES if "_DISPLAY_ARCH_CHOICES" in globals() else choices
            help_key = "Architecture"
        else:
            value = str(entry.get(key) or "")
            help_key = label
        var = tk.StringVar(value=value)
        combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=18)
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def _sync(*_args, e=entry, k=key, v=var):
            e[k] = v.get()
            self._after_entry_change()
        var.trace_add("write", _sync)
        provider = lambda v=var, k=key, labl=label, ch=list(choices or []): self._planner_option_tooltip_text("Common", k, labl, "choice", ch, v.get())
        self._planner_attach_dynamic_tooltip([row, lab, combo], provider)
    _sigless_special_option_row = _sigless_special_option_row_hover

    def _sigless_build_cli_options_panel_hover(self, parent, entry: Dict[str, Any], entry_type: str):
        specs = copy.deepcopy(_SIGLESS_CLI_SPECS.get(entry_type, {}).get("groups", []))
        if entry_type == "Checkpoint Merge" and _sigless_mode_uses_seed(entry):
            specs.insert(1, ("Seed", [("seed", "Seed", "text", "")]))
        if not specs and entry_type not in ("Checkpoint Merge", "LoRA Bake"):
            return
        opts = _sigless_ensure_cli_options(entry)
        try:
            wrapper = self._build_collapsible_section(
                parent,
                f"{entry_type} Options",
                key=f"sigless_cli_options_{entry_type}_{entry.get('id','')}",
                default_open=False,
                body_fill="x",
                body_expand=False,
                padx=6,
                pady=6,
            )
        except Exception:
            wrapper = self._build_labeled_frame(parent, f"{entry_type} Options")
        self._planner_attach_dynamic_tooltip(wrapper, lambda et=entry_type: self._right_help(f"{et} Options").get("detail", ""))

        if entry_type in ("Checkpoint Merge", "LoRA Bake"):
            basics = LabelFrame(wrapper, text="Basic", padx=6, pady=6)
            basics.pack(fill="x", pady=4)
            _sigless_special_option_row(self, basics, entry, "architecture", "Architecture", _SIGLESS_ARCH_CHOICES)
            _sigless_special_option_row(self, basics, entry, "precision", "Precision", _SIGLESS_PRECISION_CHOICES)

        for group_name, fields in specs:
            gf = LabelFrame(wrapper, text=group_name, padx=6, pady=6)
            gf.pack(fill="x", pady=4)
            self._planner_attach_dynamic_tooltip(gf, lambda gn=group_name, et=entry_type: f"{et} Options / {gn}")
            for idx, field in enumerate(fields):
                key, label, kind, default = field[:4]
                row = Frame(gf)
                row.grid(row=idx // 2, column=(idx % 2) * 2, columnspan=2, sticky="ew", padx=3, pady=2)
                gf.grid_columnconfigure(1, weight=1)
                gf.grid_columnconfigure(3, weight=1)
                lab = Label(row, text=label, width=18, anchor="w")
                lab.pack(side="left")
                control = None
                choices = field[4] if len(field) > 4 else []
                if kind == "bool":
                    var = tk.BooleanVar(value=bool(opts.get(key, default)))
                    chk = ttk.Checkbutton(row, variable=var, command=lambda v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                    chk.pack(side="left")
                    control = chk
                    provider = lambda v=var, k=key, labl=label, et=entry_type: self._planner_option_tooltip_text(et, k, labl, "bool", None, "On" if v.get() else "Off")
                elif kind == "choice":
                    var = tk.StringVar(value=str(opts.get(key, default) or default))
                    combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=16)
                    self._bind_combobox_mousewheel_passthrough(combo)
                    combo.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                    control = combo
                    provider = lambda v=var, k=key, labl=label, et=entry_type, ch=list(choices or []): self._planner_option_tooltip_text(et, k, labl, "choice", ch, v.get())
                else:
                    var = tk.StringVar(value=str(opts.get(key, default) or ""))
                    ent = Entry(row, textvariable=var, width=28)
                    ent.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                    control = ent
                    provider = lambda v=var, k=key, labl=label, et=entry_type: self._planner_option_tooltip_text(et, k, labl, "text", None, v.get())
                self._planner_attach_dynamic_tooltip([row, lab, control], provider)
        _sigless_build_unmapped_editor(self, wrapper, entry)
        reset_row = Frame(wrapper)
        reset_row.pack(fill="x", pady=(4, 0))
        def reset_options(e=entry, et=entry_type):
            e["cli_options"] = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(et, {})) if '_LORA_MERGE_DEFAULTS' in globals() else {}
            e["cli_options"].pop("rand_alpha", None)
            e["cli_options"].pop("rand_beta", None)
            self._after_entry_change()
            self._schedule_rerender_current_line()
        reset_btn = ttk.Button(reset_row, text="Reset Options", command=reset_options)
        reset_btn.pack(side="right")
        self._planner_attach_dynamic_tooltip(reset_btn, lambda: "Reset this line's structured Options to their defaults. Unmapped legacy signatures are left untouched.")
    _sigless_build_cli_options_panel = _sigless_build_cli_options_panel_hover

    # Patch alpha/beta ratio section so Ratio Mode choices explain Single/Block/Elemental/Randomize.
    def _build_ratio_section_hover(self, parent, entry: Dict[str, Any], key: str, title: str):
        wrapper = self._build_collapsible_section(
            parent,
            title,
            key=f"ratio_{entry.get('id', '')}_{key}",
            default_open=True,
            body_fill="x",
            body_expand=False,
            padx=6,
            pady=6,
        )
        ratio = entry.setdefault(key, default_ratio("Single"))
        row = Frame(wrapper)
        row.pack(fill="x", pady=3)
        label_widget = Label(row, text="Ratio Mode", width=18, anchor="w")
        label_widget.pack(side="left")
        mode_var = tk.StringVar(value=ratio.get("mode", "Single"))
        combo = ttk.Combobox(row, textvariable=mode_var, values=RATIO_MODES, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def sync_mode(*_args):
            ratio["mode"] = mode_var.get()
            if ratio["mode"] == "Single" and not ratio.get("value"):
                ratio["value"] = "0.5"
            elif ratio["mode"] == "Randomize" and not ratio.get("value"):
                ratio["value"] = "0.0,1.0"
            elif ratio["mode"] == "Block weight" and not ratio.get("value"):
                ratio["value"] = self._serialize_block_values([0.0] * len(self._current_block_names()), trailing_comma=False)
            self._schedule_rerender_current_line()
        mode_var.trace_add("write", sync_mode)
        provider = lambda v=mode_var: self._planner_option_tooltip_text("Ratio", "Ratio Mode", "Ratio Mode", "choice", RATIO_MODES, v.get())
        self._planner_attach_dynamic_tooltip([row, label_widget, combo], provider)
        self._add_right_inline_help(wrapper, "Ratio Mode")
        self._build_ratio_value_widget(wrapper, ratio, allow_block_weight=True)
        self._planner_build_ratio_preset_buttons(wrapper, ratio)
    ModelPlannerApp._build_ratio_section = _build_ratio_section_hover

    def _render_checkpoint_merge_entry_hover(self, parent, entry: Dict[str, Any]):
        _sigless_ensure_cli_options(entry)
        frame = self._build_labeled_frame(parent, "Checkpoint Merge")
        ckpts = self._collect_available_models(self.current_index).get("Checkpoint", [])
        mode_labels = [f"{m['key']} - {m['label']}" for m in self.merge_modes]
        current_mode = entry.get("merge_mode", self.merge_modes[0]["key"])
        current_label = next((f"{m['key']} - {m['label']}" for m in self.merge_modes if m["key"] == current_mode), mode_labels[0])
        row = Frame(frame)
        row.pack(fill="x", pady=3)
        lab = Label(row, text="Merge Mode", width=18, anchor="w")
        lab.pack(side="left")
        var = tk.StringVar(value=current_label)
        combo = ttk.Combobox(row, textvariable=var, values=mode_labels, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def sync_mode(*_args):
            label = var.get()
            key2 = label.split(" - ", 1)[0]
            entry["merge_mode"] = key2
            self._schedule_rerender_current_line()
        var.trace_add("write", sync_mode)
        self._planner_attach_dynamic_tooltip([row, lab, combo], lambda v=var: self._planner_mode_choice_tooltip(v.get()))
        mode_info = self.merge_mode_map.get(entry.get("merge_mode"), self.merge_modes[0])
        model_count = _sigless_model_count(self, entry, mode_info)
        if model_count >= 1:
            self._build_combo_row(frame, "Model 0", entry, "model0", ckpts or [""])
        if model_count >= 2:
            self._build_combo_row(frame, "Model 1", entry, "model1", ckpts or [""])
        if model_count >= 3:
            self._build_combo_row(frame, "Model 2", entry, "model2", ckpts or [""])
        if mode_info.get("key") != "CLIPXOR":
            self._build_ratio_section(parent, entry, "alpha", "Alpha")
        if mode_info.get("needs_beta"):
            self._build_ratio_section(parent, entry, "beta", "Beta")
        out_frame = self._build_labeled_frame(parent, "Output")
        self._build_entry_row(out_frame, "Output Name", entry, "output_name")
        _sigless_build_cli_options_panel(self, parent, entry, "Checkpoint Merge")
    ModelPlannerApp._render_checkpoint_merge_entry = _render_checkpoint_merge_entry_hover

    # Expose helper methods on the class.
    ModelPlannerApp._planner_attach_dynamic_tooltip = _planner_attach_dynamic_tooltip
    ModelPlannerApp._planner_option_tooltip_text = _planner_option_tooltip_text
    ModelPlannerApp._planner_model_choice_tooltip_text = _planner_model_choice_tooltip_text
    ModelPlannerApp._planner_mode_choice_tooltip = _planner_mode_choice_tooltip
except Exception:
    pass


# -----------------------------------------------------------------------------
# Final hover/dropdown refinement: compact mode help, per-item dropdown help,
# line-type help, and stronger Dev Mode visual states.
# -----------------------------------------------------------------------------
try:
    RIGHT_PANEL_FIELD_HELP.setdefault("LoRA Merge", {
        "short": "Merge multiple LoRA/LyCORIS models into one LoRA.",
        "detail": "This line merges several LoRA or LyCORIS inputs into a new LoRA file without baking them into a checkpoint. The result is registered as a LoRA and can be selected later by LoRA Bake or another LoRA Merge line.",
    })
    RIGHT_PANEL_FIELD_HELP["LoRA Merge"].setdefault("detail", "Merge multiple LoRA/LyCORIS models into one LoRA without baking into a checkpoint.")

    def _planner_line_type_tooltip_text(self, current: str = "") -> str:
        current = str(current or "").strip()
        title = "Model Merge Type"
        if not current:
            return f"{title}\n\nSelect the type of plan line to edit."
        info = self._right_help(current) if hasattr(self, "_right_help") else RIGHT_PANEL_FIELD_HELP.get(current, {})
        detail = str((info or {}).get("detail") or (info or {}).get("short") or "").strip()
        if not detail:
            detail = "No detailed description is registered for this plan line type yet."
        return f"{title}\n\nSelected: {current}\n{detail}"

    def _planner_mode_choice_tooltip_compact(self, current_label: str = "") -> str:
        raw = str(current_label or "").strip()
        current_key = raw.split(" - ", 1)[0].strip()
        label = ""
        mode_info = None
        try:
            for m in list(getattr(self, "merge_modes", []) or []):
                if str(m.get("key") or "") == current_key:
                    mode_info = m
                    label = str(m.get("label") or current_key)
                    break
        except Exception:
            mode_info = None
        if not current_key:
            return "Merge Mode\n\nNo mode is currently selected."
        lines = ["Merge Mode", "", f"Selected: {current_key}" + (f" - {label}" if label and label != current_key else "")]
        detail = _planner_mode_detail_text(current_key, mode_info, fallback="Backend merge mode. Check merge.py / merge_modes.py for exact tensor behavior.") if "_MODE_HELP_TEXT" in globals() else "Backend merge mode."
        lines.append(detail)
        req = []
        try:
            if mode_info and mode_info.get("needs_m2"):
                req.append("Model 2")
            if mode_info and mode_info.get("needs_beta"):
                req.append("Beta")
        except Exception:
            pass
        if req:
            lines.extend(["", "Required inputs: " + ", ".join(req)])
        return "\n".join(lines).strip()

    def _planner_mode_dropdown_item_tooltip(self, value: str = "") -> str:
        return self._planner_mode_choice_tooltip_compact(value)

    def _planner_choice_dropdown_item_tooltip(self, label_or_key: str, value: str = "") -> str:
        value = str(value or "").strip()
        if str(label_or_key or "") == "Model Merge Type":
            return self._planner_line_type_tooltip_text(value)
        if not value:
            return ""
        table = None
        try:
            table = _CHOICE_HELP_TEXT.get(label_or_key) or _CHOICE_HELP_TEXT.get(str(label_or_key).replace(" ", "_")) or _CHOICE_HELP_TEXT.get(str(label_or_key).lower())
        except Exception:
            table = None
        detail = ""
        if isinstance(table, dict):
            detail = str(table.get(value) or table.get(value.lower()) or "")
        if not detail:
            try:
                detail = str(_OPTION_HELP_TEXT.get(label_or_key) or _OPTION_HELP_TEXT.get(str(label_or_key).replace(" ", "_")) or _OPTION_HELP_TEXT.get(str(label_or_key).lower()) or "")
            except Exception:
                detail = ""
        if not detail:
            detail = str(value)
        return f"{label_or_key}\n\nSelected: {value}\n{detail}".strip()

    def _planner_hide_dropdown_tooltip(self, _event=None):
        tip = getattr(self, "_planner_dropdown_tip", None)
        if tip is not None:
            try:
                tip.destroy()
            except Exception:
                pass
            self._planner_dropdown_tip = None

    def _planner_show_dropdown_tooltip(self, text: str, x_root: int, y_root: int):
        text = str(text or "").strip()
        if not text:
            self._planner_hide_dropdown_tooltip()
            return
        old_text = getattr(self, "_planner_dropdown_tip_text", None)
        tip = getattr(self, "_planner_dropdown_tip", None)
        if tip is not None and old_text == text:
            try:
                tip.wm_geometry(f"+{int(x_root) + 18}+{int(y_root) + 12}")
                return
            except Exception:
                self._planner_hide_dropdown_tooltip()
        self._planner_hide_dropdown_tooltip()
        try:
            colors = self._theme_colors() if hasattr(self, "_theme_colors") else {"surface": "#131a30", "panel": "#0f1528", "text": "#eef2ff", "border": "#2a355d"}
        except Exception:
            colors = {"surface": "#131a30", "panel": "#0f1528", "text": "#eef2ff", "border": "#2a355d"}
        try:
            tip = tk.Toplevel(self.root)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{int(x_root) + 18}+{int(y_root) + 12}")
            tip.configure(bg=colors.get("panel", "#0f1528"))
            card = Frame(tip, bg=colors.get("surface", "#131a30"), highlightthickness=1, highlightbackground=colors.get("border", "#2a355d"), padx=2, pady=2)
            card.pack(fill="both", expand=True)
            Label(card, text=text, justify="left", anchor="w", wraplength=720, bg=colors.get("surface", "#131a30"), fg=colors.get("text", "#eef2ff"), padx=12, pady=10, font=("MS Gothic", 10)).pack(fill="both", expand=True)
            self._planner_dropdown_tip = tip
            self._planner_dropdown_tip_text = text
        except Exception:
            self._planner_dropdown_tip = None
            self._planner_dropdown_tip_text = None

    def _planner_combobox_listbox(self, combo):
        try:
            popdown = combo.tk.eval("ttk::combobox::PopdownWindow %s" % str(combo))
            # Tk's ttk combobox popdown usually contains .f.l. Keep a short fallback search.
            for suffix in (".f.l", ".f.listbox", ".listbox"):
                try:
                    return combo.nametowidget(popdown + suffix)
                except Exception:
                    pass
            try:
                return combo.nametowidget(popdown).children.get("f").children.get("l")
            except Exception:
                return None
        except Exception:
            return None

    def _planner_attach_combobox_dropdown_tooltips(self, combo, item_provider):
        if combo is None or not callable(item_provider):
            return
        try:
            if getattr(combo, "_planner_dropdown_tooltip_bound", False):
                return
            setattr(combo, "_planner_dropdown_tooltip_bound", True)
        except Exception:
            pass
        def install(_event=None, c=combo):
            def _do_install():
                lb = self._planner_combobox_listbox(c)
                if lb is None:
                    return
                try:
                    if getattr(lb, "_planner_dropdown_tooltip_bound", False):
                        return
                    setattr(lb, "_planner_dropdown_tooltip_bound", True)
                except Exception:
                    pass
                def on_motion(ev, listbox=lb):
                    try:
                        idx = listbox.nearest(ev.y)
                        if idx < 0 or idx >= listbox.size():
                            self._planner_hide_dropdown_tooltip()
                            return
                        value = str(listbox.get(idx))
                        text = item_provider(value)
                        self._planner_show_dropdown_tooltip(text, ev.x_root, ev.y_root)
                    except Exception:
                        self._planner_hide_dropdown_tooltip()
                def on_leave(_ev=None):
                    self._planner_hide_dropdown_tooltip()
                try:
                    lb.bind("<Motion>", on_motion, add="+")
                    lb.bind("<Leave>", on_leave, add="+")
                    lb.bind("<ButtonRelease-1>", on_leave, add="+")
                    lb.bind("<FocusOut>", on_leave, add="+")
                except Exception:
                    pass
            try:
                c.after(80, _do_install)
                c.after(220, _do_install)
            except Exception:
                pass
        try:
            combo.bind("<ButtonPress-1>", install, add="+")
            combo.bind("<Alt-Down>", install, add="+")
            combo.bind("<Down>", install, add="+")
            combo.bind("<<ComboboxSelected>>", lambda _e: self._planner_hide_dropdown_tooltip(), add="+")
            combo.bind("<Destroy>", lambda _e: self._planner_hide_dropdown_tooltip(), add="+")
        except Exception:
            pass

    def _planner_model_item_provider(self, label: str):
        return lambda value, lab=label: self._planner_model_choice_tooltip_text(lab, value, [value])

    # Replace mode tooltip with compact selected-only output.
    ModelPlannerApp._planner_mode_choice_tooltip = _planner_mode_choice_tooltip_compact
    ModelPlannerApp._planner_mode_dropdown_item_tooltip = _planner_mode_dropdown_item_tooltip
    ModelPlannerApp._planner_line_type_tooltip_text = _planner_line_type_tooltip_text
    ModelPlannerApp._planner_choice_dropdown_item_tooltip = _planner_choice_dropdown_item_tooltip
    ModelPlannerApp._planner_hide_dropdown_tooltip = _planner_hide_dropdown_tooltip
    ModelPlannerApp._planner_show_dropdown_tooltip = _planner_show_dropdown_tooltip
    ModelPlannerApp._planner_combobox_listbox = _planner_combobox_listbox
    ModelPlannerApp._planner_attach_combobox_dropdown_tooltips = _planner_attach_combobox_dropdown_tooltips

    # Patch combo rows: selected-only hover on the combobox itself; per-item hover inside the dropdown list.
    _refined_base_combo_row = globals().get("_hover_prev_build_combo_row", getattr(ModelPlannerApp, "_build_combo_row", None))
    def _build_combo_row_dropdown_refined(self, parent, label: str, entry: Dict[str, Any], key: str, values: List[str], callback=None):
        before = set(parent.winfo_children())
        var = _refined_base_combo_row(self, parent, label, entry, key, values, callback) if callable(_refined_base_combo_row) else None
        try:
            new_children = [w for w in parent.winfo_children() if w not in before]
            target_row = new_children[-1] if new_children else None
            widgets = [target_row]
            combos = []
            labels = []
            if target_row is not None:
                for child in target_row.winfo_children():
                    widgets.append(child)
                    if isinstance(child, ttk.Combobox) or child.winfo_class() == "TCombobox":
                        combos.append(child)
                    elif child.winfo_class() == "Label":
                        labels.append(child)
            is_modelish = any(t in str(label).lower() for t in ("model", "checkpoint", "lora", "lycoris", "local selection")) and str(label) != "Model Merge Type"
            if str(label) == "Model Merge Type":
                provider = lambda v=var: self._planner_line_type_tooltip_text(v.get() if v is not None else "")
                item_provider = lambda value: self._planner_line_type_tooltip_text(value)
            elif is_modelish:
                provider = lambda v=var, lab=label: self._planner_model_choice_tooltip_text(lab, v.get() if v is not None else "", None)
                item_provider = self._planner_model_item_provider(label) if hasattr(self, "_planner_model_item_provider") else (lambda value, lab=label: self._planner_model_choice_tooltip_text(lab, value, [value]))
            else:
                provider = lambda v=var, lab=label, k=key: self._planner_option_tooltip_text("", k, lab, "choice", None, v.get() if v is not None else "")
                item_provider = lambda value, lab=label, k=key: self._planner_choice_dropdown_item_tooltip(lab if lab else k, value)
            self._planner_attach_dynamic_tooltip(widgets + labels + combos, provider)
            for combo in combos:
                self._planner_attach_combobox_dropdown_tooltips(combo, item_provider)
        except Exception:
            pass
        return var
    ModelPlannerApp._build_combo_row = _build_combo_row_dropdown_refined

    # Patch source-backed model rows for dropdown item model help.
    _refined_source_row = globals().get("_hover_prev_source_row", getattr(ModelPlannerApp, "_build_source_backed_ref_row", None))
    if callable(_refined_source_row):
        def _build_source_backed_ref_row_dropdown_refined(self, parent, label: str, get_value, set_value, get_source, set_source, candidates, kind_options, allow_local: bool = True):
            before = set(parent.winfo_children())
            var = _refined_source_row(self, parent, label, get_value, set_value, get_source, set_source, candidates, kind_options, allow_local)
            try:
                new_children = [w for w in parent.winfo_children() if w not in before]
                target = new_children[-1] if new_children else None
                widgets = [target]
                combos = []
                if target is not None:
                    stack = list(target.winfo_children())
                    while stack:
                        w = stack.pop(0)
                        widgets.append(w)
                        try:
                            if isinstance(w, ttk.Combobox) or w.winfo_class() == "TCombobox":
                                combos.append(w)
                            stack.extend(w.winfo_children())
                        except Exception:
                            pass
                provider = lambda v=var, lab=label: self._planner_model_choice_tooltip_text(lab, v.get(), None)
                self._planner_attach_dynamic_tooltip(widgets, provider)
                item_provider = self._planner_model_item_provider(label) if hasattr(self, "_planner_model_item_provider") else (lambda value, lab=label: self._planner_model_choice_tooltip_text(lab, value, [value]))
                for combo in combos:
                    self._planner_attach_combobox_dropdown_tooltips(combo, item_provider)
            except Exception:
                pass
            return var
        ModelPlannerApp._build_source_backed_ref_row = _build_source_backed_ref_row_dropdown_refined

    # Special Architecture / Precision rows: no all-choice wall on normal hover, but dropdown item help reacts per row.
    def _sigless_special_option_row_dropdown_refined(self, parent, entry: Dict[str, Any], key: str, label: str, choices: List[str]):
        row = Frame(parent)
        row.pack(fill="x", pady=2)
        lab = Label(row, text=label, width=18, anchor="w")
        lab.pack(side="left")
        if key == "precision":
            value = _planner_display_precision_name(entry.get(key)) if "_planner_display_precision_name" in globals() else str(entry.get(key) or "")
            choices = _DISPLAY_PRECISION_CHOICES if "_DISPLAY_PRECISION_CHOICES" in globals() else choices
            help_key = "Precision"
        elif key == "architecture":
            value = _planner_display_arch_name(entry.get(key)) if "_planner_display_arch_name" in globals() else str(entry.get(key) or "")
            choices = _DISPLAY_ARCH_CHOICES if "_DISPLAY_ARCH_CHOICES" in globals() else choices
            help_key = "Architecture"
        else:
            value = str(entry.get(key) or "")
            help_key = label
        var = tk.StringVar(value=value)
        combo = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=18)
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def _sync(*_args, e=entry, k=key, v=var):
            e[k] = v.get()
            self._after_entry_change()
        var.trace_add("write", _sync)
        provider = lambda v=var, k=key, labl=help_key: self._planner_option_tooltip_text("Common", k, labl, "choice", None, v.get())
        self._planner_attach_dynamic_tooltip([row, lab, combo], provider)
        self._planner_attach_combobox_dropdown_tooltips(combo, lambda value, labl=help_key: self._planner_choice_dropdown_item_tooltip(labl, value))
    _sigless_special_option_row = _sigless_special_option_row_dropdown_refined

    # Options panel: refine choice combobox hover/dropdown behavior.
    def _sigless_build_cli_options_panel_dropdown_refined(self, parent, entry: Dict[str, Any], entry_type: str):
        specs = copy.deepcopy(_SIGLESS_CLI_SPECS.get(entry_type, {}).get("groups", []))
        if entry_type == "Checkpoint Merge" and _sigless_mode_uses_seed(entry):
            specs.insert(1, ("Seed", [("seed", "Seed", "text", "")]))
        if not specs and entry_type not in ("Checkpoint Merge", "LoRA Bake"):
            return
        opts = _sigless_ensure_cli_options(entry)
        try:
            wrapper = self._build_collapsible_section(parent, f"{entry_type} Options", key=f"sigless_cli_options_{entry_type}_{entry.get('id','')}", default_open=False, body_fill="x", body_expand=False, padx=6, pady=6)
        except Exception:
            wrapper = self._build_labeled_frame(parent, f"{entry_type} Options")
        self._planner_attach_dynamic_tooltip(wrapper, lambda et=entry_type: self._right_help(f"{et} Options").get("detail", ""))
        if entry_type in ("Checkpoint Merge", "LoRA Bake"):
            basics = LabelFrame(wrapper, text="Basic", padx=6, pady=6)
            basics.pack(fill="x", pady=4)
            _sigless_special_option_row(self, basics, entry, "architecture", "Architecture", _SIGLESS_ARCH_CHOICES)
            _sigless_special_option_row(self, basics, entry, "precision", "Precision", _SIGLESS_PRECISION_CHOICES)
        for group_name, fields in specs:
            gf = LabelFrame(wrapper, text=group_name, padx=6, pady=6)
            gf.pack(fill="x", pady=4)
            self._planner_attach_dynamic_tooltip(gf, lambda gn=group_name, et=entry_type: f"{et} Options / {gn}")
            for idx, field in enumerate(fields):
                key, label, kind, default = field[:4]
                row = Frame(gf)
                row.grid(row=idx // 2, column=(idx % 2) * 2, columnspan=2, sticky="ew", padx=3, pady=2)
                gf.grid_columnconfigure(1, weight=1)
                gf.grid_columnconfigure(3, weight=1)
                lab = Label(row, text=label, width=18, anchor="w")
                lab.pack(side="left")
                choices = field[4] if len(field) > 4 else []
                control = None
                if kind == "bool":
                    var = tk.BooleanVar(value=bool(opts.get(key, default)))
                    control = ttk.Checkbutton(row, variable=var, command=lambda v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                    control.pack(side="left")
                    provider = lambda v=var, k=key, labl=label, et=entry_type: self._planner_option_tooltip_text(et, k, labl, "bool", None, "On" if v.get() else "Off")
                elif kind == "choice":
                    current = str(opts.get(key, default) or default)
                    if key == "merge_arch" and "_planner_display_arch_name" in globals():
                        current = _planner_display_arch_name(current)
                    var = tk.StringVar(value=current)
                    control = ttk.Combobox(row, textvariable=var, values=choices, state="readonly", width=16)
                    self._bind_combobox_mousewheel_passthrough(control)
                    control.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                    provider = lambda v=var, k=key, labl=label, et=entry_type: self._planner_option_tooltip_text(et, k, labl, "choice", None, v.get())
                    self._planner_attach_combobox_dropdown_tooltips(control, lambda value, k=key, labl=label: self._planner_choice_dropdown_item_tooltip(k if k in _CHOICE_HELP_TEXT else labl, value))
                else:
                    var = tk.StringVar(value=str(opts.get(key, default) or ""))
                    control = Entry(row, textvariable=var, width=28)
                    control.pack(side="left", fill="x", expand=True)
                    var.trace_add("write", lambda *_args, v=var, k=key, d=default, e=entry: _sigless_option_changed(self, e, k, v.get(), d))
                    provider = lambda v=var, k=key, labl=label, et=entry_type: self._planner_option_tooltip_text(et, k, labl, "text", None, v.get())
                self._planner_attach_dynamic_tooltip([row, lab, control], provider)
        _sigless_build_unmapped_editor(self, wrapper, entry)
        reset_row = Frame(wrapper)
        reset_row.pack(fill="x", pady=(4, 0))
        def reset_options(e=entry, et=entry_type):
            e["cli_options"] = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(et, {})) if '_LORA_MERGE_DEFAULTS' in globals() else {}
            e["cli_options"].pop("rand_alpha", None)
            e["cli_options"].pop("rand_beta", None)
            self._after_entry_change()
            self._schedule_rerender_current_line()
        reset_btn = ttk.Button(reset_row, text="Reset Options", command=reset_options)
        reset_btn.pack(side="right")
        self._planner_attach_dynamic_tooltip(reset_btn, lambda: "Reset this line's structured Options to their defaults. Unmapped legacy signatures are left untouched.")
    _sigless_build_cli_options_panel = _sigless_build_cli_options_panel_dropdown_refined

    # Ratio Mode: compact current hover + per-choice dropdown hover.
    def _build_ratio_section_dropdown_refined(self, parent, entry: Dict[str, Any], key: str, title: str):
        wrapper = self._build_collapsible_section(parent, title, key=f"ratio_{entry.get('id', '')}_{key}", default_open=True, body_fill="x", body_expand=False, padx=6, pady=6)
        ratio = entry.setdefault(key, default_ratio("Single"))
        row = Frame(wrapper)
        row.pack(fill="x", pady=3)
        label_widget = Label(row, text="Ratio Mode", width=18, anchor="w")
        label_widget.pack(side="left")
        mode_var = tk.StringVar(value=ratio.get("mode", "Single"))
        combo = ttk.Combobox(row, textvariable=mode_var, values=RATIO_MODES, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def sync_mode(*_args):
            ratio["mode"] = mode_var.get()
            if ratio["mode"] == "Single" and not ratio.get("value"):
                ratio["value"] = "0.5"
            elif ratio["mode"] == "Randomize" and not ratio.get("value"):
                ratio["value"] = "0.0,1.0"
            elif ratio["mode"] == "Block weight" and not ratio.get("value"):
                ratio["value"] = self._serialize_block_values([0.0] * len(self._current_block_names()), trailing_comma=False)
            self._schedule_rerender_current_line()
        mode_var.trace_add("write", sync_mode)
        provider = lambda v=mode_var: self._planner_option_tooltip_text("Ratio", "Ratio Mode", "Ratio Mode", "choice", None, v.get())
        self._planner_attach_dynamic_tooltip([row, label_widget, combo], provider)
        self._planner_attach_combobox_dropdown_tooltips(combo, lambda value: self._planner_choice_dropdown_item_tooltip("Ratio Mode", value))
        self._add_right_inline_help(wrapper, "Ratio Mode")
        self._build_ratio_value_widget(wrapper, ratio, allow_block_weight=True)
        self._planner_build_ratio_preset_buttons(wrapper, ratio)
    ModelPlannerApp._build_ratio_section = _build_ratio_section_dropdown_refined

    # Checkpoint Merge renderer: add dropdown item hover for merge modes.
    def _render_checkpoint_merge_entry_dropdown_refined(self, parent, entry: Dict[str, Any]):
        _sigless_ensure_cli_options(entry)
        frame = self._build_labeled_frame(parent, "Checkpoint Merge")
        ckpts = self._collect_available_models(self.current_index).get("Checkpoint", [])
        mode_labels = [f"{m['key']} - {m['label']}" for m in self.merge_modes]
        current_mode = entry.get("merge_mode", self.merge_modes[0]["key"])
        current_label = next((f"{m['key']} - {m['label']}" for m in self.merge_modes if m["key"] == current_mode), mode_labels[0])
        row = Frame(frame)
        row.pack(fill="x", pady=3)
        lab = Label(row, text="Merge Mode", width=18, anchor="w")
        lab.pack(side="left")
        var = tk.StringVar(value=current_label)
        combo = ttk.Combobox(row, textvariable=var, values=mode_labels, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)
        def sync_mode(*_args):
            label = var.get()
            key2 = label.split(" - ", 1)[0]
            entry["merge_mode"] = key2
            self._schedule_rerender_current_line()
        var.trace_add("write", sync_mode)
        self._planner_attach_dynamic_tooltip([row, lab, combo], lambda v=var: self._planner_mode_choice_tooltip(v.get()))
        self._planner_attach_combobox_dropdown_tooltips(combo, lambda value: self._planner_mode_dropdown_item_tooltip(value))
        mode_info = self.merge_mode_map.get(entry.get("merge_mode"), self.merge_modes[0])
        model_count = _sigless_model_count(self, entry, mode_info)
        if model_count >= 1:
            self._build_combo_row(frame, "Model 0", entry, "model0", ckpts or [""])
        if model_count >= 2:
            self._build_combo_row(frame, "Model 1", entry, "model1", ckpts or [""])
        if model_count >= 3:
            self._build_combo_row(frame, "Model 2", entry, "model2", ckpts or [""])
        if mode_info.get("key") != "CLIPXOR":
            self._build_ratio_section(parent, entry, "alpha", "Alpha")
        if mode_info.get("needs_beta"):
            self._build_ratio_section(parent, entry, "beta", "Beta")
        out_frame = self._build_labeled_frame(parent, "Output")
        self._build_entry_row(out_frame, "Output Name", entry, "output_name")
        _sigless_build_cli_options_panel(self, parent, entry, "Checkpoint Merge")
    ModelPlannerApp._render_checkpoint_merge_entry = _render_checkpoint_merge_entry_dropdown_refined

    # Dev Mode: muted real button colors when disabled, theme button colors when enabled.
    def _planner_blend_hex(self, a: str, b: str = "#808080", ratio: float = 0.55) -> str:
        """Blend two #RRGGBB colors.

        This method is intentionally registered on ModelPlannerApp below because the
        Developer Mode button renderer calls it through ``self``. Earlier patch
        layers defined it only as a local function, which caused an undefined
        attribute error when the refined Dev Mode panel was built.
        """
        def _rgb(x):
            x = str(x or "#808080").lstrip("#")
            if len(x) != 6:
                x = "808080"
            return [int(x[i:i+2], 16) for i in (0, 2, 4)]
        ar = _rgb(a); br = _rgb(b)
        rr = [int(ar[i] * (1.0 - ratio) + br[i] * ratio) for i in range(3)]
        return "#" + "".join(f"{max(0, min(255, v)):02x}" for v in rr)

    ModelPlannerApp._planner_blend_hex = _planner_blend_hex

    def _build_left_panel_refined_dev_buttons(self, parent):
        self._planner_dev_ensure_config()
        base_builder = globals().get("_dev_base_build_left_panel")
        result = base_builder(self, parent) if callable(base_builder) else None
        try:
            section = self._build_collapsible_section(parent, "Developer (Experiment) Mode", key="left_developer_mode", default_open=bool(self.config.get("dev_mode", False)), body_fill="x", body_expand=False)
        except Exception:
            section = LabelFrame(parent, text="Developer (Experiment) Mode", padx=8, pady=8)
            section.pack(fill="x", pady=6)
        self.dev_mode_var = tk.BooleanVar(value=bool(self.config.get("dev_mode", False)))
        button_refs = []
        colors = self._theme_colors() if hasattr(self, "_theme_colors") else {"button_bg": "#e7ecfb", "button_fg": "#25315b", "button_hover": "#dce4fb", "button_pressed": "#ced9f7", "muted": "#777777", "text": "#111111", "border": "#cccccc"}
        disabled_bg = self._planner_blend_hex(colors.get("button_bg", "#e7ecfb"), "#808080", 0.62)
        disabled_active = self._planner_blend_hex(colors.get("button_hover", colors.get("button_bg", "#e7ecfb")), "#808080", 0.68)
        def _button_colors(enabled: bool):
            if enabled:
                return {
                    "state": "normal", "fg": colors.get("button_fg", colors.get("text", "#111111")), "background": colors.get("button_bg", "#e7ecfb"),
                    "activeforeground": colors.get("button_fg", colors.get("text", "#111111")), "activebackground": colors.get("button_hover", "#dce4fb"),
                    "disabledforeground": colors.get("muted", "#777777"), "relief": "raised", "bd": 1, "highlightthickness": 1, "highlightbackground": colors.get("border", "#cccccc"),
                }
            return {
                "state": "disabled", "fg": colors.get("muted", "#777777"), "background": disabled_bg,
                "activeforeground": colors.get("muted", "#777777"), "activebackground": disabled_active,
                "disabledforeground": colors.get("muted", "#777777"), "relief": "groove", "bd": 1, "highlightthickness": 1, "highlightbackground": colors.get("border", "#cccccc"),
            }
        def refresh_dev_buttons():
            enabled = self._planner_dev_mode_enabled()
            self.config["dev_mode"] = enabled
            for btn in button_refs:
                try:
                    btn.configure(**_button_colors(enabled))
                except Exception:
                    try:
                        btn.configure(state=("normal" if enabled else "disabled"))
                    except Exception:
                        pass
            self._schedule_config_save()
        top = Frame(section)
        top.pack(fill="x", pady=(0, 6))
        chk = ttk.Checkbutton(top, variable=self.dev_mode_var, command=refresh_dev_buttons)
        chk.pack(side="left", padx=(0, 4))
        header_btn = tk.Button(top, text="Developer Mode", padx=8, pady=2, relief="groove", state="disabled", background=disabled_bg, disabledforeground=colors.get("muted", "#777777"), highlightthickness=1, highlightbackground=colors.get("border", "#cccccc"))
        header_btn.pack(side="left", fill="x", expand=True)
        grid = Frame(section)
        grid.pack(fill="x")
        for c in range(3):
            grid.grid_columnconfigure(c, weight=1)
        actions = [
            ("T2I Settings", lambda: self._planner_dev_only(self._planner_open_t2i_settings)),
            ("Ratio Sweep", lambda: self._planner_dev_only(self._planner_open_ratio_sweep)),
            ("Compare Images", lambda: self._planner_dev_only(self._planner_show_image_compare)),
            ("Ratio Diff", lambda: self._planner_dev_only(self._planner_show_ratio_diff)),
            ("Model Anatomy", lambda: self._planner_dev_only(self._planner_model_anatomy_scan)),
            ("LoRA Check", lambda: self._planner_dev_only(self._planner_lora_compatibility_check)),
        ]
        for idx, (label, cmd) in enumerate(actions):
            btn = tk.Button(grid, text=label, command=cmd, padx=8, pady=4, takefocus=False)
            btn.grid(row=idx//3, column=idx%3, sticky="ew", padx=3, pady=3)
            button_refs.append(btn)
        self._developer_mode_buttons = [(b, True) for b in button_refs]
        refresh_dev_buttons()
        return result
    ModelPlannerApp._build_left_panel = _build_left_panel_refined_dev_buttons
except Exception:
    pass


# -----------------------------------------------------------------------------
# Final source-backed model selector restoration:
# keep hover/dropdown help while preserving Download / Local / Clear buttons.
# -----------------------------------------------------------------------------
try:
    def _planner_build_source_model_row_final(self, parent, entry: Dict[str, Any], slot: str, label: str, candidates, kind_options):
        """Build a model reference row that supports embedded Download/Local sources.

        This intentionally routes model-like rows through _build_source_backed_ref_row
        instead of the plain combobox row so the selector can keep both:
        - compact hover/dropdown explanations
        - Download..., Local..., and Clear buttons
        """
        if hasattr(self, "_build_source_backed_ref_row") and callable(getattr(self, "_build_source_backed_ref_row", None)):
            return self._build_source_backed_ref_row(
                parent,
                label,
                lambda e=entry, s=slot: e.get(s, ""),
                lambda value, e=entry, s=slot: e.__setitem__(s, value),
                lambda e=entry, s=slot: self._get_entry_slot_source(e, s),
                lambda spec, e=entry, s=slot: self._set_entry_slot_source(e, s, spec),
                candidates or [""],
                kind_options,
                allow_local=True,
            )
        return self._build_combo_row(parent, label, entry, slot, candidates or [""])

    def _planner_attach_merge_mode_help_final(self, row, label_widget, combo, mode_var):
        """Attach selected-mode and dropdown-item help when those helpers are available."""
        try:
            if hasattr(self, "_planner_attach_dynamic_tooltip") and callable(getattr(self, "_planner_attach_dynamic_tooltip", None)):
                self._planner_attach_dynamic_tooltip(
                    [row, label_widget, combo],
                    lambda v=mode_var: self._planner_mode_choice_tooltip(v.get())
                    if hasattr(self, "_planner_mode_choice_tooltip")
                    else str(v.get() or ""),
                )
            elif hasattr(self, "_attach_tooltip"):
                self._attach_tooltip([row, label_widget, combo], self._right_help("Merge Mode").get("detail", ""))
        except Exception:
            pass
        try:
            if (
                hasattr(self, "_planner_attach_combobox_dropdown_tooltips")
                and hasattr(self, "_planner_mode_dropdown_item_tooltip")
            ):
                self._planner_attach_combobox_dropdown_tooltips(
                    combo,
                    lambda value: self._planner_mode_dropdown_item_tooltip(value),
                )
        except Exception:
            pass

    def _render_checkpoint_merge_entry_source_buttons_final(self, parent, entry: Dict[str, Any]):
        ensure_cli = globals().get("_sigless_ensure_cli_options")
        if callable(ensure_cli):
            ensure_cli(entry)

        frame = self._build_labeled_frame(parent, "Checkpoint Merge")
        ckpts = self._collect_available_models(self.current_index).get("Checkpoint", [])

        mode_labels = [f"{m['key']} - {m['label']}" for m in self.merge_modes]
        current_mode = entry.get("merge_mode", self.merge_modes[0]["key"])
        current_label = next(
            (f"{m['key']} - {m['label']}" for m in self.merge_modes if m["key"] == current_mode),
            mode_labels[0] if mode_labels else str(current_mode or ""),
        )

        row = Frame(frame)
        row.pack(fill="x", pady=3)
        lab = Label(row, text="Merge Mode", width=18, anchor="w")
        lab.pack(side="left")
        mode_var = tk.StringVar(value=current_label)
        combo = ttk.Combobox(row, textvariable=mode_var, values=mode_labels, state="readonly")
        self._bind_combobox_mousewheel_passthrough(combo)
        combo.pack(side="left", fill="x", expand=True, padx=4)

        def sync_mode(*_args):
            label = mode_var.get()
            key2 = label.split(" - ", 1)[0]
            entry["merge_mode"] = key2
            self._schedule_rerender_current_line()

        mode_var.trace_add("write", sync_mode)
        self._planner_attach_merge_mode_help_final(row, lab, combo, mode_var)

        mode_info = self.merge_mode_map.get(entry.get("merge_mode"), self.merge_modes[0])
        model_count_func = globals().get("_sigless_model_count")
        if callable(model_count_func):
            model_count = model_count_func(self, entry, mode_info)
        else:
            model_count = 3 if mode_info.get("needs_m2") else 2

        if model_count >= 1:
            self._planner_build_source_model_row_final(frame, entry, "model0", "Model 0", ckpts, ["Checkpoint"])
        if model_count >= 2:
            self._planner_build_source_model_row_final(frame, entry, "model1", "Model 1", ckpts, ["Checkpoint"])
        if model_count >= 3:
            self._planner_build_source_model_row_final(frame, entry, "model2", "Model 2", ckpts, ["Checkpoint"])

        if mode_info.get("key") != "CLIPXOR":
            self._build_ratio_section(parent, entry, "alpha", "Alpha")
        if mode_info.get("needs_beta"):
            self._build_ratio_section(parent, entry, "beta", "Beta")

        out_frame = self._build_labeled_frame(parent, "Output")
        self._build_entry_row(out_frame, "Output Name", entry, "output_name")

        build_cli = globals().get("_sigless_build_cli_options_panel") or globals().get("_lm_build_cli_options_panel")
        if callable(build_cli):
            build_cli(self, parent, entry, "Checkpoint Merge")
        else:
            self._build_text_row(parent, "Additional Signatures", entry, "raw_signatures", height=5)

    ModelPlannerApp._planner_build_source_model_row_final = _planner_build_source_model_row_final
    ModelPlannerApp._planner_attach_merge_mode_help_final = _planner_attach_merge_mode_help_final
    ModelPlannerApp._render_checkpoint_merge_entry = _render_checkpoint_merge_entry_source_buttons_final

    # Keep explicit hover on Add LoRA controls in the final sigless renderers too.
    _source_buttons_prev_lora_bake_renderer = getattr(ModelPlannerApp, "_render_lora_bake_entry", None)
    def _render_lora_bake_entry_source_buttons_hover_final(self, parent, entry: Dict[str, Any]):
        if callable(_source_buttons_prev_lora_bake_renderer):
            _source_buttons_prev_lora_bake_renderer(self, parent, entry)
            try:
                # Previous renderer already uses source-backed rows; this pass only restores
                # the Add LoRA hover text when the final sigless renderer omitted it.
                for frame in parent.winfo_children():
                    for child in frame.winfo_children():
                        try:
                            if child.winfo_class() == "TButton" and str(child.cget("text")) == "+ Add LoRA":
                                detail = self._right_help("+ Add LoRA").get("detail", "")
                                if detail:
                                    self._attach_tooltip(child, detail)
                        except Exception:
                            pass
            except Exception:
                pass
            return
        return None
    ModelPlannerApp._render_lora_bake_entry = _render_lora_bake_entry_source_buttons_hover_final

    _source_buttons_prev_lora_merge_renderer = getattr(ModelPlannerApp, "_render_lora_merge_entry", None)
    def _render_lora_merge_entry_source_buttons_hover_final(self, parent, entry: Dict[str, Any]):
        if callable(_source_buttons_prev_lora_merge_renderer):
            _source_buttons_prev_lora_merge_renderer(self, parent, entry)
            try:
                for frame in parent.winfo_children():
                    for child in frame.winfo_children():
                        try:
                            if child.winfo_class() == "TButton" and str(child.cget("text")) == "+ Add LoRA":
                                detail = self._right_help("+ Add LoRA").get("detail", "")
                                if detail:
                                    self._attach_tooltip(child, detail)
                        except Exception:
                            pass
            except Exception:
                pass
            return
        return None
    ModelPlannerApp._render_lora_merge_entry = _render_lora_merge_entry_source_buttons_hover_final
except Exception:
    pass


# -----------------------------------------------------------------------------
# Final safety patch: do not expose outputs from disabled or unresolved producers.
# -----------------------------------------------------------------------------
try:
    def _planner_source_spec_is_usable_final(self, spec) -> bool:
        """Return True when an embedded Download/Local source has enough data to run."""
        if not isinstance(spec, dict):
            return False
        mode = str(spec.get('mode') or '').strip().lower()
        if mode == 'download':
            return bool(str(spec.get('link') or '').strip())
        if mode == 'local':
            return bool(str(spec.get('local_path') or '').strip())
        return bool(mode)

    def _planner_bucket_has_ref_final(self, available, kind: str, ref: str) -> bool:
        ref = str(ref or '').strip()
        if not ref:
            return False
        kind = str(kind or 'Checkpoint').strip()
        if kind == 'Checkpoint':
            return ref in (available.get('Checkpoint') or []) or ref in (available.get('Checkpoint') or {})
        if kind in {'LoRA', 'LyCORIS'}:
            return (
                ref in (available.get('LoRA') or []) or ref in (available.get('LoRA') or {})
                or ref in (available.get('LyCORIS') or []) or ref in (available.get('LyCORIS') or {})
            )
        return ref in (available.get(kind) or []) or ref in (available.get(kind) or {})

    def _planner_ref_resolves_final(self, entry, slot: str, available, kind: str = 'Checkpoint') -> bool:
        spec = self._get_entry_slot_source(entry, slot) if hasattr(self, '_get_entry_slot_source') else None
        ref = str(entry.get(slot) or '').strip()
        if isinstance(spec, dict):
            return self._planner_source_spec_is_usable_final(spec)
        return self._planner_bucket_has_ref_final(available, kind, ref)

    def _planner_lora_ref_resolves_final(self, lora_entry, available) -> bool:
        spec = self._get_lora_source(lora_entry) if hasattr(self, '_get_lora_source') else None
        name = str((lora_entry or {}).get('name') or '').strip()
        if isinstance(spec, dict):
            return self._planner_source_spec_is_usable_final(spec)
        return self._planner_bucket_has_ref_final(available, 'LoRA', name)

    def _planner_merge_requires_model2_final(self, entry) -> bool:
        mode_raw = str(entry.get('merge_mode') or 'WS').strip()
        try:
            mode_key = self._resolve_merge_mode_key(mode_raw) if hasattr(self, '_resolve_merge_mode_key') else mode_raw
        except Exception:
            mode_key = mode_raw
        mode_info = {}
        try:
            mode_info = self.merge_mode_map.get(mode_key, {}) if hasattr(self, 'merge_mode_map') else {}
        except Exception:
            mode_info = {}
        opts = entry.get('cli_options') if isinstance(entry.get('cli_options'), dict) else {}
        return bool(mode_info.get('needs_m2')) or bool(opts.get('turbo')) or bool(opts.get('deturbo')) or bool(str(entry.get('model2') or '').strip())

    def _planner_entry_can_produce_final(self, entry, available) -> bool:
        """Only register an output alias when the line itself can resolve its inputs.

        This prevents a disabled upstream Download/Local/Merge line from indirectly
        leaking a broken downstream output into later selector choices.
        """
        if not isinstance(entry, dict) or self._entry_is_disabled(entry):
            return False
        etype = entry.get('type')
        if etype == 'Download Model':
            return bool(str(entry.get('model_name') or '').strip()) and bool(str(entry.get('link') or '').strip())
        if etype == 'Local Model':
            path = str(entry.get('local_path') or '').strip()
            alias = str(entry.get('model_name') or '').strip() or (Path(path).stem if path else '')
            return bool(alias and path)
        if etype == 'Checkpoint Merge':
            # Model 0 is always required for merge.py style checkpoint outputs.
            if not self._planner_ref_resolves_final(entry, 'model0', available, 'Checkpoint'):
                return False
            # Most user-facing merge modes need Model 1; keep RM/NoIn as special metadata/no-input cases.
            mode = str(entry.get('merge_mode') or '').strip().upper()
            if mode not in {'RM', 'NOIN'}:
                if not self._planner_ref_resolves_final(entry, 'model1', available, 'Checkpoint'):
                    return False
            if self._planner_merge_requires_model2_final(entry):
                if not self._planner_ref_resolves_final(entry, 'model2', available, 'Checkpoint'):
                    return False
            return bool(str(entry.get('output_name') or '').strip())
        if etype == 'LoRA Bake':
            if not str(entry.get('output_name') or '').strip():
                return False
            if not self._planner_ref_resolves_final(entry, 'checkpoint', available, 'Checkpoint'):
                return False
            for lora in entry.get('loras', []) or []:
                if not self._planner_lora_ref_resolves_final(lora, available):
                    return False
            return True
        if etype == 'LoRA Merge':
            if not str(entry.get('output_name') or '').strip():
                return False
            loras = entry.get('loras', []) or []
            if not loras:
                return False
            for lora in loras:
                if not self._planner_lora_ref_resolves_final(lora, available):
                    return False
            return True
        return True

    def _collect_available_models_dependency_safe_final(self, upto_index: int) -> Dict[str, List[str]]:
        available: Dict[str, Dict[str, str]] = {'Checkpoint': {}, 'LoRA': {}, 'LyCORIS': {}}
        entries = (self.plan_data or {}).get('entries', []) or []
        for entry in entries[:max(0, int(upto_index or 0))]:
            if self._entry_is_disabled(entry):
                continue
            etype = entry.get('type')
            if etype == 'Remove Model':
                name = str(entry.get('model') or '').strip()
                if name:
                    for bucket in available.values():
                        bucket.pop(name, None)
                continue

            # Embedded Download/Local sources on an active line are real producers,
            # but only if their own source fields are usable.
            try:
                for _slot, alias, kind, spec in self._iter_embedded_sources(entry):
                    alias = str(alias or '').strip()
                    kind = str(kind or 'Checkpoint').strip() or 'Checkpoint'
                    if alias and self._planner_source_spec_is_usable_final(spec):
                        available.setdefault(kind, {})[alias] = kind
            except Exception:
                pass

            if not self._planner_entry_can_produce_final(entry, available):
                continue

            if etype == 'Download Model':
                name = str(entry.get('model_name') or '').strip()
                kind = str(entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if name:
                    available.setdefault(kind, {})[name] = kind
            elif etype == 'Local Model':
                path = str(entry.get('local_path') or '').strip()
                name = str(entry.get('model_name') or '').strip() or (Path(path).stem if path else '')
                kind = str(entry.get('model_type') or 'Checkpoint').strip() or 'Checkpoint'
                if name:
                    available.setdefault(kind, {})[name] = kind
            elif etype in ('Checkpoint Merge', 'LoRA Bake'):
                name = str(entry.get('output_name') or '').strip()
                if name:
                    available['Checkpoint'][name] = 'Checkpoint'
            elif etype == 'LoRA Merge':
                name = str(entry.get('output_name') or '').strip()
                if name:
                    available['LoRA'][name] = 'LoRA'
        return {k: sorted(v.keys()) for k, v in available.items()}

    def _entry_produced_aliases_dependency_safe_final(self, entry: Dict[str, Any]) -> List[str]:
        if self._entry_is_disabled(entry):
            return []
        etype = entry.get('type')
        produced = []
        if etype == 'Download Model':
            name = str(entry.get('model_name') or '').strip()
            if name:
                produced.append(name)
        elif etype == 'Local Model':
            path = str(entry.get('local_path') or '').strip()
            name = str(entry.get('model_name') or '').strip() or (Path(path).stem if path else '')
            if name:
                produced.append(name)
        elif etype in ('Checkpoint Merge', 'LoRA Bake', 'LoRA Merge'):
            name = str(entry.get('output_name') or '').strip()
            if name:
                produced.append(name)
        return produced

    ModelPlannerApp._planner_source_spec_is_usable_final = _planner_source_spec_is_usable_final
    ModelPlannerApp._planner_bucket_has_ref_final = _planner_bucket_has_ref_final
    ModelPlannerApp._planner_ref_resolves_final = _planner_ref_resolves_final
    ModelPlannerApp._planner_lora_ref_resolves_final = _planner_lora_ref_resolves_final
    ModelPlannerApp._planner_merge_requires_model2_final = _planner_merge_requires_model2_final
    ModelPlannerApp._planner_entry_can_produce_final = _planner_entry_can_produce_final
    ModelPlannerApp._collect_available_models = _collect_available_models_dependency_safe_final
    ModelPlannerApp._entry_produced_aliases = _entry_produced_aliases_dependency_safe_final
except Exception:
    pass


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelPlannerApp(root)
    root.mainloop()
