"""Developer tools responsibilities."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import _DEV_DEFAULT_SHORTCUTS
from .constants import _DEV_DEFAULT_T2I_SETTINGS
from PIL import Image
from PIL import ImageTk
from pathlib import Path
from plan import default_ratio
from plan import normalize_plan
from tkinter import Canvas
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import LabelFrame
from tkinter import Scrollbar
from tkinter import Text
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk
from typing import List
import copy
import json
import os
import re
import time
import tkinter as tk


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
