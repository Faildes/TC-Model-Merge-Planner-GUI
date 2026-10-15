"""Tools support responsibilities."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import TCCM_PROJECT_EXTENSION
from .constants import TCCM_PROJECT_FORMAT
from pathlib import Path
from plan import make_entry
from tkinter import Canvas
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import Scrollbar
from tkinter import Text
from tkinter import colorchooser
from tkinter import messagebox
from tkinter import ttk
from typing import Any
from typing import Dict
from typing import List
import copy
import json
import os
import re
import time
import tkinter as tk
import tkinter.font as tkfont


def _signature_autocomplete_candidates(self) -> List[str]:
    modes = [f"@m {m['key']}" for m in self.merge_modes]
    items = [
        "@p half",
        "@p bhalf",
        "@p quarter",
        "@p int8",
        "@p fp32",
        "@clipxor",
        "@save_component unet,clip",
        "@c 0",
        "@c 1",
        "@c 2",
        "@f 1,2,3",
        "@seed 1234",
        "@rank 32",
        "@arch sdxl",
        "@arch flux",
        "--bake_fp32",
        "--save_precision fp16",
        "--save_precision half",
        "--save_precision bf16",
        "--save_precision fp8",
        "--save_precision int8",
        "--save_precision fp32",
        "--clipxor",
        "--save-component unet,clip,vae",
        "--prune",
        "--save_safetensors",
    ]
    items.extend(modes)
    return items


def _entry_type_color(self, etype: str) -> str:
    return {
        "Download Model": "#1f5fbf",
        "Local Model": "#2e7d32",
        "Remove Model": "#6b7280",
        "Checkpoint Merge": "#7c3aed",
        "LoRA Bake": "#d97706",
    }.get(str(etype or ""), "#333333")


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


def _planner_visible_line_types(self):
    return ["Checkpoint Merge", "LoRA Bake"]


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


def _signature_autocomplete_candidates_extended(self):
    from .tools_support import _signature_autocomplete_candidates as _sig_orig_candidates

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
        "--merge_loras", "--merge_rank 64", "--merge_arch auto", "--merge_arch sdxl", "--merge_arch flux", "--merge_arch zi", "--merge_arch am", "--merge_arch krea2",
        "--merge_norm none", "--merge_norm sqrt", "--merge_norm mean", "--merge_scale 1.0", "--merge_unet_only",
        "--merge_clamp_q 0.99", "--merge_intermediate_mult 4",
    ]
    return list(dict.fromkeys(base + extra))


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


def _planner_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _planner_safe_title(self) -> str:
    try:
        title = self.entries["title"].get().strip()
    except Exception:
        title = ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", title or "merge_plan").strip("_") or "merge_plan"


def _planner_effective_json_payload(self) -> Dict[str, Any]:
    effective = self._planner_effective_plan_data() if hasattr(self, "_planner_effective_plan_data") else {"entries": copy.deepcopy(self.plan_data.get("entries", []))}
    return {
        "version": 3,
        "format": "tccm-effective-plan",
        "saved_at": _planner_timestamp(),
        "source_format": TCCM_PROJECT_FORMAT,
        "entries": copy.deepcopy((effective or {}).get("entries", []) or []),
    }


def _planner_write_json_file(path: str | os.PathLike, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _planner_read_json_or_none(path: str | os.PathLike):
    try:
        raw = Path(path).read_text(encoding="utf-8")
        stripped = raw.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(raw)
    except Exception:
        return None
    return None


def _planner_export_json_common(self, *, effective: bool, clipboard: bool = False, current_line: bool = False):
    from .document_support import _planner_project_json_payload

    if current_line:
        entries = self.plan_data.get("entries", []) or []
        payload = copy.deepcopy(entries[self.current_index]) if entries and 0 <= self.current_index < len(entries) else {}
        default_name = f"{_planner_safe_title(self)}_line_{self.current_index + 1}.json"
    elif effective:
        payload = _planner_effective_json_payload(self)
        default_name = f"{_planner_safe_title(self)}.effective.json"
    else:
        payload = _planner_project_json_payload(self)
        default_name = f"{_planner_safe_title(self)}{TCCM_PROJECT_EXTENSION}"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if clipboard:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="Copied JSON to clipboard")
        except Exception as e:
            self._show_detailed_error("Copy JSON Error", e, context="Operation: copy_json")
        return None
    initial_dir = str(Path(self._ensure_plan_path()).parent)
    path = filedialog.asksaveasfilename(
        title="Export JSON",
        defaultextension=".json",
        initialfile=default_name,
        initialdir=initial_dir,
        filetypes=[("TCCM Project", "*.tccm.json"), ("JSON", "*.json"), ("All files", "*.*")],
    )
    path = self._normalize_user_path(path)
    if not path:
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    self.status_label.config(text=f"Exported JSON: {self._path_display_name(path)}")
    return path


def _export_full_json(self):
    return self._planner_export_json_common(effective=False, clipboard=False)


def _export_effective_json(self):
    return self._planner_export_json_common(effective=True, clipboard=False)


def _copy_full_json(self):
    return self._planner_export_json_common(effective=False, clipboard=True)


def _copy_effective_json(self):
    return self._planner_export_json_common(effective=True, clipboard=True)


def _copy_current_line_json(self):
    return self._planner_export_json_common(effective=False, clipboard=True, current_line=True)


def _planner_meta_payload_id_aware(self):
    entries = []
    for entry in self.plan_data.get("entries", []) or []:
        if isinstance(entry, dict):
            entries.append({
                "id": str(entry.get("id") or ""),
                "type": str(entry.get("type") or ""),
                "output_name": str(entry.get("output_name") or entry.get("model_name") or entry.get("model") or ""),
                "memo": str(entry.get("memo") or ""),
                "_locked": bool(entry.get("_locked")),
                "_disabled": bool(entry.get("_disabled")),
                "_row_color": str(entry.get("_row_color") or ""),
            })
    return {"version": 2, "entries": entries}


def _planner_apply_loaded_meta_id_aware(self, payload):
    self._planner_apply_entry_defaults()
    entries = self.plan_data.get("entries", []) or []
    meta_entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(meta_entries, list):
        return
    by_id = {str(m.get("id") or ""): m for m in meta_entries if isinstance(m, dict) and str(m.get("id") or "")}
    for idx, entry in enumerate(entries):
        meta = by_id.get(str(entry.get("id") or ""))
        if meta is None and idx < len(meta_entries) and isinstance(meta_entries[idx], dict):
            meta = meta_entries[idx]
        if not isinstance(meta, dict):
            continue
        if "memo" in meta:
            entry["memo"] = str(meta.get("memo") or "")
        entry["_locked"] = bool(meta.get("_locked"))
        entry["_disabled"] = bool(meta.get("_disabled"))
        entry["_row_color"] = str(meta.get("_row_color") or "")


def _tccm_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _tccm_safe_filename(value: str, default: str = "item") -> str:
    raw = str(value or default).strip() or default
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or default


def _tccm_norm_text(value) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _tccm_display_name(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in ("output_name", "model_name", "model", "name"):
        val = _tccm_norm_text(entry.get(key))
        if val:
            return val
    typ = str(entry.get("type") or "Line")
    return typ


def _tccm_entry_output(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    typ = str(entry.get("type") or "")
    if typ in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
        return _tccm_norm_text(entry.get("output_name"))
    if typ in ("Download Model", "Local Model"):
        return _tccm_norm_text(entry.get("model_name"))
    return ""


def _tccm_entry_summary(entry: dict, index: int) -> str:
    status = "disabled" if bool(entry.get("_disabled")) else "enabled"
    name = _tccm_display_name(entry)
    return f"#{index + 1}: {entry.get('type', 'Line')} · {name or '(unnamed)'} · {status}"


def _tccm_make_text_window(self, title: str, text: str, *, width: int = 1100, height: int = 720, monospace: bool = True):
    win = tk.Toplevel(self.root)
    win.title(title)
    win.geometry(f"{width}x{height}+120+120")
    outer = Frame(win, padx=8, pady=8)
    outer.pack(fill="both", expand=True)
    toolbar = Frame(outer)
    toolbar.pack(fill="x", pady=(0, 6))
    def copy_all():
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(widget.get("1.0", "end-1c"))
            self.status_label.config(text=f"Copied {title} to clipboard")
        except Exception as exc:
            self._show_detailed_error("Copy Error", exc, context=title)
    ttk.Button(toolbar, text="Copy", command=copy_all).pack(side="left")
    ttk.Button(toolbar, text="Close", command=win.destroy).pack(side="right")
    holder = Frame(outer)
    holder.pack(fill="both", expand=True)
    widget = Text(holder, wrap="none" if monospace else "word", font=("Consolas", 10) if monospace else ("MS Gothic", 10))
    yscroll = Scrollbar(holder, orient="vertical", command=widget.yview)
    xscroll = Scrollbar(holder, orient="horizontal", command=widget.xview)
    widget.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    widget.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    holder.grid_rowconfigure(0, weight=1)
    holder.grid_columnconfigure(0, weight=1)
    widget.insert("1.0", text)
    widget.configure(state="disabled")
    return win, widget


def _tccm_ensure_extension_state(self):
    if not isinstance(getattr(self, "plan_variants", None), list):
        self.plan_variants = []
    if not isinstance(getattr(self, "run_profiles", None), list):
        self.run_profiles = []
    if not hasattr(self, "current_variant_name"):
        self.current_variant_name = ""


def _tccm_analyze_plan(self) -> dict:
    from .source_support import _tccm_aliases
    from .source_support import _tccm_entry_sources

    entries = copy.deepcopy((getattr(self, "plan_data", {}) or {}).get("entries", []) or [])
    outputs: dict[str, list[dict]] = {}
    canonical_outputs: list[dict] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        out = _tccm_entry_output(entry)
        if not out:
            continue
        item = {
            "index": idx,
            "line": idx + 1,
            "entry_id": str(entry.get("id") or ""),
            "type": str(entry.get("type") or ""),
            "name": out,
            "disabled": bool(entry.get("_disabled")),
            "summary": _tccm_entry_summary(entry, idx),
        }
        canonical_outputs.append(item)
        for alias in _tccm_aliases(out):
            outputs.setdefault(alias, []).append(item)
    duplicate_aliases = []
    for alias, producers in sorted(outputs.items(), key=lambda kv: kv[0].lower()):
        enabled = [p for p in producers if not p.get("disabled")]
        if len(enabled) > 1:
            duplicate_aliases.append({"alias": alias, "producers": enabled})
    disabled_refs = []
    forward_refs = []
    untracked_refs = []
    provenance = []
    ref_counts: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("_disabled")):
            continue
        refs = _tccm_entry_sources(entry)
        row_sources = []
        for ref in refs:
            matched = []
            for alias in _tccm_aliases(ref):
                matched.extend(outputs.get(alias, []))
            seen = {}
            for m in matched:
                seen[(m["line"], m["name"], m["disabled"])] = m
            matched = list(seen.values())
            if matched:
                ref_counts[ref] = ref_counts.get(ref, 0) + 1
                disabled_matches = [m for m in matched if m.get("disabled")]
                prior_enabled = [m for m in matched if not m.get("disabled") and int(m.get("index", 0)) < idx]
                later_enabled = [m for m in matched if not m.get("disabled") and int(m.get("index", 0)) > idx]
                if disabled_matches and not prior_enabled:
                    disabled_refs.append({"line": idx + 1, "entry": _tccm_entry_summary(entry, idx), "ref": ref, "producers": disabled_matches})
                elif later_enabled and not prior_enabled:
                    forward_refs.append({"line": idx + 1, "entry": _tccm_entry_summary(entry, idx), "ref": ref, "producers": later_enabled})
                row_sources.append({"ref": ref, "matched": matched, "status": "tracked"})
            else:
                untracked_refs.append({"line": idx + 1, "entry": _tccm_entry_summary(entry, idx), "ref": ref})
                row_sources.append({"ref": ref, "matched": [], "status": "external_or_untracked"})
        provenance.append({
            "line": idx + 1,
            "entry_id": str(entry.get("id") or ""),
            "type": str(entry.get("type") or ""),
            "disabled": bool(entry.get("_disabled")),
            "output": _tccm_entry_output(entry),
            "sources": row_sources,
            "memo": str(entry.get("memo") or ""),
        })
    unused_outputs = []
    for item in canonical_outputs:
        if item.get("disabled"):
            continue
        aliases = _tccm_aliases(item.get("name"))
        used = any((ref in ref_counts) or bool(_tccm_aliases(ref) & aliases) for ref in ref_counts.keys())
        if not used and item.get("type") not in ("Remove Model",):
            unused_outputs.append(item)
    errors = len(duplicate_aliases) + len(disabled_refs)
    warnings = len(forward_refs)
    info_count = len(untracked_refs) + len(unused_outputs)
    return {
        "entry_count": len(entries),
        "enabled_count": sum(1 for e in entries if isinstance(e, dict) and not e.get("_disabled")),
        "disabled_count": sum(1 for e in entries if isinstance(e, dict) and e.get("_disabled")),
        "output_count": len(canonical_outputs),
        "duplicate_aliases": duplicate_aliases,
        "disabled_refs": disabled_refs,
        "forward_refs": forward_refs,
        "untracked_refs": untracked_refs,
        "unused_outputs": unused_outputs,
        "provenance": provenance,
        "errors": errors,
        "warnings": warnings,
        "info_count": info_count,
    }


def _show_plan_health_dashboard(self):
    from .validation_support import _tccm_format_analysis_report

    analysis = self._tccm_analyze_plan()
    self._tccm_make_text_window("Plan Health Dashboard", _tccm_format_analysis_report(analysis))


def _show_dependency_chain_viewer(self):
    analysis = self._tccm_analyze_plan()
    lines = ["Dependency Chain Viewer", "=" * 78, ""]
    for row in analysis.get("provenance", []):
        status = "DISABLED" if row.get("disabled") else "ENABLED"
        out = row.get("output") or "(no output)"
        lines.append(f"Line {row['line']} [{status}] {row.get('type')} -> {out}")
        if row.get("memo"):
            lines.append(f"  memo: {row['memo']}")
        srcs = row.get("sources") or []
        if not srcs:
            lines.append("  sources: none")
        for src in srcs:
            matched = src.get("matched") or []
            if not matched:
                lines.append(f"  uses {src['ref']} -> external/untracked")
            else:
                parts = []
                for m in matched:
                    flag = "disabled" if m.get("disabled") else "enabled"
                    parts.append(f"line {m['line']} ({flag})")
                lines.append(f"  uses {src['ref']} -> {', '.join(parts)}")
        lines.append("")
    self._tccm_make_text_window("Dependency Chain Viewer", "\n".join(lines))


def _show_output_collision_checker(self):
    analysis = self._tccm_analyze_plan()
    duplicates = analysis.get("duplicate_aliases") or []
    if not duplicates:
        messagebox.showinfo("Output Name Collision Checker", "No enabled output-name collisions were detected.")
        return
    lines = ["Output Name Collision Checker", "=" * 78, ""]
    for item in duplicates:
        lines.append(f"Alias: {item['alias']}")
        for p in item.get("producers", []):
            lines.append(f"  Line {p['line']}: {p['type']} -> {p['name']}")
        lines.append("")
    self._tccm_make_text_window("Output Name Collision Checker", "\n".join(lines))


def _tccm_provenance_payload(self) -> dict:
    analysis = self._tccm_analyze_plan()
    return {
        "version": 1,
        "format": "tccm-model-provenance",
        "generated_at": _tccm_timestamp(),
        "summary": {
            "entry_count": analysis.get("entry_count", 0),
            "enabled_count": analysis.get("enabled_count", 0),
            "disabled_count": analysis.get("disabled_count", 0),
            "errors": analysis.get("errors", 0),
            "warnings": analysis.get("warnings", 0),
        },
        "provenance": analysis.get("provenance", []),
    }


def _show_model_provenance(self):
    text = json.dumps(self._tccm_provenance_payload(), ensure_ascii=False, indent=2)
    self._tccm_make_text_window("Model Line Provenance", text)


def _export_model_provenance_json(self):
    default = _tccm_safe_filename(getattr(self, "entries", {}).get("title").get() if "title" in getattr(self, "entries", {}) else "merge_plan") + ".provenance.json"
    path = filedialog.asksaveasfilename(
        title="Export Model Provenance JSON",
        defaultextension=".json",
        initialfile=default,
        filetypes=[("JSON", "*.json"), ("All files", "*.*")],
    )
    path = self._normalize_user_path(path)
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(self._tccm_provenance_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    self.status_label.config(text=f"Exported provenance: {self._path_display_name(path)}")


def _tccm_open_plan_dialog_no_native_filter(self) -> str:
    from .lifecycle_support import _tccm_plan_dialog_initialdir

    path = filedialog.askopenfilename(
        title="Open TCCM Project / Legacy Plan",
        initialdir=_tccm_plan_dialog_initialdir(self),
        filetypes=[
            ("Planner files", "*.tccm.json *.tccm *.json *.txt"),
            ("All files", "*.*"),
        ],
    )
    return self._normalize_user_path(path)


def _browse_plan_file_macos_safe(self):
    path = _tccm_open_plan_dialog_no_native_filter(self)
    if not path:
        return
    self.entries["filepath"].delete(0, tk.END)
    self.entries["filepath"].insert(0, path)
    self._schedule_config_save()
    try:
        self.status_label.config(text=f"Selected plan path: {self._path_display_name(path)}")
    except Exception:
        pass


def _load_preset_json_universal_json(self):
    from .document_support import _tccm_json_loader_normalize_entries_for_insert
    from .document_support import _tccm_json_loader_plan_from_any_file

    opener = _tccm_open_plan_dialog_no_native_filter
    if callable(opener):
        path = opener(self)
    else:
        path = self._normalize_user_path(filedialog.askopenfilename(title="Load preset / project / plan"))
    if not path:
        return
    try:
        loaded = _tccm_json_loader_plan_from_any_file(self, path)
        raw_entries = loaded.get("entries", []) or []
        payload = _tccm_json_loader_normalize_entries_for_insert(self, raw_entries, regenerate_ids=True)
        if not payload:
            messagebox.showwarning("Load Preset", "No loadable planner entries were found in this file.")
            return
        push = getattr(self, "_planner_push_history", None)
        if callable(push):
            push()
        entries = self.plan_data.setdefault("entries", [])
        insert_at = max(self._planner_get_selected_indices() or [self.current_index]) + 1 if entries else 0
        entries[insert_at:insert_at] = payload
        self.current_index = insert_at + len(payload) - 1
        try:
            self._save_plan_to_file()
        except Exception:
            save_project = getattr(self, "_planner_save_project_json", None)
            if callable(save_project):
                save_project(create_backup=False, reason="load-preset")
        self._refresh_line_selector()
        self._render_current_line()
        try:
            self._select_model_indices(list(range(insert_at, insert_at + len(payload))))
        except Exception:
            pass
        kind = loaded.get("kind", "preset")
        if kind == "project":
            status_kind = ".tccm.json project entries"
        elif kind == "preset-json":
            status_kind = "preset JSON"
        else:
            status_kind = "plan entries"
        self.status_label.config(text=f"Loaded {status_kind}: {Path(path).name}")
        return payload
    except Exception as exc:
        self._show_detailed_error("Load Preset Error", exc)


def _save_preset_json_universal_json(self):
    indices = self._planner_get_selected_indices() or [self.current_index]
    payload = [copy.deepcopy(self.plan_data["entries"][i]) for i in indices if 0 <= i < len(self.plan_data.get("entries", []))]
    path = filedialog.asksaveasfilename(
        title="Save preset as JSON",
        defaultextension=".json",
        initialfile="preset.json",
        filetypes=[("JSON", "*.json"), ("TCCM Project", "*.tccm.json"), ("All files", "*.*")]
    )
    path = self._normalize_user_path(path)
    if not path:
        return
    if not path.lower().endswith(".json"):
        path += ".json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    self.status_label.config(text=f"Saved preset: {Path(path).name}")


def _planner_normalize_row_color_spec(self, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"default", "none", "off"}:
        return ""
    presets = {name.lower(): name for name in self._planner_row_color_presets().keys()}
    if raw.lower() in presets:
        return presets[raw.lower()]
    if re.fullmatch(r"#[0-9a-fA-F]{3}", raw):
        return "#" + "".join(ch * 2 for ch in raw[1:]).upper()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", raw):
        return raw.upper()
    return ""


def _entry_row_color_final(self, entry: Dict[str, Any]) -> str:
    if not isinstance(entry, dict):
        return ""
    # row_color is the public/canonical field. _row_color remains as a
    # backwards-compatible sidecar key for existing projects.
    raw = entry.get("row_color")
    if raw in (None, ""):
        raw = entry.get("_row_color")
    return _planner_normalize_row_color_spec(self, raw)


def _entry_row_color_hex(self, entry: Dict[str, Any]) -> str:
    spec = _entry_row_color_final(self, entry)
    if not spec:
        return ""
    if spec.startswith("#"):
        return spec
    return str(self._planner_row_color_presets().get(spec) or "")


def _planner_apply_entry_defaults_row_color(self):
    from .editor_support import _planner_apply_entry_defaults_with_lora_merge as _row_prev_defaults

    if callable(_row_prev_defaults):
        _row_prev_defaults(self)
    for entry in self.plan_data.get("entries", []) or []:
        if not isinstance(entry, dict):
            continue
        spec = _planner_normalize_row_color_spec(self, entry.get("row_color") or entry.get("_row_color"))
        entry["row_color"] = spec
        entry["_row_color"] = spec


def _apply_plan_listbox_item_styles_final(self):
    if self.plan_listbox is None:
        return
    def rgb(value: str):
        raw = str(value or "").strip().lstrip("#")
        if len(raw) == 3:
            raw = "".join(c * 2 for c in raw)
        if len(raw) != 6:
            return None
        try:
            return tuple(int(raw[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            return None
    def mix(a: str, b: str, ratio: float) -> str:
        aa, bb = rgb(a), rgb(b)
        if aa is None or bb is None:
            return a or b
        r = max(0.0, min(1.0, float(ratio)))
        cc = tuple(round(x * (1-r) + y * r) for x, y in zip(aa, bb))
        return "#%02x%02x%02x" % cc
    def contrast(bg: str, dark: str, light: str) -> str:
        c = rgb(bg)
        if c is None:
            return light
        lum = .2126*c[0] + .7152*c[1] + .0722*c[2]
        return dark if lum >= 150 else light

    problems = getattr(self, "_plan_problem_map_cache", {}) or {}
    colors = self._theme_colors()
    light = str(getattr(self, "theme_mode", "dark") or "dark").lower() == "light"
    dark_text, light_text = "#17203a", "#f3f6ff"
    error_bg = "#f7c9c9" if light else "#5a1823"
    warn_bg = "#ffe2ad" if light else "#5a3b10"
    draft_bg = "#eee1b8" if light else "#493c1c"
    error_fg = "#8b0000" if light else "#ffd7dc"
    warn_fg = "#6d3b00" if light else "#fff0cf"

    entries = self.plan_data.get("entries", []) or []
    for vis_idx, model_idx in enumerate(self.visible_entry_indices):
        if not (0 <= model_idx < len(entries)):
            continue
        entry = entries[model_idx]
        sev = self._planner_problem_severity(problems.get(model_idx, []))
        custom = _entry_row_color_hex(self, entry)
        bg = custom or colors["entry_bg"]
        fg = contrast(bg, dark_text, light_text) if custom else (self._entry_type_color(entry.get("type", "")) or colors["entry_fg"])
        if self._entry_is_disabled(entry):
            bg, fg = colors["subtle"], colors["muted"]
        elif sev == "error":
            bg = mix(bg, error_bg, 0.82 if custom else 1.0)
            fg = error_fg
        elif sev == "warning":
            bg = mix(bg, warn_bg, 0.66 if custom else 1.0)
            fg = warn_fg
        elif sev == "draft":
            bg = mix(bg, draft_bg, 0.50 if custom else 1.0)
        elif self._entry_is_locked(entry):
            fg = "#1f5fbf" if light else "#b6c7ff"

        # Crucially, selection keeps the row/error identity instead of
        # replacing it with one global select color.
        select_bg = mix(bg, colors.get("select_bg", bg), 0.24)
        select_fg = contrast(select_bg, dark_text, light_text)
        if sev == "error":
            select_bg = mix(bg, error_bg, 0.25)
            select_fg = error_fg
        elif sev == "warning":
            select_bg = mix(bg, warn_bg, 0.25)
            select_fg = warn_fg
        try:
            self.plan_listbox.itemconfig(vis_idx, foreground=fg, background=bg,
                selectforeground=select_fg, selectbackground=select_bg)
        except Exception:
            try:
                self.plan_listbox.itemconfigure(vis_idx, fg=fg, bg=bg)
            except Exception:
                pass


def _show_row_color_dialog_final(self):
    indices = self._planner_get_selected_indices() or ([self.current_index] if self.plan_data.get("entries") else [])
    if not indices:
        return
    win = tk.Toplevel(self.root)
    win.title("Row Color")
    win.geometry("470x225+180+180")
    outer = Frame(win, padx=10, pady=10)
    outer.pack(fill="both", expand=True)
    Label(outer, text=f'Apply to lines: {", ".join(str(i + 1) for i in indices)}', anchor="w").pack(fill="x", pady=(0, 8))
    presets = ["Default"] + list(self._planner_row_color_presets().keys()) + ["Custom"]
    current = _entry_row_color_final(self, self.plan_data["entries"][indices[-1]])
    mode_var = tk.StringVar(value=(current if current and not current.startswith("#") else ("Custom" if current else "Default")))
    custom_var = tk.StringVar(value=current if current.startswith("#") else "#4A6FA5")
    combo = ttk.Combobox(outer, textvariable=mode_var, values=presets, state="readonly")
    combo.pack(fill="x")
    custom_row = Frame(outer)
    custom_row.pack(fill="x", pady=(8, 0))
    Label(custom_row, text="Custom HEX", width=14, anchor="w").pack(side="left")
    custom_entry = Entry(custom_row, textvariable=custom_var)
    custom_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
    def pick():
        _rgb, hx = colorchooser.askcolor(color=custom_var.get(), parent=win, title="Choose Row Color")
        if hx:
            custom_var.set(str(hx).upper())
            mode_var.set("Custom")
    ttk.Button(custom_row, text="Pick…", command=pick).pack(side="right")
    preview = Label(outer, text="Row color preview", anchor="center")
    preview.pack(fill="x", pady=(8, 0))
    def refresh_preview(*_):
        spec = custom_var.get() if mode_var.get() == "Custom" else mode_var.get()
        norm = _planner_normalize_row_color_spec(self, spec)
        hx = norm if norm.startswith("#") else self._planner_row_color_presets().get(norm, self._theme_colors()["entry_bg"])
        try:
            raw = str(hx or "").lstrip("#")
            if len(raw) == 3:
                raw = "".join(ch * 2 for ch in raw)
            values = tuple(int(raw[i:i+2], 16) for i in (0, 2, 4)) if len(raw) == 6 else (0, 0, 0)
            preview.configure(bg=hx, fg="#111111" if sum(values) > 390 else "#ffffff")
        except Exception:
            pass
    mode_var.trace_add("write", refresh_preview)
    custom_var.trace_add("write", refresh_preview)
    refresh_preview()
    def apply_color():
        raw = custom_var.get() if mode_var.get() == "Custom" else mode_var.get()
        spec = _planner_normalize_row_color_spec(self, raw)
        if mode_var.get() == "Custom" and not spec:
            messagebox.showerror("Row Color", "Enter #RGB or #RRGGBB.", parent=win)
            return
        self._planner_push_history()
        for idx in indices:
            if 0 <= idx < len(self.plan_data.get("entries", [])):
                self.plan_data["entries"][idx]["row_color"] = spec
                self.plan_data["entries"][idx]["_row_color"] = spec
        self._save_plan_to_file()
        self._refresh_line_selector()
        self._render_current_line()
        self._select_model_indices(indices)
        self.status_label.config(text=f"Applied row color: {spec or 'Default'}")
        win.destroy()
    buttons = Frame(outer)
    buttons.pack(fill="x", pady=(10, 0))
    ttk.Button(buttons, text="Apply", command=apply_color).pack(side="left")
    ttk.Button(buttons, text="Close", command=win.destroy).pack(side="right")
    try:
        colors = self._theme_colors()
        win.configure(bg=colors["bg"])
        self._apply_theme_to_children(win, colors)
        refresh_preview()
    except Exception:
        pass
