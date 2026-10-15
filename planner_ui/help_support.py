"""Help support responsibilities."""
from __future__ import annotations

from .constants import RATIO_MODES
from .settings import _CHOICE_HELP_TEXT
from .constants import _DISPLAY_ARCH_CHOICES
from .constants import _DISPLAY_PRECISION_CHOICES
from .constants import _LORA_MERGE_DEFAULTS
from .settings import _MODE_HELP_TEXT
from .settings import _OPTION_HELP_TEXT
from .constants import _SIGLESS_ARCH_CHOICES
from .constants import _SIGLESS_CLI_SPECS
from .constants import _SIGLESS_PRECISION_CHOICES
from .settings import RIGHT_PANEL_FIELD_HELP
from pathlib import Path
from plan import default_ratio
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import LabelFrame
from tkinter import ttk
from typing import Any
from typing import Dict
from typing import List
import copy
import tkinter as tk


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


def _build_plan_item_tooltip_text_with_causes(self, model_idx: int, entry: Dict[str, Any], problems: List[str] | None = None) -> str:
    from .help_support import _build_plan_item_tooltip_text as _planner_prev_build_tooltip

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


def _build_plan_item_tooltip_text_with_lora_merge(self, model_idx: int, entry: Dict[str, Any], problems: List[str] | None = None) -> str:
    from .help_support import _build_plan_item_tooltip_text_with_causes as _lm_orig_tooltip

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
    from .widgets import _PlannerDynamicTooltip

    if not isinstance(widgets, (list, tuple, set)):
        widgets = [widgets]
    for widget in widgets:
        if widget is not None:
            _PlannerDynamicTooltip(widget, provider, delay=delay, wraplength=wraplength)


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
    detail = _planner_mode_detail_text(current_key, mode_info, fallback="Backend merge mode. Check merge.py / merge_modes.py for exact tensor behavior.") if True else "Backend merge mode."
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


def _build_combo_row_dropdown_refined(self, parent, label: str, entry: Dict[str, Any], key: str, values: List[str], callback=None):
    from .tools_support import _build_combo_row as _refined_base_combo_row

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


def _sigless_special_option_row_dropdown_refined(self, parent, entry: Dict[str, Any], key: str, label: str, choices: List[str]):
    from .editor_support import _planner_display_arch_name
    from .editor_support import _planner_display_precision_name

    row = Frame(parent)
    row.pack(fill="x", pady=2)
    lab = Label(row, text=label, width=18, anchor="w")
    lab.pack(side="left")
    if key == "precision":
        value = _planner_display_precision_name(entry.get(key)) if True else str(entry.get(key) or "")
        choices = _DISPLAY_PRECISION_CHOICES if True else choices
        help_key = "Precision"
    elif key == "architecture":
        value = _planner_display_arch_name(entry.get(key)) if True else str(entry.get(key) or "")
        choices = _DISPLAY_ARCH_CHOICES if True else choices
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


def _sigless_build_cli_options_panel_dropdown_refined(self, parent, entry: Dict[str, Any], entry_type: str):
    from .editor_support import _planner_display_arch_name
    from .editor_support import _sigless_build_unmapped_editor
    from .editor_support import _sigless_ensure_cli_options_display as _sigless_ensure_cli_options
    from .editor_support import _sigless_mode_uses_seed
    from .editor_support import _sigless_option_changed_display as _sigless_option_changed
    from .help_support import _sigless_special_option_row_dropdown_refined as _sigless_special_option_row

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
                if key == "merge_arch" and True:
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
        e["cli_options"] = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(et, {})) if True else {}
        e["cli_options"].pop("rand_alpha", None)
        e["cli_options"].pop("rand_beta", None)
        self._after_entry_change()
        self._schedule_rerender_current_line()
    reset_btn = ttk.Button(reset_row, text="Reset Options", command=reset_options)
    reset_btn.pack(side="right")
    self._planner_attach_dynamic_tooltip(reset_btn, lambda: "Reset this line's structured Options to their defaults. Unmapped legacy signatures are left untouched.")


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
