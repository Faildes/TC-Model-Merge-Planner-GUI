"""Source support responsibilities."""
from __future__ import annotations

from . import dialogs as filedialog
from pathlib import Path
from plan import make_entry
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk
from typing import Any
from typing import Dict
from typing import List
import copy
import os
import re
import tkinter as tk


def _apply_resource_limits_to_env(env: Dict[str, str]) -> Dict[str, str]:
    from .resources import _planner_safe_cpu_threads

    threads = _planner_safe_cpu_threads()
    env.setdefault("CHATTIORI_CPU_THREADS", str(threads))
    env.setdefault("CHATTIORI_INT8_COMPUTE_DTYPE", "fp16")
    env.setdefault("CHATTIORI_SIM_SCRATCH_CACHE", "4")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(name, str(threads))
    return env


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


def _planner_add_local_choice(self, choices: Dict[str, str], path: str, preferred_label: str = ''):
    from .document_support import _planner_norm_path_key

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


def _scan_local_model_choices_retained(self) -> Dict[str, str]:
    from .source_support import _scan_local_model_choices as _planner_prev_scan_local_model_choices

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


def _planner_display_for_local_path(self, choices: Dict[str, str], path: str) -> str:
    from .document_support import _planner_norm_path_key

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


def _planner_source_signature(self, mode: str, value: str) -> str:
    from .document_support import _planner_norm_path_key

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


def _iter_embedded_sources_with_lora_merge(self, entry):
    from .source_support import _iter_embedded_sources as _lm_orig_iter_embedded_sources

    if entry.get("type") == "LoRA Merge":
        for idx, lora in enumerate(entry.get("loras", []) or []):
            spec = self._get_lora_source(lora)
            alias = str(lora.get("name") or self._source_alias_from_spec(spec)).strip()
            if spec and alias:
                yield f"lora:{idx}", alias, str(spec.get("kind") or "LoRA"), spec
        return
    if _lm_orig_iter_embedded_sources:
        yield from _lm_orig_iter_embedded_sources(self, entry)


def _entry_consumed_aliases_with_lora_merge(self, entry: Dict[str, Any]) -> List[str]:
    from .source_support import _entry_consumed_aliases as _lm_orig_entry_consumed

    if hasattr(self, "_entry_is_disabled") and self._entry_is_disabled(entry):
        return []
    if entry.get("type") == "LoRA Merge":
        return [str(lora.get("name") or "").strip() for lora in entry.get("loras", []) or [] if str(lora.get("name") or "").strip()]
    return _lm_orig_entry_consumed(self, entry) if _lm_orig_entry_consumed else []


def _build_source_backed_ref_row_dropdown_refined(self, parent, label: str, get_value, set_value, get_source, set_source, candidates, kind_options, allow_local: bool = True):
    from .source_support import _build_source_backed_ref_row as _refined_source_row

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


def _render_checkpoint_merge_entry_source_buttons_final(self, parent, entry: Dict[str, Any]):
    from .editor_support import _lm_build_cli_options_panel
    from .help_support import _sigless_build_cli_options_panel_dropdown_refined as _sigless_build_cli_options_panel
    from .editor_support import _sigless_ensure_cli_options_display as _sigless_ensure_cli_options
    from .editor_support import _sigless_model_count

    ensure_cli = _sigless_ensure_cli_options
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
    model_count_func = _sigless_model_count
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

    self._build_ratio_section(parent, entry, "alpha", "Alpha")
    if mode_info.get("needs_beta"):
        self._build_ratio_section(parent, entry, "beta", "Beta")

    out_frame = self._build_labeled_frame(parent, "Output")
    self._build_entry_row(out_frame, "Output Name", entry, "output_name")

    build_cli = _sigless_build_cli_options_panel or _lm_build_cli_options_panel
    if callable(build_cli):
        build_cli(self, parent, entry, "Checkpoint Merge")
    else:
        self._build_text_row(parent, "Additional Signatures", entry, "raw_signatures", height=5)


def _render_lora_bake_entry_source_buttons_hover_final(self, parent, entry: Dict[str, Any]):
    from .editor_support import _render_lora_bake_entry_sigless as _source_buttons_prev_lora_bake_renderer

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


def _render_lora_merge_entry_source_buttons_hover_final(self, parent, entry: Dict[str, Any]):
    from .editor_support import _render_lora_merge_entry_sigless as _source_buttons_prev_lora_merge_renderer

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
    from planner_core.merge_validation import required_model_slots
    return 'model2' in required_model_slots(entry) or bool(mode_info.get('needs_m2')) or bool(str(entry.get('model2') or '').strip())


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
        from planner_core.merge_validation import required_model_slots, turbo_problem
        if turbo_problem(entry):
            return False
        if 'model1' in required_model_slots(entry):
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
        if not entry.get('loras'):
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


def _tccm_aliases(value) -> set[str]:
    from .tools_support import _tccm_norm_text

    raw = _tccm_norm_text(value)
    if not raw:
        return set()
    aliases = {raw}
    try:
        base = Path(raw).name
        if base:
            aliases.add(base)
            aliases.add(Path(base).stem)
            if not base.lower().endswith(".safetensors"):
                aliases.add(base + ".safetensors")
        if not raw.lower().endswith(".safetensors"):
            aliases.add(raw + ".safetensors")
    except Exception:
        pass
    return {a for a in aliases if a}


def _tccm_entry_sources(entry: dict) -> list[str]:
    from .tools_support import _tccm_norm_text

    if not isinstance(entry, dict):
        return []
    typ = str(entry.get("type") or "")
    out: list[str] = []
    def add(value):
        value = _tccm_norm_text(value)
        if value and value not in out:
            out.append(value)
    if typ == "Checkpoint Merge":
        for key in ("model0", "model1", "model2"):
            add(entry.get(key))
    elif typ == "LoRA Bake":
        add(entry.get("checkpoint"))
        for item in entry.get("loras") or []:
            if isinstance(item, dict):
                add(item.get("name"))
            else:
                add(item)
    elif typ == "LoRA Merge":
        for item in entry.get("loras") or []:
            if isinstance(item, dict):
                add(item.get("name"))
            else:
                add(item)
    elif typ == "Remove Model":
        add(entry.get("model"))
    return out
