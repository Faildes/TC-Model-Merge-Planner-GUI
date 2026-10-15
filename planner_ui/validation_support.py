"""Validation support responsibilities."""
from __future__ import annotations

from pathlib import Path
from tkinter import Frame
from tkinter import Label
from tkinter import Scrollbar
from tkinter import Text
from tkinter import ttk
from typing import Dict
from typing import List
from urllib.error import HTTPError as _planner_HTTPError
from urllib.error import URLError as _planner_URLError
from urllib.parse import urlsplit as _planner_urlsplit
from urllib.request import Request as _planner_Request
from urllib.request import urlopen as _planner_urlopen
import re
import re as _planner_patch_re
import tkinter as tk


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


def _show_prevalidation_with_download_checks(self):
    from .validation_support import _show_prevalidation_with_causes as _planner_prev_show_prevalidation_for_download_patch

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


def _tccm_format_analysis_report(analysis: dict) -> str:
    lines = []
    lines.append("TCCM Plan Health Dashboard")
    lines.append("=" * 78)
    lines.append(f"Entries         : {analysis.get('entry_count', 0)}")
    lines.append(f"Enabled entries : {analysis.get('enabled_count', 0)}")
    lines.append(f"Disabled entries: {analysis.get('disabled_count', 0)}")
    lines.append(f"Produced outputs: {analysis.get('output_count', 0)}")
    lines.append(f"Errors          : {analysis.get('errors', 0)}")
    lines.append(f"Warnings        : {analysis.get('warnings', 0)}")
    lines.append(f"Info items      : {analysis.get('info_count', 0)}")
    lines.append("")
    if not analysis.get("errors") and not analysis.get("warnings"):
        lines.append("Result: no blocking internal dependency issues detected.")
        lines.append("")
    if analysis.get("duplicate_aliases"):
        lines.append("[Errors] Output name collisions")
        for item in analysis["duplicate_aliases"]:
            lines.append(f"- Alias: {item['alias']}")
            for p in item.get("producers", []):
                lines.append(f"    Line {p['line']}: {p['type']} -> {p['name']}")
        lines.append("")
    if analysis.get("disabled_refs"):
        lines.append("[Errors] Enabled lines that depend on disabled outputs")
        for item in analysis["disabled_refs"]:
            lines.append(f"- Line {item['line']} uses {item['ref']}")
            lines.append(f"    {item['entry']}")
            for p in item.get("producers", []):
                lines.append(f"    Produced by disabled line {p['line']}: {p['name']}")
        lines.append("")
    if analysis.get("forward_refs"):
        lines.append("[Warnings] Forward references")
        for item in analysis["forward_refs"]:
            lines.append(f"- Line {item['line']} uses {item['ref']} before it is produced")
            for p in item.get("producers", []):
                lines.append(f"    Later producer line {p['line']}: {p['name']}")
        lines.append("")
    if analysis.get("untracked_refs"):
        lines.append("[Info] External or untracked model references")
        lines.append("These may be normal if the model already exists in your model directory.")
        for item in analysis["untracked_refs"][:120]:
            lines.append(f"- Line {item['line']}: {item['ref']}")
        if len(analysis["untracked_refs"]) > 120:
            lines.append(f"  ... and {len(analysis['untracked_refs']) - 120} more")
        lines.append("")
    if analysis.get("unused_outputs"):
        lines.append("[Info] Produced outputs not used by later planner lines")
        for item in analysis["unused_outputs"][:120]:
            lines.append(f"- Line {item['line']}: {item['name']}")
        if len(analysis["unused_outputs"]) > 120:
            lines.append(f"  ... and {len(analysis['unused_outputs']) - 120} more")
        lines.append("")
    return "\n".join(lines)


def _planner_problem_severity_final(self, problems: List[str]) -> str:
    vals = [str(p).strip() for p in (problems or []) if str(p or "").strip()]
    if any(p.upper().startswith("ERROR:") for p in vals):
        return "error"
    if any(p.upper().startswith("WARN:") for p in vals):
        return "warning"
    if any(p.upper().startswith("DRAFT:") for p in vals):
        return "draft"
    return "ok"


def _plan_entry_problem_map_linear(self) -> Dict[int, List[str]]:
    """Source-aware validation in a single forward availability pass."""
    entries = (self.plan_data or {}).get("entries", []) or []
    analysis = self._planner_analysis()
    dead_entries = set(analysis.get("dead_entries", set()))
    unref_aliases = analysis.get("unreferenced_aliases", {}) or {}
    same_source_by_idx: Dict[int, set[str]] = {}
    for _sig, recs in (analysis.get("same_source_aliases") or {}).items():
        aliases = sorted({str(r.get("alias") or "") for r in recs if str(r.get("alias") or "")})
        if len(aliases) <= 1:
            continue
        text = "WARN: Same Source Registered With Different Aliases -> " + ", ".join(aliases)
        for rec in recs:
            try:
                same_source_by_idx.setdefault(int(rec.get("idx")), set()).add(text)
            except Exception:
                pass

    eager_link_validation = bool(getattr(self, "_planner_eager_link_validation", False))
    available = {"Checkpoint": set(), "LoRA": set(), "LyCORIS": set()}
    problems_by_idx: Dict[int, List[str]] = {}

    def problem(level: str, idx: int, etype: str, message: str) -> str:
        return f"{level}: Line {idx + 1} ({etype}): {message}"

    def link_issue(link: str):
        fn = _planner_download_link_issue
        if callable(fn):
            try:
                return fn(self, str(link or "").strip(), eager=eager_link_validation)
            except Exception:
                pass
        text = str(link or "").strip()
        if not text:
            return "Link Is Empty"
        if not re.match(r"^https?://", text, flags=re.IGNORECASE):
            return f"Download Link Format Is Invalid -> {text}"
        return None

    def embedded_issue(spec, label: str):
        if not isinstance(spec, dict):
            return None
        mode = str(spec.get("mode") or "").strip().lower()
        if mode == "download":
            issue = link_issue(str(spec.get("link") or ""))
            return f"{label} {issue}" if issue else None
        if mode == "local" and not str(spec.get("local_path") or "").strip():
            return f"{label} Local Path Is Empty"
        return None

    for idx, entry in enumerate(entries):
        etype = str(entry.get("type") or "")
        probs: List[str] = []
        disabled = self._entry_is_disabled(entry)
        if disabled:
            probs.append("WARN: Entry Is Disabled And Excluded From Export/Runtime")
        if self._entry_is_locked(entry):
            probs.append("WARN: Entry Is Locked Against Accidental Edits")

        if etype == "Download Model":
            name = str(entry.get("model_name") or "").strip()
            kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
            if not name:
                probs.append(problem("ERROR", idx, etype, "Model Name Is Empty"))
            issue = link_issue(str(entry.get("link") or ""))
            if issue:
                probs.append(problem("ERROR", idx, etype, issue))
            if name and not disabled:
                available.setdefault(kind, set()).add(name)

        elif etype == "Local Model":
            path = str(entry.get("local_path") or "").strip()
            alias = str(entry.get("model_name") or "").strip()
            kind = str(entry.get("model_type") or "Checkpoint").strip() or "Checkpoint"
            if not path:
                probs.append(problem("ERROR", idx, etype, "Local Path Is Empty"))
            if path and not disabled:
                available.setdefault(kind, set()).add(alias or Path(path).stem)

        elif etype == "Remove Model":
            name = str(entry.get("model") or "").strip()
            if not name:
                probs.append(problem("ERROR", idx, etype, "Model Is Empty"))
            if name and not disabled:
                for bucket in available.values():
                    bucket.discard(name)

        elif etype == "Checkpoint Merge":
            from planner_core.merge_validation import required_model_slots, turbo_problem
            issue = turbo_problem(entry)
            if issue:
                probs.append(problem("ERROR", idx, etype, issue))
            required = [(slot, slot.replace("model", "Model ")) for slot in required_model_slots(entry)]
            required.append(("output_name", "Output Name"))
            for key, label in required:
                if not str(entry.get(key) or "").strip():
                    probs.append(problem("ERROR", idx, etype, f"{label} Is Empty"))
            for slot, label in (("model0", "Model 0"), ("model1", "Model 1"), ("model2", "Model 2")):
                ref = str(entry.get(slot) or "").strip()
                if not ref:
                    continue
                spec = self._get_entry_slot_source(entry, slot)
                issue = embedded_issue(spec, label)
                if issue:
                    probs.append(problem("ERROR", idx, etype, issue))
                elif isinstance(spec, dict):
                    if not disabled:
                        available["Checkpoint"].add(ref)
                elif ref not in available["Checkpoint"]:
                    probs.append(problem("ERROR", idx, etype, f"Checkpoint Reference Not Available -> {ref}"))
            out = str(entry.get("output_name") or "").strip()
            if out and not disabled:
                available["Checkpoint"].add(out)

        elif etype == "LoRA Bake":
            checkpoint = str(entry.get("checkpoint") or "").strip()
            spec = self._get_entry_slot_source(entry, "checkpoint")
            if not checkpoint:
                probs.append(problem("ERROR", idx, etype, "Checkpoint Is Empty"))
            else:
                issue = embedded_issue(spec, "Checkpoint")
                if issue:
                    probs.append(problem("ERROR", idx, etype, issue))
                elif isinstance(spec, dict):
                    if not disabled:
                        available["Checkpoint"].add(checkpoint)
                elif checkpoint not in available["Checkpoint"]:
                    probs.append(problem("ERROR", idx, etype, f"Checkpoint Reference Not Available -> {checkpoint}"))
            if not str(entry.get("output_name") or "").strip():
                probs.append(problem("ERROR", idx, etype, "Output Name Is Empty"))
            if not entry.get("loras"):
                probs.append(problem("ERROR", idx, etype, "At Least One LoRA Is Required"))
            for li, lora in enumerate(entry.get("loras", []) or []):
                name = str(lora.get("name") or "").strip()
                label = f"LoRA {li + 1}"
                if not name:
                    probs.append(problem("ERROR", idx, etype, f"{label} Name Is Empty"))
                    continue
                lspec = self._get_lora_source(lora)
                issue = embedded_issue(lspec, label)
                if issue:
                    probs.append(problem("ERROR", idx, etype, issue))
                elif isinstance(lspec, dict):
                    if not disabled:
                        kind = str(lspec.get("kind") or "LoRA")
                        available.setdefault(kind, set()).add(name)
                elif name not in available["LoRA"] and name not in available["LyCORIS"]:
                    probs.append(problem("ERROR", idx, etype, f"LoRA Reference Not Available -> {name}"))
            out = str(entry.get("output_name") or "").strip()
            if out and not disabled:
                available["Checkpoint"].add(out)

        elif etype == "LoRA Merge":
            out = str(entry.get("output_name") or "").strip()
            if not out:
                probs.append(problem("ERROR", idx, etype, "Output Name Is Empty"))
            loras = entry.get("loras", []) or []
            if not loras:
                probs.append(problem("ERROR", idx, etype, "No LoRAs Selected"))
            for li, lora in enumerate(loras):
                name = str(lora.get("name") or "").strip()
                label = f"LoRA {li + 1}"
                if not name:
                    probs.append(problem("ERROR", idx, etype, f"{label} Name Is Empty"))
                    continue
                lspec = self._get_lora_source(lora)
                issue = embedded_issue(lspec, label)
                if issue:
                    probs.append(problem("ERROR", idx, etype, issue))
                elif isinstance(lspec, dict):
                    if not disabled:
                        kind = str(lspec.get("kind") or "LoRA")
                        available.setdefault(kind, set()).add(name)
                elif name not in available["LoRA"] and name not in available["LyCORIS"]:
                    probs.append(problem("ERROR", idx, etype, f"LoRA Reference Not Available -> {name}"))
            if out and not disabled:
                available["LoRA"].add(out)

        # Preserve analysis warnings without an O(n^2) availability scan.
        for alias in self._entry_produced_aliases(entry):
            if alias in unref_aliases:
                probs.append(f"WARN: Produced Alias Is Currently Unreferenced -> {alias}")
        if idx in same_source_by_idx:
            probs.extend(sorted(same_source_by_idx[idx]))
        if idx in dead_entries:
            probs.append("WARN: Merge Output Is Not Required By The Final Active Line")
        problems_by_idx[idx] = probs
    return problems_by_idx
