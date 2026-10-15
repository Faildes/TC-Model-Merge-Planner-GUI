"""Plan editing controls and callbacks."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import DOWNLOAD_TYPES
from .developer_tools import _planner_load_alpha_preset as _method_planner_load_alpha_preset
from .developer_tools import _planner_save_alpha_preset as _method_planner_save_alpha_preset
from .editor_support import _base_render_current_line_with_lora_merge as _method_base_render_current_line
from .editor_support import _line_summary_with_lora_merge as _method_line_summary
from .editor_support import _on_plan_list_motion_sigless as _method_on_plan_list_motion
from .editor_support import _planner_convert_entry_type_preserving_common_fields_with_lora_merge as _method_planner_convert_entry_type_preserving_common_fields
from .editor_support import _planner_visible_line_types_with_lora_merge as _method_planner_visible_line_types
from .help_support import _planner_blend_hex as _method_planner_blend_hex
from .tools_support import _planner_apply_entry_defaults_row_color as _method_planner_apply_entry_defaults
from .tools_support import _show_plan_context_menu as _method_show_plan_context_menu
from .tools_support import _tccm_make_text_window as _method_tccm_make_text_window
from pathlib import Path
from plan import default_ratio
from plan import export_plan_records_txt
from plan import make_entry
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import LabelFrame
from tkinter import Scrollbar
from tkinter import Text
from tkinter import messagebox
from tkinter import ttk
from typing import Any
from typing import Dict
from typing import List
import copy
import difflib
import json
import re
import tempfile
import tkinter as tk
import traceback


class PlanEditingMixin:
    """Plan editing behavior for the Planner."""

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


    _planner_apply_entry_defaults = _method_planner_apply_entry_defaults


    def _entry_is_locked(self, entry: Dict[str, Any]) -> bool:
        return bool(entry.get('_locked')) if isinstance(entry, dict) else False


    def _entry_is_disabled(self, entry: Dict[str, Any]) -> bool:
        return bool(entry.get('_disabled')) if isinstance(entry, dict) else False


    _planner_convert_entry_type_preserving_common_fields = _method_planner_convert_entry_type_preserving_common_fields


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


    def _event_has_open_modifier(self, event) -> bool:
        state = int(getattr(event, "state", 0) or 0)
        return bool(state & 0x0004 or state & 0x0008 or state & 0x0010 or state & 0x0080)


    @staticmethod
    def _is_shift_pressed(event) -> bool:
        state = int(getattr(event, "state", 0) or 0)
        return bool(state & 0x0001)


    def _add_tag_for_span(self, widget: Text, tag: str, start: int, end: int):
        if end <= start:
            return
        widget.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")


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


    def _on_root_resize(self, event=None):
        if event is not None and getattr(event, "widget", None) is not self.root:
            return
        self._schedule_responsive_layout()


    def _reset_plan(self):
        self._planner_push_history()
        result = self._base_reset_plan()
        self._refresh_line_selector()
        self._planner_refresh_plan_meta()
        return result


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


    def _format_exception_text(self, title: str, exc: Exception, *, plan_entry: Dict[str, Any] | None = None) -> str:
        from .errors import PlanCompileError

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


    def _base_reset_plan(self):
        self.plan_data = self._planner_default_visible_plan()
        self.current_index = 0
        self._refresh_line_selector()
        self._render_current_line()
        self.status_label.config(text=f"Reset the Plan")


    def _planner_default_visible_plan(self):
        return {"entries": [make_entry("Checkpoint Merge")]}


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


    def _planner_unused_summary(self):
        analysis = self._planner_analysis()
        dead_count = len(analysis['dead_entries'])
        extra = len(analysis['unreferenced_aliases'])
        if dead_count <= 0 and extra <= 0:
            return 'Unused: 0 dead line(s)'
        return f'Unused: {dead_count} dead line(s) / {extra} unreferenced alias(es)'


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


    _planner_save_alpha_preset = _method_planner_save_alpha_preset


    _planner_load_alpha_preset = _method_planner_load_alpha_preset


    _planner_blend_hex = _method_planner_blend_hex


    _tccm_make_text_window = _method_tccm_make_text_window


    def _strip_mode_signatures(self, raw_text: str) -> str:
        text = str(raw_text or "")
        if not text:
            return ""
        text = re.sub(r"(?<!\S)@(?:m|mode)\s+\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"[ 	]+", " ", text)
        text = re.sub(r"[ 	]*\n[ 	]*", "\n", text)
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines).strip()


    @staticmethod
    def _memo_preview_text(text: str, head: int = 12, tail: int = 10) -> str:
        raw = " ".join(str(text or "").split())
        if not raw:
            return ""
        if len(raw) <= head + tail + 3:
            return raw
        return f"{raw[:head]}…{raw[-tail:]}"


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


    _show_plan_context_menu = _method_show_plan_context_menu


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


    def _schedule_rerender_current_line(self):
        if self._rerender_after_id is not None:
            try:
                self.root.after_cancel(self._rerender_after_id)
            except Exception:
                pass
        self._rerender_after_id = self.root.after(1, self._perform_scheduled_rerender)


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


    _on_plan_list_motion = _method_on_plan_list_motion


    def _set_active_multiline_text(self, widget: Text | None):
        try:
            if widget is not None and widget.winfo_exists():
                self._active_multiline_text = widget
                return
        except Exception:
            pass
        self._active_multiline_text = None


    def _current_entry_context(self) -> str:
        try:
            entries = self.plan_data.get("entries", [])
            if not entries:
                return ""
            entry = entries[self.current_index]
            return "Current Plan Entry:\n" + json.dumps(entry, ensure_ascii=False, indent=2)
        except Exception:
            return ""


    def _select_directory(self, entry_key):
        directory_path = filedialog.askdirectory(title=f"Select {entry_key.replace('_', ' ').title()}")
        directory_path = self._normalize_user_path(directory_path)
        if directory_path:
            self.entries[entry_key].delete(0, tk.END)
            self.entries[entry_key].insert(0, directory_path)
            self._schedule_config_save()


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


    _line_summary = _method_line_summary


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


    def _change_line_type(self, new_type: str):
        entries = self.plan_data.get('entries', [])
        if 0 <= self.current_index < len(entries) and self._entry_is_locked(entries[self.current_index]):
            messagebox.showwarning('Locked Line', f'Line {self.current_index + 1} is locked and cannot change type.')
            return
        self._planner_push_history()
        if new_type not in self._planner_visible_line_types() and new_type not in self._planner_internal_line_types():
            new_type = 'Checkpoint Merge'
        return self._base_change_line_type(new_type)


    def _render_download_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Download Model")
        self._build_entry_row(frame, "Model Name", entry, "model_name")
        self._build_entry_row(frame, "Link", entry, "link")
        self._build_combo_row(frame, "Type", entry, "model_type", DOWNLOAD_TYPES)


    def _render_remove_entry(self, parent, entry: Dict[str, Any]):
        frame = self._build_labeled_frame(parent, "Remove Model")
        available = self._collect_available_models(self.current_index)
        candidates = available.get("Checkpoint", []) + available.get("LoRA", []) + available.get("LyCORIS", [])
        if entry.get("model") and entry["model"] not in candidates:
            candidates = [entry["model"]] + candidates
        self._build_combo_row(frame, "Model", entry, "model", candidates or [""])


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


    def _update_status_from_line(self, line: str) -> str:
        from .runner import RunnerConsoleWindow

        stripped = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", str(line or "")).strip()
        if not stripped:
            return "info"
        lower = stripped.lower()
        msg = stripped[:180]

        def push_progress(text: str, status_prefix: str = "Progress"):
            text = str(text or "").strip()
            if not text:
                return
            fraction = RunnerConsoleWindow._extract_progress_fraction(text)
            self.root.after(0, lambda: self.status_label.config(text=f"{status_prefix}: {text[:100]}"))
            self.console.set_progress(text)
            if fraction is not None:
                self.console.set_progress_fraction(fraction, text)

        proc_marker = "[planner-process]"
        marker_pos = stripped.find(proc_marker)
        if marker_pos >= 0:
            payload = stripped[marker_pos + len(proc_marker):].lstrip(" \t:")
            parts = payload.split("\t")
            if len(parts) >= 4:
                status = parts[0].strip().upper()
                ident = parts[1].strip() or "-"
                kind = parts[2].strip() or "process"
                if status in {"END", "FAILED"} and len(parts) >= 5:
                    rc = parts[3].strip()
                    detail = "\t".join(parts[4:]).strip()
                    if rc:
                        detail = f"rc={rc} | {detail}" if detail else f"rc={rc}"
                else:
                    detail = "\t".join(parts[3:]).strip()
                self.console.process_event(status, kind, ident, detail)
                self.console.set_step(detail[:120] or f"{kind} {status.lower()}")
                return "error" if status == "FAILED" else "process"

        progress_marker = "[planner-progress]"
        progress_pos = stripped.find(progress_marker)
        if progress_pos >= 0:
            progress_msg = stripped[progress_pos + len(progress_marker):].strip() or msg
            push_progress(progress_msg, "Progress")
            m = re.match(r"(\d+)\s*/\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*(.*)", progress_msg)
            if m:
                idx, total, step_kind, label = m.groups()
                ident = f"step {idx}/{total}"
                previous = getattr(self, "_last_console_plan_step", None)
                if previous and previous[0] != ident:
                    self.console.process_event("DONE", "plan", previous[0], previous[1])
                detail = f"{step_kind.strip()}: {label.strip()}".strip(": ")
                self._last_console_plan_step = (ident, detail)
                self.console.process_event("RUNNING", "plan", ident, detail)
                self.console.set_step(detail or ident)
            return "cell"

        if stripped.startswith("[#") or ("dl:" in lower and ("eta:" in lower or "%" in stripped)):
            push_progress(msg, "Downloading")
            return "download"

        if "%|" in stripped or "it/s" in lower or "s/it" in lower:
            push_progress(msg, "Progress")
            if any(k in lower for k in ["merging", "difference", "baking", "preparing merged"]):
                return "merge"
            return "info"

        if stripped.startswith("$"):
            if "merge.py" in lower or "lora_bake.py" in lower:
                push_progress(msg, "Merging")
                self.console.set_step(msg[:120])
                return "merge"
            if any(k in lower for k in ["aria2", "wget", "curl"]):
                push_progress(msg, "Downloading")
                self.console.set_step(msg[:120])
                return "download"

        if "traceback (most recent call last)" in lower or lower.startswith("error:") or " failed" in lower or "exception" in lower:
            return "error"
        if "warning" in lower or "warn:" in lower:
            return "warning"
        if any(k in lower for k in ["uploading", "upload file", "hugging face", "huggingface"]):
            return "upload"
        if any(k in lower for k in ["download", "aria2", "custom_model"]):
            push_progress(msg, "Downloading")
            return "download"
        if any(k in lower for k in ["merge.py", "merging", "lora_bake.py", "checkpoint merge", "lora bake", "getting difference"]):
            push_progress(msg, "Merging")
            return "merge"
        if any(k in lower for k in ["saving", "output", "register"]):
            push_progress(msg, "Saving")
            return "save"
        if any(k in lower for k in ["done!", "completed successfully", "success"]):
            return "success"
        return "info"


    _base_render_current_line = _method_base_render_current_line


    def _base_change_line_type(self, new_type: str):
        self.plan_data["entries"][self.current_index] = make_entry(new_type)
        self._schedule_rerender_current_line()


    def _flush_after_entry_change(self):
        self._after_entry_change_id = None
        self._refresh_line_selector()
        self._save_plan_to_file()
        self._planner_refresh_plan_meta()
        self._update_current_line_indicator()


    def _planner_restore_plan_list_focus(self):
        if self.plan_listbox is None:
            return
        try:
            if self.plan_listbox.winfo_exists():
                self.plan_listbox.focus_set()
        except Exception:
            pass


    def _planner_deepcopy_plan(self):
        return copy.deepcopy(self.plan_data)


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


    _planner_visible_line_types = _method_planner_visible_line_types


    def _planner_internal_line_types(self):
        return ["Download Model", "Local Model", "Remove Model"]


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


    def _selected_entries_support_ws_collapse(self):
        indices = self._planner_get_selected_indices()
        if not indices:
            return False
        entries = self.plan_data.get("entries", [])
        selected = [entries[idx] for idx in indices if 0 <= idx < len(entries)]
        return bool(selected) and all(entry.get("type") == "Checkpoint Merge" for entry in selected)
