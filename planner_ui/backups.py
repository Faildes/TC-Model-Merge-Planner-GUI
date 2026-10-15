"""Backups controls and callbacks."""
from __future__ import annotations

from .developer_tools import _planner_save_version_snapshot as _method_planner_save_version_snapshot
from .document_support import _planner_maybe_create_backup_advanced as _method_planner_maybe_create_backup
from .document_support import _planner_offer_newer_backup_restore as _method_planner_offer_newer_backup_restore
from .document_support import _show_backup_diff_manager as _method_show_backup_diff_manager
from pathlib import Path
from tkinter import Frame
from tkinter import Label
from tkinter import Scrollbar
from tkinter import Text
from tkinter import messagebox
from tkinter import ttk
import json
import os
import tkinter as tk


class BackupsMixin:
    """Backups behavior for the Planner."""

    _planner_offer_newer_backup_restore = _method_planner_offer_newer_backup_restore


    _show_backup_diff_manager = _method_show_backup_diff_manager


    def _planner_plan_snapshot(self):
        return json.dumps(self.plan_data, ensure_ascii=False, sort_keys=True)


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


    _planner_save_version_snapshot = _method_planner_save_version_snapshot


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


    _planner_maybe_create_backup = _method_planner_maybe_create_backup


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
