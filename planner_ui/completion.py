"""Completion controls and callbacks."""
from __future__ import annotations

from .tools_support import _signature_autocomplete_candidates_extended as _method_signature_autocomplete_candidates
from tkinter import Canvas
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import Scrollbar
from tkinter import Text
from typing import Dict
from typing import List
import re
import tkinter as tk


class CompletionMixin:
    """Completion behavior for the Planner."""

    @staticmethod
    def _autocomplete_item(insert: str, effect: str = "", *, label: str = "") -> Dict[str, str]:
        display = label or (f"{insert}    — {effect}" if effect else insert)
        return {"insert": insert, "display": display, "effect": effect}


    _signature_autocomplete_candidates = _method_signature_autocomplete_candidates


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
            "@clipxor", "@save_component", "@save-component",
        }
        valid_values = {
            "@m": {m["key"].lower() for m in self.merge_modes},
            "@mode": {m["key"].lower() for m in self.merge_modes},
            "@p": {"half", "fp16", "float16", "f16", "bhalf", "bf16", "bfloat16", "quarter", "fp8", "float8", "int8", "i8", "fp32", "float32", "full", "single"},
            "@precision": {"half", "fp16", "float16", "f16", "bhalf", "bf16", "bfloat16", "quarter", "fp8", "float8", "int8", "i8", "fp32", "float32", "full", "single"},
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


    def _entry_token_bounds(self, widget: Entry):
        try:
            text = widget.get()
            pos = int(widget.index("insert"))
        except Exception:
            return "", 0, 0
        left = re.split(r"[\s,]", text[:pos])[-1]
        start = pos - len(left)
        return left, start, pos
