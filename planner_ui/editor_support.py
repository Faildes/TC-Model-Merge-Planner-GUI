"""Editor support responsibilities."""
from __future__ import annotations

from . import dialogs as filedialog
from .constants import BASE_MODEL_OPTIONS
from .constants import INIT_CONFIG
from .constants import LINE_TYPES
from .constants import SDXL_BLOCKS
from .constants import TCCM_PROJECT_EXTENSION
from .constants import TCCM_PROJECT_FORMAT
from .constants import _ARCH_DISPLAY_ALIASES
from .constants import _LORA_MERGE_CLI_SPECS
from .constants import _LORA_MERGE_DEFAULTS
from .constants import _PRECISION_DISPLAY_ALIASES
from .constants import _SIGLESS_SEED_MODES
from pathlib import Path
from plan import default_ratio
from plan import load_plan_records
from plan import make_entry
from plan import normalize_plan
from tkinter import Entry
from tkinter import Frame
from tkinter import Label
from tkinter import LabelFrame
from tkinter import Text
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk
from typing import Any
from typing import Dict
import copy
import json
import re
import tkinter as tk


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


def _planner_visible_line_types_with_lora_merge(self):
    from .tools_support import _planner_visible_line_types as _lm_orig_visible_line_types

    vals = list(_lm_orig_visible_line_types(self) if _lm_orig_visible_line_types else LINE_TYPES)
    if "LoRA Merge" not in vals:
        vals.append("LoRA Merge")
    return vals


def _entry_type_color_with_lora_merge(self, etype: str) -> str:
    from .tools_support import _entry_type_color as _lm_orig_entry_type_color

    if str(etype or "") == "LoRA Merge":
        return "#0ea5e9"
    return _lm_orig_entry_type_color(self, etype) if _lm_orig_entry_type_color else "#333333"


def _line_summary_with_lora_merge(self, entry: Dict[str, Any]) -> str:
    from .tools_support import _line_summary as _lm_orig_line_summary

    if entry.get("type") == "LoRA Merge":
        n = len(entry.get("loras", []) or [])
        base = f"LoRA Merge - {entry.get('output_name') or '(unset)'} ({n} LoRA{'s' if n != 1 else ''})"
        return base + "  ?"
    return _lm_orig_line_summary(self, entry) if _lm_orig_line_summary else str(entry.get("type") or "")


def _planner_convert_entry_type_preserving_common_fields_with_lora_merge(self, entry: Dict[str, Any], new_type: str) -> Dict[str, Any]:
    from .tools_support import _planner_convert_entry_type_preserving_common_fields as _lm_orig_convert_entry

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


def _planner_apply_entry_defaults_with_lora_merge(self):
    from .tools_support import _planner_apply_entry_defaults as _lm_orig_apply_defaults

    if _lm_orig_apply_defaults:
        _lm_orig_apply_defaults(self)
    for entry in self.plan_data.get("entries", []) or []:
        if isinstance(entry, dict) and entry.get("type") in _LORA_MERGE_DEFAULTS:
            _lm_ensure_cli_options(entry)
            if entry.get("type") == "LoRA Merge":
                entry.setdefault("loras", [])


def _sigless_ensure_cli_options(entry: Dict[str, Any]) -> Dict[str, Any]:
    etype = str(entry.get("type") or "")
    defaults = copy.deepcopy(_LORA_MERGE_DEFAULTS.get(etype, {})) if True else {}
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
    from .editor_support import _sigless_ensure_cli_options_display as _sigless_ensure_cli_options

    key = str(mode_info.get("key") or entry.get("merge_mode") or "").upper()
    opts = _sigless_ensure_cli_options(entry)
    if bool(opts.get("turbo")) or bool(opts.get("deturbo")):
        return 3
    if key in {"NOIN", "RM", "COMP"}:
        return 2 if bool(opts.get("clipxor")) else 1
    if mode_info.get("needs_m2"):
        return 3
    return 2


def _sigless_option_changed(self, entry: Dict[str, Any], key: str, value, default):
    from .editor_support import _sigless_ensure_cli_options_display as _sigless_ensure_cli_options

    opts = _sigless_ensure_cli_options(entry)
    if isinstance(default, bool):
        opts[key] = bool(value)
    else:
        opts[key] = str(value or "")
    self._after_entry_change()


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


def _build_ratio_value_widget_sigless(self, parent, ratio: Dict[str, Any], allow_block_weight: bool):
    from .editor_support import _build_ratio_value_widget as _sigless_orig_build_ratio_value_widget

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


def _render_lora_bake_entry_sigless(self, parent, entry: Dict[str, Any]):
    from .help_support import _sigless_build_cli_options_panel_dropdown_refined as _sigless_build_cli_options_panel
    from .editor_support import _sigless_ensure_cli_options_display as _sigless_ensure_cli_options

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
    from .help_support import _sigless_build_cli_options_panel_dropdown_refined as _sigless_build_cli_options_panel
    from .editor_support import _sigless_ensure_cli_options_display as _sigless_ensure_cli_options

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


def _on_plan_list_motion_sigless(self, event):
    from .tools_support import _on_plan_list_motion as _sigless_orig_on_plan_motion

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


def _planner_display_precision_name(value):
    raw = str(value or "").strip()
    key = raw.lower().replace(" ", "")
    return _PRECISION_DISPLAY_ALIASES.get(key, raw.upper() if raw else "")


def _planner_display_arch_name(value):
    raw = str(value or "").strip()
    key = raw.lower().replace(" ", "").replace("_", "-")
    return _ARCH_DISPLAY_ALIASES.get(key, raw[:1].upper() + raw[1:] if raw else "")


def _sigless_ensure_cli_options_display(entry: Dict[str, Any]) -> Dict[str, Any]:
    from .editor_support import _sigless_ensure_cli_options as _old_sigless_ensure_cli_options

    opts = _old_sigless_ensure_cli_options(entry)
    if entry.get("type") == "LoRA Merge" and "merge_arch" in opts:
        opts["merge_arch"] = _planner_display_arch_name(opts.get("merge_arch") or "Auto")
        entry["cli_options"] = opts
    if entry.get("architecture"):
        entry["architecture"] = _planner_display_arch_name(entry.get("architecture"))
    if entry.get("precision"):
        entry["precision"] = _planner_display_precision_name(entry.get("precision"))
    return opts


def _sigless_option_changed_display(self, entry: Dict[str, Any], key: str, value, default):
    from .editor_support import _sigless_option_changed as _old_sigless_option_changed

    if key == "merge_arch":
        value = _planner_display_arch_name(value)
    return _old_sigless_option_changed(self, entry, key, value, default)


def _build_left_panel_refined_dev_buttons(self, parent):
    from .editor_support import _build_left_panel as _dev_base_build_left_panel

    self._planner_dev_ensure_config()
    base_builder = _dev_base_build_left_panel
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


def _safe_migration_tool(self):
    from .tools_support import _planner_read_json_or_none
    from .tools_support import _tccm_timestamp

    src = filedialog.askopenfilename(title="Select legacy planner text / JSON", filetypes=[("Planner files", "*.txt *.json *.tccm *.tccm.json"), ("All files", "*.*")])
    src = self._normalize_user_path(src)
    if not src:
        return
    default = Path(src).stem
    if default.endswith(".tccm"):
        default = default[:-5]
    dst = filedialog.asksaveasfilename(title="Save migrated .tccm.json", defaultextension=TCCM_PROJECT_EXTENSION, initialfile=f"{default}{TCCM_PROJECT_EXTENSION}", initialdir=str(Path(src).parent), filetypes=[("TCCM Project", "*.tccm.json"), ("All files", "*.*")])
    dst = self._normalize_user_path(dst)
    if not dst:
        return
    if not dst.lower().endswith(TCCM_PROJECT_EXTENSION):
        dst = str(Path(dst).with_suffix("")) + TCCM_PROJECT_EXTENSION
    old_plan = copy.deepcopy(getattr(self, "plan_data", {}))
    old_index = int(getattr(self, "current_index", 0) or 0)
    old_variants = copy.deepcopy(getattr(self, "plan_variants", []))
    old_profiles = copy.deepcopy(getattr(self, "run_profiles", []))
    try:
        payload = _planner_read_json_or_none(src) if True else None
        is_project = isinstance(payload, dict) and payload.get("format") == TCCM_PROJECT_FORMAT
        if is_project:
            self._planner_restore_project_payload(payload, source_path=src)
        else:
            raw = normalize_plan(load_plan_records(src))
            collapse = getattr(self, "_collapse_internal_plan_entries", None)
            self.plan_data = collapse(raw) if callable(collapse) else raw
            sidecar = Path(src + ".planner_meta.json")
            if sidecar.exists():
                try:
                    self._planner_apply_loaded_meta(json.loads(sidecar.read_text(encoding="utf-8")))
                except Exception:
                    pass
            if hasattr(self, "_planner_apply_entry_defaults"):
                self._planner_apply_entry_defaults()
            self.current_index = 0
            self.plan_variants = []
            self.run_profiles = []
        out_payload = self._planner_project_json_payload()
        out_payload["migrated_from"] = src
        out_payload["migrated_at"] = _tccm_timestamp()
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        load_now = messagebox.askyesno("Migration Complete", f"Migrated to:\n{dst}\n\nLoad the migrated project now?")
        if load_now:
            self.entries["filepath"].delete(0, tk.END)
            self.entries["filepath"].insert(0, dst)
            self._planner_restore_project_payload(out_payload, source_path=dst)
            self._refresh_line_selector()
            self._render_current_line()
            self._planner_save_project_json(dst, reason="migration-load")
        else:
            self.plan_data = old_plan
            self.current_index = old_index
            self.plan_variants = old_variants
            self.run_profiles = old_profiles
            self._refresh_line_selector()
            self._render_current_line()
        self.status_label.config(text=f"Migrated project: {self._path_display_name(dst)}")
    except Exception as exc:
        self.plan_data = old_plan
        self.current_index = old_index
        self.plan_variants = old_variants
        self.run_profiles = old_profiles
        self._show_detailed_error("Safe Migration Error", exc)
