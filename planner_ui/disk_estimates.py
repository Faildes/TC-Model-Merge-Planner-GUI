"""Disk estimates responsibilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
import os


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
    from .editor_support import _planner_display_arch_name

    entries = self.plan_data.get("entries", []) if isinstance(getattr(self, "plan_data", None), dict) else []

    # FP16 / pruned checkpoint-size heuristics requested for peak estimation.
    checkpoint_size_by_base = {
        "SD1.5": 1.99,
        "SDXL": 6.46,
        "Flux": 22.17,
        "ZImage": 11.46,
        "Anima": 3.90,
        "Krea2": 25.0,
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
            if callable(_planner_display_arch_name):
                shown = _planner_display_arch_name(raw)
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
            "k2": "Krea2",
            "krea2": "Krea2",
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
