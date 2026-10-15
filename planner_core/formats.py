"""Formats responsibilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
import json
import os
import shlex


def parse_legacy_text_plan(text: str) -> Dict[str, Any]:
    from .schema import _infer_ratio_mode
    from .syntax import _needs_quote
    from .syntax import _parse_lora_pairs
    from .syntax import _parse_tail_at
    from .syntax import _split
    from .schema import make_entry
    from .schema import normalize_plan
    from .syntax import quoter

    entries: List[Dict[str, Any]] = []
    temp = lambda x: f"TEMP{x}" if x and x[0]=="_" else x
    for raw_line in text.splitlines():
        t = raw_line.strip()
        if not t or t.startswith("//") or t.startswith("#"):
            continue

        if t.startswith("+"):
            body = t[1:]
            parts = [p.strip() for p in body.split(",")]
            entry = make_entry("Download Model")
            entry["model_name"] = temp(parts[0] if parts else "")
            entry["link"] = parts[1] if len(parts) > 1 else ""
            if "%LR" in t:
                entry["model_type"] = "LoRA"
            entries.append(entry)
            continue

        if t.upper().startswith("LC"):
            parts = (t.split(",", 3) + ["", "", "", ""])[:4]
            _, path, model_type, alias = parts
            entry = make_entry("Local Model")
            entry["local_path"] = path.strip()
            entry["model_name"] = alias.strip() or os.path.splitext(os.path.basename(path.strip()))[0]
            entry["model_type"] = model_type.strip() or "Checkpoint"
            entries.append(entry)
            continue

        if t.upper().startswith("-"):
            entry = make_entry("Remove Model")
            entry["model"] = temp(t[1:].strip())
            entries.append(entry)
            continue

        if t.startswith("LM"):
            toks = _split(t)
            if len(toks) < 3:
                continue
            cut = len(toks)
            for i, tk in enumerate(toks):
                if tk.startswith("@"):
                    cut = i
                    break
            core, at = toks[:cut], _parse_tail_at(toks[cut:])
            tail_opts = []
            precision = "half"
            if at["precision"] is not None:
                precision = ("bhalf" if at["precision"].lower() in ("bhalf","bf16","bfloat16") else ("quarter" if at["precision"].lower() in ("quarter","fp8","float8") else ("int8" if at["precision"].lower() in ("int8","i8") else ("fp32" if at["precision"].lower() in ("fp32","float32","full") else "half"))))
            for d in at["extras"]:
                if d.startswith("--"):
                    tail_opts.append(d)
            tail_str = "" if not tail_opts else " ".join(tail_opts)
            entry = make_entry("LoRA Merge")
            entry["output_name"] = temp(core[-1])
            entry["loras"] = []
            entry["precision"] = precision
            entry["additional_signatures"] = tail_str
            entry["raw_signatures"] = " ".join(toks[cut:])
            for name, ratio in _parse_lora_pairs(" ".join(core[1:-1]).strip()):
                ratio_mode = "Elemental" if any(ch in ratio for ch in "[]{}") or "\n" in ratio else "Single"
                entry["loras"].append({"name": temp(name), "ratio": {"mode": ratio_mode, "value": ratio}})
            entries.append(entry)
            continue

        if t.startswith("CM"):
            # print(t[2:].strip())
            toks = _split(t[2:].strip())
            if len(toks) < 3:
                continue
            cut = len(toks)
            for i, tk in enumerate(toks):
                if tk.startswith("@") and tk.lower() not in ("@r", "@rand"):
                    cut = i
                    break
            core, at = toks[:cut], _parse_tail_at(toks[cut:])
            op1 = core[1].upper()
            tail_opts = []
            precision = "half"
            if at["cosine"] is not None: tail_opts.append(f"--cosine{at['cosine']}")
            if at["fine"]: tail_opts.append(f'--fine={"\""+at["fine"]+"\"" if _needs_quote(at["fine"]) else at["fine"]}')
            if at["seed"] is not None: tail_opts.append(f"--seed {at['seed']}")
            if at["precision"] is not None:
                precision = ("bhalf" if at["precision"].lower() in ("bhalf","bf16","bfloat16") else ("quarter" if at["precision"].lower() in ("quarter","fp8","float8") else ("int8" if at["precision"].lower() in ("int8","i8") else ("fp32" if at["precision"].lower() in ("fp32","float32","full") else "half"))))
            for d in at["extras"]:
                if d.startswith("--"): tail_opts.append(d)
            tail_str = "" if not tail_opts else " ".join(tail_opts)
            entry = make_entry("Checkpoint Merge")
            entry["merge_mode"] = at.get("mode") or "WS"
            entry["model0"] = temp(core[0])
            entry["precision"] = precision
            entry["additional_signatures"] = tail_str
            entry["raw_signatures"] = " ".join(toks[cut:])

            if op1 == "+":
                entry["model1"] = temp(core[2])
                if len(core) >= 8 and core[3].upper() in ("+T", "+S"):
                    entry["merge_mode"] = at.get("mode") or ("TRS" if core[3].upper() == "+T" else "ST")
                    entry["model2"] = temp(core[4])
                    r_a = core[5].lower() in ("@r", "@rand")
                    a_idx = 6 if r_a else 5
                    r_b = core[a_idx + 1].lower() in ("@r", "@rand")
                    b_idx = a_idx + 2 if r_b else a_idx + 1
                    entry["alpha"] = {"mode": _infer_ratio_mode(core[a_idx], allow_block_weight=True, randomized=r_a), "value": quoter(core[a_idx])}
                    entry["beta"] = {"mode": _infer_ratio_mode(core[b_idx], allow_block_weight=True, randomized=r_b), "value": quoter(core[b_idx])}
                    entry["output_name"] = temp(core[b_idx + 1])
                elif len(core) >= 7 and core[3] == "-":
                    entry["merge_mode"] = at.get("mode") or "AD"
                    entry["model2"] = temp(core[4])
                    r_a = core[5].lower() in ("@r", "@rand")
                    a_idx = 6 if r_a else 5
                    entry["alpha"] = {"mode": _infer_ratio_mode(core[a_idx], allow_block_weight=True, randomized=r_a), "value": quoter(core[a_idx])}
                    entry["output_name"] = temp(core[a_idx + 1])
                else:
                    entry["merge_mode"] = at.get("mode") or "WS"
                    r_a = core[3].lower() in ("@r", "@rand")
                    a_idx = 4 if r_a else 3
                    entry["alpha"] = {"mode": _infer_ratio_mode(core[a_idx], allow_block_weight=True, randomized=r_a), "value": quoter(core[a_idx])}
                    entry["output_name"] = temp(core[a_idx + 1])
            elif op1 == "+D":
                entry["merge_mode"] = at.get("mode") or "DARE"
                entry["model1"] = temp(core[2])
                r_a = core[3].lower() in ("@r", "@rand")
                a_idx = 4 if r_a else 3
                r_b = core[a_idx + 1].lower() in ("@r", "@rand")
                b_idx = a_idx + 2 if r_b else a_idx + 1
                entry["alpha"] = {"mode": _infer_ratio_mode(core[a_idx], allow_block_weight=True, randomized=r_a), "value": quoter(core[a_idx])}
                entry["beta"] = {"mode": _infer_ratio_mode(core[b_idx], allow_block_weight=True, randomized=r_b), "value": quoter(core[b_idx])}
                entry["output_name"] = temp(core[b_idx + 1])
            elif op1 == "#S":
                entry["merge_mode"] = at.get("mode") or "SWAP"
                entry["model1"] = temp(core[2])
                r_a = core[3].lower() in ("@r", "@rand")
                a_idx = 4 if r_a else 3
                entry["alpha"] = {"mode": _infer_ratio_mode(core[a_idx], allow_block_weight=True, randomized=r_a), "value": quoter(core[a_idx])}
                entry["output_name"] = temp(core[a_idx + 1])
            elif op1 == "#X":
                entry["merge_mode"] = at.get("mode") or "CLIPXOR"
                entry["model1"] = temp(core[2])
                entry["output_name"] = temp(core[3])
            elif op1 == "#T":
                entry["merge_mode"] = at.get("mode") or "TF"
                entry["model1"] = temp(core[2])
                entry["output_name"] = temp(core[3])
            elif op1 == "+F":
                entry["merge_mode"] = at.get("mode") or "FWM"
                entry["model1"] = temp(core[2])
                r_a = core[3].lower() in ("@r", "@rand")
                a_idx = 4 if r_a else 3
                entry["alpha"] = {"mode": _infer_ratio_mode(core[a_idx], allow_block_weight=True, randomized=r_a), "value": quoter(core[a_idx])}
                entry["output_name"] = temp(core[a_idx + 1])
            entries.append(entry)
            continue

        if t.startswith("LB"):
            toks = _split(t)
            if len(toks) < 4:
                continue
            cut = len(toks)
            for i, tk in enumerate(toks):
                if tk.startswith("@"):
                    cut = i; break
            core, at = toks[:cut], _parse_tail_at(toks[cut:])
            tail_opts = []
            precision = "half"
            if at["precision"] is not None:
                precision = ("bhalf" if at["precision"].lower() in ("bhalf","bf16","bfloat16") else ("quarter" if at["precision"].lower() in ("quarter","fp8","float8") else ("int8" if at["precision"].lower() in ("int8","i8") else ("fp32" if at["precision"].lower() in ("fp32","float32","full") else "half"))))
            for d in at["extras"]:
                if d.startswith("--"): tail_opts.append(d)
            tail_str = "" if not tail_opts else " ".join(tail_opts)
            entry = make_entry("LoRA Bake")
            entry["checkpoint"] = temp(core[1])
            entry["output_name"] = temp(core[-1])
            entry["loras"] = []
            entry["precision"] = precision
            entry["additional_signatures"] = tail_str
            entry["raw_signatures"] = " ".join(toks[cut:])
            for name, ratio in _parse_lora_pairs(" ".join(core[2:-1]).strip()):
                ratio_mode = "Elemental" if any(ch in ratio for ch in "[]{}") or "\n" in ratio else "Single"
                entry["loras"].append({"name": temp(name), "ratio": {"mode": ratio_mode, "value": ratio}})
            entries.append(entry)
            continue

    return normalize_plan({"version": 2, "format": "legacy-import", "entries": entries})


def load_plan_records(filepath: str) -> Dict[str, Any]:
    from .schema import default_plan
    from .schema import normalize_plan

    path = Path(filepath)
    if not path.exists():
        return default_plan()
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        return normalize_plan(json.loads(raw))
    return parse_legacy_text_plan(raw)


def save_plan_records(filepath: str, plan: Dict[str, Any]) -> None:
    from .schema import normalize_plan

    path = Path(filepath)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_plan(plan)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_record_to_legacy_line(entry: Dict[str, Any]) -> str:
    from .options import _legacy_signature_text_base as _legacy_signature_text
    from .schema import _ratio_text_base as _ratio_text

    mode = (entry.get("merge_mode") or "WS").strip() or "WS"
    temp = lambda x: f"TEMP{x}" if x and x[0]=="_" else x
    m0 = temp((entry.get("model0") or "").strip())
    m1 = temp((entry.get("model1") or "").strip())
    m2 = temp((entry.get("model2") or "").strip())
    a = _ratio_text(entry.get("alpha"))
    b = _ratio_text(entry.get("beta"))
    out = (entry.get("output_name") or "").strip()
    sig = _legacy_signature_text(entry)

    if mode == "WS":
        line = f"CM {m0} + {m1} {a} {out}"
    elif mode == "ST":
        line = f"CM {m0} + {m1} +S {m2} {a} {b} {out}"
    elif mode == "TRS":
        line = f"CM {m0} + {m1} +T {m2} {a} {b} {out}"
    elif mode == "AD":
        line = f"CM {m0} + {m1} - {m2} {a} {out}"
    elif mode == "DARE":
        line = f"CM {m0} +D {m1} {a} {b} {out}"
    elif mode == "SWAP":
        line = f"CM {m0} #S {m1} {a} {out}"
    elif mode == "CLIPXOR":
        line = f"CM {m0} #X {m1} {out}"
    elif mode == "TF":
        line = f"CM {m0} #T {m1} {out}"
    elif mode == "FWM":
        line = f"CM {m0} +F {m1} {a} {out}"
    else:
        if m2 and b:
            line = f"CM {m0} + {m1} +S {m2} {a} {b} {out} @mode {mode}"
        elif m2:
            line = f"CM {m0} + {m1} - {m2} {a} {out} @mode {mode}"
        elif b:
            line = f"CM {m0} +D {m1} {a} {b} {out} @mode {mode}"
        else:
            line = f"CM {m0} + {m1} {a} {out} @mode {mode}"

    if sig:
        line += f" {sig}"
    return line.strip()




def export_plan_records_txt_base(filepath: str, plan: Dict[str, Any]) -> None:
    from .options import _legacy_signature_text_base as _legacy_signature_text
    from .schema import _normalize_ratio_spec
    from .schema import normalize_plan

    path = Path(filepath)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_plan(plan)
    lines: List[str] = []
    for entry in normalized.get("entries", []):
        etype = entry.get("type")
        if etype == "Download Model":
            name = (entry.get("model_name") or "").strip()
            link = (entry.get("link") or "").strip()
            model_type = (entry.get("model_type") or "Checkpoint").strip()
            if not name and not link:
                continue
            line = f"+{name}"
            if link:
                line += f", {link}"
            if model_type in ("LoRA", "LyCORIS"):
                line += ", %LR"
            lines.append(line)
        elif etype == "Local Model":
            local_path = (entry.get("local_path") or "").strip()
            model_type = (entry.get("model_type") or "Checkpoint").strip()
            model_name = (entry.get("model_name") or "").strip()
            if local_path:
                stem = os.path.splitext(os.path.basename(local_path))[0]
                suffix = f", {model_name}" if model_name and model_name != stem else ""
                lines.append(f"LC, {local_path}, {model_type}{suffix}")
        elif etype == "Remove Model":
            model = (entry.get("model") or "").strip()
            if model:
                lines.append(f"-{model}")
        elif etype == "Checkpoint Merge":
            if (entry.get("model0") or "").strip() and (entry.get("model1") or "").strip() and (entry.get("output_name") or "").strip():
                lines.append(_merge_record_to_legacy_line(entry))
        elif etype == "LoRA Bake":
            checkpoint = (entry.get("checkpoint") or "").strip()
            output_name = (entry.get("output_name") or "").strip()
            loras = []
            for lora in entry.get("loras", []) or []:
                name = (lora.get("name") or "").strip()
                if not name:
                    continue
                ratio = _normalize_ratio_spec(lora.get("ratio"), allow_block_weight=False, default_single="1.0")["value"] or "1.0"
                loras.append(f"{name}:{ratio}")
            if checkpoint and output_name and loras:
                line = f"LB {checkpoint} {','.join(loras)} {output_name}"
                sig = _legacy_signature_text(entry)
                if sig:
                    line += f" {sig}"
                lines.append(line)
        elif etype == "LoRA Merge":
            output_name = (entry.get("output_name") or "").strip()
            loras = []
            for lora in entry.get("loras", []) or []:
                name = (lora.get("name") or "").strip()
                if not name:
                    continue
                ratio = _normalize_ratio_spec(lora.get("ratio"), allow_block_weight=False, default_single="1.0")["value"] or "1.0"
                loras.append(f"{name}:{ratio}")
            if output_name and loras:
                line = f"LM {','.join(loras)} {output_name}"
                sig = _legacy_signature_text(entry)
                if sig:
                    line += f" {sig}"
                lines.append(line)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")




def _legacy_sig_quote(value: Any) -> str:
    text = str(value)
    return shlex.quote(text)
