from __future__ import annotations

import json
import errno
import os
import random
import shlex
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Tuple


class PlanCompileError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        entry_index: int | None = None,
        entry_type: str | None = None,
        entry: Dict[str, Any] | None = None,
        entry_id: str | None = None,
        entry_payload: Any = None,
        cause: Exception | None = None,
        source_lines: List[str] | None = None,
    ):
        self.entry_index = entry_index
        self.entry_type = entry_type
        self.entry_id = entry_id
        self.entry = entry_payload if entry_payload is not None else (entry or {})
        self.entry_payload = self.entry
        self.cause = cause
        self.source_lines = list(source_lines or [])

        parts = [message]
        if entry_index is not None:
            parts.append(f"entry_index={entry_index}")
        if entry_type:
            parts.append(f"entry_type={entry_type}")
        if entry_id:
            parts.append(f"entry_id={entry_id}")
        if cause is not None:
            parts.append(f"cause={type(cause).__name__}: {cause}")
        super().__init__(" | ".join(parts))


hexchars = "0123456789abcdef"
rnm = lambda n: ''.join(random.choices(hexchars, k=n))
_uid = lambda: f"{rnm(8)}-{rnm(4)}-{rnm(4)}-{rnm(4)}-{rnm(12)}"

SDXL_BLOCKS = [
    "BASE", "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08",
    "MID00", "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08",
]


def _planner_dir() -> Path:
    return Path(__file__).resolve().parent


def _toolpath_candidates(base: Path | None = None) -> List[Path]:
    root = base or _planner_dir()
    return [
        root / "tools" / "chattiori_model_merge",
        root / "tools" / "chattiori_model_merger",
    ]


def _preferred_toolpath() -> str:
    candidates = _toolpath_candidates()
    for path in candidates:
        if (path / "merge.py").exists() or (path / "lora_bake.py").exists() or path.exists():
            return str(path)
    return str(candidates[0])


def _notebook_runtime_workpath(workpath: str) -> str:
    """Return the runtime root that should appear in generated scripts/notebooks."""
    raw = str(workpath or "").strip() or "."
    raw = raw.rstrip("/\\") or raw
    if os.path.normpath(raw) == os.path.normpath("/kaggle"):
        return "/kaggle/working"
    return raw


# ----------------------------
# Generic helpers
# ----------------------------
def _ensure_dirs(root: str, subdirs: List[str]):
    for d in [root, *[os.path.join(root, x) for x in subdirs]]:
        os.makedirs(d, exist_ok=True)


def _nb_json(cells: List[str]) -> str:
    def wrap(src: str) -> str:
        return json.dumps({
            "cell_type": "code",
            "execution_count": None,
            "id": _uid(),
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.splitlines()],
        })

    return '{{"cells":[{cells}],"metadata":{{"kernelspec":{{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"}},"language_info":{{"name":"python","version":"3.10.6"}}}},"nbformat":4,"nbformat_minor":5}}'.format(
        cells=",".join(map(wrap, cells))
    )


def _split(s: str):
    return shlex.split(s, posix=True)


def _split_top_level(s: str, sep: str = ",") -> List[str]:
    opens = {"(": ")", "[": "]", "{": "}"}
    stack: List[str] = []
    out: List[str] = []
    buf: List[str] = []
    for ch in s:
        if ch in opens:
            stack.append(opens[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif ch == sep and not stack:
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
            continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        out.append(part)
    return out


def _ensure_st(val: str) -> str:
    v = val.strip()
    return v if v.lower().endswith(".safetensors") else f"{v}.safetensors"


def _parse_lora_pairs(raw: str):
    items = _split_top_level(raw.strip(), ",")
    out = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        if ":" in it:
            name, ratio = it.split(":", 1)
            out.append((name.strip(), ratio.strip()))
        else:
            out.append((it.strip(), "1.0"))
    return out


def _needs_quote(val: str) -> bool:
    try:
        float(val)
        return False
    except:
        return True


def quoter(val: str) -> str:
    return f'"{val}"' if _needs_quote(val) else val


def _parse_tail_at(tokens):
    out = {
        "cosine": None,
        "fine": None,
        "seed": None,
        "mode": None,
        "precision": None,
        "rank": None,
        "arch": None,
        "extras": [],
    }
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("@"):
            out["extras"].append(t)
            i += 1
            continue
        k, v = t[1:].lower(), None
        if "=" in k:
            k, v = k.split("=", 1)
            v = v.strip('"').strip("'")
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("@"):
            v = tokens[i + 1].strip('"').strip("'")
            i += 1

        if k in ("cosine0", "cosine1", "cosine2", "c0", "c1", "c2"):
            out["cosine"] = int(k[-1])
        elif k in ("cosine", "c") and v is not None:
            out["cosine"] = int(v)
        elif k in ("fine", "f") and v is not None:
            out["fine"] = v
        elif k in ("s", "seed") and v is not None:
            out["seed"] = int(v)
        elif k in ("m", "mode") and v is not None:
            out["mode"] = v.upper()
        elif k in ("p", "precision") and v is not None:
            out["precision"] = v
        elif k in ("rank", "rk", "rnk") and v is not None:
            out["rank"] = int(v)
        elif k in ("arch", "a") and v is not None:
            out["arch"] = v.lower()
        elif t.startswith("@"):
            if k in ("bake_fp32","b32"):
                out["extras"].append("--bake_fp32")
            else:
                out["extras"].append(f"--{k} {v}")
        else:
            out["extras"].append(tokens[i])
        i += 1
    return out


def _infer_ratio_mode(value: str, allow_block_weight: bool = True, randomized: bool = False) -> str:
    s = str(value or "").strip()
    if randomized:
        return "Randomize"
    if not s:
        return "Single"
    if ":" in s:
        return "Elemental"
    if allow_block_weight and "," in s:
        return "Block weight"
    return "Single"


def default_ratio(mode: str = "Single") -> Dict[str, Any]:
    return {
        "mode": mode,
        "value": "0.5" if mode == "Single" else ("0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" if mode == "Block weight" else ""),
    }


def _normalize_ratio_spec(spec: Any, *, allow_block_weight: bool = True, default_single: str = "0.5") -> Dict[str, Any]:
    if isinstance(spec, dict):
        mode = str(spec.get("mode") or _infer_ratio_mode(spec.get("value", ""), allow_block_weight=allow_block_weight))
        value = str(spec.get("value", "")).strip()
    else:
        value = str(spec or "").strip()
        mode = _infer_ratio_mode(value, allow_block_weight=allow_block_weight)
    if mode == "Single" and not value:
        value = default_single
    if mode == "Block weight" and not value:
        value = default_ratio("Block weight")["value"]
    return {"mode": mode, "value": value}


def make_entry(entry_type: str = "Checkpoint Merge") -> Dict[str, Any]:
    base = {"id": _uid(), "type": entry_type}
    if entry_type == "Download Model":
        base.update({"model_name": "", "link": "", "model_type": "Checkpoint"})
    elif entry_type == "Local Model":
        base.update({"local_path": "", "model_name": "", "model_type": "Checkpoint"})
    elif entry_type == "Remove Model":
        base.update({"model": ""})
    elif entry_type == "Checkpoint Merge":
        base.update({
            "merge_mode": "WS",
            "model0": "",
            "model1": "",
            "model2": "",
            "alpha": default_ratio("Single"),
            "beta": default_ratio("Single"),
            "output_name": "",
            "precision": "",
            "additional_signatures": "",
            "raw_signatures": "",
        })
    elif entry_type == "LoRA Bake":
        base.update({
            "checkpoint": "",
            "loras": [],
            "output_name": "",
            "precision": "",
            "additional_signatures": "",
            "raw_signatures": "",
        })
    return base


def default_plan() -> Dict[str, Any]:
    return {"version": 2, "format": "planner-json", "final_memo": "", "history": [], "entries": [make_entry("Checkpoint Merge")]}


# ----------------------------
# Structured plan I/O
# ----------------------------
def normalize_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    plan = default_plan()
    if isinstance(data, dict):
        plan["version"] = data.get("version", 2)
        plan["format"] = data.get("format", "planner-json")
        plan["final_memo"] = str(data.get("final_memo", plan.get("final_memo", "")) or "")
        hist = data.get("history", plan.get("history", []))
        plan["history"] = hist if isinstance(hist, list) else []
        meta = data.get("meta", {})
        if isinstance(meta, dict):
            plan["meta"] = meta
        plan["entries"] = []
        for raw in data.get("entries", []):
            entry = make_entry(raw.get("type", "Checkpoint Merge"))
            entry.update(raw)
            entry.setdefault("id", _uid())
            if entry["type"] == "Checkpoint Merge":
                entry["alpha"] = _normalize_ratio_spec(entry.get("alpha"), allow_block_weight=True, default_single="0.5")
                entry["beta"] = _normalize_ratio_spec(entry.get("beta"), allow_block_weight=True, default_single="0.5")
            if entry["type"] in ("LoRA Bake", "LoRA Merge"):
                entry.setdefault("loras", [])
                normalized_loras = []
                for lora in entry.get("loras", []):
                    normalized_loras.append({
                        "name": lora.get("name", ""),
                        "ratio": _normalize_ratio_spec(lora.get("ratio"), allow_block_weight=False, default_single="1.0"),
                    })
                entry["loras"] = normalized_loras
            plan["entries"].append(entry)
    if not plan["entries"]:
        plan["entries"] = [make_entry("Checkpoint Merge")]
    return plan


def parse_legacy_text_plan(text: str) -> Dict[str, Any]:
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
                precision = "bhalf" if at["precision"].lower() in ("bhalf","bf16","bfloat16") else ("quarter" if at["precision"].lower() in ("quarter","fp8","float8") else "half")
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
                precision = "bhalf" if at["precision"].lower() in ("bhalf","bf16","bfloat16") else ("quarter" if at["precision"].lower() in ("quarter","fp8","float8") else "half")
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
                precision = "bhalf" if at["precision"].lower() in ("bhalf","bf16","bfloat16") else ("quarter" if at["precision"].lower() in ("quarter","fp8","float8") else "half")
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
    path = Path(filepath)
    if not path.exists():
        return default_plan()
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        return normalize_plan(json.loads(raw))
    return parse_legacy_text_plan(raw)


def save_plan_records(filepath: str, plan: Dict[str, Any]) -> None:
    path = Path(filepath)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_plan(plan)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")



def _ratio_text(spec: Dict[str, Any] | None) -> str:
    spec = _normalize_ratio_spec(spec, allow_block_weight=True, default_single="0.5")
    mode = str(spec.get("mode", "Single"))
    value = str(spec.get("value", "")).strip()
    if mode == "Single":
        return value or "0.5"
    value = f'"{value[0].strip("\'\"")}{value[1:-1]}{value[-1].strip("\'\"")}"'
    return value


def _merge_record_to_legacy_line(entry: Dict[str, Any]) -> str:
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


def export_plan_records_txt(filepath: str, plan: Dict[str, Any]) -> None:
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
                # print(lora)
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
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ----------------------------
# Notebook compilation
# ----------------------------
INSTALL_TPL = Template(r'''import os, platform, shutil, subprocess, sys

IGNORE_INSTALL_DEPS = $ignore_install_deps
USE_ONLINE = $use_online
WORKPATH_RAW = r"$workpath"
DIFFUSION = "$diffusion"


def _planner_normalize_workpath(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(path or ".")))
    # Kaggle's /kaggle root is read-only for user-created files. Use /kaggle/working.
    if os.path.normpath(path) == os.path.normpath("/kaggle"):
        return os.path.join(path, "working")
    return path


def _planner_working_dir(path: str) -> str:
    path = _planner_normalize_workpath(path)
    # If the user already chose /kaggle/working, do not append another /working.
    if os.path.basename(os.path.normpath(path)).lower() == "working":
        return path
    return os.path.join(path, "working")


WORKING_DIR = _planner_working_dir(WORKPATH_RAW)
LOCAL_REPO_DIR = r"$toolpath"
REPO_DIR = os.path.join(WORKING_DIR, "tools", "chattiori_model_merger") if USE_ONLINE else LOCAL_REPO_DIR
if USE_ONLINE:
    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
print(f"[install] use_online={USE_ONLINE} working_dir={WORKING_DIR} repo_dir={REPO_DIR}")


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd, *, check=False):
    pretty = " ".join(str(x) for x in cmd)
    print(f"$ {pretty}")
    return subprocess.run(cmd, check=check)


def ensure(*args: str):
    _run([sys.executable, "-m", "pip", "install", *args], check=False)

def install_system_tools():
    system = platform.system()
    print(f"[install] platform={system}")
    if system == "Linux":
        if _cmd_exists("apt-get"):
            _run(["apt-get", "update", "-qq"], check=False)
            _run(["apt-get", "install", "-y", "-qq", "aria2", "git"], check=False)
        elif _cmd_exists("dnf"):
            _run(["dnf", "install", "-y", "aria2", "git"], check=False)
        elif _cmd_exists("yum"):
            _run(["yum", "install", "-y", "aria2", "git"], check=False)
        elif _cmd_exists("apk"):
            _run(["apk", "add", "aria2", "git"], check=False)
        elif _cmd_exists("pacman"):
            _run(["pacman", "-Sy", "--noconfirm", "aria2", "git"], check=False)
        else:
            print("[install] Unsupported Linux package manager. Skipping system package installation.")
    elif system == "Darwin":
        if _cmd_exists("brew"):
            _run(["brew", "install", "aria2", "git"], check=False)
        else:
            print("[install] Homebrew not found. Skipping aria2/git installation on macOS.")
    elif system == "Windows":
        if _cmd_exists("winget"):
            _run(["winget", "install", "-e", "--id", "aria2.aria2", "--accept-package-agreements", "--accept-source-agreements"], check=False)
            _run(["winget", "install", "-e", "--id", "Git.Git", "--accept-package-agreements", "--accept-source-agreements"], check=False)
        elif _cmd_exists("choco"):
            _run(["choco", "install", "-y", "aria2", "git"], check=False)
        else:
            print("[install] winget/choco not found. Skipping aria2/git installation on Windows.")
    else:
        print(f"[install] Unsupported OS for automatic system dependency installation: {system}")

packages = [("torch",),
        ("torchvision",),
        ("lora",),
        ("fake_useragent",),
        ("diffusers",),
        ("torchsde",),
        ("git+https://github.com/huggingface/diffusers",),
        ("git+https://github.com/Faildes/sd_embed_negpip.git",),
        ("git+https://github.com/Faildes/Chattiori_ImageKit",),
        ("-U", "peft"),
        ("torchao", "--extra-index-url", "https://download.pytorch.org/whl/cu121"),]
if DIFFUSION:
    packages.append((DIFFUSION,))

if IGNORE_INSTALL_DEPS:
    print("[install] Ignore Install Deps is enabled. Skipping dependency installation and repo setup.")
else:
    for pkg in packages:
        ensure(*pkg)

    install_system_tools()
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        _run(["git", "-C", REPO_DIR, "fetch", "origin", "notebook"], check=False)
        _run(["git", "-C", REPO_DIR, "checkout", "notebook"], check=False)
        _run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", "notebook"], check=False)
    else:
        _run(["git", "clone", "https://github.com/Faildes/Chattiori-Model-Merger", "-b", "notebook", REPO_DIR], check=False)

    req = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(req):
        _run([sys.executable, "-m", "pip", "install", "-r", req], check=False)
''')

PRELUDE_TPL = Template(r'''# Planner runtime prelude
import gc
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import traceback
import errno
import copy
import time
from pathlib import Path

import filelock
import requests
import torch
from fake_useragent import UserAgent

HFToken = "$hf_token"
CVToken = "$cv_token"
VAE_URL = "$vae_url".strip()
VAE_NAME = "$vae_name".strip() or "VAE"
BAKE_VAE = $bake_vae
USE_ONLINE = $use_online
WORKPATH_RAW = r"$workpath"


def _planner_normalize_workpath(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(path or ".")))
    # Kaggle's /kaggle root is read-only for user-created files. Use /kaggle/working.
    if os.path.normpath(path) == os.path.normpath("/kaggle"):
        return os.path.join(path, "working")
    return path


def _planner_working_dir(path: str) -> str:
    path = _planner_normalize_workpath(path)
    # If the user already chose /kaggle/working, do not append another /working.
    if os.path.basename(os.path.normpath(path)).lower() == "working":
        return path
    return os.path.join(path, "working")


workpath = _planner_normalize_workpath(WORKPATH_RAW)
WORKING_DIR = _planner_working_dir(workpath)
_md = r"$model_dir"
_vd = r"$vae_dir"

models_dir = _md if _md else f"{workpath}/tmp/models"
vae_dir = _vd if _vd else f"{workpath}/tmp/vae"
emb_dir = f"{workpath}/tmp/embeddings"
local_merge_repo_dir = r"$toolpath"
merge_repo_dir = os.path.join(WORKING_DIR, "tools", "chattiori_model_merger") if USE_ONLINE else local_merge_repo_dir
MERGE_PY = os.path.join(merge_repo_dir, "merge.py")
LORA_BAKE_PY = os.path.join(merge_repo_dir, "lora_bake.py")
print(f"[planner] use_online={USE_ONLINE} merge_repo_dir={merge_repo_dir}")
for p in (f"{workpath}/tmp", models_dir, vae_dir, emb_dir):
    os.makedirs(p, exist_ok=True)

MODEL_REGISTRY = {}
MODEL_REGISTRY_HISTORY = []
MODEL_INSTALL_COUNTER = 0
REMOVED_MODELS = set()
SOURCE_MODEL_CACHE = {}
PLANNER_REGISTRY_DIR = os.path.join(models_dir, "_planner_registry")
PLANNER_CACHE_TTL_HOURS = float(os.environ.get("PLANNER_CACHE_TTL_HOURS", "72") or 72)
PLANNER_LOW_DISK_GB = float(os.environ.get("PLANNER_LOW_DISK_GB", "8") or 8)
_PIP_CACHE_PURGED = False


def _path_is_relative_to(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _is_planner_registry_path(path):
    return bool(path) and _path_is_relative_to(path, PLANNER_REGISTRY_DIR)


def _is_model_dir_path(path):
    return bool(path) and _path_is_relative_to(path, models_dir)


def _safe_disk_remove(path, *, source_kind="registered", reason="cleanup"):
    path = str(path or "")
    if not path or not os.path.lexists(path):
        return False
    source_kind = str(source_kind or "registered")
    owns_path = (
        source_kind in ("generated", "download")
        or _is_planner_registry_path(path)
    )
    # Existing user files and raw local paths outside the planner registry must never be removed.
    if not owns_path:
        return False
    try:
        label = os.path.basename(path.rstrip(os.sep)) or path
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"[cleanup] removed {label} ({source_kind}, {reason})")
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[cleanup] could not remove {path}: {e}")
        return False


def _prune_empty_parent_dirs(path):
    try:
        parent = Path(path).resolve().parent
        root = Path(PLANNER_REGISTRY_DIR).resolve()
        while parent != root and root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    except Exception:
        pass


def _release_record(record, *, reason="cleanup", unregister=True):
    if not record:
        return False
    alias = str(record.get("alias") or "")
    path = str(record.get("path") or "")
    source_kind = str(record.get("source_kind") or "registered")
    removed = _safe_disk_remove(path, source_kind=source_kind, reason=reason)
    if removed:
        _prune_empty_parent_dirs(path)
    if unregister and alias and MODEL_REGISTRY.get(alias) is record:
        MODEL_REGISTRY.pop(alias, None)
    return removed


def _drop_source_cache_path(path):
    if not path:
        return
    try:
        target = str(Path(path).resolve())
    except Exception:
        target = os.path.abspath(str(path))
    for key, cached in list(SOURCE_MODEL_CACHE.items()):
        try:
            cached_resolved = str(Path(cached).resolve())
        except Exception:
            cached_resolved = os.path.abspath(str(cached))
        if cached_resolved == target:
            SOURCE_MODEL_CACHE.pop(key, None)


def _current_active_paths():
    paths = set()
    for info in MODEL_REGISTRY.values():
        path = str(info.get("path") or "")
        if path:
            try:
                paths.add(str(Path(path).resolve()))
            except Exception:
                paths.add(os.path.abspath(path))
    for path in SOURCE_MODEL_CACHE.values():
        if path:
            try:
                paths.add(str(Path(path).resolve()))
            except Exception:
                paths.add(os.path.abspath(path))
    return paths


def _prune_orphan_registry_dirs(max_age_hours=None, *, aggressive=False):
    root = Path(PLANNER_REGISTRY_DIR)
    if not root.exists():
        return 0
    now = time.time()
    max_age = PLANNER_CACHE_TTL_HOURS if max_age_hours is None else float(max_age_hours)
    active_paths = _current_active_paths()
    removed = 0
    for child in sorted(root.iterdir(), key=lambda p: str(p)):
        try:
            resolved = str(child.resolve())
            if any(ap == resolved or ap.startswith(resolved + os.sep) for ap in active_paths):
                continue
            age_hours = (now - child.stat().st_mtime) / 3600.0
            if aggressive or age_hours >= max_age:
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
                removed += 1
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"[cleanup] could not prune {child}: {e}")
    if removed:
        print(f"[cleanup] pruned {removed} stale planner cache item(s)")
    return removed


def _prune_dead_hash_cache():
    global cache_data
    if cache_data is None:
        return
    changed = False
    active_titles = set()
    for info in MODEL_REGISTRY.values():
        alias = str(info.get("alias") or "")
        mode = str(info.get("mode") or "checkpoint")
        if alias:
            active_titles.add(f"{mode}/{alias}")
    for section in ("hashes", "hashes-addnet"):
        data = cache_data.get(section)
        if not isinstance(data, dict):
            continue
        for title in list(data.keys()):
            if title.startswith(("checkpoint/", "lora/")) and active_titles and title not in active_titles:
                data.pop(title, None)
                changed = True
    if changed:
        dump_cache()


def _disk_free_gb(path=None):
    try:
        total, used, free = shutil.disk_usage(path or models_dir)
        return free / (2**30), total / (2**30)
    except Exception:
        total, used, free = shutil.disk_usage("/")
        return free / (2**30), total / (2**30)


def flush(light=True):
    global _PIP_CACHE_PURGED
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    free_gb, total_gb = _disk_free_gb()
    if not light:
        _prune_orphan_registry_dirs(aggressive=free_gb < PLANNER_LOW_DISK_GB)
    if (not light or free_gb < PLANNER_LOW_DISK_GB) and not _PIP_CACHE_PURGED:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
        _PIP_CACHE_PURGED = True
    if free_gb < PLANNER_LOW_DISK_GB:
        print(f"[cleanup] low disk: {free_gb:.2f}GB/{total_gb:.2f}GB free")


def release_registered_model(name, *, reason="last-use"):
    alias = str(name or "").strip()
    if not alias:
        return
    info = MODEL_REGISTRY.pop(alias, None)
    REMOVED_MODELS.add(alias)
    if not info:
        print(f"[cleanup] {alias}: already absent ({reason})")
        return
    _drop_source_cache_path(info.get("path", ""))
    _release_record(info, reason=reason, unregister=False)
    print(f"[cleanup] {alias}: released ({reason})")


def release_models_after_step(names, *, reason="last-use"):
    for name in names or []:
        release_registered_model(name, reason=reason)
    _prune_dead_hash_cache()
    flush(light=True)


def _ensure_runtime_path():
    extras = [
        os.path.expanduser("~/.local/bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/opt/local/bin",
        "/usr/bin",
        "/bin",
    ]
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    for p in extras:
        if p and p not in parts and os.path.isdir(p):
            parts.append(p)
    os.environ["PATH"] = os.pathsep.join(parts)


def _resolve_executable(name):
    _ensure_runtime_path()
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidates = [
        name,
        os.path.expanduser(f"~/.local/bin/{name}"),
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/local/bin/{name}",
        f"/usr/bin/{name}",
        f"/bin/{name}",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        errno.ENOENT,
        f"Executable not found: {name}. PATH={os.environ.get('PATH','')}",
        name,
    )


SHELL_META_RE = re.compile(r'[ \t\n\r|&;<>()[\]{}$`!*?~"\'\\]')


def _is_progressish_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    lower = stripped.lower()
    return (
        stripped.startswith("[#")
        or "%|" in stripped
        or "it/s" in lower
        or "s/it" in lower
        or ("dl:" in lower and ("eta:" in lower or "%" in stripped))
    )


def _stream_subprocess_output(proc, *, progress_prefix: bool = False):
    current = ""
    while True:
        chunk = proc.stdout.read(1)
        if chunk == "":
            break
        if chunk == "\r":
            text = current.strip()
            if text:
                if progress_prefix or _is_progressish_text(text):
                    print(f"[planner-progress] {text}")
                else:
                    print(text)
            current = ""
            continue
        if chunk == "\n":
            text = current.rstrip()
            if text:
                if progress_prefix or _is_progressish_text(text):
                    print(f"[planner-progress] {text}")
                else:
                    print(text)
            else:
                print()
            current = ""
            continue
        current += chunk
    tail = current.strip()
    if tail:
        if progress_prefix or _is_progressish_text(tail):
            print(f"[planner-progress] {tail}")
        else:
            print(tail)


def run_cmd(cmd, cwd=None, check_path: bool=False, path: str="", ignore_meta: bool=False):
    if not cmd:
        raise ValueError("cmd must not be empty")
    cmd = [x.strip() for x in cmd]
    prefer_stream = os.path.basename(str(cmd[0])).lower() == "aria2c"
    if not prefer_stream:
        try:
            cmd_ipython = copy.deepcopy(cmd)
            if not ignore_meta:
                for i, value in enumerate(cmd_ipython):
                    value = str(value)

                    if SHELL_META_RE.search(value):
                        value = (
                            value
                            .replace("\\", "\\\\")
                            .replace('"', '\\"')
                            .replace("$", "\\$")
                            .replace("`", "\\`")
                            .replace("!", "\\!")
                        )
                        cmd_ipython[i] = f'"{value}"'

            try:
                !{" ".join(cmd_ipython)}
                if check_path and not os.path.exists(path):
                    raise FileNotFoundError(path)
                return
            finally:
                flush(light=True)
        except:
            pass

    try:
        cmd[0] = _resolve_executable(str(cmd[0]))
    except FileNotFoundError as e:
        print(f"[run_cmd] missing executable: {cmd[0]}")
        print(f"[run_cmd] cwd={cwd}")
        print(f"[run_cmd] PATH={os.environ.get('PATH','')}")
        raise
    pretty = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"$ {pretty}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    try:
        _stream_subprocess_output(proc, progress_prefix=prefer_stream)
        code = proc.wait()
        if check_path and not os.path.exists(path):
            raise FileNotFoundError(path)
        if code != 0:
            raise RuntimeError(f"Command failed with exit code {code}: {pretty}")
    finally:
        flush(light=True)


def register_model(name, path, mode="checkpoint", *, source_kind="registered", source_identity="", original_path=""):
    global MODEL_INSTALL_COUNTER
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    alias = str(name or "").strip()
    if not alias:
        raise ValueError("Model alias is empty")
    MODEL_INSTALL_COUNTER += 1
    record = {
        "alias": alias,
        "path": str(path),
        "basename": os.path.basename(path),
        "filename": os.path.basename(path),
        "mode": mode,
        "source_kind": str(source_kind or "registered"),
        "source_identity": str(source_identity or ""),
        "original_path": str(original_path or ""),
        "install_index": MODEL_INSTALL_COUNTER,
        "install_stage": f"install_{MODEL_INSTALL_COUNTER:04d}",
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    previous = MODEL_REGISTRY.get(alias)
    if previous and os.path.abspath(str(previous.get("path", ""))) != os.path.abspath(str(path)):
        _drop_source_cache_path(previous.get("path", ""))
        _release_record(previous, reason="superseded", unregister=False)
    MODEL_REGISTRY[alias] = record
    MODEL_REGISTRY_HISTORY.append(dict(record))
    if alias in REMOVED_MODELS:
        REMOVED_MODELS.remove(alias)
    print(f"[register] {alias} -> {record['basename']} ({mode}, {record['install_stage']}, {record['source_kind']})")
    return str(path)


def resolve_model_path(name):
    info = MODEL_REGISTRY.get(name)
    if info:
        return info["path"]
    p = Path(models_dir) / f"{name}.safetensors"
    if p.exists():
        return str(p)
    p = Path(models_dir) / f"{name}.ckpt"
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"Model not found: {name}")


def model_file(name):
    path = resolve_model_path(name)
    try:
        p = Path(path).resolve()
        base = Path(models_dir).resolve()
        rel = os.path.relpath(str(p), str(base))
        if rel != ".." and not rel.startswith(".." + os.sep) and not os.path.isabs(rel):
            return rel.replace(os.sep, "/")
    except Exception:
        pass
    return str(path)


def remove_registered_model(name):
    release_registered_model(name, reason="explicit-remove")
    free_gb, total_gb = _disk_free_gb()
    print(f"Remain Storage: {free_gb:.2f}GB/{total_gb:.2f}GB")


def get_vae_path(warn=True):
    for name in ("$vae_name.safetensors", "$vae_name.ckpt"):
        candidate = os.path.join(vae_dir, name)
        if os.path.exists(candidate):
            return candidate
    if warn:
        print(f"⚠️ No VAE found in {vae_dir}, merges may fail")
    return None

vae_path = get_vae_path(warn=False) if BAKE_VAE else None

pref = {"format": "SafeTensor", "size": "pruned", "fp": "fp16"}
cache_filename = os.path.join(models_dir, "cache.json")
cache_data = None


def cache(subsection):
    global cache_data
    if cache_data is None:
        with filelock.FileLock(f"{cache_filename}.lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)
    s = cache_data.get(subsection, {})
    cache_data[subsection] = s
    return s


def dump_cache():
    with filelock.FileLock(f"{cache_filename}.lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    ondisk_mtime = os.path.getmtime(filename)
    if title not in hashes:
        return None
    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)
    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None
    return cached_sha256


def addnet_hash_safetensors(b):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")
    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value
    print(f"Calculating sha256 for {filename}: ", end='')
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")
    hashes[title] = {"mtime": os.path.getmtime(filename), "sha256": sha256_value}
    dump_cache()
    return sha256_value


def sha256_set(filename, title, sha256_value, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    hashes[title] = {"mtime": os.path.getmtime(filename), "sha256": sha256_value}
    dump_cache()


def make_pref(p, mode):
    pref_set = {
        "size": ["full", "pruned"],
        "fp": ["fp16", "bf16", "fp8", "fp32"],
        "format": ["PickleTensor", "SafeTensor"],
    }

    def ordered(values, preferred):
        preferred = str(preferred or "").strip()
        out = [preferred] if preferred in values else []
        out.extend([v for v in values if v not in out])
        return out

    p = dict(p or {})
    if mode == "lora":
        return [{"format": fmt} for fmt in ordered(pref_set["format"], p.get("format", "SafeTensor"))]

    if mode == "checkpoint":
        size_order = ordered(pref_set["size"], p.get("size", "full"))
        fp_order = ordered(pref_set["fp"], p.get("fp", "fp16"))
        format_order = ordered(pref_set["format"], p.get("format", "SafeTensor"))
        res = []
        for fmt in format_order:
            for fp in fp_order:
                for size in size_order:
                    res.append({"size": size, "fp": fp, "format": fmt})
        return res
    raise ValueError(f"Unknown mode: {mode}")


def _metadata_matches_pref(metadata, pref_candidate):
    """Civitai may add extra keys such as isRequired; compare pref as a subset."""
    metadata = metadata or {}
    for key, expected in (pref_candidate or {}).items():
        if str(metadata.get(key)) != str(expected):
            return False
    return True


def _is_civitai_payload_file(file_info):
    """Exclude Required Components such as VAE / Text Encoder from model downloads."""
    if not isinstance(file_info, dict):
        return False
    if file_info.get("type") != "Model":
        return False
    return True


def _select_civitai_file(files, prefs):
    files = list(files or [])
    model_files = [f for f in files if _is_civitai_payload_file(f)]

    # Prefer real model payload files. Only fall back to all files for unusual APIs
    # where the file type is absent, not for Required Components.
    candidates = model_files or [f for f in files if not (f.get("metadata") or {}).get("isRequired")] or files
    if not candidates:
        return None

    for pref_candidate in prefs:
        for f in candidates:
            if _metadata_matches_pref(f.get("metadata", {}), pref_candidate):
                return f

    for f in candidates:
        if f.get("primary") is True:
            return f

    return candidates[0]


def get_dl(url, version=None, mode="checkpoint"):
    prefs = make_pref(pref, mode)
    if "civitai" in url:
        file = None
        if "/api/" in url:
            dllink = url
            dlname = None
            ext = 1
            sha256_value = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR"
        else:
            cid = re.sub(r"\D", "", re.search(r"models/[0-9]+", url).group())
            if "modelVersionId=" in url and version is None:
                version = re.sub(r"\D", "", re.search(r"modelVersionId=[0-9]+", url).group())
            api = f"https://civitai.red/api/v1/models/{cid}" if "civitai.red" in url else f"https://civitai.com/api/v1/models/{cid}" 
            response = requests.get(api)
            if response.status_code != 200:
                return None
            d = response.json()
            model_name = d["name"]
            model_version = version if version is not None else d["modelVersions"][0]["name"]
            model = d["modelVersions"][0]
            for k in d["modelVersions"]:
                if k["name"] == model_version or str(k["id"]) == str(model_version):
                    model = k
                    model_version = k["name"]
                    break
            file = _select_civitai_file(model.get("files", []), prefs)
            if file is None:
                return None
            dllink = file["downloadUrl"]
            sha256_value = file.get("hashes", {}).get("SHA256", "").lower() or None
            ext = 1 if file.get("metadata", {}).get("format") == "SafeTensor" else 0
            dlname = model_name + "-" + model_version
        filename = None
        try:
            filename = file.get("name") if file is not None else None
        except Exception:
            filename = None
        if not filename:
            filename = os.path.basename(str(dllink).split("?", 1)[0]) or ((dlname or "model") + (".safetensors" if ext == 1 else ".ckpt"))
        return {"url": dllink, "name": dlname, "format": ext, "sha256": sha256_value, "filename": filename}

    if "huggingface" in url:
        url_set = url.replace("https://huggingface.co/", "").split("/")
        base = "https://huggingface.co/"
        api = base
        dllink = base
        dname = url_set[-1].rsplit(".", 1)
        dlname = dname[0]
        ext = 1 if len(dname) > 1 and dname[1] == "safetensors" else 0
        for i, s in enumerate(url_set):
            if i == 2:
                api += "raw/"
                dllink += "resolve/"
            else:
                api += f"{s}/"
                dllink += f"{s}/"
        res = requests.get(api)
        if res.status_code != 200:
            return None
        match = re.search(r"sha256:[0-9a-f]+", res.text)
        sha256_value = match.group().replace("sha256:", "") if match else None
        filename = url_set[-1].split("?", 1)[0] or (dlname + (".safetensors" if ext == 1 else ".ckpt"))
        return {"url": dllink, "name": dlname, "format": ext, "sha256": sha256_value, "filename": filename}

    return None


def _safe_model_stem(value):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "model")).strip("._-")
    return stem or "model"


def _source_identity_hash(*parts):
    text = "|".join(str(x or "") for x in parts)
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]


def _safe_model_filename(value, ext="safetensors"):
    ext = str(ext or "safetensors").lstrip(".")
    raw = os.path.basename(str(value or "").split("?", 1)[0])
    if not raw:
        raw = f"model.{ext}"
    suffix = Path(raw).suffix or f".{ext}"
    stem = _safe_model_stem(Path(raw).stem)
    safe_suffix = re.sub(r"[^A-Za-z0-9.]+", "", suffix) or f".{ext}"
    return f"{stem}{safe_suffix}"


def _source_backed_destination(alias, ext, mode, source_identity, kind="src", filename=None):
    ext = str(ext or "safetensors").lstrip(".")
    suffix = _source_identity_hash(mode, alias, source_identity)
    folder = os.path.join(models_dir, "_planner_registry", f"{kind}_{suffix}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, _safe_model_filename(filename or f"{alias}.{ext}", ext))


def model(name, format=1, mode="checkpoint"):
    ext = "ckpt" if format == 0 else "safetensors"
    path = f"{models_dir}/{name}.{ext}"
    if os.path.exists(path):
        sha256_set(path, f"{mode}/{name}", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR")
    return register_model(name, path, mode, source_kind="existing")


def _aria_headers(token):
    return ["--header", f"Authorization: Bearer {token}"] if token else []


def custom_model(url, checkpoint_name=None, mode="checkpoint"):
    user_token = HFToken if "huggingface" in str(url) else CVToken
    parse = {"url": url, "version": None, "mode": mode} if not isinstance(url, list) else {"url": url[0], "version": url[1], "mode": mode}
    g = get_dl(**parse)
    if not g:
        raise RuntimeError(f"Could not resolve download info for {url}")
    url = g["url"]
    checkpoint_name = g["name"] if checkpoint_name is None else checkpoint_name
    sha256_value = g["sha256"]
    ext = "ckpt" if g["format"] == 0 else "safetensors"
    source_identity = sha256_value or url
    filename = g.get("filename") or f"{_safe_model_stem(checkpoint_name)}.{ext}"
    cache_key = (mode, checkpoint_name, source_identity)
    dst = SOURCE_MODEL_CACHE.get(cache_key) or _source_backed_destination(checkpoint_name, ext, mode, source_identity, kind="dl", filename=filename)
    SOURCE_MODEL_CACHE[cache_key] = dst
    if os.path.exists(dst):
        if sha256_value is not None:
            sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
        return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=url)
    out_name = os.path.basename(dst)
    out_dir = os.path.dirname(dst)
    if "huggingface" in url:
        run_cmd(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", *_aria_headers(user_token), url, "-d", out_dir, "-o", out_name], check_path=True, path=dst)
    else:
        headers = {
            "User-Agent": UserAgent().chrome,
            "Authorization": f"Bearer {user_token}",
        }
        response = requests.get(url, headers=headers, allow_redirects=False)
        download_link = response.headers.get("Location") or url
        run_cmd(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", download_link, "-d", out_dir, "-o", out_name], check_path=True, path=dst)
    if sha256_value is not None:
        sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
    return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=url)


def custom_vae(url, vae_name="VAE"):
    url = str(url or "").strip()
    vae_name = str(vae_name or "VAE").strip() or "VAE"
    if not url:
        return get_vae_path()
    user_token = HFToken if "huggingface" in url else CVToken
    if "civitai" in url:
        if "/api/" in url:
            ext = "safetensors" if "SafeTensor" in url else "ckpt"
            headers = {"User-Agent": UserAgent().chrome, "Authorization": f"Bearer {user_token}"}
            response = requests.get(url, headers=headers, allow_redirects=False)
            download_link = response.headers.get("Location") or url
            run_cmd(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", download_link, "-d", vae_dir, "-o", f"{vae_name}.{ext}"], check_path=True, path=os.path.join(vae_dir, f"{vae_name}.{ext}"))
            # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{vae_dir}" -o {vae_name}.{ext}
            return os.path.join(vae_dir, f"{vae_name}.{ext}")
        pref_order = ["SafeTensor", "PickleTensor"]
        cid_match = re.search(r"models/[0-9]+", url)
        if not cid_match:
            raise RuntimeError(f"Could not resolve civitai VAE model id from {url}")
        cid = re.sub(r"\D", "", cid_match.group())
        version_match = re.search(r"modelVersionId=[0-9]+", url)
        version = re.sub(r"\D", "", version_match.group()) if version_match else None
        api = f"https://civitai.red/api/v1/models/{cid}" if "civitai.red" in url else f"https://civitai.com/api/v1/models/{cid}" 
        response = requests.get(api)
        if response.status_code != 200:
            raise RuntimeError("ERROR: VAE Not Found")
        d = response.json()
        model_name = vae_name or d["name"]
        model_version = version if version is not None else d["modelVersions"][0]["name"]
        model = d["modelVersions"][0]
        for k in d["modelVersions"]:
            if k["name"] == model_version or str(k["id"]) == str(model_version):
                model = k
                break
        file = None
        meta_list = [a.get("metadata", {}).get("format") for a in model["files"]]
        for pref_candidate in pref_order:
            if pref_candidate in meta_list:
                file = model["files"][meta_list.index(pref_candidate)]
                break
        if file is None:
            file = model["files"][0]
        ext = "safetensors" if file.get("metadata", {}).get("format") == "SafeTensor" else "ckpt"
        headers = {"User-Agent": UserAgent().chrome, "Authorization": f"Bearer {user_token}"}
        response = requests.get(file["downloadUrl"], headers=headers, allow_redirects=False)
        download_link = response.headers.get("Location") or file["downloadUrl"]
        run_cmd(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", download_link, "-d", vae_dir, "-o", f"{model_name}.{ext}"], check_path=True, path=os.path.join(vae_dir, f"{model_name}.{ext}"))
        # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{vae_dir}" -o {model_name}.{ext}
        return os.path.join(vae_dir, f"{model_name}.{ext}")
    if "huggingface" in url:
        filename = url.split("/")[-1]
        ext = filename.split(".")[-1] if "." in filename else "safetensors"
        resolved = url.replace("/blob/main/", "/resolve/main/") if "/blob/main/" in url else url
        user_header = f"\"Authorization: Bearer {user_token}\""
        run_cmd(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", *_aria_headers(user_token), resolved, "-d", vae_dir, "-o", f"{vae_name}.{ext}"], check_path=True, path=os.path.join(vae_dir, f"{vae_name}.{ext}"))
        # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M --header={user_header} "{resolved}" -d "{vae_dir}" -o {vae_name}.{ext}
        return os.path.join(vae_dir, f"{vae_name}.{ext}")
    return None


def old_custom_model(url, checkpoint_name=None, format=1, sha256_value=None, mode="checkpoint"):
    checkpoint_name = checkpoint_name or _safe_model_stem(Path(str(url).split("?")[0]).stem or "model")
    ext = "ckpt" if format == 0 else "safetensors"
    source_identity = sha256_value or url
    filename = os.path.basename(str(url).split("?", 1)[0]) or f"{_safe_model_stem(checkpoint_name)}.{ext}"
    cache_key = (mode, checkpoint_name, source_identity)
    dst = SOURCE_MODEL_CACHE.get(cache_key) or _source_backed_destination(checkpoint_name, ext, mode, source_identity, kind="dl", filename=filename)
    SOURCE_MODEL_CACHE[cache_key] = dst
    if os.path.exists(dst):
        if sha256_value is not None:
            sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
        return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=url)
    out_name = os.path.basename(dst)
    out_dir = os.path.dirname(dst)
    if "huggingface" in str(url):
        run_cmd(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", *_aria_headers(HFToken), url, "-d", out_dir, "-o", out_name], check_path=True, path=dst)
    else:
        headers = {"User-Agent": UserAgent().chrome, "Authorization": f"Bearer {CVToken}"}
        response = requests.get(url, headers=headers, allow_redirects=False)
        download_link = response.headers.get("Location") or url
        run_cmd(["aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M", download_link, "-d", out_dir, "-o", out_name], check_path=True, path=dst)
    if sha256_value is not None:
        sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
    return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=url)


def local_model(src, alias=None, mode="checkpoint"):
    src = os.path.abspath(os.path.expanduser(str(src)))
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    alias = alias or Path(src).stem
    ext = (Path(src).suffix or ".safetensors").lstrip(".")
    try:
        stat = os.stat(src)
        source_identity = f"{src}:{stat.st_size}:{stat.st_mtime_ns}"
    except Exception:
        source_identity = src
    cache_key = (mode, alias, source_identity)
    dst = SOURCE_MODEL_CACHE.get(cache_key) or _source_backed_destination(alias, ext, mode, source_identity, kind="local", filename=Path(src).name)
    SOURCE_MODEL_CACHE[cache_key] = dst
    # Do not duplicate local models.  Use a lightweight link inside the planner registry
    # when possible so merge.py can still receive a path relative to models_dir.
    if os.path.abspath(src) != os.path.abspath(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            if os.path.lexists(dst) and os.path.islink(dst) and os.path.realpath(dst) != src:
                os.unlink(dst)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        except Exception:
            try:
                if not os.path.exists(dst):
                    os.link(src, dst)
            except Exception:
                dst = src
    return register_model(alias, dst, mode, source_kind="local", source_identity=source_identity, original_path=src)


def ratio_value(spec):
    if not spec:
        return "0.5"
    mode = spec.get("mode", "Single")
    value = str(spec.get("value", "")).strip()
    if mode == "Block weight":
        if not value:
            value = ",".join(["0"] * 20)
        return value
    return value or "0.5"


def ratio_args(flag, spec):
    mode = ""
    if isinstance(spec, dict):
        mode = str(spec.get("mode", ""))
    value = ratio_value(spec)
    text = str(value).strip()
    is_rand = mode.lower() == "randomize" or text.lower().startswith(("@r", "@rand"))
    if is_rand:
        text = re.sub(r"^@(r|rand)\s*", "", text, flags=re.IGNORECASE).strip()
    name = f"--rand_{flag}" if is_rand else f"--{flag}"
    return [name, text if is_rand else value]


def signature_args(text):
    text = (text or "").strip()
    if not text:
        return []
    return shlex.split(text.replace("\n", " "))


def _print_plan_failure(entry_index, entry_type, entry_id, entry_payload, body_lines):
    print("\n[PLAN FAILURE]")
    print(f"step={entry_index + 1} type={entry_type} id={entry_id}")
    try:
        print(json.dumps(entry_payload, ensure_ascii=False, indent=2))
    except Exception:
        print(repr(entry_payload))
    print("\n[STEP SOURCE]")
    for i, src in enumerate(body_lines, start=1):
        print(f"{i:02d}: {src}")
    traceback.print_exc()


def run_notebook_bang(source, cwd=None):
    source = str(source or "").strip()
    if not source:
        return
    lines = [ln.rstrip() for ln in source.splitlines() if ln.strip()]
    if not lines:
        return
    normalized = []
    for idx, ln in enumerate(lines):
        stripped = ln.lstrip()
        if idx == 0 and stripped.startswith("!"):
            stripped = stripped[1:]
        normalized.append(stripped)
    cmd = "\n".join(normalized).strip()
    shell_cmd = f"cd {shlex.quote(cwd)} && {cmd}" if cwd else cmd
    print(f"$ {shell_cmd}")
    ip = None
    try:
        ip = get_ipython()
    except Exception:
        ip = None
    if ip is not None:
        rc = ip.system(shell_cmd)
        if rc not in (None, 0):
            raise RuntimeError(f"Notebook shell command failed with exit code {rc}: {shell_cmd}")
        return
    proc = subprocess.Popen(shell_cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=os.environ.copy())
    for line in proc.stdout:
        print(line, end="")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Notebook shell command failed with exit code {code}: {shell_cmd}")


if BAKE_VAE and VAE_URL:
    try:
        custom_vae(VAE_URL, VAE_NAME)
    except Exception as e:
        print(f"VAE download failed: {e}")

vae_path = get_vae_path(warn=True) if BAKE_VAE else None
flush(light=False)
%cd {merge_repo_dir}
''')

UPLOAD_TPL = Template(r'''# Optional upload helper
import os
from huggingface_hub import create_repo, upload_file

UPLOAD_AFTER_MERGE=$upload_after_merge
repo_id = "$repo".strip()
final_model = "$final".strip()
model_dir = r"$model_dir"
final_path = os.path.join(model_dir, f"{final_model}.safetensors")

if UPLOAD_AFTER_MERGE:
    if repo_id and final_model and HFToken and os.path.exists(final_path):
        create_repo(repo_id=repo_id, token=HFToken, exist_ok=True)
        upload_file(path_or_fileobj=final_path, path_in_repo=os.path.basename(final_path), repo_id=repo_id, token=HFToken)
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
        print(f"Uploaded {final_path} -> {repo_id}")
    else:
        print("Upload helper idle. Set repo/token or produce a final model first.")
''')

T2I_CFG_TPL = Template(r'''# Pipe Config (short)
RUN_T2I = $run_t2i

if RUN_T2I:
    import os
    import torch
    import diffusers
    $pipelines
    base = "$base".strip()
    pipeline = $pipeline

    checkpoint = "$final".strip()
    ext = "safetensors"
    model_type = "fp16"
    scheduler = "euler_a"

    SCHEDULERS = {
        "unipc": [diffusers.schedulers.UniPCMultistepScheduler, {}, "UniPC"],
        "euler_a": [diffusers.schedulers.EulerAncestralDiscreteScheduler, {}, "Euler a"],
        "euler": [diffusers.schedulers.EulerDiscreteScheduler, {}, "Euler"],
        "ddim": [diffusers.schedulers.DDIMScheduler, {}, "DDIM"],
        "ddpm": [diffusers.schedulers.DDPMScheduler, {}, "DDPM"],
        "deis": [diffusers.schedulers.DEISMultistepScheduler, {}, "DEIS"],
        "dpm++_2m": [diffusers.schedulers.DPMSolverMultistepScheduler, {}, "DPM++ 2M"],
        "dpm++_2m_karras": [diffusers.schedulers.DPMSolverMultistepScheduler, {"use_karras_sigmas": True}, "DPM++ 2M Karras"],
    }
    mt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

    cpath = os.path.join(r"$model_dir", f"{checkpoint}.{ext}")
    dtype = mt[model_type]
    scheduler_cls, scheduler_kwargs, scheduler_name = SCHEDULERS[scheduler]

    if checkpoint and os.path.exists(cpath):
        base_pipe = pipeline.from_single_file(cpath, torch_dtype=dtype, use_safetensors=True, variant="fp16")
        scd = scheduler_cls.from_config(base_pipe.scheduler.config, **scheduler_kwargs)
        pipe = pipeline.from_single_file(cpath, torch_dtype=dtype, scheduler=scd, use_safetensors=True, variant="fp16")
        if base == "Anima":
            pipe.scheduler.set_sampling_config(
                sampler="euler_a_rf",      # flowmatch_euler | euler | euler_a_rf | euler_ancestral_rf
                sigma_schedule="normal",     # uniform | beta | simple | normal
            )
        pipe.safety_checker = None
        pipe = pipe.to("cuda:0" if torch.cuda.is_available() else "cpu")
        init_pipe = pipe
        scd_name = scheduler_name
        print(f"Loaded pipeline for {checkpoint}")
    else:
        init_pipe = None
        init_refiner = None
        scd_name = "N/A"
        print("No final checkpoint found for t2i.")
''')

T2I_RUN_TPL = Template(r'''# t2i
RUN_T2I = $run_t2i
T2I_SETTINGS_JSON = r"""$t2i_settings_json"""

def bpro(prompt):
    k = prompt.split(",")
    thu = []
    for g in k:
        f = g.count(" ")
        thu.append([g, f+1])
    off = 0
    nl = []
    t = 0
    for x in thu:
        if "BREAK" in x[0]:
            tok = t+off
            add = tok % 75
            nl += [" "]*add
            off += add
            continue
        t += x[1]
        nl.append(x[0])
    return ",".join(nl)
    
if RUN_T2I:
    import os
    import json
    import random
    from PIL import Image
    from IPython.display import display
    from sd_embed.embedding_funcs import $encoder
    
    base = "$base".strip()
    pipe = globals().get("init_pipe")
    encoder = $encoder
    default_settings = {
        "prompts": [{"name": "default", "prompt": "masterpiece, best quality, scenery", "negative": "lowres, bad anatomy, watermark"}],
        "seed": -1,
        "steps": 20,
        "width": 768,
        "height": 1152,
        "cfg": 4.5,
        "num_gen": 1,
    }
    try:
        loaded_settings = json.loads(T2I_SETTINGS_JSON) if T2I_SETTINGS_JSON.strip() else {}
        if isinstance(loaded_settings, dict):
            default_settings.update(loaded_settings)
    except Exception as e:
        print(f"[t2i] Failed to parse settings, using defaults: {e}")
    settings = default_settings
    prompt_items = settings.get("prompts") or default_settings["prompts"]
    if isinstance(prompt_items, dict):
        prompt_items = [prompt_items]
    prompt_items = [p for p in prompt_items if isinstance(p, dict)] or default_settings["prompts"]
    w = int(settings.get("width") or settings.get("w") or 768)
    h = int(settings.get("height") or settings.get("h") or 1152)
    steps = int(settings.get("steps") or 20)
    guidance = float(settings.get("cfg", settings.get("guidance", 4.5)) or 4.5)
    num_gen = max(1, int(settings.get("num_gen") or 1))
    base_seed = int(settings.get("seed", -1) if str(settings.get("seed", "")).strip() else -1)
    _t2i_root = globals().get("WORKING_DIR", r"$workpath")
    idir = os.path.join(_t2i_root, "t2i_images")
    os.makedirs(idir, exist_ok=True)

    if pipe is None:
        print("t2i helper idle. Configure or build a final model first.")
    else:
        manifest = []
        for prompt_idx, item in enumerate(prompt_items):
            prompt = str(item.get("prompt") or "").strip() or default_settings["prompts"][0]["prompt"]
            neg = str(item.get("negative") or item.get("neg") or settings.get("negative") or default_settings["prompts"][0]["negative"])
            if base != "SDXL":
                embeds, negative_embeds=encoder(pipe,prompt=bpro(prompt),neg_prompt=neg)
            else:
                embeds, negative_embeds, pooled, neg_pooled=encoder(pipe,prompt=bpro(prompt),neg_prompt=neg)
            name = str(item.get("name") or f"prompt_{prompt_idx+1}").strip() or f"prompt_{prompt_idx+1}"
            safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:64] or f"prompt_{prompt_idx+1}"
            for i in range(num_gen):
                seed = base_seed if base_seed >= 0 else random.randrange(4294967294)
                if base_seed >= 0 and (prompt_idx or i):
                    seed = (base_seed + prompt_idx * max(1, num_gen) + i) % 4294967294
                generator = torch.Generator(device="cpu").manual_seed(seed)
                if base != "SDXL":
                    image = pipe(
                        prompt=None, 
                        prompt_embeds=embeds, 
                        negative_prompt_embeds=negative_embeds, 
                        height=h, width=w, 
                        num_inference_steps=steps, 
                        guidance_scale=guidance, 
                        generator=generator).images[0]
                else:
                    image = pipe(
                        prompt=None, 
                        prompt_embeds=embeds, 
                        pooled_prompt_embeds=pooled, 
                        negative_prompt_embeds=negative_embeds, 
                        negative_pooled_prompt_embeds=neg_pooled, 
                        height=h, width=w, 
                        num_inference_steps=steps, 
                        guidance_scale=guidance, 
                        generator=generator).images[0]
                out_path = os.path.join(idir, f"{safe_name}_{i:03d}_{seed}.png")
                image.save(out_path)
                manifest.append({"path": out_path, "prompt_name": name, "prompt": prompt, "negative": neg, "seed": seed, "width": w, "height": h, "steps": steps, "cfg": guidance})
                display(image.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS))
                print(f"Saved: {out_path}")
        manifest_path = os.path.join(idir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[t2i] Manifest: {manifest_path}")
''')
ZIP_TPL = Template(r'''# Image ZIP
RUN_T2I = $run_t2i

if RUN_T2I:
    import os
    import zipfile
    from pathlib import Path

    name = "download"
    _zip_root = globals().get("WORKING_DIR", r"$workpath")
    dst = os.path.join(_zip_root, f"{name}.zip")
    idir = os.path.join(_zip_root, "t2i_images")
    if os.path.exists(dst):
        os.remove(dst)
    paths = [str(p) for p in Path(idir).rglob("*") if p.is_file()]
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, os.path.join(name, os.path.relpath(p, idir)))
    print(f"Done! -> {dst}")
''')


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalize_precision_name(value: Any) -> str:
    p = str(value or "").strip().lower()
    if not p:
        return ""
    if p in ("bhalf", "bf16", "bfloat16"):
        return "bhalf"
    if p in ("quarter", "fp8", "float8"):
        return "quarter"
    if p in ("fp32", "float32", "full"):
        return "fp32"
    if p in ("half", "fp16", "float16"):
        return "half"
    return p


def _tail_text_to_cli_signatures(raw_text: str) -> str:
    tokens = shlex.split((raw_text or "").replace("\n", " "))
    if not tokens:
        return ""
    tail = _parse_tail_at(tokens)
    out: List[str] = []
    if tail.get("cosine") is not None:
        out.append(f"--cosine{tail['cosine']}")
    if tail.get("fine"):
        fine = str(tail["fine"])
        out.append(f'--fine={"\"" + fine + "\"" if _needs_quote(fine) else fine}')
    if tail.get("seed") is not None:
        out.append(f"--seed {tail['seed']}")
    if tail.get("rank") is not None:
        out.append(f"--rank {tail['rank']}")
    if tail.get("arch"):
        out.append(f"--arch {tail['arch']}")
    for extra in tail.get("extras") or []:
        extra = str(extra).strip()
        if extra:
            out.append(extra)
    return " ".join(out).strip()


def _command_signatures(entry: Dict[str, Any]) -> str:
    additional = str(entry.get("additional_signatures") or "").strip()
    if additional:
        return additional
    return _tail_text_to_cli_signatures(str(entry.get("raw_signatures") or ""))


def _precision_from_signatures(text: str) -> str:
    tokens = shlex.split((text or "").replace("\n", " "))
    tail = _parse_tail_at(tokens)
    return _normalize_precision_name(tail.get("precision"))


def _entry_precision(entry: Dict[str, Any], command_signatures: str = "") -> str:
    precision = _normalize_precision_name(entry.get("precision"))
    # additional_signatures may contain @p/@precision when a JSON plan is fed directly.
    parsed_additional = _precision_from_signatures(command_signatures or str(entry.get("additional_signatures") or ""))
    if parsed_additional:
        precision = parsed_additional
    # raw_signatures is the user-facing field in the planner UI. Let it override stale
    # imported precision values so editing @p in Additional Signatures is respected.
    parsed_raw = _precision_from_signatures(str(entry.get("raw_signatures") or ""))
    if parsed_raw:
        precision = parsed_raw
    return precision or "half"


def _precision_args(entry: Dict[str, Any], command_signatures: str = "") -> str:
    p = _entry_precision(entry, command_signatures=command_signatures)
    save_flag = "--save_half"
    if p in ("bhalf", "bf16", "bfloat16"):
        save_flag = "--save_bhalf"
    elif p in ("quarter", "fp8", "float8"):
        save_flag = "--save_quarter"
    elif p in ("fp32", "float32", "full"):
        save_flag = "--save_full"
    return _json_literal([save_flag, "--prune", "--save_safetensors"])


def _legacy_signature_text(entry: Dict[str, Any]) -> str:
    sig = (entry.get("raw_signatures") or "").strip() or (entry.get("additional_signatures") or "").strip()
    precision = _normalize_precision_name(entry.get("precision"))
    if precision and precision != "half" and not _precision_from_signatures(sig):
        sig = f"{sig} @p {precision}".strip()
    return sig


def _entry_to_lines(entry: Dict[str, Any], *, bake_vae: bool = True) -> Tuple[List[str], str | None]:
    etype = entry.get("type")
    lines: List[str] = []
    produced: str | None = None
    temp = lambda x: f"TEMP{x}" if x and x[0]=="_" else x

    if etype == "Download Model":
        model_name = temp((entry.get("model_name") or "").strip())
        link = (entry.get("link") or "").strip()
        model_type = (entry.get("model_type") or "Checkpoint").strip()
        if not model_name and not link:
            return [], None
        if link and not model_name:
            raise PlanCompileError("Download Model requires Model Name when Link is set", entry_type=etype, entry_id=entry.get("id"), entry_payload=entry)
        mode = "lora" if model_type in ("LoRA", "LyCORIS") else "checkpoint"
        if model_name and link:
            lines.append(f'custom_model({_json_literal(link)}, checkpoint_name={_json_literal(model_name)}, mode={_json_literal(mode)})')
        elif model_name:
            lines.append(f'model({_json_literal(model_name)}, format=1, mode={_json_literal(mode)})')
        return lines, model_name or None

    if etype == "Local Model":
        local_path = (entry.get("local_path") or "").strip()
        if local_path:
            alias = temp(str((entry.get("model_name") or "").strip() or Path(local_path).stem))
            mode = "lora" if (entry.get("model_type") or "Checkpoint") in ("LoRA", "LyCORIS") else "checkpoint"
            lines.append(f'local_model({_json_literal(local_path)}, alias={_json_literal(alias)}, mode={_json_literal(mode)})')
            produced = alias
        return lines, produced

    if etype == "Remove Model":
        model_name = (entry.get("model") or "").strip()
        if model_name:
            lines.append(f'remove_registered_model({_json_literal(model_name)})')
            lines.append('flush()')
        return lines, None

    if etype == "Checkpoint Merge":
        merge_mode = (entry.get("merge_mode") or "WS").strip() or "WS"
        model0 = (entry.get("model0") or "").strip()
        model1 = (entry.get("model1") or "").strip()
        model2 = (entry.get("model2") or "").strip()
        output_name = (entry.get("output_name") or "").strip()
        if model0 and model1 and output_name:
            command_signatures = _command_signatures(entry)
            precision_args = _precision_args(entry, command_signatures)
            lines.extend([
                f'beta = {entry.get("beta","") != ""}',
                'cmd = [sys.executable, "merge.py", ' + _json_literal(merge_mode) + ', models_dir + "/", model_file(' + _json_literal(model0) + '), model_file(' + _json_literal(model1) + ')]',
                f'if {bool(model2)!r}:\n        cmd.append(model_file({_json_literal(model2)}))',
                *(['if vae_path:\n        cmd += ["--vae", vae_path]'] if bake_vae else []),
                'cmd += ratio_args("alpha", ' + _json_literal(entry.get("alpha") or default_ratio("Single")) + ')',
                'if beta:\n        cmd += ratio_args("beta", ' + _json_literal(_normalize_ratio_spec(entry.get("beta"), allow_block_weight=True, default_single="0.5")) + ')',
                'cmd += ' + precision_args,
                'cmd += ["--output", ' + _json_literal(output_name) + ']',
                'cmd += shlex.split(' + _json_literal(command_signatures) + ')' if command_signatures else 'cmd += []',
                'run_cmd(cmd, cwd=merge_repo_dir, check_path=True, path=os.path.join(models_dir, ' + _json_literal(f"{output_name}.safetensors") + '), ignore_meta=True)',
                'register_model(' + _json_literal(output_name) + ', os.path.join(models_dir, ' + _json_literal(f"{output_name}.safetensors") + '), "checkpoint", source_kind="generated")',
                'flush()',
            ])
            produced = output_name
        return lines, produced

    if etype == "LoRA Bake":
        checkpoint = (entry.get("checkpoint") or "").strip()
        output_name = (entry.get("output_name") or "").strip()
        loras = entry.get("loras") or []
        if checkpoint and output_name and loras:
            parts = []
            for lora in loras:
                name = (lora.get("name") or "").strip()
                if not name:
                    continue
                ratio = _normalize_ratio_spec(lora.get("ratio"), allow_block_weight=False, default_single="1.0")["value"] or "1.0"
                parts.append((name, ratio))
            lines.append('lora_items = []')
            for name, ratio in parts:
                lines.append('lora_items.append(f"{model_file(' + _json_literal(name).replace("\"","'") + ')}:" + "' + ratio.replace('\\', '\\\\').replace('"', '\\"')+ '")')
            command_signatures = _command_signatures(entry)
            precision_args = _precision_args(entry, command_signatures)
            lines.extend([
                'cmd = [sys.executable, "lora_bake.py", models_dir + "/", model_file(' + _json_literal(checkpoint) + '), ",".join(lora_items)]',
                'cmd += ' + precision_args,
                'cmd += ["--output", ' + _json_literal(output_name) + ']',
                'cmd += shlex.split(' + _json_literal(command_signatures) + ')' if command_signatures else 'cmd += []',
                'run_cmd(cmd, cwd=merge_repo_dir, check_path=True, path=os.path.join(models_dir, ' + _json_literal(f"{output_name}.safetensors") + '), ignore_meta=True)',
                'register_model(' + _json_literal(output_name) + ', os.path.join(models_dir, ' + _json_literal(f"{output_name}.safetensors") + '), "checkpoint", source_kind="generated")',
                'flush()',
            ])
            produced = output_name
        return lines, produced

    return lines, None



def _entry_reference_aliases(entry: Dict[str, Any]) -> List[str]:
    etype = entry.get("type")
    refs: List[str] = []
    def add(value: Any):
        if isinstance(value, dict):
            # Embedded source specs are resolved at the owning step and do not refer to
            # an existing alias unless an explicit name/alias is supplied.
            value = value.get("name") or value.get("alias") or value.get("model_name") or ""
        text = str(value or "").strip()
        if text:
            refs.append(text)
    if etype == "Checkpoint Merge":
        add(entry.get("model0"))
        add(entry.get("model1"))
        add(entry.get("model2"))
    elif etype == "LoRA Bake":
        add(entry.get("checkpoint"))
        for lora in entry.get("loras", []) or []:
            add(lora.get("name"))
    elif etype == "Remove Model":
        # Preserve explicit Remove Model semantics: an alias with a later Remove line
        # should remain available until that line runs, even if it is not used by a
        # merge/bake step before then.
        add(entry.get("model"))
    return refs


def _entry_produced_aliases_for_cleanup(entry: Dict[str, Any]) -> List[str]:
    etype = entry.get("type")
    temp = lambda x: f"TEMP{x}" if x and str(x)[0] == "_" else x
    out: List[str] = []
    if etype == "Download Model":
        name = temp(str(entry.get("model_name") or "").strip())
        if name:
            out.append(name)
    elif etype == "Local Model":
        local_path = str(entry.get("local_path") or "").strip()
        if local_path:
            alias = temp(str((entry.get("model_name") or "").strip() or Path(local_path).stem))
            if alias:
                out.append(alias)
    elif etype in ("Checkpoint Merge", "LoRA Bake"):
        name = temp(str(entry.get("output_name") or "").strip())
        if name:
            out.append(name)
    return out


def _auto_release_aliases_by_entry(entries: List[Dict[str, Any]], final_alias: str | None) -> Dict[int, List[str]]:
    # Returns 1-based entry index -> aliases that can be released immediately after
    # that entry succeeds. The final produced checkpoint is intentionally retained.
    refs_by_idx = [_entry_reference_aliases(entry) for entry in entries]
    remaining: Dict[str, int] = {}
    for refs in refs_by_idx:
        for alias in refs:
            remaining[alias] = remaining.get(alias, 0) + 1
    keep = {str(final_alias or "").strip()} if final_alias else set()
    release_by_idx: Dict[int, List[str]] = {}
    active_aliases: set[str] = set()
    for entry_index, entry in enumerate(entries, start=1):
        for alias in _entry_produced_aliases_for_cleanup(entry):
            active_aliases.add(alias)
        for alias in refs_by_idx[entry_index - 1]:
            if alias in remaining:
                remaining[alias] -= 1
                if remaining[alias] <= 0:
                    remaining.pop(alias, None)
        releasable = []
        for alias in sorted(active_aliases):
            if alias in keep or remaining.get(alias, 0) > 0:
                continue
            releasable.append(alias)
        if releasable:
            release_by_idx[entry_index] = releasable
            for alias in releasable:
                active_aliases.discard(alias)
    return release_by_idx


def _entry_progress_label(entry: Dict[str, Any], index: int, total: int) -> str:
    etype = str(entry.get("type") or "Step")
    label = ""
    temp = lambda x: f"TEMP{x}" if x and x[0]=="_" else x
    if etype == "Download Model":
        label = str(entry.get("model_name") or entry.get("link") or "download")
    elif etype == "Local Model":
        label = Path(str(entry.get("local_path") or "local")).name
    elif etype == "Remove Model":
        label = str(entry.get("model") or "remove")
    elif etype == "Checkpoint Merge":
        label = str(entry.get("output_name") or entry.get("merge_mode") or "merge")
    elif etype == "LoRA Bake":
        label = str(entry.get("output_name") or entry.get("checkpoint") or "lora bake")
    label = temp(label)
    return f"[planner-progress] {index}/{total} | {etype} | {label}"


def planit_records(plan: Dict[str, Any], workpath: str, model_dir: str = "", vae_dir: str = "", bake_vae: bool = True) -> Tuple[List[str], str | None]:
    del workpath, model_dir, vae_dir
    entries = normalize_plan(plan).get("entries", [])
    total = max(1, len(entries))
    compiled: List[Tuple[int, Dict[str, Any], List[str], str | None]] = []
    final: str | None = None
    for entry_index, entry in enumerate(entries, start=1):
        try:
            lines, produced = _entry_to_lines(entry, bake_vae=bake_vae)
        except Exception as e:
            raise PlanCompileError(
                f"Failed to compile plan entry #{entry_index} ({entry.get('type', 'Unknown')}): {e}",
                entry_index=entry_index,
                entry_type=entry.get('type', 'Unknown'),
                entry_id=entry.get('id'),
                entry_payload=entry,
                cause=e,
            ) from e
        compiled.append((entry_index, entry, lines, produced))
        if produced:
            final = produced

    release_by_idx = _auto_release_aliases_by_entry(entries, final)
    res: List[str] = []
    for entry_index, entry, lines, produced in compiled:
        if not lines:
            continue
        progress_line = f"print({_json_literal(_entry_progress_label(entry, entry_index, total))})"
        cleanup_aliases = release_by_idx.get(entry_index, [])
        cleanup_line = (
            "release_models_after_step("
            + _json_literal(cleanup_aliases)
            + f", reason=\"last-use after step {entry_index}\")"
        )
        res.append("\n".join([progress_line, *lines, cleanup_line]))
    res.append("\n".join([
        "print('[planner-progress] cleanup | final runtime cleanup')",
        "_prune_dead_hash_cache()",
        "_prune_orphan_registry_dirs(aggressive=False)",
        "flush(light=False)",
    ]))
    return res, final


def planit(filepath, workpath, model_dir="", vae_dir="", bake_vae: bool = True):
    plan = load_plan_records(filepath)
    return planit_records(plan, workpath, model_dir, vae_dir, bake_vae=bake_vae)


def create_plan(filepath: str, workpath: str, saveas: str, title: str,
                vae: str, CivitAPI: str, HuggingAPI: str, UR: str,
                model_dir: str = "", vae_dir: str = "", vae_name: str = "VAE",
                bake_vae: bool = True,
                ignore_install_deps: bool = False, upload_after_merge: bool = False, run_t2i: bool = False):
    # Exporting a txt runner must not touch runtime-only directories on the local machine.
    # The generated runner creates workpath/tmp, models, embeddings, and vae when it executes.
    runtime_workpath = _notebook_runtime_workpath(workpath)
    res, _ = planit(filepath, runtime_workpath, model_dir, vae_dir, bake_vae=bake_vae)
    prelude = PRELUDE_TPL.safe_substitute(
        workpath=runtime_workpath,
        toolpath=_preferred_toolpath(),
        hf_token=HuggingAPI,
        cv_token=CivitAPI,
        vae_url=vae,
        vae_name=vae_name,
        bake_vae="True" if bake_vae else "False",
        model_dir=model_dir,
        vae_dir=vae_dir,
    )
    with open(saveas, "w", encoding="utf-8") as f:
        f.write(f"#{title}\n\n")
        f.write(prelude)
        f.write("\n\n".join(res))


def create_plan_ipynb(filepath: str, workpath: str, saveas: str, title: str,
                      vae: str, CivitAPI: str, HuggingAPI: str, UR: str,
                      model_dir: str = "", vae_dir: str = "", vae_name: str = "VAE",
                      base_model: str = "SDXL", bake_vae: bool = True,
                      ignore_install_deps: bool = False, use_online: bool = False,
                      upload_after_merge: bool = False, run_t2i: bool = False,
                      t2i_settings: Dict[str, Any] | None = None):
    # Export as notebook should only write an .ipynb. Do not create /kaggle or other
    # runtime directories on the machine that is doing the export. The notebook itself
    # creates those directories when it is executed in the target environment.
    runtime_workpath = _notebook_runtime_workpath(workpath)
    act_model_dir = model_dir if model_dir else f"{runtime_workpath}/tmp/models"
    use_online_literal = "True" if use_online else "False"
    diffusion = "git+https://github.com/Faildes/diffusers-anima@from_multiple_models" if base_model == "Anima" else ""
    install = INSTALL_TPL.safe_substitute(
        workpath=runtime_workpath,
        toolpath=_preferred_toolpath(),
        ignore_install_deps=ignore_install_deps,
        use_online=use_online_literal,
        diffusion=diffusion,
    )
    prelude = PRELUDE_TPL.safe_substitute(
        workpath=runtime_workpath,
        toolpath=_preferred_toolpath(),
        use_online=use_online_literal,
        hf_token=HuggingAPI,
        cv_token=CivitAPI,
        vae_url=vae,
        vae_name=vae_name,
        bake_vae="True" if bake_vae else "False",
        model_dir=model_dir,
        vae_dir=vae_dir,
    )
    res, final = planit(filepath, runtime_workpath, model_dir, vae_dir, bake_vae=bake_vae)
    plan_cell = f"#{title}\n\n" + prelude + "\n\n" + "\n\n".join(res)
    upload = UPLOAD_TPL.safe_substitute(workpath=runtime_workpath, final=final or "", repo=UR, model_dir=act_model_dir, upload_after_merge=upload_after_merge)
    base = "StableDiffusionXL" if base_model == "SDXL" else ("Flux" if base_model == "Flux" else ("StableDiffusion" if base_model == "SD1.5" else base_model))
    pipelines = "from diffusers_anima import AnimaPipeline\n    import diffusers_anima" if base_model == "Anima" else f"from diffusers import {base}Pipeline"
    pipeline = f"{base}Pipeline"
    t2i_cfg = T2I_CFG_TPL.safe_substitute(workpath=runtime_workpath, final=final or "", model_dir=act_model_dir, run_t2i=run_t2i, base=base, pipelines=pipelines, pipeline=pipeline)
    t2i_settings_json = json.dumps(t2i_settings or {}, ensure_ascii=False)
    encoder = f"get_weighted_text_embeddings_{base_model.lower().replace('.','').replace('flux','flux1')}"
    t2i_run = T2I_RUN_TPL.safe_substitute(workpath=runtime_workpath, run_t2i=run_t2i, t2i_settings_json=t2i_settings_json, base=base, encoder=encoder)
    zipc = ZIP_TPL.safe_substitute(workpath=runtime_workpath, run_t2i=run_t2i)
    cells = [install, plan_cell, upload, t2i_cfg, t2i_run, zipc]
    with open(saveas, "w", encoding="utf-8") as f:
        f.write(_nb_json(cells))


# -----------------------------------------------------------------------------
# Structured plan compatibility layer
# - LoRA Merge entries
# - structured CLI options
# - legacy @signature import/export bridge
# -----------------------------------------------------------------------------
try:
    _PLANNER_ORIG_MAKE_ENTRY = make_entry
    _PLANNER_ORIG_DEFAULT_PLAN = default_plan
    _PLANNER_ORIG_NORMALIZE_PLAN = normalize_plan
    _PLANNER_ORIG_EXPORT_TXT = export_plan_records_txt
    _PLANNER_ORIG_ENTRY_TO_LINES = _entry_to_lines
    _PLANNER_ORIG_COMMAND_SIGNATURES = _command_signatures
    _PLANNER_ORIG_LEGACY_SIGNATURE_TEXT = _legacy_signature_text
    _PLANNER_ORIG_ENTRY_REFS = _entry_reference_aliases
    _PLANNER_ORIG_ENTRY_PRODUCED = _entry_produced_aliases_for_cleanup
    _PLANNER_ORIG_PROGRESS_LABEL = _entry_progress_label
except Exception:
    pass


def _planner_default_cli_options(entry_type: str) -> Dict[str, Any]:
    if entry_type == "Checkpoint Merge":
        return {
            "m0_name": "", "m1_name": "", "m2_name": "",
            "use_dif_10": False, "use_dif_20": False, "use_dif_21": False,
            "rand_alpha": "", "rand_beta": "",
            "cosine0": False, "cosine1": False, "cosine2": False,
            "keep_ema": False, "delete_source": False, "no_metadata": False,
            "force": False, "turbo": False, "deturbo": False,
            "seed": "", "rebasin": "", "memo": "", "fine": "", "fine_sat": "",
            "cfg_sens": "", "cfg_sens_targets": "",
            "sat_boost": "", "sat_boost_side": "alpha", "sat_boost_tags": "",
            "sat_profile": "legacy", "sat_delta_cap_pct": "", "sat_boost_mix": "",
            "boost_clamp": "auto", "vae_sat": "",
        }
    if entry_type == "LoRA Bake":
        return {
            "dare": False, "keep_ema": False, "no_metadata": False,
            "memo": "", "bake_clip_scale": "", "bake_unet_only": False,
            "bake_norm": "sqrt", "bake_scale": "", "bake_rank_cap": "",
            "bake_clamp_q": "", "bake_delta_cap": "", "bake_fp32": False,
            "bake_guard": "auto", "bake_guard_cap": "", "bake_guard_skip": "",
            "bake_budget_report": False,
        }
    if entry_type == "LoRA Merge":
        return {
            "merge_rank": "64", "merge_arch": "auto", "merge_norm": "none",
            "merge_scale": "", "merge_unet_only": False,
            "merge_clamp_q": "", "merge_intermediate_mult": "",
            "no_metadata": False, "memo": "",
        }
    return {}


def _planner_clean_cli_options(entry_type: str, raw: Any) -> Dict[str, Any]:
    defaults = _planner_default_cli_options(entry_type)
    if not isinstance(raw, dict):
        raw = {}
    out = dict(defaults)
    for k, v in raw.items():
        if k in out:
            out[k] = v
    return out


def make_entry(entry_type: str = "Checkpoint Merge") -> Dict[str, Any]:
    if entry_type == "LoRA Merge":
        return {
            "id": _uid(),
            "type": "LoRA Merge",
            "loras": [],
            "output_name": "",
            "precision": "",
            "additional_signatures": "",
            "raw_signatures": "",
            "cli_options": _planner_default_cli_options("LoRA Merge"),
        }
    entry = _PLANNER_ORIG_MAKE_ENTRY(entry_type)
    if entry_type in ("Checkpoint Merge", "LoRA Bake"):
        entry.setdefault("cli_options", _planner_default_cli_options(entry_type))
    return entry


def default_plan() -> Dict[str, Any]:
    plan = _PLANNER_ORIG_DEFAULT_PLAN()
    plan.setdefault("final_memo", "")
    plan.setdefault("history", [])
    plan.setdefault("meta", {})
    return plan


def normalize_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    plan = default_plan()
    if isinstance(data, dict):
        plan["version"] = data.get("version", 2)
        plan["format"] = data.get("format", "planner-json")
        plan["final_memo"] = str(data.get("final_memo", plan.get("final_memo", "")) or "")
        hist = data.get("history", plan.get("history", []))
        plan["history"] = hist if isinstance(hist, list) else []
        meta = data.get("meta", {})
        if isinstance(meta, dict):
            plan["meta"] = meta
        plan["entries"] = []
        for raw in data.get("entries", []):
            if not isinstance(raw, dict):
                continue
            entry = make_entry(raw.get("type", "Checkpoint Merge"))
            entry.update(raw)
            entry.setdefault("id", _uid())
            if entry["type"] == "Checkpoint Merge":
                entry["alpha"] = _normalize_ratio_spec(entry.get("alpha"), allow_block_weight=True, default_single="0.5")
                entry["beta"] = _normalize_ratio_spec(entry.get("beta"), allow_block_weight=True, default_single="0.5")
                entry["cli_options"] = _planner_clean_cli_options("Checkpoint Merge", entry.get("cli_options"))
            if entry["type"] in ("LoRA Bake", "LoRA Merge"):
                entry.setdefault("loras", [])
                normalized_loras = []
                for lora in entry.get("loras", []):
                    if not isinstance(lora, dict):
                        continue
                    normalized_loras.append({
                        "name": lora.get("name", ""),
                        "ratio": _normalize_ratio_spec(lora.get("ratio"), allow_block_weight=False, default_single="1.0"),
                        **({"_source": lora.get("_source")} if isinstance(lora.get("_source"), dict) else {}),
                    })
                entry["loras"] = normalized_loras
                entry["cli_options"] = _planner_clean_cli_options(entry["type"], entry.get("cli_options"))
            plan["entries"].append(entry)
    if not plan["entries"]:
        plan["entries"] = [make_entry("Checkpoint Merge")]
    return plan


def _planner_quote_cli_value(value: Any) -> str:
    return shlex.quote(str(value))


_CLI_OPTION_SPECS = {
    "Checkpoint Merge": [
        ("value", "m0_name", "--m0_name", ""), ("value", "m1_name", "--m1_name", ""), ("value", "m2_name", "--m2_name", ""),
        ("flag", "use_dif_10", "--use_dif_10", False), ("flag", "use_dif_20", "--use_dif_20", False), ("flag", "use_dif_21", "--use_dif_21", False),
        ("value", "rand_alpha", "--rand_alpha", ""), ("value", "rand_beta", "--rand_beta", ""),
        ("flag", "cosine0", "--cosine0", False), ("flag", "cosine1", "--cosine1", False), ("flag", "cosine2", "--cosine2", False),
        ("flag", "keep_ema", "--keep_ema", False), ("flag", "delete_source", "--delete_source", False), ("flag", "no_metadata", "--no_metadata", False),
        ("flag", "force", "--force", False), ("flag", "turbo", "--turbo", False), ("flag", "deturbo", "--deturbo", False),
        ("value", "seed", "--seed", ""), ("value", "rebasin", "--rebasin", ""), ("value", "memo", "--memo", ""), ("value", "fine", "--fine", ""), ("value", "fine_sat", "--fine_sat", ""),
        ("value", "cfg_sens", "--cfg_sens", ""), ("value", "cfg_sens_targets", "--cfg_sens_targets", ""),
        ("value", "sat_boost", "--sat_boost", ""), ("choice", "sat_boost_side", "--sat_boost_side", "alpha"), ("value", "sat_boost_tags", "--sat_boost_tags", ""),
        ("choice", "sat_profile", "--sat_profile", "legacy"), ("value", "sat_delta_cap_pct", "--sat_delta_cap_pct", ""), ("value", "sat_boost_mix", "--sat_boost_mix", ""),
        ("choice", "boost_clamp", "--boost_clamp", "auto"), ("value", "vae_sat", "--vae_sat", ""),
    ],
    "LoRA Bake": [
        ("flag", "dare", "--dare", False), ("flag", "keep_ema", "--keep_ema", False), ("flag", "no_metadata", "--no_metadata", False),
        ("value", "memo", "--memo", ""), ("value", "bake_clip_scale", "--bake_clip_scale", ""), ("flag", "bake_unet_only", "--bake_unet_only", False),
        ("choice", "bake_norm", "--bake_norm", "sqrt"), ("value", "bake_scale", "--bake_scale", ""), ("value", "bake_rank_cap", "--bake_rank_cap", ""),
        ("value", "bake_clamp_q", "--bake_clamp_q", ""), ("value", "bake_delta_cap", "--bake_delta_cap", ""), ("flag", "bake_fp32", "--bake_fp32", False),
        ("choice", "bake_guard", "--bake_guard", "auto"), ("value", "bake_guard_cap", "--bake_guard_cap", ""), ("value", "bake_guard_skip", "--bake_guard_skip", ""),
        ("flag", "bake_budget_report", "--bake_budget_report", False),
    ],
    "LoRA Merge": [
        ("value", "merge_rank", "--merge_rank", "64"), ("choice", "merge_arch", "--merge_arch", "auto"), ("choice", "merge_norm", "--merge_norm", "none"),
        ("value", "merge_scale", "--merge_scale", ""), ("flag", "merge_unet_only", "--merge_unet_only", False),
        ("value", "merge_clamp_q", "--merge_clamp_q", ""), ("value", "merge_intermediate_mult", "--merge_intermediate_mult", ""),
        ("flag", "no_metadata", "--no_metadata", False), ("value", "memo", "--memo", ""),
    ],
}


def _structured_entry_signatures(entry: Dict[str, Any]) -> str:
    etype = str(entry.get("type") or "")
    opts = _planner_clean_cli_options(etype, entry.get("cli_options"))
    parts: List[str] = []
    for kind, key, flag, default in _CLI_OPTION_SPECS.get(etype, []):
        value = opts.get(key, default)
        if kind == "flag":
            if bool(value):
                parts.append(flag)
            continue
        text = str(value or "").strip()
        default_text = str(default or "").strip()
        if not text:
            continue
        if kind == "choice" and text == default_text:
            continue
        if kind == "value" and text == default_text and key in {"merge_rank"}:
            continue
        parts.append(f"{flag} {_planner_quote_cli_value(text)}")
    return " ".join(parts).strip()


def _command_signatures(entry: Dict[str, Any]) -> str:
    parts = []
    structured = _structured_entry_signatures(entry)
    if structured:
        parts.append(structured)
    additional = str(entry.get("additional_signatures") or "").strip()
    raw = str(entry.get("raw_signatures") or "").strip()
    if additional:
        parts.append(additional)
    elif raw:
        parts.append(_tail_text_to_cli_signatures(raw))
    return " ".join(x for x in parts if str(x).strip()).strip()


def _legacy_signature_text(entry: Dict[str, Any]) -> str:
    sig = _command_signatures(entry)
    precision = _normalize_precision_name(entry.get("precision"))
    if precision and precision != "half" and not _precision_from_signatures(sig):
        sig = f"{sig} @p {precision}".strip()
    return sig


def _lora_items_lines_from_entry(entry: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
    parts = []
    for lora in entry.get("loras") or []:
        name = str(lora.get("name") or "").strip()
        if not name:
            continue
        ratio = _normalize_ratio_spec(lora.get("ratio"), allow_block_weight=False, default_single="1.0")["value"] or "1.0"
        parts.append((name, ratio))
    lines = ['lora_items = []']
    for name, ratio in parts:
        safe_ratio = str(ratio).replace('\\', '\\\\').replace('"', '\\"')
        lines.append('lora_items.append(f"{model_file(' + _json_literal(name).replace('"', "'") + ')}:" + "' + safe_ratio + '")')
    return lines, parts


def _entry_to_lines(entry: Dict[str, Any], *, bake_vae: bool = True) -> Tuple[List[str], str | None]:
    if entry.get("type") != "LoRA Merge":
        return _PLANNER_ORIG_ENTRY_TO_LINES(entry, bake_vae=bake_vae)
    lines: List[str] = []
    output_name = str(entry.get("output_name") or "").strip()
    if not output_name:
        return [], None
    lora_lines, parts = _lora_items_lines_from_entry(entry)
    if not parts:
        return [], None
    command_signatures = _command_signatures(entry)
    precision_args = _precision_args(entry, command_signatures)
    lines.extend(lora_lines)
    lines.extend([
        'cmd = [sys.executable, "lora_bake.py", models_dir + "/", "", ",".join(lora_items), "--merge_loras"]',
        'cmd += ' + precision_args,
        'cmd += ["--output", ' + _json_literal(output_name) + ']',
        'cmd += shlex.split(' + _json_literal(command_signatures) + ')' if command_signatures else 'cmd += []',
        'run_cmd(cmd, cwd=merge_repo_dir, check_path=True, path=os.path.join(models_dir, ' + _json_literal(f"{output_name}.safetensors") + '), ignore_meta=True)',
        'register_model(' + _json_literal(output_name) + ', os.path.join(models_dir, ' + _json_literal(f"{output_name}.safetensors") + '), "lora", source_kind="generated")',
        'flush()',
    ])
    return lines, output_name


def export_plan_records_txt(filepath: str, plan: Dict[str, Any]) -> None:
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


def _entry_reference_aliases(entry: Dict[str, Any]) -> List[str]:
    if entry.get("type") != "LoRA Merge":
        return _PLANNER_ORIG_ENTRY_REFS(entry)
    refs: List[str] = []
    for lora in entry.get("loras", []) or []:
        name = str(lora.get("name") or "").strip()
        if name:
            refs.append(name)
    return refs


def _entry_produced_aliases_for_cleanup(entry: Dict[str, Any]) -> List[str]:
    if entry.get("type") == "LoRA Merge":
        name = str(entry.get("output_name") or "").strip()
        return [f"TEMP{name}" if name.startswith("_") else name] if name else []
    return _PLANNER_ORIG_ENTRY_PRODUCED(entry)


def _entry_progress_label(entry: Dict[str, Any], index: int, total: int) -> str:
    if entry.get("type") == "LoRA Merge":
        label = str(entry.get("output_name") or "lora merge")
        if label.startswith("_"):
            label = f"TEMP{label}"
        return f"[planner-progress] {index}/{total} | LoRA Merge | {label}"
    return _PLANNER_ORIG_PROGRESS_LABEL(entry, index, total)


# -----------------------------------------------------------------------------
# Signature-less structured options / legacy @-signature bridge
# -----------------------------------------------------------------------------
try:
    _SIGLESS_PREV_MAKE_ENTRY = make_entry
    _SIGLESS_PREV_NORMALIZE_PLAN = normalize_plan
    _SIGLESS_PREV_DEFAULT_CLI_OPTIONS = _planner_default_cli_options
    _SIGLESS_PREV_CLEAN_CLI_OPTIONS = _planner_clean_cli_options
    _SIGLESS_PREV_STRUCTURED_SIGNATURES = _structured_entry_signatures
    _SIGLESS_PREV_COMMAND_SIGNATURES = _command_signatures
    _SIGLESS_PREV_LEGACY_SIGNATURE_TEXT = _legacy_signature_text
    _SIGLESS_PREV_RATIO_TEXT = _ratio_text
except Exception:
    pass

_PRECISION_CHOICES = {"half", "bhalf", "bf16", "bfloat16", "quarter", "fp8", "float8", "fp32", "float32", "full"}
_ARCH_CHOICES = {"auto", "sd", "sd15", "sd1.5", "sdxl", "xl", "flux", "zimage", "zi", "anima", "am"}

def _planner_default_cli_options(entry_type: str) -> Dict[str, Any]:
    if entry_type == "Checkpoint Merge":
        return {
            "m0_name": "", "m1_name": "", "m2_name": "",
            "use_dif_10": False, "use_dif_20": False, "use_dif_21": False,
            "cosine0": False, "cosine1": False, "cosine2": False,
            "keep_ema": False, "delete_source": False, "no_metadata": False,
            "force": False, "turbo": False, "deturbo": False,
            "seed": "", "rebasin": "", "memo": "", "fine": "", "fine_sat": "",
            "cfg_sens": "", "cfg_sens_targets": "",
            "sat_boost": "", "sat_boost_side": "alpha", "sat_boost_tags": "",
            "sat_profile": "legacy", "sat_delta_cap_pct": "", "sat_boost_mix": "",
            "boost_clamp": "auto", "vae_sat": "",
        }
    if entry_type == "LoRA Bake":
        return {
            "dare": False, "keep_ema": False, "no_metadata": False,
            "memo": "", "bake_clip_scale": "", "bake_unet_only": False,
            "bake_norm": "sqrt", "bake_scale": "", "bake_rank_cap": "",
            "bake_clamp_q": "", "bake_delta_cap": "", "bake_fp32": False,
            "bake_guard": "auto", "bake_guard_cap": "", "bake_guard_skip": "",
            "bake_budget_report": False,
        }
    if entry_type == "LoRA Merge":
        return {
            "merge_rank": "64", "merge_arch": "auto", "merge_norm": "none",
            "merge_scale": "", "merge_unet_only": False,
            "merge_clamp_q": "", "merge_intermediate_mult": "",
            "no_metadata": False, "memo": "",
        }
    return {}


def _planner_clean_cli_options(entry_type: str, raw: Any) -> Dict[str, Any]:
    defaults = _planner_default_cli_options(entry_type)
    if not isinstance(raw, dict):
        raw = {}
    out = dict(defaults)
    for k, v in raw.items():
        if k in out:
            out[k] = v
    return out


def make_entry(entry_type: str = "Checkpoint Merge") -> Dict[str, Any]:
    entry = _SIGLESS_PREV_MAKE_ENTRY(entry_type)
    if entry_type in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
        entry.setdefault("cli_options", _planner_default_cli_options(entry_type))
        entry.setdefault("architecture", "")
        entry.setdefault("unmapped_signatures", "")
        entry["additional_signatures"] = ""
        entry["raw_signatures"] = ""
    return entry


def _ratio_text(spec: Dict[str, Any] | None) -> str:
    spec = _normalize_ratio_spec(spec, allow_block_weight=True, default_single="0.5")
    mode = str(spec.get("mode", "Single"))
    value = str(spec.get("value", "")).strip()
    if mode == "Randomize":
        return "@r " + (value.strip("\"'") or "0.0,1.0")
    if mode == "Single":
        return value or "0.5"
    value = f'"{value[0].strip("\'\"")}{value[1:-1]}{value[-1].strip("\'\"")}"' if value else '""'
    return value


def _sigless_tokenize(text: str) -> List[str]:
    try:
        return shlex.split((text or "").replace("\n", " "))
    except Exception:
        return str(text or "").replace("\n", " ").split()


def _sigless_read_value(tokens: List[str], i: int) -> Tuple[str | None, int]:
    if i + 1 < len(tokens) and not str(tokens[i + 1]).startswith("@") and not str(tokens[i + 1]).startswith("--"):
        return str(tokens[i + 1]), i + 2
    return None, i + 1


def _sigless_option_maps(entry_type: str):
    specs = _CLI_OPTION_SPECS.get(entry_type, [])
    by_flag = {}
    by_key = {}
    for kind, key, flag, default in specs:
        by_flag[flag.lstrip("-").lower()] = (kind, key, flag, default)
        by_key[key.lower()] = (kind, key, flag, default)
    return by_flag, by_key


def _sigless_apply_legacy_signatures(entry: Dict[str, Any]) -> None:
    etype = str(entry.get("type") or "")
    if etype not in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
        return
    opts = _planner_clean_cli_options(etype, entry.get("cli_options"))
    by_flag, by_key = _sigless_option_maps(etype)
    raw_primary = str(entry.get("raw_signatures") or "").strip()
    additional_primary = str(entry.get("additional_signatures") or "").strip()
    unmapped_primary = str(entry.get("unmapped_signatures") or "").strip()
    # Legacy import stores both raw @-tokens and a derived CLI tail; prefer raw when present
    # to avoid duplicating unknown tokens as both @foo and --foo.
    raw_text = " ".join(x for x in [raw_primary or additional_primary, unmapped_primary] if x)
    tokens = _sigless_tokenize(raw_text)
    unmapped: List[str] = []
    i = 0
    while i < len(tokens):
        tok = str(tokens[i])
        original = tok
        if tok.startswith("@"):
            key = tok[1:].lower()
            value, ni = _sigless_read_value(tokens, i)
            if key in ("m", "mode") and value and etype == "Checkpoint Merge":
                entry["merge_mode"] = value.upper()
            elif key in ("p", "precision") and value:
                entry["precision"] = _normalize_precision_name(value) or value
            elif key in ("a", "arch", "architecture") and value:
                entry["architecture"] = value.lower()
            elif key in ("c", "cosine") and value is not None and etype == "Checkpoint Merge":
                for n in ("0", "1", "2"):
                    opts[f"cosine{n}"] = (str(value) == n)
            elif key in ("cosine0", "c0") and etype == "Checkpoint Merge":
                opts["cosine0"] = True
            elif key in ("cosine1", "c1") and etype == "Checkpoint Merge":
                opts["cosine1"] = True
            elif key in ("cosine2", "c2") and etype == "Checkpoint Merge":
                opts["cosine2"] = True
            elif key in ("s", "seed") and value is not None and etype == "Checkpoint Merge":
                opts["seed"] = value
            elif key in ("f", "fine") and value is not None and etype == "Checkpoint Merge":
                opts["fine"] = value
            elif key in ("rand_alpha", "ra") and value is not None and etype == "Checkpoint Merge":
                entry["alpha"] = {"mode": "Randomize", "value": str(value).strip("\"'")}
            elif key in ("rand_beta", "rb") and value is not None and etype == "Checkpoint Merge":
                entry["beta"] = {"mode": "Randomize", "value": str(value).strip("\"'")}
            elif key in by_key:
                kind, opt_key, _flag, default = by_key[key]
                if kind == "flag":
                    opts[opt_key] = True
                elif value is not None:
                    opts[opt_key] = value
                else:
                    unmapped.append(original)
            else:
                # Unknown @token is intentionally preserved for the Options panel.
                if value is None:
                    unmapped.append(original)
                else:
                    unmapped.append(f"{original} {shlex.quote(str(value))}")
            i = ni
            continue
        if tok.startswith("--"):
            key = tok[2:].replace("-", "_").lower()
            value, ni = _sigless_read_value(tokens, i)
            if key in ("rand_alpha", "ra") and value is not None and etype == "Checkpoint Merge":
                entry["alpha"] = {"mode": "Randomize", "value": str(value).strip("\"'")}
            elif key in ("rand_beta", "rb") and value is not None and etype == "Checkpoint Merge":
                entry["beta"] = {"mode": "Randomize", "value": str(value).strip("\"'")}
            elif key in ("arch", "architecture") and value:
                entry["architecture"] = value.lower()
            elif key in by_flag:
                kind, opt_key, _flag, default = by_flag[key]
                if kind == "flag":
                    opts[opt_key] = True
                elif value is not None:
                    opts[opt_key] = value
                else:
                    unmapped.append(original)
            else:
                if value is None:
                    unmapped.append(original)
                else:
                    unmapped.append(f"{original} {shlex.quote(str(value))}")
            i = ni
            continue
        unmapped.append(original)
        i += 1
    entry["cli_options"] = _planner_clean_cli_options(etype, opts)
    entry["unmapped_signatures"] = " ".join(unmapped).strip()
    entry["additional_signatures"] = ""
    entry["raw_signatures"] = ""


def normalize_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    plan = _SIGLESS_PREV_NORMALIZE_PLAN(data)
    for entry in plan.get("entries", []) or []:
        if not isinstance(entry, dict):
            continue
        etype = str(entry.get("type") or "")
        if etype in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
            entry.setdefault("architecture", "")
            entry.setdefault("unmapped_signatures", "")
            entry["cli_options"] = _planner_clean_cli_options(etype, entry.get("cli_options"))
            _sigless_apply_legacy_signatures(entry)
            if etype == "Checkpoint Merge":
                entry["alpha"] = _normalize_ratio_spec(entry.get("alpha"), allow_block_weight=True, default_single="0.5")
                entry["beta"] = _normalize_ratio_spec(entry.get("beta"), allow_block_weight=True, default_single="0.5")
                for _rk in ("alpha", "beta"):
                    if str(entry[_rk].get("mode") or "") == "Randomize":
                        entry[_rk]["value"] = str(entry[_rk].get("value") or "").strip("\"'")
    return plan


_CLI_OPTION_SPECS = {
    "Checkpoint Merge": [
        ("value", "m0_name", "--m0_name", ""), ("value", "m1_name", "--m1_name", ""), ("value", "m2_name", "--m2_name", ""),
        ("flag", "use_dif_10", "--use_dif_10", False), ("flag", "use_dif_20", "--use_dif_20", False), ("flag", "use_dif_21", "--use_dif_21", False),
        ("flag", "cosine0", "--cosine0", False), ("flag", "cosine1", "--cosine1", False), ("flag", "cosine2", "--cosine2", False),
        ("flag", "keep_ema", "--keep_ema", False), ("flag", "delete_source", "--delete_source", False), ("flag", "no_metadata", "--no_metadata", False),
        ("flag", "force", "--force", False), ("flag", "turbo", "--turbo", False), ("flag", "deturbo", "--deturbo", False),
        ("value", "seed", "--seed", ""), ("value", "rebasin", "--rebasin", ""), ("value", "memo", "--memo", ""), ("value", "fine", "--fine", ""), ("value", "fine_sat", "--fine_sat", ""),
        ("value", "cfg_sens", "--cfg_sens", ""), ("value", "cfg_sens_targets", "--cfg_sens_targets", ""),
        ("value", "sat_boost", "--sat_boost", ""), ("choice", "sat_boost_side", "--sat_boost_side", "alpha"), ("value", "sat_boost_tags", "--sat_boost_tags", ""),
        ("choice", "sat_profile", "--sat_profile", "legacy"), ("value", "sat_delta_cap_pct", "--sat_delta_cap_pct", ""), ("value", "sat_boost_mix", "--sat_boost_mix", ""),
        ("choice", "boost_clamp", "--boost_clamp", "auto"), ("value", "vae_sat", "--vae_sat", ""),
    ],
    "LoRA Bake": [
        ("flag", "dare", "--dare", False), ("flag", "keep_ema", "--keep_ema", False), ("flag", "no_metadata", "--no_metadata", False),
        ("value", "memo", "--memo", ""), ("value", "bake_clip_scale", "--bake_clip_scale", ""), ("flag", "bake_unet_only", "--bake_unet_only", False),
        ("choice", "bake_norm", "--bake_norm", "sqrt"), ("value", "bake_scale", "--bake_scale", ""), ("value", "bake_rank_cap", "--bake_rank_cap", ""),
        ("value", "bake_clamp_q", "--bake_clamp_q", ""), ("value", "bake_delta_cap", "--bake_delta_cap", ""), ("flag", "bake_fp32", "--bake_fp32", False),
        ("choice", "bake_guard", "--bake_guard", "auto"), ("value", "bake_guard_cap", "--bake_guard_cap", ""), ("value", "bake_guard_skip", "--bake_guard_skip", ""),
        ("flag", "bake_budget_report", "--bake_budget_report", False),
    ],
    "LoRA Merge": [
        ("value", "merge_rank", "--merge_rank", "64"), ("choice", "merge_arch", "--merge_arch", "auto"), ("choice", "merge_norm", "--merge_norm", "none"),
        ("value", "merge_scale", "--merge_scale", ""), ("flag", "merge_unet_only", "--merge_unet_only", False),
        ("value", "merge_clamp_q", "--merge_clamp_q", ""), ("value", "merge_intermediate_mult", "--merge_intermediate_mult", ""),
        ("flag", "no_metadata", "--no_metadata", False), ("value", "memo", "--memo", ""),
    ],
}


def _structured_entry_signatures(entry: Dict[str, Any]) -> str:
    etype = str(entry.get("type") or "")
    opts = _planner_clean_cli_options(etype, entry.get("cli_options"))
    parts: List[str] = []
    # Architecture is kept as a structured option and legacy @arch bridge for
    # Checkpoint Merge / LoRA Bake. The uploaded merge.py and bake path do not
    # expose a generic --arch argument, so it is not emitted into runtime cmd here.
    for kind, key, flag, default in _CLI_OPTION_SPECS.get(etype, []):
        value = opts.get(key, default)
        if kind == "flag":
            if bool(value):
                parts.append(flag)
            continue
        text = str(value or "").strip()
        default_text = str(default or "").strip()
        if not text:
            continue
        if kind == "choice" and text == default_text:
            continue
        if kind == "value" and text == default_text and key in {"merge_rank"}:
            continue
        parts.append(f"{flag} {_planner_quote_cli_value(text)}")
    return " ".join(parts).strip()


def _command_signatures(entry: Dict[str, Any]) -> str:
    parts = []
    structured = _structured_entry_signatures(entry)
    if structured:
        parts.append(structured)
    unmapped = str(entry.get("unmapped_signatures") or "").strip()
    if unmapped:
        parts.append(_tail_text_to_cli_signatures(unmapped) if unmapped.lstrip().startswith("@") else unmapped)
    return " ".join(x for x in parts if str(x).strip()).strip()


def _legacy_sig_quote(value: Any) -> str:
    text = str(value)
    return shlex.quote(text)


def _structured_entry_legacy_signatures(entry: Dict[str, Any]) -> str:
    etype = str(entry.get("type") or "")
    opts = _planner_clean_cli_options(etype, entry.get("cli_options"))
    parts: List[str] = []
    arch = str(entry.get("architecture") or "").strip()
    if arch and arch.lower() != "auto" and etype in ("Checkpoint Merge", "LoRA Bake"):
        parts.append(f"@arch {_legacy_sig_quote(arch)}")
    # Prefer compact @c for cosine routing.
    if etype == "Checkpoint Merge":
        for n in ("0", "1", "2"):
            if opts.get(f"cosine{n}"):
                parts.append(f"@c {n}")
        skip_keys = {"cosine0", "cosine1", "cosine2"}
    else:
        skip_keys = set()
    for kind, key, flag, default in _CLI_OPTION_SPECS.get(etype, []):
        if key in skip_keys:
            continue
        value = opts.get(key, default)
        legacy_name = flag.lstrip("-").replace("-", "_")
        if kind == "flag":
            if bool(value):
                parts.append(f"@{legacy_name}")
            continue
        text = str(value or "").strip()
        default_text = str(default or "").strip()
        if not text:
            continue
        if kind == "choice" and text == default_text:
            continue
        if kind == "value" and text == default_text and key in {"merge_rank"}:
            continue
        parts.append(f"@{legacy_name} {_legacy_sig_quote(text)}")
    return " ".join(parts).strip()


def _legacy_signature_text(entry: Dict[str, Any]) -> str:
    parts = []
    structured = _structured_entry_legacy_signatures(entry)
    if structured:
        parts.append(structured)
    precision = _normalize_precision_name(entry.get("precision"))
    if precision and precision != "half":
        parts.append(f"@p {precision}")
    unmapped = str(entry.get("unmapped_signatures") or "").strip()
    if unmapped:
        parts.append(unmapped)
    return " ".join(parts).strip()


# -----------------------------------------------------------------------------
# Final arch/precision display bridge
# UI may store user-friendly values (SDXL/ZImage/Anima, FP16/BF16/FP8/FP32),
# while command/legacy output must use backend-compatible tokens.
# -----------------------------------------------------------------------------
_PRECISION_RUNTIME_ALIASES = {
    "": "", "half": "half", "fp16": "half", "float16": "half", "16": "half",
    "bhalf": "bhalf", "bf16": "bhalf", "bfloat16": "bhalf",
    "quarter": "quarter", "fp8": "quarter", "float8": "quarter", "8": "quarter",
    "fp32": "fp32", "float32": "fp32", "full": "fp32", "32": "fp32",
}
_PRECISION_DISPLAY_ALIASES = {"": "", "half": "FP16", "bhalf": "BF16", "quarter": "FP8", "fp32": "FP32"}
_ARCH_RUNTIME_ALIASES = {
    "": "", "auto": "auto",
    "sd": "sd", "sd15": "sd", "sd1.5": "sd", "sd-1.5": "sd", "sd_1.5": "sd",
    "xl": "sdxl", "sdxl": "sdxl", "sd-xl": "sdxl", "sd_xl": "sdxl",
    "flux": "flux",
    "zi": "zi", "zimage": "zi", "z-image": "zi", "z_image": "zi",
    "am": "am", "anima": "am",
}
_ARCH_DISPLAY_ALIASES = {"": "", "auto": "Auto", "sd": "SD1.5", "sdxl": "SDXL", "flux": "Flux", "zi": "ZImage", "am": "Anima"}

def _normalize_precision_name(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "")
    return _PRECISION_RUNTIME_ALIASES.get(key, key)

def _display_precision_name(value: Any) -> str:
    return _PRECISION_DISPLAY_ALIASES.get(_normalize_precision_name(value), str(value or "").upper() if value else "")

def _normalize_arch_runtime(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "").replace("_", "-")
    return _ARCH_RUNTIME_ALIASES.get(key, key)

def _display_arch_name(value: Any) -> str:
    runtime = _normalize_arch_runtime(value)
    if runtime in _ARCH_DISPLAY_ALIASES:
        return _ARCH_DISPLAY_ALIASES[runtime]
    raw = str(value or "").strip()
    return raw[:1].upper() + raw[1:] if raw else ""

_FINAL_PREV_PLANNER_CLEAN_CLI_OPTIONS = _planner_clean_cli_options
def _planner_clean_cli_options(entry_type: str, raw: Any) -> Dict[str, Any]:
    out = _FINAL_PREV_PLANNER_CLEAN_CLI_OPTIONS(entry_type, raw)
    if entry_type == "LoRA Merge" and "merge_arch" in out:
        out["merge_arch"] = _display_arch_name(out.get("merge_arch") or "Auto")
    return out

_FINAL_PREV_SIGLESS_APPLY_LEGACY = _sigless_apply_legacy_signatures
def _sigless_apply_legacy_signatures(entry: Dict[str, Any]) -> None:
    _FINAL_PREV_SIGLESS_APPLY_LEGACY(entry)
    etype = str(entry.get("type") or "")
    if etype in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
        if entry.get("architecture"):
            entry["architecture"] = _display_arch_name(entry.get("architecture"))
        if entry.get("precision"):
            entry["precision"] = _display_precision_name(entry.get("precision"))
        if isinstance(entry.get("cli_options"), dict) and entry.get("type") == "LoRA Merge":
            entry["cli_options"]["merge_arch"] = _display_arch_name(entry["cli_options"].get("merge_arch") or "Auto")

_FINAL_PREV_NORMALIZE_PLAN = normalize_plan
def normalize_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    plan = _FINAL_PREV_NORMALIZE_PLAN(data)
    for entry in plan.get("entries", []) or []:
        if not isinstance(entry, dict):
            continue
        etype = str(entry.get("type") or "")
        if etype in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
            entry["architecture"] = _display_arch_name(entry.get("architecture"))
            entry["precision"] = _display_precision_name(entry.get("precision"))
            entry["cli_options"] = _planner_clean_cli_options(etype, entry.get("cli_options"))
    return plan

def _structured_entry_signatures(entry: Dict[str, Any]) -> str:
    etype = str(entry.get("type") or "")
    opts = _planner_clean_cli_options(etype, entry.get("cli_options"))
    parts: List[str] = []
    for kind, key, flag, default in _CLI_OPTION_SPECS.get(etype, []):
        value = opts.get(key, default)
        if key == "merge_arch":
            text = _normalize_arch_runtime(value or default)
            default_text = _normalize_arch_runtime(default)
        else:
            text = str(value or "").strip()
            default_text = str(default or "").strip()
        if kind == "flag":
            if bool(value):
                parts.append(flag)
            continue
        if not text:
            continue
        if kind == "choice" and text == default_text:
            continue
        if kind == "value" and text == default_text and key in {"merge_rank"}:
            continue
        parts.append(f"{flag} {_planner_quote_cli_value(text)}")
    return " ".join(parts).strip()

def _structured_entry_legacy_signatures(entry: Dict[str, Any]) -> str:
    etype = str(entry.get("type") or "")
    opts = _planner_clean_cli_options(etype, entry.get("cli_options"))
    parts: List[str] = []
    arch_runtime = _normalize_arch_runtime(entry.get("architecture"))
    if arch_runtime and arch_runtime != "auto" and etype in ("Checkpoint Merge", "LoRA Bake"):
        parts.append(f"@arch {_legacy_sig_quote(arch_runtime)}")
    if etype == "Checkpoint Merge":
        for n in ("0", "1", "2"):
            if opts.get(f"cosine{n}"):
                parts.append(f"@c {n}")
        skip_keys = {"cosine0", "cosine1", "cosine2"}
    else:
        skip_keys = set()
    for kind, key, flag, default in _CLI_OPTION_SPECS.get(etype, []):
        if key in skip_keys:
            continue
        value = opts.get(key, default)
        legacy_name = flag.lstrip("-").replace("-", "_")
        if key == "merge_arch":
            text = _normalize_arch_runtime(value or default)
            default_text = _normalize_arch_runtime(default)
        else:
            text = str(value or "").strip()
            default_text = str(default or "").strip()
        if kind == "flag":
            if bool(value):
                parts.append(f"@{legacy_name}")
            continue
        if not text:
            continue
        if kind == "choice" and text == default_text:
            continue
        if kind == "value" and text == default_text and key in {"merge_rank"}:
            continue
        parts.append(f"@{legacy_name} {_legacy_sig_quote(text)}")
    return " ".join(parts).strip()

def _legacy_signature_text(entry: Dict[str, Any]) -> str:
    parts = []
    structured = _structured_entry_legacy_signatures(entry)
    if structured:
        parts.append(structured)
    precision = _normalize_precision_name(entry.get("precision"))
    if precision and precision != "half":
        parts.append(f"@p {precision}")
    unmapped = str(entry.get("unmapped_signatures") or "").strip()
    if unmapped:
        parts.append(unmapped)
    return " ".join(parts).strip()


# FP32 is the default full precision path in merge.py/lora_bake.py; there is no
# --save_full CLI flag, so emit no save_* precision flag for FP32.
def _precision_args(entry: Dict[str, Any], command_signatures: str = "") -> str:
    p = _entry_precision(entry, command_signatures=command_signatures)
    args = ["--prune", "--save_safetensors"]
    if p in ("bhalf", "bf16", "bfloat16"):
        args.insert(0, "--save_bhalf")
    elif p in ("quarter", "fp8", "float8"):
        args.insert(0, "--save_quarter")
    elif p in ("half", "fp16", "float16"):
        args.insert(0, "--save_half")
    return _json_literal(args)


# -----------------------------------------------------------------------------
# Advanced project metadata compatibility patch
# Preserve planner-side project extensions when plan.py normalizes project JSON.
# -----------------------------------------------------------------------------
try:
    _tccm_adv_default_plan_base = default_plan
    def default_plan() -> Dict[str, Any]:
        plan = _tccm_adv_default_plan_base()
        plan.setdefault("variants", [])
        plan.setdefault("run_profiles", [])
        plan.setdefault("current_variant", "")
        plan.setdefault("schema_version", 3)
        return plan

    _tccm_adv_normalize_plan_base = normalize_plan
    def normalize_plan(data: Dict[str, Any]) -> Dict[str, Any]:
        plan = _tccm_adv_normalize_plan_base(data)
        if isinstance(data, dict):
            for key in ("variants", "run_profiles", "current_variant", "runtime", "schema_version", "migrated_from", "migrated_at"):
                if key in data:
                    plan[key] = data.get(key)
        plan.setdefault("variants", [])
        plan.setdefault("run_profiles", [])
        plan.setdefault("current_variant", "")
        plan.setdefault("schema_version", 3)
        return plan
except Exception:
    pass
