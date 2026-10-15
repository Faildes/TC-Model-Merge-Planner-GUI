"""Options responsibilities."""
from __future__ import annotations

from .constants import _ARCH_DISPLAY_ALIASES
from .constants import _ARCH_RUNTIME_ALIASES
from .constants import _CLI_OPTION_SPECS
from .constants import _PRECISION_DISPLAY_ALIASES
from .constants import _PRECISION_RUNTIME_ALIASES
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import shlex


def _tail_text_to_cli_signatures(raw_text: str) -> str:
    from .syntax import _needs_quote
    from .syntax import _parse_tail_at

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




def _precision_from_signatures(text: str) -> str:
    from .syntax import _parse_tail_at

    tokens = shlex.split((text or "").replace("\n", " "))
    tail = _parse_tail_at(tokens)
    precision = _normalize_precision_name(tail.get("precision"))
    if precision:
        return precision
    for i, token in enumerate(tokens):
        low = str(token).lower()
        if low.startswith("--save_precision=") or low.startswith("--save-precision="):
            return _normalize_precision_name(str(token).split("=", 1)[1])
        if low in ("--save_precision", "--save-precision") and i + 1 < len(tokens):
            return _normalize_precision_name(tokens[i + 1])
        legacy = {"--save_half": "half", "--save_bhalf": "bhalf", "--save_quarter": "quarter", "--save_int8": "int8", "--save_full": "fp32"}
        if low in legacy:
            return legacy[low]
    return ""


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


def _canonical_save_precision(value: Any) -> str:
    p = _normalize_precision_name(value)
    return {"half": "fp16", "bhalf": "bf16", "quarter": "fp8", "int8": "int8", "fp32": "fp32"}.get(p, "fp16")








def _structured_entry_signatures(entry: Dict[str, Any]) -> str:
    from .schema import _planner_quote_cli_value

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


def _default_cli_options_base(entry_type: str) -> Dict[str, Any]:
    if entry_type == "Checkpoint Merge":
        return {
            "m0_name": "", "m1_name": "", "m2_name": "",
            "use_dif_10": False, "use_dif_20": False, "use_dif_21": False,
            "cosine0": False, "cosine1": False, "cosine2": False, "clipxor": False,
            "keep_ema": False, "delete_source": False, "no_metadata": False,
            "force": False, "turbo": False, "deturbo": False, "turbo_method": "delta",
            "seed": "", "rebasin": "", "memo": "", "save_component": "", "fine": "", "fine_sat": "",
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
            "anima_lora_source": "auto", "anima_inserted": "original", "anima_inserted_scale": "",
        }
    if entry_type == "LoRA Merge":
        return {
            "merge_rank": "64", "merge_arch": "auto", "merge_norm": "none",
            "merge_scale": "", "merge_unet_only": False,
            "merge_clamp_q": "", "merge_intermediate_mult": "",
            "no_metadata": False, "memo": "",
        }
    return {}


def _clean_cli_options_structured(entry_type: str, raw: Any) -> Dict[str, Any]:
    defaults = _planner_default_cli_options(entry_type)
    if not isinstance(raw, dict):
        raw = {}
    out = dict(defaults)
    for k, v in raw.items():
        if k in out:
            out[k] = v
    return out


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
            option_text = tok[2:]
            inline_value = None
            if "=" in option_text:
                option_text, inline_value = option_text.split("=", 1)
            key = option_text.replace("-", "_").lower()
            if inline_value is None:
                value, ni = _sigless_read_value(tokens, i)
            else:
                value, ni = inline_value, i + 1
            if key in ("save_precision", "saveprecision") and value is not None:
                entry["precision"] = _normalize_precision_name(value) or value
            elif key in ("save_half",):
                entry["precision"] = "half"
            elif key in ("save_bhalf",):
                entry["precision"] = "bhalf"
            elif key in ("save_quarter",):
                entry["precision"] = "quarter"
            elif key in ("save_int8",):
                entry["precision"] = "int8"
            elif key in ("save_full",):
                entry["precision"] = "fp32"
            elif key in ("rand_alpha", "ra") and value is not None and etype == "Checkpoint Merge":
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


def _command_signatures_base(entry: Dict[str, Any]) -> str:
    from .options import _structured_entry_signatures_base as _structured_entry_signatures

    parts = []
    structured = _structured_entry_signatures(entry)
    if structured:
        parts.append(structured)
    unmapped = str(entry.get("unmapped_signatures") or "").strip()
    if unmapped:
        parts.append(_tail_text_to_cli_signatures(unmapped) if unmapped.lstrip().startswith("@") else unmapped)
    return " ".join(x for x in parts if str(x).strip()).strip()


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


def _planner_clean_cli_options(entry_type: str, raw: Any) -> Dict[str, Any]:
    from .options import _clean_cli_options_structured as _FINAL_PREV_PLANNER_CLEAN_CLI_OPTIONS

    out = _FINAL_PREV_PLANNER_CLEAN_CLI_OPTIONS(entry_type, raw)
    if entry_type == "LoRA Merge" and "merge_arch" in out:
        out["merge_arch"] = _display_arch_name(out.get("merge_arch") or "Auto")
    return out


def _sigless_apply_legacy_signatures_base(entry: Dict[str, Any]) -> None:
    from .options import _sigless_apply_legacy_signatures as _FINAL_PREV_SIGLESS_APPLY_LEGACY

    _FINAL_PREV_SIGLESS_APPLY_LEGACY(entry)
    etype = str(entry.get("type") or "")
    if etype in ("Checkpoint Merge", "LoRA Bake", "LoRA Merge"):
        if entry.get("architecture"):
            entry["architecture"] = _display_arch_name(entry.get("architecture"))
        if entry.get("precision"):
            entry["precision"] = _display_precision_name(entry.get("precision"))
        if isinstance(entry.get("cli_options"), dict) and entry.get("type") == "LoRA Merge":
            entry["cli_options"]["merge_arch"] = _display_arch_name(entry["cli_options"].get("merge_arch") or "Auto")




def _structured_entry_signatures_base(entry: Dict[str, Any]) -> str:
    from .schema import _planner_quote_cli_value

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
    from .formats import _legacy_sig_quote

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


def _legacy_signature_text_base(entry: Dict[str, Any]) -> str:
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


def _precision_args(entry: Dict[str, Any], command_signatures: str = "") -> str:
    from .compiler import _json_literal

    precision = _canonical_save_precision(_entry_precision(entry, command_signatures=command_signatures))
    return _json_literal(["--save_precision", precision, "--prune", "--save_safetensors"])


def _planner_default_cli_options(entry_type: str) -> Dict[str, Any]:
    from .options import _default_cli_options_base as _TCCM_OPTIONIZED_PREV_DEFAULT_CLI_OPTIONS

    opts = _TCCM_OPTIONIZED_PREV_DEFAULT_CLI_OPTIONS(entry_type)
    if entry_type == "Checkpoint Merge":
        opts.setdefault("clipxor", False)
        opts.setdefault("save_component", "")
    return opts
