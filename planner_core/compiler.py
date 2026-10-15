"""Compiler responsibilities."""
from __future__ import annotations

from .templates import INSTALL_TPL
from .templates import PRELUDE_TPL
from .templates import T2I_CFG_TPL
from .templates import T2I_RUN_TPL
from .templates import UPLOAD_TPL
from .templates import ZIP_TPL
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
import json


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _entry_to_lines(entry: Dict[str, Any], *, bake_vae: bool = True) -> Tuple[List[str], str | None]:
    from .errors import PlanCompileError
    from .options import _command_signatures_base as _command_signatures
    from .schema import _normalize_ratio_spec
    from .options import _precision_args
    from .schema import default_ratio

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
        from .merge_validation import required_model_slots, turbo_problem
        issue = turbo_problem(entry)
        if issue:
            raise PlanCompileError(issue, entry_type=etype, entry_id=entry.get("id"))
        required = required_model_slots(entry)
        opts = entry.get("cli_options") or {}
        if (opts.get("turbo") or opts.get("deturbo")) and any(not entry.get(slot) for slot in required):
            raise PlanCompileError("Turbo requires Model 0 (target), Model 1 (Turbo) and Model 2 (Base)", entry_type=etype)
        if model0 and (model1 or "model1" not in required) and output_name:
            command_signatures = _command_signatures(entry)
            precision_args = _precision_args(entry, command_signatures)
            lines.extend([
                f'beta = {entry.get("beta","") != ""}',
                'cmd = [sys.executable, "merge.py", ' + _json_literal(merge_mode) + ', models_dir + "/", model_file(' + _json_literal(model0) + ')' + (', model_file(' + _json_literal(model1) + ')' if model1 else '') + ']',
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
    from .schema import _entry_produced_aliases_for_cleanup_base as _entry_produced_aliases_for_cleanup
    from .schema import _entry_reference_aliases_base as _entry_reference_aliases

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
    from .errors import PlanCompileError
    from .compiler import _entry_progress_label_base as _entry_progress_label
    from .compiler import _entry_to_lines_base as _entry_to_lines
    from .schema import normalize_plan

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
    from .formats import load_plan_records

    plan = load_plan_records(filepath)
    return planit_records(plan, workpath, model_dir, vae_dir, bake_vae=bake_vae)


def create_plan(filepath: str, workpath: str, saveas: str, title: str,
                vae: str, CivitAPI: str, HuggingAPI: str, UR: str,
                model_dir: str = "", vae_dir: str = "", vae_name: str = "VAE",
                bake_vae: bool = True,
                ignore_install_deps: bool = False, upload_after_merge: bool = False, run_t2i: bool = False):
    # Exporting a txt runner must not touch runtime-only directories on the local machine.
    # The generated runner creates workpath/tmp, models, embeddings, and vae when it executes.
    from .paths import _notebook_runtime_workpath
    from .paths import _preferred_toolpath

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
    from .syntax import _nb_json
    from .paths import _notebook_runtime_workpath
    from .paths import _preferred_toolpath

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


def _entry_to_lines_base(entry: Dict[str, Any], *, bake_vae: bool = True) -> Tuple[List[str], str | None]:
    from .compiler import _entry_to_lines as _PLANNER_ORIG_ENTRY_TO_LINES
    from .options import _command_signatures_base as _command_signatures
    from .schema import _lora_items_lines_from_entry
    from .options import _precision_args

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


def _entry_progress_label_base(entry: Dict[str, Any], index: int, total: int) -> str:
    from .compiler import _entry_progress_label as _PLANNER_ORIG_PROGRESS_LABEL

    if entry.get("type") == "LoRA Merge":
        label = str(entry.get("output_name") or "lora merge")
        if label.startswith("_"):
            label = f"TEMP{label}"
        return f"[planner-progress] {index}/{total} | LoRA Merge | {label}"
    return _PLANNER_ORIG_PROGRESS_LABEL(entry, index, total)
