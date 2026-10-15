"""Compatibility entry point; implementation lives in planner_core."""
from __future__ import annotations
from importlib import import_module

_EXPORTS = {'PlanCompileError': ('errors', 'PlanCompileError'),
 'hexchars': ('constants', 'hexchars'),
 'rnm': ('syntax', 'rnm'),
 '_uid': ('syntax', '_uid'),
 'SDXL_BLOCKS': ('constants', 'SDXL_BLOCKS'),
 '_planner_dir': ('paths', '_planner_dir'),
 '_toolpath_candidates': ('paths', '_toolpath_candidates'),
 '_preferred_toolpath': ('paths', '_preferred_toolpath'),
 '_notebook_runtime_workpath': ('paths', '_notebook_runtime_workpath'),
 '_ensure_dirs': ('syntax', '_ensure_dirs'),
 '_nb_json': ('syntax', '_nb_json'),
 '_split': ('syntax', '_split'),
 '_split_top_level': ('syntax', '_split_top_level'),
 '_ensure_st': ('syntax', '_ensure_st'),
 '_parse_lora_pairs': ('syntax', '_parse_lora_pairs'),
 '_needs_quote': ('syntax', '_needs_quote'),
 'quoter': ('syntax', 'quoter'),
 '_parse_tail_at': ('syntax', '_parse_tail_at'),
 '_infer_ratio_mode': ('schema', '_infer_ratio_mode'),
 'default_ratio': ('schema', 'default_ratio'),
 '_normalize_ratio_spec': ('schema', '_normalize_ratio_spec'),
 'make_entry': ('schema', 'make_entry'),
 'default_plan': ('schema', 'default_plan'),
 'normalize_plan': ('schema', 'normalize_plan'),
 'parse_legacy_text_plan': ('formats', 'parse_legacy_text_plan'),
 'load_plan_records': ('formats', 'load_plan_records'),
 'save_plan_records': ('formats', 'save_plan_records'),
 '_ratio_text': ('schema', '_ratio_text_base'),
 '_merge_record_to_legacy_line': ('formats', '_merge_record_to_legacy_line'),
 'export_plan_records_txt': ('formats', 'export_plan_records_txt_base'),
 'INSTALL_TPL': ('templates', 'INSTALL_TPL'),
 'PRELUDE_TPL': ('templates', 'PRELUDE_TPL'),
 'UPLOAD_TPL': ('templates', 'UPLOAD_TPL'),
 'T2I_CFG_TPL': ('templates', 'T2I_CFG_TPL'),
 'T2I_RUN_TPL': ('templates', 'T2I_RUN_TPL'),
 'ZIP_TPL': ('templates', 'ZIP_TPL'),
 '_json_literal': ('compiler', '_json_literal'),
 '_normalize_precision_name': ('options', '_normalize_precision_name'),
 '_tail_text_to_cli_signatures': ('options', '_tail_text_to_cli_signatures'),
 '_command_signatures': ('options', '_command_signatures_base'),
 '_precision_from_signatures': ('options', '_precision_from_signatures'),
 '_entry_precision': ('options', '_entry_precision'),
 '_canonical_save_precision': ('options', '_canonical_save_precision'),
 '_precision_args': ('options', '_precision_args'),
 '_legacy_signature_text': ('options', '_legacy_signature_text_base'),
 '_entry_to_lines': ('compiler', '_entry_to_lines_base'),
 '_entry_reference_aliases': ('schema', '_entry_reference_aliases_base'),
 '_entry_produced_aliases_for_cleanup': ('schema', '_entry_produced_aliases_for_cleanup_base'),
 '_auto_release_aliases_by_entry': ('compiler', '_auto_release_aliases_by_entry'),
 '_entry_progress_label': ('compiler', '_entry_progress_label_base'),
 'planit_records': ('compiler', 'planit_records'),
 'planit': ('compiler', 'planit'),
 'create_plan': ('compiler', 'create_plan'),
 'create_plan_ipynb': ('compiler', 'create_plan_ipynb'),
 '_planner_default_cli_options': ('options', '_planner_default_cli_options'),
 '_planner_clean_cli_options': ('options', '_planner_clean_cli_options'),
 '_planner_quote_cli_value': ('schema', '_planner_quote_cli_value'),
 '_CLI_OPTION_SPECS': ('constants', '_CLI_OPTION_SPECS'),
 '_structured_entry_signatures': ('options', '_structured_entry_signatures_base'),
 '_lora_items_lines_from_entry': ('schema', '_lora_items_lines_from_entry'),
 '_PRECISION_CHOICES': ('constants', '_PRECISION_CHOICES'),
 '_ARCH_CHOICES': ('constants', '_ARCH_CHOICES'),
 '_sigless_tokenize': ('options', '_sigless_tokenize'),
 '_sigless_read_value': ('options', '_sigless_read_value'),
 '_sigless_option_maps': ('options', '_sigless_option_maps'),
 '_sigless_apply_legacy_signatures': ('options', '_sigless_apply_legacy_signatures_base'),
 '_legacy_sig_quote': ('formats', '_legacy_sig_quote'),
 '_structured_entry_legacy_signatures': ('options', '_structured_entry_legacy_signatures'),
 '_PRECISION_RUNTIME_ALIASES': ('constants', '_PRECISION_RUNTIME_ALIASES'),
 '_PRECISION_DISPLAY_ALIASES': ('constants', '_PRECISION_DISPLAY_ALIASES'),
 '_ARCH_RUNTIME_ALIASES': ('constants', '_ARCH_RUNTIME_ALIASES'),
 '_ARCH_DISPLAY_ALIASES': ('constants', '_ARCH_DISPLAY_ALIASES'),
 '_display_precision_name': ('options', '_display_precision_name'),
 '_normalize_arch_runtime': ('options', '_normalize_arch_runtime'),
 '_display_arch_name': ('options', '_display_arch_name'),
 '_checkpoint_specs': ('constants', '_checkpoint_specs')}

__all__ = [name for name in _EXPORTS if not name.startswith("_")]

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module("planner_core." + module), attribute)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
