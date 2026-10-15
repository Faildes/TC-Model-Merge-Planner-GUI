# Function and method index


## planner_ui/app.py

- `ModelPlannerApp`

## planner_ui/appearance.py

- `AppearanceMixin`
- `AppearanceMixin._planner_disable_widget_subtree`
- `AppearanceMixin._toggle_theme_mode`
- `AppearanceMixin._theme_colors`
- `AppearanceMixin._button_style_for_text`
- `AppearanceMixin._apply_theme_mode`
- `AppearanceMixin._apply_theme_to_children`
- `AppearanceMixin._planner_row_color_presets`

## planner_ui/backups.py

- `BackupsMixin`
- `BackupsMixin._planner_plan_snapshot`
- `BackupsMixin._planner_restore_snapshot`
- `BackupsMixin._planner_backup_root`
- `BackupsMixin._show_backup_manager`

## planner_ui/completion.py

- `CompletionMixin`
- `CompletionMixin._autocomplete_item`
- `CompletionMixin._hide_autocomplete`
- `CompletionMixin._normalize_autocomplete_suggestions`
- `CompletionMixin._autocomplete_has_items`
- `CompletionMixin._style_autocomplete_popup`
- `CompletionMixin._select_autocomplete_index`
- `CompletionMixin._show_autocomplete`
- `CompletionMixin._move_autocomplete`
- `CompletionMixin._accept_autocomplete`
- `CompletionMixin._apply_signature_completion`
- `CompletionMixin._apply_elemental_completion`
- `CompletionMixin._refresh_signature_invalid_highlight`
- `CompletionMixin._refresh_elemental_invalid_highlight`
- `CompletionMixin._bind_entry_autocomplete`
- `CompletionMixin._bind_text_autocomplete`
- `CompletionMixin._configure_syntax_tags`
- `CompletionMixin._clear_syntax_highlight`
- `CompletionMixin._text_token_bounds`
- `CompletionMixin._absorb_mode_signature_from_entry`
- `CompletionMixin._entry_token_bounds`

## planner_ui/developer_tools.py

- `_planner_dev_ensure_config`
- `_planner_dev_mode_enabled`
- `_planner_dev_only`
- `_planner_get_t2i_settings`
- `_planner_set_t2i_settings`
- `_planner_save_version_snapshot`
- `_planner_show_versions`
- `_planner_open_t2i_settings`
- `_parse_sweep_values`
- `_planner_save_alpha_preset`
- `_planner_load_alpha_preset`
- `_planner_model_anatomy_scan`
- `_planner_show_image_compare`
- `_planner_optimize_plan_full`

## planner_ui/discovery.py

- `_coerce_block_names`
- `_load_module_from_candidates`
- `discover_block_sets`
- `discover_merge_modes`

## planner_ui/disk_estimates.py

- `_planner_nearest_existing_path`
- `_planner_disk_usage_for_path`
- `_planner_unique_disk_key`
- `_planner_collect_disk_usages`
- `_planner_estimate_disk_runtime_robust`

## planner_ui/document_support.py

- `_save_current_state_to_config`
- `_planner_norm_path_key`
- `_save_current_state_to_config_dev`
- `_planner_is_project_json_path`
- `_planner_project_json_payload`
- `_planner_restore_project_payload`
- `_ensure_plan_path_project_json`
- `_save_plan_to_file_project_json`
- `_write_temp_plan_text_project_json`
- `_get_notebook_paths_project_json`
- `_suggest_export_txt_path_project_json`
- `_new_plan_project_json`
- `_load_plan_from_path_project_json`
- `_planner_offer_newer_backup_restore`
- `_create_menu_project_json`
- `_create_menu_project_json_open_dialog`
- `_tccm_current_runtime_payload`
- `_tccm_apply_runtime_payload`
- `_tccm_backup_pin_path`
- `_tccm_backup_is_pinned`
- `_show_backup_diff_manager`
- `_tccm_project_json_schema`
- `_export_project_json_schema`
- `_planner_project_json_payload_advanced`
- `_planner_restore_project_payload_advanced`
- `_tccm_is_project_path`
- `_planner_save_project_json_advanced`
- `_planner_maybe_create_backup_advanced`
- `_new_plan_advanced`
- `_load_plan_from_path_macos_safe`
- `_open_plan_dialog_project_json_macos_safe`
- `_planner_force_project_extension_macos_safe`
- `_tccm_json_loader_read_json`
- `_tccm_json_loader_is_entry_dict`
- `_tccm_json_loader_is_project_dict`
- `_tccm_json_loader_extract_project_plan`
- `_tccm_json_loader_raw_entries_from_json`
- `_tccm_json_loader_normalize_entries_for_insert`
- `_tccm_json_loader_plan_from_any_file`
- `_restore_session_state_universal_json`
- `_tccm_atomic_normalize_path`
- `_tccm_atomic_init`
- `_tccm_atomic_runtime_payload`
- `_tccm_atomic_apply_runtime`
- `_tccm_atomic_project_payload`
- `_tccm_atomic_set_filepath_widget`
- `_tccm_atomic_active_path`
- `_tccm_atomic_cancel_or_flush_pending_edit`
- `_tccm_atomic_save_project`
- `_tccm_atomic_autosave`
- `_tccm_atomic_save_config`
- `_tccm_atomic_browse_plan_file`
- `_tccm_atomic_new_plan`
- `_tccm_v2_norm`
- `_tccm_v2_active_path`
- `_tccm_v2_set_path_widget`
- `_tccm_v2_set_title_widget`
- `_tccm_v2_default_title_for_path`
- `_tccm_v2_project_payload`
- `_tccm_v2_apply_runtime`
- `_tccm_v2_save_project`
- `_tccm_v2_save_button`
- `_tccm_v2_load`

## planner_ui/editor_support.py

- `_build_left_panel`
- `_build_ratio_value_widget`
- `_planner_rerender_preserve_scroll`
- `_add_lora_block_preserve_scroll`
- `_remove_lora_block_preserve_scroll`
- `_move_lora_block`
- `_planner_open_ratio_sweep`
- `_planner_ratio_to_lines`
- `_planner_show_ratio_diff`
- `_planner_lora_compatibility_check`
- `_lm_ensure_cli_options`
- `_lm_option_changed`
- `_lm_build_cli_options_panel`
- `_base_render_current_line_with_lora_merge`
- `_planner_visible_line_types_with_lora_merge`
- `_entry_type_color_with_lora_merge`
- `_line_summary_with_lora_merge`
- `_planner_convert_entry_type_preserving_common_fields_with_lora_merge`
- `_planner_apply_entry_defaults_with_lora_merge`
- `_sigless_ensure_cli_options`
- `_sigless_mode_uses_seed`
- `_sigless_model_count`
- `_sigless_option_changed`
- `_sigless_build_unmapped_editor`
- `_sigless_ratio_presets`
- `_planner_save_ratio_preset`
- `_planner_load_ratio_preset`
- `_planner_build_ratio_preset_buttons`
- `_build_ratio_value_widget_sigless`
- `_render_lora_bake_entry_sigless`
- `_render_lora_merge_entry_sigless`
- `_on_plan_help_click_sigless`
- `_on_plan_list_motion_sigless`
- `_planner_display_precision_name`
- `_planner_display_arch_name`
- `_sigless_ensure_cli_options_display`
- `_sigless_option_changed_display`
- `_build_left_panel_refined_dev_buttons`
- `_safe_migration_tool`

## planner_ui/elemental.py

- `ElementalMixin`
- `ElementalMixin._elemental_candidate_json_filename`
- `ElementalMixin._elemental_candidate_json_path`
- `ElementalMixin._load_elemental_catalog`
- `ElementalMixin._normalize_elemental_layer_name`
- `ElementalMixin._elemental_valid_elements_for_layer`
- `ElementalMixin._elemental_alias_lookup`
- `ElementalMixin._elemental_layer_order`
- `ElementalMixin._expand_elemental_layer_token`
- `ElementalMixin._resolve_elemental_layers_expr`
- `ElementalMixin._elemental_valid_elements_for_layers`
- `ElementalMixin._elemental_separator_for_widget`
- `ElementalMixin._elemental_context`
- `ElementalMixin._elemental_suggestion_items`
- `ElementalMixin._on_elemental_text_motion`
- `ElementalMixin._normalize_named_effect_items`
- `ElementalMixin._element_lookup_from_items`
- `ElementalMixin._resolve_layer_element_items`
- `ElementalMixin._block_effect_hint_fallback`
- `ElementalMixin._block_effect_hint`
- `ElementalMixin._current_block_names`

## planner_ui/execution.py

- `ExecutionMixin`
- `ExecutionMixin._export_as_notebook`
- `ExecutionMixin._build_runner_notebook`
- `ExecutionMixin._build_notebook_run_command`
- `ExecutionMixin._run_target_notebook`
- `ExecutionMixin._upload_merge_result`

## planner_ui/experiments.py

- `ExperimentsMixin`
- `ExperimentsMixin._show_unused_models`

## planner_ui/help_support.py

- `_build_plan_item_tooltip_text`
- `_build_plan_item_tooltip_text_with_causes`
- `_build_plan_item_tooltip_text_with_lora_merge`
- `_planner_mode_catalog_entry`
- `_planner_mode_detail_text`
- `_planner_attach_dynamic_tooltip`
- `_planner_choice_help_lines`
- `_planner_option_tooltip_text`
- `_planner_model_choice_tooltip_text`
- `_planner_line_type_tooltip_text`
- `_planner_mode_choice_tooltip_compact`
- `_planner_mode_dropdown_item_tooltip`
- `_planner_choice_dropdown_item_tooltip`
- `_planner_hide_dropdown_tooltip`
- `_planner_show_dropdown_tooltip`
- `_planner_attach_combobox_dropdown_tooltips`
- `_build_combo_row_dropdown_refined`
- `_sigless_special_option_row_dropdown_refined`
- `_sigless_build_cli_options_panel_dropdown_refined`
- `_build_ratio_section_dropdown_refined`
- `_planner_blend_hex`

## planner_ui/help_ui.py

- `HelpUiMixin`
- `HelpUiMixin._extract_elemental_hover_token`
- `HelpUiMixin._hide_elemental_alias_hover`
- `HelpUiMixin._show_elemental_alias_hover_popup`
- `HelpUiMixin._hide_plan_item_hover`
- `HelpUiMixin._show_plan_item_hover`
- `HelpUiMixin._left_help`
- `HelpUiMixin._attach_tooltip`
- `HelpUiMixin._open_help_link`
- `HelpUiMixin._add_inline_help`
- `HelpUiMixin._right_help`
- `HelpUiMixin._add_right_inline_help`

## planner_ui/keyboard.py

- `KeyboardMixin`
- `KeyboardMixin._planner_widget_state`
- `KeyboardMixin._planner_is_editable_text_widget`
- `KeyboardMixin._planner_widget_is_descendant`
- `KeyboardMixin._planner_focus_in_plan_scope`
- `KeyboardMixin._planner_is_text_focus`
- `KeyboardMixin._planner_register_shortcut_bindtag`
- `KeyboardMixin._planner_register_shortcut_bindtag_on_focus`
- `KeyboardMixin._planner_register_shortcut_bindtag_on_tree`
- `KeyboardMixin._shortcut_show_diff`
- `KeyboardMixin._shortcut_validate`
- `KeyboardMixin._shortcut_save_preset`
- `KeyboardMixin._shortcut_load_preset`
- `KeyboardMixin._set_widget_state_if_present`
- `KeyboardMixin._planner_focus_allows_line_shortcuts`
- `KeyboardMixin._shortcut_undo`
- `KeyboardMixin._shortcut_redo`
- `KeyboardMixin._shortcut_copy_lines`
- `KeyboardMixin._shortcut_paste_lines`
- `KeyboardMixin._shortcut_delete_lines`
- `KeyboardMixin._shortcut_duplicate_lines`
- `KeyboardMixin._shortcut_add_line`
- `KeyboardMixin._shortcut_move_lines_up`
- `KeyboardMixin._shortcut_move_lines_down`
- `KeyboardMixin._shortcut_remove_line`

## planner_ui/layout_ui.py

- `LayoutUiMixin`
- `LayoutUiMixin._build_status_bar`
- `LayoutUiMixin._build_collapsible_section`
- `LayoutUiMixin._check_aria2`
- `LayoutUiMixin._play_notification_sound`
- `LayoutUiMixin._build_labeled_frame`
- `LayoutUiMixin._build_text_row`
- `LayoutUiMixin._show_about`
- `LayoutUiMixin._run_image_converter`
- `LayoutUiMixin._create_layout`
- `LayoutUiMixin._build_left_fixed_header`
- `LayoutUiMixin._build_right_panel`
- `LayoutUiMixin._is_compact_layout`
- `LayoutUiMixin._set_active_view`
- `LayoutUiMixin._cycle_active_view`
- `LayoutUiMixin._update_view_switch_label`
- `LayoutUiMixin._update_responsive_layout`
- `LayoutUiMixin._schedule_responsive_layout`
- `LayoutUiMixin._run_responsive_layout_update`
- `LayoutUiMixin._planner_build_fixed_status_bar`
- `LayoutUiMixin._sync_target_line_combo`
- `LayoutUiMixin._build_entry_row`

## planner_ui/lifecycle.py

- `LifecycleMixin`
- `LifecycleMixin._on_closing`
- `LifecycleMixin._schedule_config_save`
- `LifecycleMixin._on_bake_vae_toggle`
- `LifecycleMixin._planner_init_extra_state`

## planner_ui/lifecycle_support.py

- `_initialize_widgets`
- `_create_menu`
- `_initialize_application`
- `_planner_safe_init`
- `_create_menu_advanced_tools`
- `_tccm_plan_dialog_initialdir`
- `_create_menu_macos_safe_open`

## planner_ui/model_editors.py

- `ModelEditorsMixin`
- `ModelEditorsMixin._on_base_model_change`
- `ModelEditorsMixin._resolve_merge_mode_key`
- `ModelEditorsMixin._sync_lora_name`
- `ModelEditorsMixin._get_lora_source`
- `ModelEditorsMixin._set_lora_source`

## planner_ui/native_dialogs.py

- `_tccm_running_on_macos`
- `_tccm_safe_dialog_kwargs`
- `_tccm_askopenfilename_safe`
- `_tccm_askopenfilenames_safe`
- `_tccm_asksaveasfilename_safe`

## planner_ui/plan_editing.py

- `PlanEditingMixin`
- `PlanEditingMixin._show_history_view`
- `PlanEditingMixin._collapse_selected_ws_chain`
- `PlanEditingMixin._remove_dead_lines`
- `PlanEditingMixin._entry_is_locked`
- `PlanEditingMixin._entry_is_disabled`
- `PlanEditingMixin._toggle_selected_locked`
- `PlanEditingMixin._toggle_selected_disabled`
- `PlanEditingMixin._event_has_open_modifier`
- `PlanEditingMixin._is_shift_pressed`
- `PlanEditingMixin._add_tag_for_span`
- `PlanEditingMixin._perform_scheduled_rerender`
- `PlanEditingMixin._on_root_resize`
- `PlanEditingMixin._reset_plan`
- `PlanEditingMixin._show_detailed_error`
- `PlanEditingMixin._format_exception_text`
- `PlanEditingMixin._base_reset_plan`
- `PlanEditingMixin._planner_default_visible_plan`
- `PlanEditingMixin._invoke_plan_reorder_button`
- `PlanEditingMixin._planner_unused_summary`
- `PlanEditingMixin._update_plan_action_buttons`
- `PlanEditingMixin._collapse_internal_plan_entries`
- `PlanEditingMixin._planner_restore_visible_plan`
- `PlanEditingMixin._planner_locked_indices`
- `PlanEditingMixin._planner_guard_unlocked`
- `PlanEditingMixin._planner_replace_in_values`
- `PlanEditingMixin._show_bulk_edit_dialog`
- `PlanEditingMixin._hide_plan_drag_guide`
- `PlanEditingMixin._show_plan_drag_guide`
- `PlanEditingMixin._strip_mode_signatures`
- `PlanEditingMixin._memo_preview_text`
- `PlanEditingMixin._hide_plan_context_menu`
- `PlanEditingMixin._show_plan_diff_preview`
- `PlanEditingMixin._schedule_rerender_current_line`
- `PlanEditingMixin._filter_candidates`
- `PlanEditingMixin._set_active_multiline_text`
- `PlanEditingMixin._current_entry_context`
- `PlanEditingMixin._select_directory`
- `PlanEditingMixin._refresh_line_selector`
- `PlanEditingMixin._on_line_selection_change`
- `PlanEditingMixin._add_line`
- `PlanEditingMixin._remove_line`
- `PlanEditingMixin._move_line_up`
- `PlanEditingMixin._move_line_down`
- `PlanEditingMixin._render_current_line`
- `PlanEditingMixin._change_line_type`
- `PlanEditingMixin._render_download_entry`
- `PlanEditingMixin._render_remove_entry`
- `PlanEditingMixin._after_entry_change`
- `PlanEditingMixin._update_status_from_line`
- `PlanEditingMixin._base_change_line_type`
- `PlanEditingMixin._flush_after_entry_change`
- `PlanEditingMixin._planner_restore_plan_list_focus`
- `PlanEditingMixin._planner_deepcopy_plan`
- `PlanEditingMixin._update_current_line_indicator`
- `PlanEditingMixin._planner_get_selected_indices`
- `PlanEditingMixin._planner_push_history`
- `PlanEditingMixin._planner_schedule_history_checkpoint`
- `PlanEditingMixin._planner_commit_history_checkpoint`
- `PlanEditingMixin._visible_entries_for_filter`
- `PlanEditingMixin._on_plan_list_select`
- `PlanEditingMixin._select_model_indices`
- `PlanEditingMixin._copy_selected_lines`
- `PlanEditingMixin._paste_copied_lines`
- `PlanEditingMixin._delete_selected_lines`
- `PlanEditingMixin._duplicate_selected_lines`
- `PlanEditingMixin._plan_list_context_visible_index`
- `PlanEditingMixin._plan_list_prepare_context_selection`
- `PlanEditingMixin._selected_entries_support_remove_context`
- `PlanEditingMixin._insert_remove_model_lines_below_selected`
- `PlanEditingMixin._plan_list_drag_start`
- `PlanEditingMixin._plan_list_drag_end`
- `PlanEditingMixin._planner_internal_line_types`
- `PlanEditingMixin._append_auto_remove_lines`
- `PlanEditingMixin._selected_entries_support_ws_collapse`

## planner_ui/project_ui.py

- `ProjectUiMixin`
- `ProjectUiMixin._normalize_user_path`
- `ProjectUiMixin._paths_equal`
- `ProjectUiMixin._path_display_name`
- `ProjectUiMixin._export_as_txt`
- `ProjectUiMixin._base_load_plan_from_path`
- `ProjectUiMixin._base_new_plan`
- `ProjectUiMixin._planner_refresh_plan_meta`
- `ProjectUiMixin._planner_meta_sidecar_path`
- `ProjectUiMixin._planner_save_meta_to_disk`
- `ProjectUiMixin._planner_load_meta_from_disk`

## planner_ui/ratios_ui.py

- `RatiosUiMixin`
- `RatiosUiMixin._is_plain_number`
- `RatiosUiMixin._serialize_block_values`
- `RatiosUiMixin._parse_block_values`
- `RatiosUiMixin._clamp_ratio_float`
- `RatiosUiMixin._coerce_ratio_float`
- `RatiosUiMixin._format_ratio_float`
- `RatiosUiMixin._commit_block_ratio_entry`
- `RatiosUiMixin._sync_block_ratio_scale`
- `RatiosUiMixin._sync_block_ratio`
- `RatiosUiMixin._sync_ratio_value`
- `RatiosUiMixin._ratio_single_float`

## planner_ui/resources.py

- `_planner_safe_cpu_threads`

## planner_ui/runner.py

- `RunnerConsoleWindow`
- `RunnerConsoleWindow.__init__`
- `RunnerConsoleWindow._task_palette`
- `RunnerConsoleWindow.apply_theme`
- `RunnerConsoleWindow._make_scrolled_text`
- `RunnerConsoleWindow.show`
- `RunnerConsoleWindow._schedule_poll`
- `RunnerConsoleWindow._enqueue`
- `RunnerConsoleWindow._on_window_close`
- `RunnerConsoleWindow._safe_configure_stop_button`
- `RunnerConsoleWindow._on_output_frame_configure`
- `RunnerConsoleWindow._on_output_canvas_configure`
- `RunnerConsoleWindow.bind_notebook`
- `RunnerConsoleWindow.set_state`
- `RunnerConsoleWindow.set_step`
- `RunnerConsoleWindow.set_operation`
- `RunnerConsoleWindow.set_progress_mode`
- `RunnerConsoleWindow.process_event`
- `RunnerConsoleWindow.set_progress`
- `RunnerConsoleWindow.set_progress_fraction`
- `RunnerConsoleWindow.attach_process`
- `RunnerConsoleWindow.log`
- `RunnerConsoleWindow.idle`
- `RunnerConsoleWindow.clear_output`
- `RunnerConsoleWindow.request_stop`
- `RunnerConsoleWindow._text_at_bottom`
- `RunnerConsoleWindow._trim_text`
- `RunnerConsoleWindow._extract_progress_fraction`
- `RunnerConsoleWindow._looks_like_progress_line`
- `RunnerConsoleWindow._clear_live_progress`
- `RunnerConsoleWindow._sync_live_progress`
- `RunnerConsoleWindow._normalize_stream_text`
- `RunnerConsoleWindow._extract_text_output`
- `RunnerConsoleWindow._render_output_block`
- `RunnerConsoleWindow._render_notebook_outputs`
- `RunnerConsoleWindow._refresh_notebook_view`
- `RunnerConsoleWindow._reset_notebook_widgets`
- `RunnerConsoleWindow._insert_idle_batch`
- `RunnerConsoleWindow._insert_log_batch`
- `RunnerConsoleWindow._append_text`
- `RunnerConsoleWindow._update_process_tree`
- `RunnerConsoleWindow._poll_queue`

## planner_ui/scrolling.py

- `ScrollingMixin`
- `ScrollingMixin._apply_scrollbar_palette`
- `ScrollingMixin._canvas_contains_widget`
- `ScrollingMixin._canvas_for_event`
- `ScrollingMixin._canvas_can_scroll`
- `ScrollingMixin._scroll_canvas_by_event`
- `ScrollingMixin._scroll_text_widget`
- `ScrollingMixin._install_panel_scroll_support`
- `ScrollingMixin._on_mousewheel`
- `ScrollingMixin._on_mousewheel_up`
- `ScrollingMixin._on_mousewheel_down`
- `ScrollingMixin._show_scrollable_text_dialog`
- `ScrollingMixin._style_scrollbar_widget`
- `ScrollingMixin._bind_combobox_mousewheel_passthrough`
- `ScrollingMixin._bind_multiline_text_scroll_priority`

## planner_ui/settings.py

- `save_config_to_disk`
- `_planner_tooltip_catalog_path_candidates`
- `_planner_load_tooltip_catalog`
- `load_config_from_disk`

## planner_ui/shortcuts.py

- `_planner_shortcut_registry`
- `_planner_bind_shortcuts_configurable`
- `_planner_show_shortcut_settings`

## planner_ui/source_support.py

- `_apply_resource_limits_to_env`
- `_scan_local_model_choices`
- `_build_source_backed_ref_row`
- `_iter_embedded_sources`
- `_entry_consumed_aliases`
- `_planner_add_local_choice`
- `_scan_local_model_choices_retained`
- `_planner_display_for_local_path`
- `_build_local_selection_row_retained`
- `_prompt_local_source_dialog_retained`
- `_planner_source_signature`
- `_planner_source_records`
- `_planner_analysis_source_aware`
- `_show_dependency_view_source_aware`
- `_make_effective_source_entry_alias_preserving`
- `_iter_embedded_sources_with_lora_merge`
- `_entry_consumed_aliases_with_lora_merge`
- `_build_source_backed_ref_row_dropdown_refined`
- `_planner_build_source_model_row_final`
- `_render_checkpoint_merge_entry_source_buttons_final`
- `_render_lora_bake_entry_source_buttons_hover_final`
- `_render_lora_merge_entry_source_buttons_hover_final`
- `_planner_source_spec_is_usable_final`
- `_planner_bucket_has_ref_final`
- `_planner_ref_resolves_final`
- `_planner_lora_ref_resolves_final`
- `_planner_merge_requires_model2_final`
- `_planner_entry_can_produce_final`
- `_collect_available_models_dependency_safe_final`
- `_entry_produced_aliases_dependency_safe_final`
- `_tccm_aliases`
- `_tccm_entry_sources`

## planner_ui/sources_ui.py

- `SourcesUiMixin`
- `SourcesUiMixin._candidate_local_model_roots`
- `SourcesUiMixin._browse_local_path`
- `SourcesUiMixin._render_local_entry`
- `SourcesUiMixin._compute_aliases`
- `SourcesUiMixin._slot_sources_dict`
- `SourcesUiMixin._get_entry_slot_source`
- `SourcesUiMixin._set_entry_slot_source`
- `SourcesUiMixin._source_alias_from_spec`
- `SourcesUiMixin._normalize_source_kind`
- `SourcesUiMixin._guess_alias_from_link`
- `SourcesUiMixin._format_source_spec_text`
- `SourcesUiMixin._ask_source_kind`
- `SourcesUiMixin._prompt_download_source_dialog`
- `SourcesUiMixin._restore_embedded_sources_into_entry`
- `SourcesUiMixin._normalize_entry_preserving_embedded_sources`
- `SourcesUiMixin._normalize_plan_preserving_embedded_sources`
- `SourcesUiMixin._gather_selected_source_specs_by_alias`

## planner_ui/tools_support.py

- `_signature_autocomplete_candidates`
- `_entry_type_color`
- `_build_combo_row`
- `_planner_visible_line_types`
- `_planner_apply_entry_defaults`
- `_line_summary`
- `_planner_convert_entry_type_preserving_common_fields`
- `_on_plan_list_motion`
- `_show_plan_context_menu`
- `_planner_show_dependency_graph`
- `_signature_autocomplete_candidates_extended`
- `_planner_combobox_listbox`
- `_planner_attach_merge_mode_help_final`
- `_planner_timestamp`
- `_planner_safe_title`
- `_planner_effective_json_payload`
- `_planner_write_json_file`
- `_planner_read_json_or_none`
- `_planner_export_json_common`
- `_export_full_json`
- `_export_effective_json`
- `_copy_full_json`
- `_copy_effective_json`
- `_copy_current_line_json`
- `_planner_meta_payload_id_aware`
- `_planner_apply_loaded_meta_id_aware`
- `_tccm_timestamp`
- `_tccm_safe_filename`
- `_tccm_norm_text`
- `_tccm_display_name`
- `_tccm_entry_output`
- `_tccm_entry_summary`
- `_tccm_make_text_window`
- `_tccm_ensure_extension_state`
- `_tccm_analyze_plan`
- `_show_plan_health_dashboard`
- `_show_dependency_chain_viewer`
- `_show_output_collision_checker`
- `_tccm_provenance_payload`
- `_show_model_provenance`
- `_export_model_provenance_json`
- `_tccm_open_plan_dialog_no_native_filter`
- `_browse_plan_file_macos_safe`
- `_load_preset_json_universal_json`
- `_save_preset_json_universal_json`
- `_planner_normalize_row_color_spec`
- `_entry_row_color_final`
- `_entry_row_color_hex`
- `_planner_apply_entry_defaults_row_color`
- `_apply_plan_listbox_item_styles_final`
- `_show_row_color_dialog_final`

## planner_ui/validation_support.py

- `_planner_collect_issue_rows`
- `_show_prevalidation_with_causes`
- `_planner_download_link_issue`
- `_planner_problem_explanation_titleized`
- `_show_prevalidation_with_download_checks`
- `_tccm_format_analysis_report`
- `_planner_problem_severity_final`
- `_plan_entry_problem_map_linear`

## planner_ui/validation_ui.py

- `ValidationUiMixin`
- `ValidationUiMixin._planner_dependency_summary`
- `ValidationUiMixin._validate_plan_entries`
- `ValidationUiMixin._normalize_loaded_plan_entries`
- `ValidationUiMixin._planner_effective_plan_data`
- `ValidationUiMixin._sanitize_effective_payload`

## planner_ui/variants.py

- `_tccm_choose_by_name`
- `_save_current_as_variant`
- `_load_variant`
- `_delete_variant`
- `_export_variants_json`
- `_save_current_as_run_profile`
- `_apply_run_profile`
- `_delete_run_profile`
- `_export_run_profiles_json`

## planner_ui/variants_ui.py

- `VariantsUiMixin`
- `VariantsUiMixin._cache_entry_variant`
- `VariantsUiMixin._restore_entry_variant`

## planner_ui/widgets.py

- `ProgressWindow`
- `ProgressWindow.__init__`
- `ProgressWindow.start`
- `ProgressWindow.update`
- `ProgressWindow._update_status`
- `ProgressWindow.finish`
- `Tooltip`
- `Tooltip.__init__`
- `Tooltip._schedule`
- `Tooltip._cancel`
- `Tooltip._theme_colors`
- `Tooltip._show`
- `Tooltip._hide`
- `_PlannerDynamicTooltip`
- `_PlannerDynamicTooltip.__init__`
- `_PlannerDynamicTooltip._schedule`

## planner_core/compiler.py

- `_json_literal`
- `_entry_to_lines`
- `_entry_reference_aliases`
- `_entry_produced_aliases_for_cleanup`
- `_auto_release_aliases_by_entry`
- `_entry_progress_label`
- `planit_records`
- `planit`
- `create_plan`
- `create_plan_ipynb`
- `_entry_to_lines_base`
- `_entry_progress_label_base`

## planner_core/errors.py

- `PlanCompileError`
- `PlanCompileError.__init__`

## planner_core/formats.py

- `parse_legacy_text_plan`
- `load_plan_records`
- `save_plan_records`
- `_merge_record_to_legacy_line`
- `export_plan_records_txt_base`
- `_legacy_sig_quote`

## planner_core/merge_validation.py

- `required_model_slots`
- `turbo_problem`

## planner_core/options.py

- `_tail_text_to_cli_signatures`
- `_precision_from_signatures`
- `_entry_precision`
- `_canonical_save_precision`
- `_structured_entry_signatures`
- `_default_cli_options_base`
- `_clean_cli_options_structured`
- `_sigless_tokenize`
- `_sigless_read_value`
- `_sigless_option_maps`
- `_sigless_apply_legacy_signatures`
- `_command_signatures_base`
- `_normalize_precision_name`
- `_display_precision_name`
- `_normalize_arch_runtime`
- `_display_arch_name`
- `_planner_clean_cli_options`
- `_sigless_apply_legacy_signatures_base`
- `_structured_entry_signatures_base`
- `_structured_entry_legacy_signatures`
- `_legacy_signature_text_base`
- `_precision_args`
- `_planner_default_cli_options`

## planner_core/paths.py

- `_planner_dir`
- `_toolpath_candidates`
- `_preferred_toolpath`
- `_notebook_runtime_workpath`

## planner_core/schema.py

- `_infer_ratio_mode`
- `default_ratio`
- `_normalize_ratio_spec`
- `make_entry_base`
- `default_plan_original`
- `make_entry_structured`
- `default_plan_structured`
- `_planner_quote_cli_value`
- `_lora_items_lines_from_entry`
- `_entry_reference_aliases_base`
- `_entry_produced_aliases_for_cleanup_base`
- `make_entry`
- `_ratio_text_base`
- `default_plan`
- `normalize_plan`

## planner_core/syntax.py

- `rnm`
- `_uid`
- `_ensure_dirs`
- `_nb_json`
- `_split`
- `_split_top_level`
- `_ensure_st`
- `_parse_lora_pairs`
- `_needs_quote`
- `quoter`
- `_parse_tail_at`
