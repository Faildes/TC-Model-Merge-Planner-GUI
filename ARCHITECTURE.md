# Model Merge Planner architecture

- `main.py`: compatibility imports and GUI startup after final method composition.
- `plan.py`: compatibility facade for existing public Planner imports.
- `planner_ui/app.py`: `ModelPlannerApp`, composed from feature Mixins without runtime class patching.
- `lifecycle.py`, `lifecycle_support.py`: initialization and shutdown; establish state before creating widgets.
- `project_ui.py`, `document_support.py`: save/load, document switching, and active document identity.
- `plan_editing.py`, `keyboard.py`: row edits, history, selection, and keyboard actions.
- `model_editors.py`, `ratios_ui.py`, `elemental.py`, `completion.py`: editors and input assistance.
- `sources_ui.py`, `source_support.py`, `validation_ui.py`, `validation_support.py`: source resolution and plan validation.
- `layout_ui.py`, `appearance.py`, `help_ui.py`, `scrolling.py`, `widgets.py`: layout, themes, help, scrolling, and widgets.
- `execution.py`, `runner.py`: execution and queue-based console updates.
- `developer_tools.py`, `experiments.py`, `disk_estimates.py`, `variants_ui.py`, `variants.py`: developer tools, comparisons, space estimates, and variants.
- `planner_core/schema.py`: row/plan normalization in one main traversal.
- `planner_core/options.py`: precision and architecture names, structured CLI options, and legacy signature conversion.
- `planner_core/merge_validation.py`: shared compiler/GUI model requirements and Turbo option checks.
- `planner_core/formats.py`: JSON and legacy TXT import/export.
- `planner_core/compiler.py`: execution order, model release, and TXT/notebook generation.
- `planner_core/templates/*.py.tpl`: notebook code templates, editable without nested string escaping.

Unless a path begins with `planner_core/`, feature module names above are relative to `planner_ui/`.

`settings.py` loads tooltip JSON from the application resource root. `paths.py` distinguishes module paths from resource paths. `dialogs.py` and `native_dialogs.py` implement platform-specific dialog handling without patching Tkinter globally.

`main_most_output_size.py` is a preserved historical alternative; use `main.py` for the current GUI.
