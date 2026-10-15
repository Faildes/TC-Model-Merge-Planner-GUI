"""Application composition. Each method is defined once, without runtime patching."""
from __future__ import annotations

from .execution import ExecutionMixin
from .project_ui import ProjectUiMixin
from .backups import BackupsMixin
from .variants_ui import VariantsUiMixin
from .ratios_ui import RatiosUiMixin
from .plan_editing import PlanEditingMixin
from .appearance import AppearanceMixin
from .help_ui import HelpUiMixin
from .layout_ui import LayoutUiMixin
from .sources_ui import SourcesUiMixin
from .validation_ui import ValidationUiMixin
from .elemental import ElementalMixin
from .completion import CompletionMixin
from .model_editors import ModelEditorsMixin
from .lifecycle import LifecycleMixin
from .scrolling import ScrollingMixin
from .keyboard import KeyboardMixin
from .experiments import ExperimentsMixin


class ModelPlannerApp(
    ExecutionMixin,
    ProjectUiMixin,
    BackupsMixin,
    VariantsUiMixin,
    RatiosUiMixin,
    PlanEditingMixin,
    AppearanceMixin,
    HelpUiMixin,
    LayoutUiMixin,
    SourcesUiMixin,
    ValidationUiMixin,
    ElementalMixin,
    CompletionMixin,
    ModelEditorsMixin,
    LifecycleMixin,
    ScrollingMixin,
    KeyboardMixin,
    ExperimentsMixin,
):
    """TC Model Merge Planner. Edit the module for the relevant feature."""
    NOTEBOOK_PARAMS_KEYS = ['workpath', 'title', 'vae', 'vae_name', 'bake_vae', 'CivitAPI', 'HuggingAPI', 'UR', 'model_dir', 'vae_dir', 'ignore_install_deps', 'use_online', 'upload_after_merge', 'run_t2i']
