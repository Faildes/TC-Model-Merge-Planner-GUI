"""Document switching must never save edited content over a pending load target."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from planner_ui import document_support as document
from planner_ui.app import ModelPlannerApp


class TextField:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def delete(self, *_args):
        self.value = ""

    def insert(self, _position, value):
        self.value = value


def app_stub(active, requested):
    return SimpleNamespace(
        _tccm_active_plan_path=str(active),
        _tccm_pending_plan_path=str(requested),
        _after_entry_change_id="pending-edit",
        _normalize_user_path=lambda value: value,
        entries={"filepath": TextField(str(requested)), "title": TextField("Old title")},
        config={}, root=SimpleNamespace(after_cancel=Mock(), after=Mock()),
        plan_data={"entries": [{"id": "old", "type": "Checkpoint Merge"}]},
        _history_undo=[1], _history_redo=[2],
        _refresh_line_selector=Mock(), _render_current_line=Mock(),
        _save_current_state_to_config=Mock(), _show_detailed_error=Mock(),
        _path_display_name=lambda value: value,
        _planner_save_project_json=Mock(), status_label=SimpleNamespace(config=Mock()),
    )


def test_save_uses_active_document_when_browse_field_points_elsewhere(tmp_path):
    active, pending = tmp_path / "active.tccm.json", tmp_path / "pending.tccm.json"
    pending.write_text("pending document must be untouched")
    app = app_stub(active, pending)
    writes = []
    def save(_app, path, **_kwargs):
        writes.append(path)
        return path
    with patch.object(document, "_tccm_atomic_save_project", side_effect=save):
        ModelPlannerApp._planner_save_project_json(app, reason="autosave")
    assert writes == [str(active)]
    assert pending.read_text() == "pending document must be untouched"
    assert app.entries["filepath"].get() == str(active)


def test_successful_load_commits_pending_edits_to_old_document(tmp_path):
    active, target = tmp_path / "old.tccm.json", tmp_path / "new.tccm.json"
    target.write_text(json.dumps({"entries": [{"id": "new-row", "type": "LoRA Bake",
                                             "checkpoint": "A", "loras": []}]}))
    before = target.read_bytes()
    app = app_stub(active, target)
    result = ModelPlannerApp._load_plan_from_path(app)
    assert result["kind"] == "preset-json"
    app.root.after_cancel.assert_called_once_with("pending-edit")
    app._planner_save_project_json.assert_called_once_with(str(active), reason="autosave")
    assert app._tccm_active_plan_path == str(target)
    assert app.entries["title"].get() == "new"
    assert app.plan_data["entries"][0]["id"] == "new-row"
    assert target.read_bytes() == before
    app._show_detailed_error.assert_not_called()


def test_failed_load_keeps_old_document_identity_and_content(tmp_path):
    active, missing = tmp_path / "old.tccm.json", tmp_path / "missing.tccm.json"
    app = app_stub(active, missing)
    assert ModelPlannerApp._load_plan_from_path(app) is None
    assert app._tccm_active_plan_path == str(active)
    assert app.entries["filepath"].get() == str(active)
    assert app.entries["title"].get() == "Old title"
    assert app.plan_data["entries"][0]["id"] == "old"
    assert app._tccm_plan_switch_in_progress is False
    app._show_detailed_error.assert_called_once()
