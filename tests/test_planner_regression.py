"""Golden cases captured from the supplied pre-refactor Planner."""
import ast
import copy
import json
from pathlib import Path

import pytest
import plan

FIXTURES = json.loads((Path(__file__).parent / "fixtures/planner_regression.json").read_text())


def without_generated_ids(value):
    if isinstance(value, dict):
        return {k: without_generated_ids(v) for k, v in value.items() if k != "id"}
    if isinstance(value, (list, tuple)):
        return [without_generated_ids(v) for v in value]
    return value


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda value: str(FIXTURES.index(value)))
def test_current_and_legacy_plan_outputs(fixture, tmp_path):
    try:
        data = plan.normalize_plan(copy.deepcopy(fixture["input"]))
        result = ["ok", without_generated_ids(data)]
    except Exception as exc:
        result = ["error", type(exc).__name__, str(exc)]
    # The only additive schema default in v3; keep all historical golden
    # compiled commands and exported text unchanged.
    expected = copy.deepcopy(fixture["normalized"])
    if expected[0] == "ok":
        for entry in expected[1]["entries"]:
            if entry.get("type") == "Checkpoint Merge":
                entry["cli_options"]["turbo_method"] = "delta"
    assert result == expected
    if result[0] != "ok":
        return
    try:
        compiled = ["ok", plan.planit_records(data, "/kaggle")]
        compiled = json.loads(json.dumps(compiled))
    except Exception as exc:
        compiled = ["error", type(exc).__name__, str(exc)]
    assert compiled == fixture["compiled"]
    output = tmp_path / "plan.txt"
    try:
        plan.export_plan_records_txt(str(output), data)
        text = ["ok", output.read_text()]
    except Exception as exc:
        text = ["error", type(exc).__name__, str(exc)]
    assert text == fixture["text"]


def test_ids_and_unknown_fields_survive_normalization():
    source = {"kind": "local", "path": "/data/donor.safetensors"}
    row = {"type": "LoRA Bake", "id": "existing-id", "checkpoint": "A", "loras": [],
           "_slot_sources": {"checkpoint": source}, "custom_field": {"keep": True}}
    data = plan.normalize_plan({"entries": [row]})
    actual = data["entries"][0]
    assert actual["id"] == "existing-id"
    assert actual["_slot_sources"]["checkpoint"] == source
    assert actual["custom_field"] == {"keep": True}


@pytest.mark.parametrize("architecture", ["SD1.5", "SDXL", "Flux", "ZImage", "Anima", "Krea2"])
def test_notebook_cells_are_valid_python(architecture, tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"entries": [{"type": "Local Model", "id": "a", "model_name": "A",
                                               "local_path": "/models/a.safetensors"},
                                              {"type": "Checkpoint Merge", "id": "b", "model0": "A",
                                               "model1": "A", "output_name": "out"}]}))
    output = tmp_path / "plan.ipynb"
    plan.create_plan_ipynb(str(source), "/kaggle", str(output), "test", "", "", "", "User/Repo",
                          base_model=architecture)
    notebook = json.loads(output.read_text())
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 6
    for cell in notebook["cells"]:
        code = "".join(cell["source"])
        ast.parse(code)
    assert "/kaggle/working" in "".join(notebook["cells"][1]["source"])
