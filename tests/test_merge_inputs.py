"""GUI and compiler input requirements must agree with the merger CLI."""
from types import SimpleNamespace
import pytest
import plan
from planner_core.compiler import _entry_to_lines
from planner_core.errors import PlanCompileError
from planner_core.merge_validation import required_model_slots,turbo_problem
from planner_ui.editor_support import _sigless_model_count
from planner_ui.source_support import _planner_entry_can_produce_final
from planner_ui.validation_support import _plan_entry_problem_map_linear

@pytest.mark.parametrize('mode,options,count',[('NoIn',{},1),('COMP',{},1),('WS',{},2),('AD',{},3),('NoIn',{'turbo':True},3),('WS',{'deturbo':True},3)])
def test_model_slot_counts(mode,options,count):
    entry={'type':'Checkpoint Merge','merge_mode':mode,'cli_options':options}
    assert len(required_model_slots(entry))==count
    assert _sigless_model_count(None,entry,{'key':mode,'needs_m2':mode=='AD'})==count

def test_noin_compiles_with_one_input():
    lines,output=_entry_to_lines({'type':'Checkpoint Merge','merge_mode':'NoIn','model0':'A','output_name':'out'})
    assert output=='out'
    command=next(line for line in lines if line.startswith('cmd ='))
    assert 'model_file("A")' in command and 'model_file("")' not in command

@pytest.mark.parametrize('slot',['model0','model1','model2'])
def test_turbo_cannot_compile_missing_reference(slot):
    entry={'type':'Checkpoint Merge','model0':'A','model1':'B','model2':'C','output_name':'out','cli_options':{'turbo':True}}
    entry[slot]=''
    with pytest.raises(PlanCompileError,match='Turbo requires'):_entry_to_lines(entry)

@pytest.mark.parametrize('options',[{'turbo':True,'deturbo':True},{'turbo':True,'use_dif_21':True},{'turbo':True,'cosine0':True}])
def test_turbo_conflicts_are_actionable(options):
    entry={'type':'Checkpoint Merge','cli_options':options}
    assert turbo_problem(entry)
    with pytest.raises(PlanCompileError):_entry_to_lines(entry)

def test_legacy_option_survives_json_and_command_generation():
    row={'type':'Checkpoint Merge','model0':'A','model1':'B','model2':'C','output_name':'out','cli_options':{'turbo':True,'turbo_method':'legacy'}}
    actual=plan.normalize_plan({'entries':[row]})['entries'][0]
    assert actual['cli_options']['turbo_method']=='legacy'
    lines,_=_entry_to_lines(actual)
    assert any('--turbo_method legacy' in line for line in lines)

def test_invalid_bake_does_not_publish_alias():
    app=SimpleNamespace(_entry_is_disabled=lambda _:False,_planner_ref_resolves_final=lambda *_:True)
    assert not _planner_entry_can_produce_final(app,{'type':'LoRA Bake','checkpoint':'A','output_name':'out','loras':[]},{})

def test_linear_validation_checks_third_model_and_accepts_noin():
    app=SimpleNamespace(plan_data={'entries':[{'type':'Local Model','model_name':'A','local_path':'/data/a'},
        {'type':'Checkpoint Merge','merge_mode':'NoIn','model0':'A','output_name':'noin'},
        {'type':'Checkpoint Merge','merge_mode':'NoIn','model0':'A','output_name':'turbo','cli_options':{'turbo':True}}]},
        _planner_analysis=lambda:{},_entry_is_disabled=lambda _:False,_entry_is_locked=lambda _:False,
        _get_entry_slot_source=lambda *_:None,_entry_produced_aliases=lambda _:[])
    result=_plan_entry_problem_map_linear(app)
    assert not any('Model 1 Is Empty' in issue for issue in result[1])
    assert any('Model 1 Is Empty' in issue for issue in result[2])
    assert any('Model 2 Is Empty' in issue for issue in result[2])
