import json
import subprocess
import sys
from pathlib import Path

def test_tooltip_customizations_are_read_at_startup(tmp_path):
    root=Path(__file__).resolve().parents[1]
    (tmp_path/'planner_tooltips.json').write_text(json.dumps({'options': {'custom': 'Edited help'}}))
    code="""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import planner_ui.paths as paths
paths.RESOURCE_ROOT=Path(sys.argv[2])
import planner_ui.settings as settings
assert settings._OPTION_HELP_TEXT['custom']=='Edited help'
"""
    subprocess.run([sys.executable,'-c',code,str(root),str(tmp_path)],check=True,timeout=15)
