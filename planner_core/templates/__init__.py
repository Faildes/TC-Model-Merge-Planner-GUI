"""Standalone script/notebook cell templates."""
from pathlib import Path
from string import Template

_ROOT = Path(__file__).resolve().parent

INSTALL_TPL = Template((_ROOT / "install_tpl.py.tpl").read_text(encoding="utf-8"))
PRELUDE_TPL = Template((_ROOT / "prelude_tpl.py.tpl").read_text(encoding="utf-8"))
UPLOAD_TPL = Template((_ROOT / "upload_tpl.py.tpl").read_text(encoding="utf-8"))
T2I_CFG_TPL = Template((_ROOT / "t2i_cfg_tpl.py.tpl").read_text(encoding="utf-8"))
T2I_RUN_TPL = Template((_ROOT / "t2i_run_tpl.py.tpl").read_text(encoding="utf-8"))
ZIP_TPL = Template((_ROOT / "zip_tpl.py.tpl").read_text(encoding="utf-8"))
