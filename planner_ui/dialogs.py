"""macOS-safe native dialogs, scoped to the Planner only."""
from tkinter.filedialog import askdirectory
from .native_dialogs import _tccm_askopenfilename_safe as askopenfilename
from .native_dialogs import _tccm_askopenfilenames_safe as askopenfilenames
from .native_dialogs import _tccm_asksaveasfilename_safe as asksaveasfilename
