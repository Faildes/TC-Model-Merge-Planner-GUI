"""Application resource root, independent of the current directory."""
from pathlib import Path
RESOURCE_ROOT = Path(__file__).resolve().parent.parent
