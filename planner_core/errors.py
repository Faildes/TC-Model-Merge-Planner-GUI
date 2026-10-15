"""Errors responsibilities."""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List


class PlanCompileError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        entry_index: int | None = None,
        entry_type: str | None = None,
        entry: Dict[str, Any] | None = None,
        entry_id: str | None = None,
        entry_payload: Any = None,
        cause: Exception | None = None,
        source_lines: List[str] | None = None,
    ):
        self.entry_index = entry_index
        self.entry_type = entry_type
        self.entry_id = entry_id
        self.entry = entry_payload if entry_payload is not None else (entry or {})
        self.entry_payload = self.entry
        self.cause = cause
        self.source_lines = list(source_lines or [])

        parts = [message]
        if entry_index is not None:
            parts.append(f"entry_index={entry_index}")
        if entry_type:
            parts.append(f"entry_type={entry_type}")
        if entry_id:
            parts.append(f"entry_id={entry_id}")
        if cause is not None:
            parts.append(f"cause={type(cause).__name__}: {cause}")
        super().__init__(" | ".join(parts))
