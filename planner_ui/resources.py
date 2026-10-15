"""Resources responsibilities."""
from __future__ import annotations

import os


def _planner_safe_cpu_threads() -> int:
    for env_name in ("PLANNER_CPU_THREADS", "CHATTIORI_CPU_THREADS"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    cpus = os.cpu_count() or 4
    return max(1, min(8, max(1, cpus // 2)))
