"""Syntax responsibilities."""
from __future__ import annotations

from typing import List
import json
import os
import shlex


def rnm(n):
    import random
    from .constants import hexchars
    return "".join(random.choices(hexchars, k=n))


def _uid():
    return f"{rnm(8)}-{rnm(4)}-{rnm(4)}-{rnm(4)}-{rnm(12)}"


def _ensure_dirs(root: str, subdirs: List[str]):
    for d in [root, *[os.path.join(root, x) for x in subdirs]]:
        os.makedirs(d, exist_ok=True)


def _nb_json(cells: List[str]) -> str:
    def wrap(src: str) -> str:
        return json.dumps({
            "cell_type": "code",
            "execution_count": None,
            "id": _uid(),
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in src.splitlines()],
        })

    return '{{"cells":[{cells}],"metadata":{{"kernelspec":{{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"}},"language_info":{{"name":"python","version":"3.10.6"}}}},"nbformat":4,"nbformat_minor":5}}'.format(
        cells=",".join(map(wrap, cells))
    )


def _split(s: str):
    return shlex.split(s, posix=True)


def _split_top_level(s: str, sep: str = ",") -> List[str]:
    opens = {"(": ")", "[": "]", "{": "}"}
    stack: List[str] = []
    out: List[str] = []
    buf: List[str] = []
    for ch in s:
        if ch in opens:
            stack.append(opens[ch])
        elif stack and ch == stack[-1]:
            stack.pop()
        elif ch == sep and not stack:
            part = "".join(buf).strip()
            if part:
                out.append(part)
            buf = []
            continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        out.append(part)
    return out


def _ensure_st(val: str) -> str:
    v = val.strip()
    return v if v.lower().endswith(".safetensors") else f"{v}.safetensors"


def _parse_lora_pairs(raw: str):
    items = _split_top_level(raw.strip(), ",")
    out = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        if ":" in it:
            name, ratio = it.split(":", 1)
            out.append((name.strip(), ratio.strip()))
        else:
            out.append((it.strip(), "1.0"))
    return out


def _needs_quote(val: str) -> bool:
    try:
        float(val)
        return False
    except:
        return True


def quoter(val: str) -> str:
    return f'"{val}"' if _needs_quote(val) else val


def _parse_tail_at(tokens):
    out = {
        "cosine": None,
        "fine": None,
        "seed": None,
        "mode": None,
        "precision": None,
        "rank": None,
        "arch": None,
        "extras": [],
    }
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("@"):
            out["extras"].append(t)
            i += 1
            continue
        k, v = t[1:].lower(), None
        if "=" in k:
            k, v = k.split("=", 1)
            v = v.strip('"').strip("'")
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("@"):
            v = tokens[i + 1].strip('"').strip("'")
            i += 1

        if k in ("cosine0", "cosine1", "cosine2", "c0", "c1", "c2"):
            out["cosine"] = int(k[-1])
        elif k in ("cosine", "c") and v is not None:
            out["cosine"] = int(v)
        elif k in ("fine", "f") and v is not None:
            out["fine"] = v
        elif k in ("s", "seed") and v is not None:
            out["seed"] = int(v)
        elif k in ("m", "mode") and v is not None:
            out["mode"] = v.upper()
        elif k in ("p", "precision") and v is not None:
            out["precision"] = v
        elif k in ("rank", "rk", "rnk") and v is not None:
            out["rank"] = int(v)
        elif k in ("arch", "a") and v is not None:
            out["arch"] = v.lower()
        elif t.startswith("@"):
            if k in ("bake_fp32","b32"):
                out["extras"].append("--bake_fp32")
            else:
                out["extras"].append(f"--{k} {v}")
        else:
            out["extras"].append(tokens[i])
        i += 1
    return out
