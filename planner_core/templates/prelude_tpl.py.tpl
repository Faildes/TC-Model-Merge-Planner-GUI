# Planner runtime prelude
import gc
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import traceback
import errno
import copy
import time
from pathlib import Path

import filelock
import requests
import torch
from fake_useragent import UserAgent

HFToken = "$hf_token"
CVToken = "$cv_token"
VAE_URL = "$vae_url".strip()
VAE_NAME = "$vae_name".strip() or "VAE"
BAKE_VAE = $bake_vae
USE_ONLINE = $use_online
WORKPATH_RAW = r"$workpath"


def _planner_normalize_workpath(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(path or ".")))
    # Kaggle's /kaggle root is read-only for user-created files. Use /kaggle/working.
    if os.path.normpath(path) == os.path.normpath("/kaggle"):
        return os.path.join(path, "working")
    return path


def _planner_working_dir(path: str) -> str:
    path = _planner_normalize_workpath(path)
    # If the user already chose /kaggle/working, do not append another /working.
    if os.path.basename(os.path.normpath(path)).lower() == "working":
        return path
    return os.path.join(path, "working")


workpath = _planner_normalize_workpath(WORKPATH_RAW)
WORKING_DIR = _planner_working_dir(workpath)
_md = r"$model_dir"
_vd = r"$vae_dir"

models_dir = _md if _md else f"{workpath}/tmp/models"
vae_dir = _vd if _vd else f"{workpath}/tmp/vae"
emb_dir = f"{workpath}/tmp/embeddings"
local_merge_repo_dir = r"$toolpath"
merge_repo_dir = os.path.join(WORKING_DIR, "tools", "chattiori_model_merger") if USE_ONLINE else local_merge_repo_dir
MERGE_PY = os.path.join(merge_repo_dir, "merge.py")
LORA_BAKE_PY = os.path.join(merge_repo_dir, "lora_bake.py")
print(f"[planner] use_online={USE_ONLINE} merge_repo_dir={merge_repo_dir}")
for p in (f"{workpath}/tmp", models_dir, vae_dir, emb_dir):
    os.makedirs(p, exist_ok=True)

MODEL_REGISTRY = {}
MODEL_REGISTRY_HISTORY = []
MODEL_INSTALL_COUNTER = 0
REMOVED_MODELS = set()
SOURCE_MODEL_CACHE = {}
DOWNLOAD_INFO_CACHE = {}
_DOWNLOAD_USER_AGENT = None
PLANNER_REGISTRY_DIR = os.path.join(models_dir, "_planner_registry")
PLANNER_CACHE_TTL_HOURS = float(os.environ.get("PLANNER_CACHE_TTL_HOURS", "72") or 72)
PLANNER_LOW_DISK_GB = float(os.environ.get("PLANNER_LOW_DISK_GB", "8") or 8)
PLANNER_PURGE_PIP_CACHE = str(os.environ.get("PLANNER_PURGE_PIP_CACHE", "0")).strip().lower() in {"1", "true", "yes", "on"}
_PIP_CACHE_PURGED = False


def _path_is_relative_to(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _is_planner_registry_path(path):
    return bool(path) and _path_is_relative_to(path, PLANNER_REGISTRY_DIR)


def _is_model_dir_path(path):
    return bool(path) and _path_is_relative_to(path, models_dir)


def _safe_disk_remove(path, *, source_kind="registered", reason="cleanup"):
    path = str(path or "")
    if not path or not os.path.lexists(path):
        return False
    source_kind = str(source_kind or "registered")
    owns_path = (
        source_kind in ("generated", "download")
        or _is_planner_registry_path(path)
    )
    # Existing user files and raw local paths outside the planner registry must never be removed.
    if not owns_path:
        return False
    try:
        label = os.path.basename(path.rstrip(os.sep)) or path
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"[cleanup] removed {label} ({source_kind}, {reason})")
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[cleanup] could not remove {path}: {e}")
        return False


def _prune_empty_parent_dirs(path):
    try:
        parent = Path(path).resolve().parent
        root = Path(PLANNER_REGISTRY_DIR).resolve()
        while parent != root and root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    except Exception:
        pass


def _release_record(record, *, reason="cleanup", unregister=True):
    if not record:
        return False
    alias = str(record.get("alias") or "")
    path = str(record.get("path") or "")
    source_kind = str(record.get("source_kind") or "registered")
    removed = _safe_disk_remove(path, source_kind=source_kind, reason=reason)
    if removed:
        _prune_empty_parent_dirs(path)
    if unregister and alias and MODEL_REGISTRY.get(alias) is record:
        MODEL_REGISTRY.pop(alias, None)
    return removed


def _drop_source_cache_path(path):
    if not path:
        return
    try:
        target = str(Path(path).resolve())
    except Exception:
        target = os.path.abspath(str(path))
    for key, cached in list(SOURCE_MODEL_CACHE.items()):
        try:
            cached_resolved = str(Path(cached).resolve())
        except Exception:
            cached_resolved = os.path.abspath(str(cached))
        if cached_resolved == target:
            SOURCE_MODEL_CACHE.pop(key, None)


def _current_active_paths():
    paths = set()
    for info in MODEL_REGISTRY.values():
        path = str(info.get("path") or "")
        if path:
            try:
                paths.add(str(Path(path).resolve()))
            except Exception:
                paths.add(os.path.abspath(path))
    for path in SOURCE_MODEL_CACHE.values():
        if path:
            try:
                paths.add(str(Path(path).resolve()))
            except Exception:
                paths.add(os.path.abspath(path))
    return paths


def _prune_orphan_registry_dirs(max_age_hours=None, *, aggressive=False):
    root = Path(PLANNER_REGISTRY_DIR)
    if not root.exists():
        return 0
    now = time.time()
    max_age = PLANNER_CACHE_TTL_HOURS if max_age_hours is None else float(max_age_hours)
    active_paths = _current_active_paths()
    removed = 0
    for child in sorted(root.iterdir(), key=lambda p: str(p)):
        try:
            resolved = str(child.resolve())
            if any(ap == resolved or ap.startswith(resolved + os.sep) for ap in active_paths):
                continue
            age_hours = (now - child.stat().st_mtime) / 3600.0
            if aggressive or age_hours >= max_age:
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
                removed += 1
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"[cleanup] could not prune {child}: {e}")
    if removed:
        print(f"[cleanup] pruned {removed} stale planner cache item(s)")
    return removed


def _prune_dead_hash_cache():
    global cache_data
    if cache_data is None:
        return
    changed = False
    active_titles = set()
    for info in MODEL_REGISTRY.values():
        alias = str(info.get("alias") or "")
        mode = str(info.get("mode") or "checkpoint")
        if alias:
            active_titles.add(f"{mode}/{alias}")
    for section in ("hashes", "hashes-addnet"):
        data = cache_data.get(section)
        if not isinstance(data, dict):
            continue
        for title in list(data.keys()):
            if title.startswith(("checkpoint/", "lora/")) and active_titles and title not in active_titles:
                data.pop(title, None)
                changed = True
    if changed:
        dump_cache()


def _disk_free_gb(path=None):
    try:
        total, used, free = shutil.disk_usage(path or models_dir)
        return free / (2**30), total / (2**30)
    except Exception:
        total, used, free = shutil.disk_usage("/")
        return free / (2**30), total / (2**30)


def flush(light=True):
    global _PIP_CACHE_PURGED
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    free_gb, total_gb = _disk_free_gb()
    if not light:
        _prune_orphan_registry_dirs(aggressive=free_gb < PLANNER_LOW_DISK_GB)
    if PLANNER_PURGE_PIP_CACHE and free_gb < PLANNER_LOW_DISK_GB and not _PIP_CACHE_PURGED:
        subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
        _PIP_CACHE_PURGED = True
    if free_gb < PLANNER_LOW_DISK_GB:
        print(f"[cleanup] low disk: {free_gb:.2f}GB/{total_gb:.2f}GB free")


def release_registered_model(name, *, reason="last-use"):
    alias = str(name or "").strip()
    if not alias:
        return
    info = MODEL_REGISTRY.pop(alias, None)
    REMOVED_MODELS.add(alias)
    if not info:
        print(f"[cleanup] {alias}: already absent ({reason})")
        return
    _drop_source_cache_path(info.get("path", ""))
    _release_record(info, reason=reason, unregister=False)
    print(f"[cleanup] {alias}: released ({reason})")


def release_models_after_step(names, *, reason="last-use"):
    for name in names or []:
        release_registered_model(name, reason=reason)
    _prune_dead_hash_cache()
    flush(light=True)


def _ensure_runtime_path():
    extras = [
        os.path.expanduser("~/.local/bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/opt/local/bin",
        "/usr/bin",
        "/bin",
    ]
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    for p in extras:
        if p and p not in parts and os.path.isdir(p):
            parts.append(p)
    os.environ["PATH"] = os.pathsep.join(parts)


def _resolve_executable(name):
    _ensure_runtime_path()
    resolved = shutil.which(name)
    if resolved:
        return resolved
    candidates = [
        name,
        os.path.expanduser(f"~/.local/bin/{name}"),
        f"/opt/homebrew/bin/{name}",
        f"/usr/local/bin/{name}",
        f"/opt/local/bin/{name}",
        f"/usr/bin/{name}",
        f"/bin/{name}",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        errno.ENOENT,
        f"Executable not found: {name}. PATH={os.environ.get('PATH','')}",
        name,
    )


SHELL_META_RE = re.compile(r'[ \t\n\r|&;<>()[\]{}$`!*?~"\'\\]')


def _is_progressish_text(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    lower = stripped.lower()
    return (
        stripped.startswith("[#")
        or "%|" in stripped
        or "it/s" in lower
        or "s/it" in lower
        or ("dl:" in lower and ("eta:" in lower or "%" in stripped))
    )


def _stream_subprocess_output(proc, *, progress_prefix: bool = False):
    current = ""
    while True:
        chunk = proc.stdout.read(1)
        if chunk == "":
            break
        if chunk == "\r":
            text = current.strip()
            if text:
                if progress_prefix or _is_progressish_text(text):
                    print(f"[planner-progress] {text}")
                else:
                    print(text)
            current = ""
            continue
        if chunk == "\n":
            text = current.rstrip()
            if text:
                if progress_prefix or _is_progressish_text(text):
                    print(f"[planner-progress] {text}")
                else:
                    print(text)
            else:
                print()
            current = ""
            continue
        current += chunk
    tail = current.strip()
    if tail:
        if progress_prefix or _is_progressish_text(tail):
            print(f"[planner-progress] {tail}")
        else:
            print(tail)


def run_cmd(cmd, cwd=None, check_path: bool=False, path: str="", ignore_meta: bool=False):
    if not cmd:
        raise ValueError("cmd must not be empty")
    cmd = [x.strip() for x in cmd]
    prefer_stream = os.path.basename(str(cmd[0])).lower() == "aria2c"
    if not prefer_stream:
        try:
            cmd_ipython = copy.deepcopy(cmd)
            if not ignore_meta:
                for i, value in enumerate(cmd_ipython):
                    value = str(value)

                    if SHELL_META_RE.search(value):
                        value = (
                            value
                            .replace("\\", "\\\\")
                            .replace('"', '\\"')
                            .replace("$", "\\$")
                            .replace("`", "\\`")
                            .replace("!", "\\!")
                        )
                        cmd_ipython[i] = f'"{value}"'

            try:
                ip = get_ipython()
                shell_line = " ".join(cmd_ipython)
                rc = ip.system(shell_line)
                # Real IPython returns None and stores the status in _exit_code;
                # the Task Center proxy returns the code directly.  Handle both
                # so a failed command can never be mistaken for success.
                if rc is None:
                    try:
                        rc = int(getattr(ip, "user_ns", {}).get("_exit_code", 0) or 0)
                    except Exception:
                        rc = 0
                if int(rc or 0) != 0:
                    raise RuntimeError(f"Notebook shell command failed with exit code {rc}: {shell_line}")
                if check_path and not os.path.exists(path):
                    raise FileNotFoundError(path)
                return
            finally:
                flush(light=True)
        except:
            pass

    try:
        cmd[0] = _resolve_executable(str(cmd[0]))
    except FileNotFoundError as e:
        print(f"[run_cmd] missing executable: {cmd[0]}")
        print(f"[run_cmd] cwd={cwd}")
        print(f"[run_cmd] PATH={os.environ.get('PATH','')}")
        raise
    pretty = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"$ {pretty}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    try:
        _stream_subprocess_output(proc, progress_prefix=prefer_stream)
        code = proc.wait()
        if check_path and not os.path.exists(path):
            raise FileNotFoundError(path)
        if code != 0:
            raise RuntimeError(f"Command failed with exit code {code}: {pretty}")
    finally:
        flush(light=True)


def register_model(name, path, mode="checkpoint", *, source_kind="registered", source_identity="", original_path=""):
    global MODEL_INSTALL_COUNTER
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    alias = str(name or "").strip()
    if not alias:
        raise ValueError("Model alias is empty")
    MODEL_INSTALL_COUNTER += 1
    record = {
        "alias": alias,
        "path": str(path),
        "basename": os.path.basename(path),
        "filename": os.path.basename(path),
        "mode": mode,
        "source_kind": str(source_kind or "registered"),
        "source_identity": str(source_identity or ""),
        "original_path": str(original_path or ""),
        "install_index": MODEL_INSTALL_COUNTER,
        "install_stage": f"install_{MODEL_INSTALL_COUNTER:04d}",
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    previous = MODEL_REGISTRY.get(alias)
    if previous and os.path.abspath(str(previous.get("path", ""))) != os.path.abspath(str(path)):
        _drop_source_cache_path(previous.get("path", ""))
        _release_record(previous, reason="superseded", unregister=False)
    MODEL_REGISTRY[alias] = record
    MODEL_REGISTRY_HISTORY.append(dict(record))
    if alias in REMOVED_MODELS:
        REMOVED_MODELS.remove(alias)
    print(f"[register] {alias} -> {record['basename']} ({mode}, {record['install_stage']}, {record['source_kind']})")
    return str(path)


def resolve_model_path(name):
    info = MODEL_REGISTRY.get(name)
    if info:
        return info["path"]
    p = Path(models_dir) / f"{name}.safetensors"
    if p.exists():
        return str(p)
    p = Path(models_dir) / f"{name}.ckpt"
    if p.exists():
        return str(p)
    raise FileNotFoundError(f"Model not found: {name}")


def model_file(name):
    path = resolve_model_path(name)
    try:
        p = Path(path).resolve()
        base = Path(models_dir).resolve()
        rel = os.path.relpath(str(p), str(base))
        if rel != ".." and not rel.startswith(".." + os.sep) and not os.path.isabs(rel):
            return rel.replace(os.sep, "/")
    except Exception:
        pass
    return str(path)


def remove_registered_model(name):
    release_registered_model(name, reason="explicit-remove")
    free_gb, total_gb = _disk_free_gb()
    print(f"Remain Storage: {free_gb:.2f}GB/{total_gb:.2f}GB")


def get_vae_path(warn=True):
    for name in ("$vae_name.safetensors", "$vae_name.ckpt"):
        candidate = os.path.join(vae_dir, name)
        if os.path.exists(candidate):
            return candidate
    if warn:
        print(f"⚠️ No VAE found in {vae_dir}, merges may fail")
    return None

vae_path = get_vae_path(warn=False) if BAKE_VAE else None

pref = {"format": "SafeTensor", "size": "pruned", "fp": "fp16"}
cache_filename = os.path.join(models_dir, "cache.json")
cache_data = None


def cache(subsection):
    global cache_data
    if cache_data is None:
        with filelock.FileLock(f"{cache_filename}.lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)
    s = cache_data.get(subsection, {})
    cache_data[subsection] = s
    return s


def dump_cache():
    with filelock.FileLock(f"{cache_filename}.lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    ondisk_mtime = os.path.getmtime(filename)
    if title not in hashes:
        return None
    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)
    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None
    return cached_sha256


def addnet_hash_safetensors(b):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024
    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")
    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sha256(filename, title, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value
    print(f"Calculating sha256 for {filename}: ", end='')
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")
    hashes[title] = {"mtime": os.path.getmtime(filename), "sha256": sha256_value}
    dump_cache()
    return sha256_value


def sha256_set(filename, title, sha256_value, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    hashes[title] = {"mtime": os.path.getmtime(filename), "sha256": sha256_value}
    dump_cache()


def make_pref(p, mode):
    pref_set = {
        "size": ["full", "pruned"],
        "fp": ["fp16", "bf16", "fp8", "fp32"],
        "format": ["PickleTensor", "SafeTensor"],
    }

    def ordered(values, preferred):
        preferred = str(preferred or "").strip()
        out = [preferred] if preferred in values else []
        out.extend([v for v in values if v not in out])
        return out

    p = dict(p or {})
    if mode == "lora":
        return [{"format": fmt} for fmt in ordered(pref_set["format"], p.get("format", "SafeTensor"))]

    if mode == "checkpoint":
        size_order = ordered(pref_set["size"], p.get("size", "full"))
        fp_order = ordered(pref_set["fp"], p.get("fp", "fp16"))
        format_order = ordered(pref_set["format"], p.get("format", "SafeTensor"))
        res = []
        for fmt in format_order:
            for fp in fp_order:
                for size in size_order:
                    res.append({"size": size, "fp": fp, "format": fmt})
        return res
    raise ValueError(f"Unknown mode: {mode}")


def _metadata_matches_pref(metadata, pref_candidate):
    """Civitai may add extra keys such as isRequired; compare pref as a subset."""
    metadata = metadata or {}
    for key, expected in (pref_candidate or {}).items():
        if str(metadata.get(key)) != str(expected):
            return False
    return True


def _is_civitai_payload_file(file_info):
    """Exclude Required Components such as VAE / Text Encoder from model downloads."""
    if not isinstance(file_info, dict):
        return False
    if file_info.get("type") != "Model":
        return False
    return True


def _select_civitai_file(files, prefs):
    files = list(files or [])
    model_files = [f for f in files if _is_civitai_payload_file(f)]

    # Prefer real model payload files. Only fall back to all files for unusual APIs
    # where the file type is absent, not for Required Components.
    candidates = model_files or [f for f in files if not (f.get("metadata") or {}).get("isRequired")] or files
    if not candidates:
        return None

    for pref_candidate in prefs:
        for f in candidates:
            if _metadata_matches_pref(f.get("metadata", {}), pref_candidate):
                return f

    for f in candidates:
        if f.get("primary") is True:
            return f

    return candidates[0]


def _metadata_get(url, **kwargs):
    last_error = None
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=(10, 30), **kwargs)
            if response.status_code not in (429, 500, 502, 503, 504):
                return response
            last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
            response.close()
        except requests.RequestException as e:
            last_error = e
        if attempt == 0:
            print(f"[planner-download] metadata HTTP retry | {url} | {last_error}")
            time.sleep(0.5)
    raise RuntimeError(f"Metadata request failed for {url}: {last_error}") from last_error


def get_dl(url, version=None, mode="checkpoint"):
    prefs = make_pref(pref, mode)
    if "civitai" in url:
        file = None
        if "/api/" in url:
            dllink = url
            dlname = None
            ext = 1
            sha256_value = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR"
        else:
            cid = re.sub(r"\D", "", re.search(r"models/[0-9]+", url).group())
            if "modelVersionId=" in url and version is None:
                version = re.sub(r"\D", "", re.search(r"modelVersionId=[0-9]+", url).group())
            api = f"https://civitai.red/api/v1/models/{cid}" if "civitai.red" in url else f"https://civitai.com/api/v1/models/{cid}" 
            response = _metadata_get(api)
            if response.status_code != 200:
                return None
            d = response.json()
            model_name = d["name"]
            model_version = version if version is not None else d["modelVersions"][0]["name"]
            model = d["modelVersions"][0]
            for k in d["modelVersions"]:
                if k["name"] == model_version or str(k["id"]) == str(model_version):
                    model = k
                    model_version = k["name"]
                    break
            file = _select_civitai_file(model.get("files", []), prefs)
            if file is None:
                return None
            dllink = file["downloadUrl"]
            sha256_value = file.get("hashes", {}).get("SHA256", "").lower() or None
            ext = 1 if file.get("metadata", {}).get("format") == "SafeTensor" else 0
            dlname = model_name + "-" + model_version
        filename = None
        try:
            filename = file.get("name") if file is not None else None
        except Exception:
            filename = None
        if not filename:
            filename = os.path.basename(str(dllink).split("?", 1)[0]) or ((dlname or "model") + (".safetensors" if ext == 1 else ".ckpt"))
        return {"url": dllink, "name": dlname, "format": ext, "sha256": sha256_value, "filename": filename}

    if "huggingface" in url:
        url_set = url.replace("https://huggingface.co/", "").split("/")
        base = "https://huggingface.co/"
        api = base
        dllink = base
        dname = url_set[-1].rsplit(".", 1)
        dlname = dname[0]
        ext = 1 if len(dname) > 1 and dname[1] == "safetensors" else 0
        for i, s in enumerate(url_set):
            if i == 2:
                api += "raw/"
                dllink += "resolve/"
            else:
                api += f"{s}/"
                dllink += f"{s}/"
        res = _metadata_get(api)
        if res.status_code != 200:
            return None
        match = re.search(r"sha256:[0-9a-f]+", res.text)
        sha256_value = match.group().replace("sha256:", "") if match else None
        filename = url_set[-1].split("?", 1)[0] or (dlname + (".safetensors" if ext == 1 else ".ckpt"))
        return {"url": dllink, "name": dlname, "format": ext, "sha256": sha256_value, "filename": filename}

    return None


def _safe_model_stem(value):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "model")).strip("._-")
    return stem or "model"


def _source_identity_hash(*parts):
    text = "|".join(str(x or "") for x in parts)
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]


def _safe_model_filename(value, ext="safetensors"):
    ext = str(ext or "safetensors").lstrip(".")
    raw = os.path.basename(str(value or "").split("?", 1)[0])
    if not raw:
        raw = f"model.{ext}"
    suffix = Path(raw).suffix or f".{ext}"
    stem = _safe_model_stem(Path(raw).stem)
    safe_suffix = re.sub(r"[^A-Za-z0-9.]+", "", suffix) or f".{ext}"
    return f"{stem}{safe_suffix}"


def _source_backed_destination(alias, ext, mode, source_identity, kind="src", filename=None):
    ext = str(ext or "safetensors").lstrip(".")
    suffix = _source_identity_hash(mode, alias, source_identity)
    folder = os.path.join(models_dir, "_planner_registry", f"{kind}_{suffix}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, _safe_model_filename(filename or f"{alias}.{ext}", ext))


def model(name, format=1, mode="checkpoint"):
    ext = "ckpt" if format == 0 else "safetensors"
    path = f"{models_dir}/{name}.{ext}"
    if os.path.exists(path):
        sha256_set(path, f"{mode}/{name}", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJKLMNOPQR")
    return register_model(name, path, mode, source_kind="existing")


def _download_user_agent():
    global _DOWNLOAD_USER_AGENT
    if _DOWNLOAD_USER_AGENT:
        return _DOWNLOAD_USER_AGENT
    try:
        _DOWNLOAD_USER_AGENT = UserAgent().chrome
    except Exception:
        _DOWNLOAD_USER_AGENT = "Mozilla/5.0"
    return _DOWNLOAD_USER_AGENT


def _aria_headers(token="", *, include_auth=True):
    args = ["--header", f"User-Agent: {_download_user_agent()}"]
    if include_auth and token:
        args += ["--header", f"Authorization: Bearer {token}"]
    return args


def _download_file_is_ready(path):
    # O(1) only: do not hash/scan multi-GB model files on the hot path.
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0 and not os.path.exists(path + ".aria2")
    except OSError:
        return False


def _download_info_cached(raw_url, version, mode):
    key = (str(mode or "checkpoint"), str(raw_url or ""), str(version or ""))
    cached = DOWNLOAD_INFO_CACHE.get(key)
    if cached is not None:
        print(f"[planner-download] metadata cache hit | {raw_url}")
        return dict(cached)
    last_error = None
    for attempt in range(2):
        try:
            info = get_dl(raw_url, version=version, mode=mode)
            if info:
                DOWNLOAD_INFO_CACHE[key] = dict(info)
                return dict(info)
            last_error = RuntimeError(f"Could not resolve download info for {raw_url}")
        except Exception as e:
            last_error = e
        if attempt == 0:
            print(f"[planner-download] metadata retry | {raw_url} | {last_error}")
            time.sleep(0.5)
    raise RuntimeError(f"Could not resolve download info for {raw_url}: {last_error}") from last_error


def _resolve_download_endpoint(url, token=""):
    # Resolve redirects without downloading the response body.  Civitai/CDN endpoints
    # occasionally return 200 directly instead of 30x; stream=True prevents Python
    # from silently downloading a multi-GB model before aria2 starts.
    headers = {"User-Agent": _download_user_agent()}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error = None
    for attempt in range(2):
        try:
            with requests.get(
                url, headers=headers, allow_redirects=False, stream=True, timeout=(10, 30)
            ) as response:
                status = int(response.status_code)
                location = response.headers.get("Location")
                if 300 <= status < 400 and location:
                    return location, False
                if status in (200, 206):
                    return url, True
                if status in (429, 500, 502, 503, 504):
                    last_error = RuntimeError(f"HTTP {status} while resolving {url}")
                else:
                    raise RuntimeError(f"HTTP {status} while resolving {url}")
        except requests.RequestException as e:
            last_error = e
        if attempt == 0:
            print(f"[planner-download] endpoint retry | {url} | {last_error}")
            time.sleep(0.5)
    raise RuntimeError(f"Could not resolve download endpoint {url}: {last_error}") from last_error


def _run_aria2_download(url, out_dir, out_name, *, token="", include_auth=False, fallback=False):
    connections = "8" if fallback else "16"
    print(f"[planner-download] aria2 start | connections={connections} | {out_name}")
    run_cmd([
        "aria2c", "--console-log-level=error", "-c", "-x", connections, "-s", connections, "-k", "1M",
        *_aria_headers(token, include_auth=include_auth),
        url, "-d", out_dir, "-o", out_name,
    ], check_path=True, path=os.path.join(out_dir, out_name))


def custom_model(url, checkpoint_name=None, mode="checkpoint"):
    raw_url = url[0] if isinstance(url, list) else url
    version = url[1] if isinstance(url, list) and len(url) > 1 else None
    user_token = HFToken if "huggingface" in str(raw_url) else CVToken
    g = _download_info_cached(raw_url, version, mode)
    resolved_url = g["url"]
    checkpoint_name = g["name"] if checkpoint_name is None else checkpoint_name
    sha256_value = g["sha256"]
    ext = "ckpt" if g["format"] == 0 else "safetensors"
    source_identity = sha256_value or resolved_url
    filename = g.get("filename") or f"{_safe_model_stem(checkpoint_name)}.{ext}"
    cache_key = (mode, checkpoint_name, source_identity)
    dst = SOURCE_MODEL_CACHE.get(cache_key) or _source_backed_destination(checkpoint_name, ext, mode, source_identity, kind="dl", filename=filename)
    SOURCE_MODEL_CACHE[cache_key] = dst
    if _download_file_is_ready(dst):
        if sha256_value is not None:
            sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
        return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=resolved_url)
    out_name = os.path.basename(dst)
    out_dir = os.path.dirname(dst)
    os.makedirs(out_dir, exist_ok=True)

    if "huggingface" in str(resolved_url):
        try:
            _run_aria2_download(resolved_url, out_dir, out_name, token=user_token, include_auth=True)
        except Exception as first_error:
            print(f"[planner-download] aria2 retry | {out_name} | {first_error}")
            time.sleep(0.5)
            _run_aria2_download(resolved_url, out_dir, out_name, token=user_token, include_auth=True, fallback=True)
    else:
        try:
            download_link, needs_auth = _resolve_download_endpoint(resolved_url, user_token)
            _run_aria2_download(download_link, out_dir, out_name, token=user_token, include_auth=needs_auth)
        except Exception as first_error:
            print(f"[planner-download] aria2/endpoint retry | {out_name} | {first_error}")
            time.sleep(0.5)
            # Re-resolve a fresh signed CDN URL only after failure.  Successful
            # downloads keep the original 16-way fast path with no extra delay.
            download_link, needs_auth = _resolve_download_endpoint(resolved_url, user_token)
            _run_aria2_download(download_link, out_dir, out_name, token=user_token, include_auth=needs_auth, fallback=True)

    if not _download_file_is_ready(dst):
        raise RuntimeError(f"Download did not complete: {dst}")
    if sha256_value is not None:
        sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
    return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=resolved_url)


def custom_vae(url, vae_name="VAE"):
    url = str(url or "").strip()
    vae_name = str(vae_name or "VAE").strip() or "VAE"
    if not url:
        return get_vae_path()
    user_token = HFToken if "huggingface" in url else CVToken
    if "civitai" in url:
        if "/api/" in url:
            ext = "safetensors" if "SafeTensor" in url else "ckpt"
            download_link, needs_auth = _resolve_download_endpoint(url, user_token)
            _run_aria2_download(download_link, vae_dir, f"{vae_name}.{ext}", token=user_token, include_auth=needs_auth)
            # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{vae_dir}" -o {vae_name}.{ext}
            return os.path.join(vae_dir, f"{vae_name}.{ext}")
        pref_order = ["SafeTensor", "PickleTensor"]
        cid_match = re.search(r"models/[0-9]+", url)
        if not cid_match:
            raise RuntimeError(f"Could not resolve civitai VAE model id from {url}")
        cid = re.sub(r"\D", "", cid_match.group())
        version_match = re.search(r"modelVersionId=[0-9]+", url)
        version = re.sub(r"\D", "", version_match.group()) if version_match else None
        api = f"https://civitai.red/api/v1/models/{cid}" if "civitai.red" in url else f"https://civitai.com/api/v1/models/{cid}" 
        response = _metadata_get(api)
        if response.status_code != 200:
            raise RuntimeError("ERROR: VAE Not Found")
        d = response.json()
        model_name = vae_name or d["name"]
        model_version = version if version is not None else d["modelVersions"][0]["name"]
        model = d["modelVersions"][0]
        for k in d["modelVersions"]:
            if k["name"] == model_version or str(k["id"]) == str(model_version):
                model = k
                break
        file = None
        meta_list = [a.get("metadata", {}).get("format") for a in model["files"]]
        for pref_candidate in pref_order:
            if pref_candidate in meta_list:
                file = model["files"][meta_list.index(pref_candidate)]
                break
        if file is None:
            file = model["files"][0]
        ext = "safetensors" if file.get("metadata", {}).get("format") == "SafeTensor" else "ckpt"
        download_link, needs_auth = _resolve_download_endpoint(file["downloadUrl"], user_token)
        _run_aria2_download(download_link, vae_dir, f"{model_name}.{ext}", token=user_token, include_auth=needs_auth)
        # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M "{download_link}" -d "{vae_dir}" -o {model_name}.{ext}
        return os.path.join(vae_dir, f"{model_name}.{ext}")
    if "huggingface" in url:
        filename = url.split("/")[-1]
        ext = filename.split(".")[-1] if "." in filename else "safetensors"
        resolved = url.replace("/blob/main/", "/resolve/main/") if "/blob/main/" in url else url
        user_header = f"\"Authorization: Bearer {user_token}\""
        _run_aria2_download(resolved, vae_dir, f"{vae_name}.{ext}", token=user_token, include_auth=True)
        # !aria2c --console-log-level=error -c -x 16 -s 16 -k 1M --header={user_header} "{resolved}" -d "{vae_dir}" -o {vae_name}.{ext}
        return os.path.join(vae_dir, f"{vae_name}.{ext}")
    return None


def old_custom_model(url, checkpoint_name=None, format=1, sha256_value=None, mode="checkpoint"):
    checkpoint_name = checkpoint_name or _safe_model_stem(Path(str(url).split("?")[0]).stem or "model")
    ext = "ckpt" if format == 0 else "safetensors"
    source_identity = sha256_value or url
    filename = os.path.basename(str(url).split("?", 1)[0]) or f"{_safe_model_stem(checkpoint_name)}.{ext}"
    cache_key = (mode, checkpoint_name, source_identity)
    dst = SOURCE_MODEL_CACHE.get(cache_key) or _source_backed_destination(checkpoint_name, ext, mode, source_identity, kind="dl", filename=filename)
    SOURCE_MODEL_CACHE[cache_key] = dst
    if _download_file_is_ready(dst):
        if sha256_value is not None:
            sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
        return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=url)
    out_name = os.path.basename(dst)
    out_dir = os.path.dirname(dst)
    if "huggingface" in str(url):
        _run_aria2_download(url, out_dir, out_name, token=HFToken, include_auth=True)
    else:
        download_link, needs_auth = _resolve_download_endpoint(url, CVToken)
        _run_aria2_download(download_link, out_dir, out_name, token=CVToken, include_auth=needs_auth)
    if sha256_value is not None:
        sha256_set(dst, f"{mode}/{checkpoint_name}", sha256_value)
    return register_model(checkpoint_name, dst, mode, source_kind="download", source_identity=source_identity, original_path=url)


def local_model(src, alias=None, mode="checkpoint"):
    src = os.path.abspath(os.path.expanduser(str(src)))
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    alias = alias or Path(src).stem
    ext = (Path(src).suffix or ".safetensors").lstrip(".")
    try:
        stat = os.stat(src)
        source_identity = f"{src}:{stat.st_size}:{stat.st_mtime_ns}"
    except Exception:
        source_identity = src
    cache_key = (mode, alias, source_identity)
    dst = SOURCE_MODEL_CACHE.get(cache_key) or _source_backed_destination(alias, ext, mode, source_identity, kind="local", filename=Path(src).name)
    SOURCE_MODEL_CACHE[cache_key] = dst
    # Do not duplicate local models.  Use a lightweight link inside the planner registry
    # when possible so merge.py can still receive a path relative to models_dir.
    if os.path.abspath(src) != os.path.abspath(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            if os.path.lexists(dst) and os.path.islink(dst) and os.path.realpath(dst) != src:
                os.unlink(dst)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        except Exception:
            try:
                if not os.path.exists(dst):
                    os.link(src, dst)
            except Exception:
                dst = src
    return register_model(alias, dst, mode, source_kind="local", source_identity=source_identity, original_path=src)


def ratio_value(spec):
    if not spec:
        return "0.5"
    mode = spec.get("mode", "Single")
    value = str(spec.get("value", "")).strip()
    if mode == "Block weight":
        if not value:
            value = ",".join(["0"] * 20)
        return value
    return value or "0.5"


def ratio_args(flag, spec):
    mode = ""
    if isinstance(spec, dict):
        mode = str(spec.get("mode", ""))
    value = ratio_value(spec)
    text = str(value).strip()
    is_rand = mode.lower() == "randomize" or text.lower().startswith(("@r", "@rand"))
    if is_rand:
        text = re.sub(r"^@(r|rand)\s*", "", text, flags=re.IGNORECASE).strip()
    name = f"--rand_{flag}" if is_rand else f"--{flag}"
    return [name, text if is_rand else value]


def signature_args(text):
    text = (text or "").strip()
    if not text:
        return []
    return shlex.split(text.replace("\n", " "))


def _print_plan_failure(entry_index, entry_type, entry_id, entry_payload, body_lines):
    print("\n[PLAN FAILURE]")
    print(f"step={entry_index + 1} type={entry_type} id={entry_id}")
    try:
        print(json.dumps(entry_payload, ensure_ascii=False, indent=2))
    except Exception:
        print(repr(entry_payload))
    print("\n[STEP SOURCE]")
    for i, src in enumerate(body_lines, start=1):
        print(f"{i:02d}: {src}")
    traceback.print_exc()


def run_notebook_bang(source, cwd=None):
    source = str(source or "").strip()
    if not source:
        return
    lines = [ln.rstrip() for ln in source.splitlines() if ln.strip()]
    if not lines:
        return
    normalized = []
    for idx, ln in enumerate(lines):
        stripped = ln.lstrip()
        if idx == 0 and stripped.startswith("!"):
            stripped = stripped[1:]
        normalized.append(stripped)
    cmd = "\n".join(normalized).strip()
    shell_cmd = f"cd {shlex.quote(cwd)} && {cmd}" if cwd else cmd
    print(f"$ {shell_cmd}")
    ip = None
    try:
        ip = get_ipython()
    except Exception:
        ip = None
    if ip is not None:
        rc = ip.system(shell_cmd)
        if rc is None:
            try:
                rc = int(getattr(ip, "user_ns", {}).get("_exit_code", 0) or 0)
            except Exception:
                rc = 0
        if int(rc or 0) != 0:
            raise RuntimeError(f"Notebook shell command failed with exit code {rc}: {shell_cmd}")
        return
    proc = subprocess.Popen(shell_cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=os.environ.copy())
    for line in proc.stdout:
        print(line, end="")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Notebook shell command failed with exit code {code}: {shell_cmd}")


if BAKE_VAE and VAE_URL:
    try:
        custom_vae(VAE_URL, VAE_NAME)
    except Exception as e:
        print(f"VAE download failed: {e}")

vae_path = get_vae_path(warn=True) if BAKE_VAE else None
flush(light=False)
if not os.path.isdir(merge_repo_dir):
    raise FileNotFoundError(f"Merge repository not found: {merge_repo_dir}")
if not os.path.isfile(MERGE_PY):
    raise FileNotFoundError(f"merge.py not found in merge repository: {MERGE_PY}")
os.chdir(merge_repo_dir)
print(os.getcwd())
