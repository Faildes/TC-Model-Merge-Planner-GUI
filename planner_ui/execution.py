"""Execution controls and callbacks."""
from __future__ import annotations

from . import dialogs as filedialog
from .developer_tools import _planner_get_t2i_settings as _method_planner_get_t2i_settings
from .developer_tools import _planner_open_t2i_settings as _method_planner_open_t2i_settings
from .developer_tools import _planner_set_t2i_settings as _method_planner_set_t2i_settings
from .document_support import _get_notebook_paths_project_json as _method_get_notebook_paths
from IPython.core.interactiveshell import InteractiveShell
from huggingface_hub import HfApi
from pathlib import Path
from plan import create_plan_ipynb
from tkinter import messagebox
from typing import Dict
from typing import List
import importlib.util
import nbformat
import os
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk


class ExecutionMixin:
    """Execution behavior for the Planner."""

    _get_notebook_paths = _method_get_notebook_paths


    def _export_as_notebook(self):
        try:
            temp_plan_path = self._write_temp_plan_text()
            default_nb_path, _ = self._get_notebook_paths()
            path = filedialog.asksaveasfilename(
                title="Export notebook",
                defaultextension=".ipynb",
                initialfile=Path(default_nb_path).name,
                initialdir=str(Path(default_nb_path).parent),
                filetypes=[("Jupyter Notebook", "*.ipynb"), ("All files", "*.*")],
            )
            if not path:
                return
            params = {key: widget.get() for key, widget in self.entries.items()}
            create_plan_ipynb(
                filepath=temp_plan_path,
                workpath=params.get("workpath", ""),
                saveas=path,
                title=params.get("title", "merge_plan"),
                vae=params.get("vae", ""),
                vae_name=params.get("vae_name", "VAE"),
                bake_vae=bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get()),
                CivitAPI=params.get("CivitAPI", ""),
                HuggingAPI=params.get("HuggingAPI", ""),
                UR=params.get("UR", ""),
                model_dir=params.get("model_dir", ""),
                vae_dir=params.get("vae_dir", ""),
                base_model=self.config.get("base_model", "SDXL"),
                ignore_install_deps=bool(getattr(self, "ignore_install_deps_var", tk.BooleanVar(value=False)).get()),
                use_online=bool(getattr(self, "use_online_var", tk.BooleanVar(value=False)).get()),
                upload_after_merge=bool(getattr(self, "upload_after_merge_var", tk.BooleanVar(value=False)).get()),
                run_t2i=bool(getattr(self, "run_t2i_var", tk.BooleanVar(value=False)).get()),
                t2i_settings=self._planner_get_t2i_settings() if hasattr(self, "_planner_get_t2i_settings") else None
            )
            self.notebook_path_var.set(path)
            self.config["saveas"] = path
            self._save_current_state_to_config()
            self.status_label.config(text=f"Exported notebook: {os.path.basename(path)}")
            messagebox.showinfo("Export", f"Notebook exported successfully.\n\n{path}")
        except Exception as e:
            self._show_detailed_error("Export Notebook Error", e, context="Operation: export_as_notebook")
        finally:
            if temp_plan_path and os.path.exists(temp_plan_path):
                try:
                    os.remove(temp_plan_path)
                except Exception:
                    pass


    def _build_runner_notebook(self, notebook_path: str) -> str:
        """
        Create a temporary execution notebook whose code cells are transformed
        with the real IPython cell transformer.
        The original generated notebook is left untouched.
        """
        if nbformat is None:
            return notebook_path

        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        shell = InteractiveShell.instance()

        compat_cell = nbformat.v4.new_code_cell(
            """
import os
import subprocess
import sys
import builtins

_runner_builtin_print = builtins.print
_runner_event_path = os.environ.get("PLANNER_EVENT_FILE", "").strip()
_runner_event_fp = None
if _runner_event_path:
    try:
        _runner_event_fp = open(_runner_event_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        _runner_event_fp = None

def _runner_event_write(text):
    if _runner_event_fp is None:
        return
    try:
        value = str(text).replace("\\r", "\\n")
        _runner_event_fp.write(value)
        if value and not value.endswith("\\n"):
            _runner_event_fp.write("\\n")
        _runner_event_fp.flush()
    except Exception:
        pass

def print(*args, sep=" ", end="\\n", file=None, flush=False):
    _runner_builtin_print(*args, sep=sep, end=end, file=file, flush=flush)
    if file is None:
        try:
            rendered = sep.join(str(x) for x in args) + end
            if "[planner-progress]" in rendered:
                _runner_event_write(rendered)
        except Exception:
            pass

try:
    from IPython import get_ipython as _real_get_ipython
except Exception:
    _real_get_ipython = None

def _runner_compact(value):
    return str(value).replace("\\t", " ").replace("\\r", " ").replace("\\n", " ")[:900]

def _runner_call(command, kind="shell"):
    kwargs = {
        "shell": True,
        "cwd": os.getcwd(),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "errors": "replace",
    }
    if os.name != "nt":
        kwargs["executable"] = os.environ.get("SHELL", "/bin/bash")
    proc = subprocess.Popen(command, **kwargs)
    detail = _runner_compact(command)
    start_marker = f"[planner-process]\\tSTART\\t{proc.pid}\\t{kind}\\t{detail}"
    _runner_event_write(start_marker)
    print(start_marker, flush=True)
    if proc.stdout is not None:
        for raw in proc.stdout:
            _runner_event_write(raw)
            print(raw, end="" if raw.endswith("\\n") else "\\n", flush=True)
    rc = proc.wait()
    state = "END" if rc == 0 else "FAILED"
    end_marker = f"[planner-process]\\t{state}\\t{proc.pid}\\t{kind}\\t{rc}\\t{detail}"
    _runner_event_write(end_marker)
    print(end_marker, flush=True)
    return rc

class _RunnerIPythonShim:
    def __init__(self, real_ip=None):
        self._real_ip = real_ip

    def system(self, cmd):
        expanded = str(cmd)
        if self._real_ip is not None:
            try:
                # InteractiveShell.system() normally performs var_expand.  The
                # Task Center proxy must preserve that behaviour for constructs
                # such as !{" ".join(cmd_ipython)} in exported notebooks.
                expanded = self._real_ip.var_expand(expanded, depth=1)
            except Exception:
                pass
        return _runner_call(expanded, "shell")

    def run_line_magic(self, magic, arg):
        expanded = str(arg)
        if self._real_ip is not None:
            try:
                # Preserve normal IPython magic interpolation, e.g.
                # %cd {merge_repo_dir} and %cd $merge_repo_dir.
                expanded = self._real_ip.var_expand(expanded, depth=1)
            except Exception:
                pass
        if magic == "cd":
            expanded = os.path.expandvars(os.path.expanduser(expanded))
            if not os.path.isdir(expanded):
                raise FileNotFoundError(f"Directory for %cd does not exist: {expanded}")
            os.chdir(expanded)
            print(os.getcwd())
            return None
        if self._real_ip is not None:
            return self._real_ip.run_line_magic(magic, expanded)
        raise RuntimeError(f"Unsupported line magic: %{magic}")

    def run_cell_magic(self, magic, line, cell):
        if magic == "bash":
            script = cell if not line else f"{line}\\n{cell}"
            return _runner_call(script, "bash")
        if self._real_ip is not None:
            return self._real_ip.run_cell_magic(magic, line, cell)
        raise RuntimeError(f"Unsupported cell magic: %%{magic}")

try:
    _runner_real_ip = _real_get_ipython() if _real_get_ipython else None
except Exception:
    _runner_real_ip = None
_runner_ip = _RunnerIPythonShim(_runner_real_ip)

def get_ipython():
    # Always return the proxy so shell/bash commands are visible to the
    # desktop task monitor; unsupported magics still delegate to real IPython.
    return _runner_ip
        """.strip()
        )

        transformed_cells = [compat_cell]
        for cell in nb.cells:
            if getattr(cell, "cell_type", None) != "code":
                transformed_cells.append(cell)
                continue

            src = cell.source or ""
            try:
                transformed = shell.transform_cell(src)
            except Exception:
                transformed = src

            cell.source = transformed
            transformed_cells.append(cell)

        nb.cells = transformed_cells

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".ipynb",
            prefix="runner_ipy_",
            dir=str(Path(notebook_path).parent),
            encoding="utf-8",
        )
        tmp_path = tmp.name
        tmp.close()

        with open(tmp_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return tmp_path


    def _build_notebook_run_command(self, notebook_path: str, executed_path: str) -> List[str]:
        papermill_spec = importlib.util.find_spec("papermill")
        if papermill_spec is not None:
            return [
                sys.executable,
                "-m",
                "papermill",
                notebook_path,
                executed_path,
                "--log-output",
                "--kernel",
                "python3",
            ]
        return [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            notebook_path,
            "--output",
            Path(executed_path).name,
            "--output-dir",
            str(Path(executed_path).parent),
            "--ExecutePreprocessor.timeout=-1",
        ]


    def _run_target_notebook(self):
        from .source_support import _apply_resource_limits_to_env

        temp_plan_path = None
        runner_notebook_path = None
        try:
            temp_plan_path = self._write_temp_plan_text()
            notebook_path, executed_path = self._get_notebook_paths()
            params = {key: widget.get() for key, widget in self.entries.items()}
            self.notebook_path_var.set(notebook_path)
            self.executed_notebook_path_var.set(executed_path)
            self._save_current_state_to_config()
            self.console.show()
            self.console.bind_notebook(executed_path)
            self.console.clear_output()
            self.console._stop_requested = False
            self._last_console_plan_step = None
            self.console.set_operation("Run Merge Notebook")
            self.console.set_progress_mode("determinate")
            self.console.set_state("BUILDING")
            self.console.set_step("Generating notebook from Plan Creator")
            self.console.process_event("BUILDING", "planner", "notebook-build", "Generating execution notebook")
            self.console.idle(f"Temporary plan path: {temp_plan_path}")
            self.console.idle(f"Notebook path: {notebook_path}")
            self.progress_indicator.start(3, "Running merge")

            create_plan_ipynb(
                filepath=temp_plan_path,
                workpath=params.get("workpath", ""),
                saveas=notebook_path,
                title=params.get("title", "merge_plan"),
                vae=params.get("vae", ""),
                vae_name=params.get("vae_name", "VAE"),
                bake_vae=bool(getattr(self, "bake_vae_var", tk.BooleanVar(value=False)).get()),
                CivitAPI=params.get("CivitAPI", ""),
                HuggingAPI=params.get("HuggingAPI", ""),
                UR=params.get("UR", ""),
                model_dir=params.get("model_dir", ""),
                vae_dir=params.get("vae_dir", ""),
                ignore_install_deps=bool(getattr(self, "ignore_install_deps_var", tk.BooleanVar(value=False)).get()),
                use_online=bool(getattr(self, "use_online_var", tk.BooleanVar(value=False)).get()),
                upload_after_merge=bool(getattr(self, "upload_after_merge_var", tk.BooleanVar(value=False)).get()),
                run_t2i=bool(getattr(self, "run_t2i_var", tk.BooleanVar(value=False)).get()),
                t2i_settings=self._planner_get_t2i_settings() if hasattr(self, "_planner_get_t2i_settings") else None
            )
            self.progress_indicator.update(1, "Notebook generated")
            self.console.process_event("DONE", "planner", "notebook-build", "Notebook generated")
            self.console.idle("Notebook generation completed.")

            runner_notebook_path = self._build_runner_notebook(notebook_path)
            if runner_notebook_path != notebook_path:
                self.console.idle(f"Runner notebook path: {runner_notebook_path}")

            def worker():
                proc = None
                event_path = None
                event_stop = threading.Event()
                event_thread = None
                recent_lines: Dict[str, float] = {}
                recent_lock = threading.Lock()

                def is_recent_duplicate(text: str) -> bool:
                    key = str(text or "").strip()
                    if not key:
                        return False
                    now = time.monotonic()
                    with recent_lock:
                        previous = recent_lines.get(key)
                        recent_lines[key] = now
                        if len(recent_lines) > 512:
                            cutoff = now - 2.0
                            for old_key, old_time in list(recent_lines.items()):
                                if old_time < cutoff:
                                    recent_lines.pop(old_key, None)
                        return previous is not None and (now - previous) < 0.70

                def consume_line(raw_line: str, pending: list[str]) -> str:
                    line = str(raw_line or "").rstrip("\r\n")
                    if not line or is_recent_duplicate(line):
                        return "skip"
                    kind = self._update_status_from_line(raw_line)
                    if self.console._looks_like_progress_line(line):
                        return "progress"
                    if kind != "info":
                        if pending:
                            self.console.log("\n".join(pending), "info")
                            pending.clear()
                        self.console.log(line, kind)
                        return "colored"
                    pending.append(line)
                    return "info"

                try:
                    self.console.set_state("RUNNING")
                    self.console.set_step("Executing notebook")
                    cmd = self._build_notebook_run_command(runner_notebook_path or notebook_path, executed_path)
                    self.console.idle("Command: " + " ".join(cmd))
                    env = _apply_resource_limits_to_env(os.environ.copy())
                    env.setdefault("PYTHONUNBUFFERED", "1")
                    self.console.idle(
                        "Resource limits: CPU threads=" + env.get("CHATTIORI_CPU_THREADS", "?")
                        + ", int8 compute=" + env.get("CHATTIORI_INT8_COMPUTE_DTYPE", "fp16")
                        + ", SIM scratch cache=" + env.get("CHATTIORI_SIM_SCRATCH_CACHE", "4")
                    )

                    # Notebook executors do not all forward cell stdout live
                    # (nbconvert is a common example).  The runner shim writes a
                    # compact event stream so child merge/download processes stay
                    # visible regardless of the execution backend.
                    event_tmp = tempfile.NamedTemporaryFile(
                        mode="w",
                        delete=False,
                        suffix=".events.log",
                        prefix="planner_runner_",
                        dir=str(Path(executed_path).parent),
                        encoding="utf-8",
                    )
                    event_path = event_tmp.name
                    event_tmp.close()
                    env["PLANNER_EVENT_FILE"] = event_path

                    event_pending: list[str] = []

                    def event_reader():
                        last_flush = time.monotonic()
                        try:
                            with open(event_path, "r", encoding="utf-8", errors="replace") as stream:
                                while True:
                                    raw = stream.readline()
                                    if raw:
                                        result = consume_line(raw, event_pending)
                                        now = time.monotonic()
                                        if event_pending and (len(event_pending) >= 20 or (now - last_flush) >= 0.12):
                                            self.console.log("\n".join(event_pending), "info")
                                            event_pending.clear()
                                            last_flush = now
                                        continue
                                    if event_stop.is_set():
                                        # One final read after the producer exits.
                                        raw = stream.readline()
                                        if not raw:
                                            break
                                        consume_line(raw, event_pending)
                                        continue
                                    time.sleep(0.06)
                        except Exception as event_error:
                            self.console.log(f"Event stream warning: {event_error}", "warning")
                        finally:
                            if event_pending:
                                self.console.log("\n".join(event_pending), "info")
                                event_pending.clear()

                    event_thread = threading.Thread(target=event_reader, daemon=True)
                    event_thread.start()

                    popen_kwargs = dict(
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=env,
                        errors="replace",
                    )
                    if os.name != "nt":
                        popen_kwargs["start_new_session"] = True
                    proc = subprocess.Popen(cmd, **popen_kwargs)
                    self.console.attach_process(proc, "Jupyter / Papermill runner")
                    self.root.after(0, lambda: self.progress_indicator.update(2, "Notebook started"))

                    pending: list[str] = []
                    last_flush = time.monotonic()
                    if proc.stdout is not None:
                        for raw_line in proc.stdout:
                            consume_line(raw_line, pending)
                            now = time.monotonic()
                            if pending and (len(pending) >= 20 or (now - last_flush) >= 0.12):
                                self.console.log("\n".join(pending), "info")
                                pending.clear()
                                last_flush = now
                    if pending:
                        self.console.log("\n".join(pending), "info")
                        pending.clear()

                    return_code = proc.wait()
                    event_stop.set()
                    if event_thread is not None:
                        event_thread.join(timeout=1.0)

                    runner_status = "STOPPED" if self.console._stop_requested else ("DONE" if return_code == 0 else "FAILED")
                    self.console.process_event(
                        runner_status,
                        "runner",
                        str(getattr(proc, "pid", "-")),
                        f"Jupyter / Papermill runner rc={return_code}",
                    )
                    last_step = getattr(self, "_last_console_plan_step", None)
                    if last_step:
                        self.console.process_event(runner_status, "plan", last_step[0], last_step[1])

                    if self.console._stop_requested:
                        self.console.set_state("STOPPED")
                        self.console.set_step("Execution stopped by user")
                        self.console.log("Execution stopped by user.", "warning")
                        self.console.set_progress_fraction(None, "Stopped")
                        self.root.after(0, lambda: self.progress_indicator.finish(False))
                        self.root.after(0, lambda: self.status_label.config(text="⏹ Merge stopped"))
                        return
                    if return_code != 0:
                        raise RuntimeError(f"Notebook execution failed with exit code {return_code}")

                    self.root.after(0, lambda: self.progress_indicator.update(3, "Notebook finished"))
                    self.console.set_state("DONE")
                    self.console.set_step("Execution completed")
                    self.console.set_progress_mode("determinate")
                    self.console.set_progress_fraction(1.0, "Completed")
                    self.console.log("Execution completed successfully.", "success")
                    self.root.after(0, lambda: self.progress_indicator.finish(True))
                    self.root.after(0, lambda: self.status_label.config(text="✅ Merge completed"))
                    self._play_notification_sound()
                except Exception as e:
                    event_stop.set()
                    if event_thread is not None and event_thread.is_alive():
                        event_thread.join(timeout=1.0)
                    detail = self._format_exception_text("Execution Error", e)
                    self.console.set_progress_mode("determinate")
                    self.console.set_state("FAILED")
                    self.console.set_step("Execution failed")
                    self.console.log(detail, "error")
                    if proc is not None:
                        self.console.process_event("FAILED", "runner", str(getattr(proc, "pid", "-")), str(e))
                    self.root.after(0, lambda: self.progress_indicator.finish(False))
                    self.root.after(0, lambda: self.status_label.config(text="❌ Merge failed"))
                    self.root.after(0, lambda: self._show_scrollable_text_dialog("Execution Error", detail))
                finally:
                    event_stop.set()
                    self.console.attach_process(None)
                    if event_path and os.path.exists(event_path):
                        try:
                            os.remove(event_path)
                        except Exception:
                            pass
                    if runner_notebook_path and runner_notebook_path != notebook_path and os.path.exists(runner_notebook_path):
                        try:
                            os.remove(runner_notebook_path)
                        except Exception:
                            pass
                    if temp_plan_path and os.path.exists(temp_plan_path):
                        os.remove(temp_plan_path)

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            self.progress_indicator.finish(False)
            if temp_plan_path and os.path.exists(temp_plan_path):
                os.remove(temp_plan_path)
            self._show_detailed_error("Run Error", e, context="Operation: run_target_notebook")


    def _upload_merge_result(self):
        repo_id = self.entries["UR"].get().strip()
        token = self.entries["HuggingAPI"].get().strip()
        model_dir = self.entries["model_dir"].get().strip() or os.path.join(self.entries["workpath"].get().strip() or ".", "tmp", "models")
        if not repo_id or not token:
            messagebox.showwarning("Missing Info", "Please provide both Repo ID and HuggingFace Token")
            return
        if not os.path.exists(model_dir):
            messagebox.showwarning("Model Directory", "Model directory does not exist.")
            return
        safetensors_files = list(Path(model_dir).glob("*.safetensors"))
        if not safetensors_files:
            messagebox.showinfo("No Models", "No .safetensors files found in model directory")
            return
        latest = max(safetensors_files, key=lambda p: p.stat().st_mtime)
        if not messagebox.askyesno("Upload", f"Upload latest model?\n\n{latest.name}\n-> {repo_id}"):
            return

        size_bytes = latest.stat().st_size
        size_gib = size_bytes / float(1024 ** 3)
        self.console.show()
        self.console.bind_notebook("")
        self.console.clear_output()
        self.console._stop_requested = False
        self.console.set_operation("Upload Latest Model")
        self.console.set_progress_mode("determinate")
        self.console.set_state("PREPARING")
        self.console.set_step("Validating upload target")
        self.console.set_progress_fraction(0.05, f"Preparing {latest.name}")
        self.console.idle(f"File: {latest}")
        self.console.idle(f"Size: {size_gib:.3f} GiB")
        self.console.idle(f"Repository: {repo_id}")
        self.progress_indicator.start(4, f"Uploading {latest.name}")

        def worker():
            try:
                api = HfApi()
                self.console.process_event("RUNNING", "hub", "repo", f"Create/check repository {repo_id}")
                self.console.set_step("Creating/checking Hugging Face repository")
                self.console.log(f"Checking repository: {repo_id}", "upload")
                api.create_repo(repo_id=repo_id, token=token, exist_ok=True)
                self.console.process_event("DONE", "hub", "repo", f"Repository ready: {repo_id}")
                self.console.set_progress_fraction(0.20, "Repository ready")
                self.root.after(0, lambda: self.progress_indicator.update(1, "Repository ready"))

                self.console.set_state("UPLOADING")
                self.console.set_step(f"Uploading {latest.name}")
                # Hub transfer APIs do not expose a stable byte callback across
                # all backends. Keep the native path upload (including its
                # optimized large-file backend) and show an indeterminate bar
                # for the network phase rather than buffering/copying the model.
                self.console.set_progress_mode("indeterminate")
                self.console.set_progress(f"Uploading {latest.name} ({size_gib:.3f} GiB) -> {repo_id}")
                self.console.process_event("UPLOADING", "upload", latest.name, f"{size_gib:.3f} GiB -> {repo_id}")
                self.console.log(f"Uploading {latest.name} ({size_gib:.3f} GiB) to {repo_id}", "upload")
                api.upload_file(
                    path_or_fileobj=str(latest),
                    path_in_repo=latest.name,
                    repo_id=repo_id,
                    token=token,
                )

                self.console.set_progress_mode("determinate")
                self.console.set_progress_fraction(1.0, f"Uploaded {latest.name}")
                self.console.process_event("DONE", "upload", latest.name, f"Uploaded to {repo_id}")
                self.console.set_state("DONE")
                self.console.set_step("Upload completed")
                self.console.log(f"Upload completed: {latest.name} -> {repo_id}", "success")
                self.root.after(0, lambda: self.progress_indicator.update(4, latest.name))
                self.root.after(0, lambda: self.progress_indicator.finish(True))
                self.root.after(0, lambda: messagebox.showinfo("Upload", f"Uploaded {latest.name} to {repo_id}"))
                self._play_notification_sound()
            except Exception as e:
                self.console.set_progress_mode("determinate")
                self.console.set_state("FAILED")
                self.console.set_step("Upload failed")
                self.console.process_event("FAILED", "upload", latest.name, str(e))
                self.root.after(0, lambda: self.progress_indicator.finish(False))
                detail = self._format_exception_text("Upload Error", e)
                self.console.log(detail, "error")
                self.root.after(0, lambda: self._show_scrollable_text_dialog("Upload Error", detail))

        threading.Thread(target=worker, daemon=True).start()


    _planner_get_t2i_settings = _method_planner_get_t2i_settings


    _planner_set_t2i_settings = _method_planner_set_t2i_settings


    _planner_open_t2i_settings = _method_planner_open_t2i_settings
