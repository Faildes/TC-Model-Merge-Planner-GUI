import os, platform, shutil, subprocess, sys

IGNORE_INSTALL_DEPS = $ignore_install_deps
USE_ONLINE = $use_online
WORKPATH_RAW = r"$workpath"
DIFFUSION = "$diffusion"


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


WORKING_DIR = _planner_working_dir(WORKPATH_RAW)
LOCAL_REPO_DIR = r"$toolpath"
REPO_DIR = os.path.join(WORKING_DIR, "tools", "chattiori_model_merger") if USE_ONLINE else LOCAL_REPO_DIR
if USE_ONLINE:
    os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
print(f"[install] use_online={USE_ONLINE} working_dir={WORKING_DIR} repo_dir={REPO_DIR}")


def _cmd_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd, *, check=False):
    pretty = " ".join(str(x) for x in cmd)
    print(f"$ {pretty}")
    return subprocess.run(cmd, check=check)


def ensure(*args: str):
    _run([sys.executable, "-m", "pip", "install", *args], check=False)

def install_system_tools():
    system = platform.system()
    print(f"[install] platform={system}")
    if system == "Linux":
        if _cmd_exists("apt-get"):
            _run(["apt-get", "update", "-qq"], check=False)
            _run(["apt-get", "install", "-y", "-qq", "aria2", "git"], check=False)
        elif _cmd_exists("dnf"):
            _run(["dnf", "install", "-y", "aria2", "git"], check=False)
        elif _cmd_exists("yum"):
            _run(["yum", "install", "-y", "aria2", "git"], check=False)
        elif _cmd_exists("apk"):
            _run(["apk", "add", "aria2", "git"], check=False)
        elif _cmd_exists("pacman"):
            _run(["pacman", "-Sy", "--noconfirm", "aria2", "git"], check=False)
        else:
            print("[install] Unsupported Linux package manager. Skipping system package installation.")
    elif system == "Darwin":
        if _cmd_exists("brew"):
            _run(["brew", "install", "aria2", "git"], check=False)
        else:
            print("[install] Homebrew not found. Skipping aria2/git installation on macOS.")
    elif system == "Windows":
        if _cmd_exists("winget"):
            _run(["winget", "install", "-e", "--id", "aria2.aria2", "--accept-package-agreements", "--accept-source-agreements"], check=False)
            _run(["winget", "install", "-e", "--id", "Git.Git", "--accept-package-agreements", "--accept-source-agreements"], check=False)
        elif _cmd_exists("choco"):
            _run(["choco", "install", "-y", "aria2", "git"], check=False)
        else:
            print("[install] winget/choco not found. Skipping aria2/git installation on Windows.")
    else:
        print(f"[install] Unsupported OS for automatic system dependency installation: {system}")

packages = [("torch",),
        ("torchvision",),
        ("lora",),
        ("fake_useragent",),
        ("diffusers",),
        ("torchsde",),
        ("git+https://github.com/huggingface/diffusers",),
        ("git+https://github.com/Faildes/sd_embed_negpip.git",),
        ("git+https://github.com/Faildes/Chattiori_ImageKit",),
        ("-U", "peft"),
        ("torchao", "--extra-index-url", "https://download.pytorch.org/whl/cu121"),]
if DIFFUSION:
    packages.append((DIFFUSION,))

if IGNORE_INSTALL_DEPS:
    print("[install] Ignore Install Deps is enabled. Skipping dependency installation and repo setup.")
else:
    for pkg in packages:
        ensure(*pkg)

    install_system_tools()
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        _run(["git", "-C", REPO_DIR, "fetch", "origin", "notebook"], check=False)
        _run(["git", "-C", REPO_DIR, "checkout", "notebook"], check=False)
        _run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", "notebook"], check=False)
    else:
        _run(["git", "clone", "https://github.com/Faildes/Chattiori-Model-Merger", "-b", "notebook", REPO_DIR], check=False)

    req = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(req):
        _run([sys.executable, "-m", "pip", "install", "-r", req], check=False)
