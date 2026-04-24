from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tempfile
import time
import ctypes
import os

LOG_PATH = Path(tempfile.gettempdir()) / "fusion_upscaler_launcher.log"
CACHE_PATH = Path(tempfile.gettempdir()) / "fusion_upscaler_python.txt"


def _log(msg: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _show_error(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Fusion Upscaler", 0x10)
    except Exception:
        pass


def _launch_app(run_exe: Path, app_path: Path, app_root: Path) -> int:
    run_cmd = [str(run_exe), str(app_path)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    child_env = os.environ.copy()
    child_env.pop("TCL_LIBRARY", None)
    child_env.pop("TK_LIBRARY", None)
    child_env.pop("_MEIPASS2", None)
    app_log = LOG_PATH.with_name("fusion_upscaler_app.log")

    with app_log.open("a", encoding="utf-8") as logf:
        logf.write("\n--- app launch ---\n")
        logf.write(f"cmd: {' '.join(run_cmd)}\n")

    proc = subprocess.Popen(
        run_cmd,
        cwd=str(app_root),
        creationflags=flags,
        stdout=app_log.open("a", encoding="utf-8"),
        stderr=app_log.open("a", encoding="utf-8"),
        env=child_env,
    )

    time.sleep(0.4)
    if proc.poll() is not None:
        _log(f"app exited quickly with code {proc.returncode}")
        details = ""
        try:
            details = app_log.read_text(encoding="utf-8")[-1600:]
        except Exception:
            pass
        _show_error(
            "The desktop app closed immediately.\n\n"
            f"Exit code: {proc.returncode}\n"
            f"Log: {app_log}\n\n"
            f"Recent output:\n{details}"
        )
        return proc.returncode or 1

    _log(f"launch ok -> pid {proc.pid}")
    return 0


def main() -> None:
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    candidates = [
        base_dir,
        base_dir.parent,
        base_dir.parent.parent,
        Path.cwd(),
        Path("C:/Projects/image-upscaler"),
    ]
    app_root = next((p for p in candidates if (p / "desktop_app.py").exists()), None)

    if app_root is None:
        _log("app_root not found")
        _show_error("Could not find desktop_app.py. Keep the launcher inside the project/dist folder.")
        return

    _log(f"app_root={app_root}")
    app_path = app_root / "desktop_app.py"
    requirements_path = app_root / "requirements.txt"

    if CACHE_PATH.exists():
        try:
            cached = Path(CACHE_PATH.read_text(encoding="utf-8").strip())
            if cached.exists():
                _log(f"trying cached interpreter: {cached}")
                if _launch_app(cached, app_path, app_root) == 0:
                    return
        except Exception as exc:
            _log(f"cached interpreter failed: {exc}")

    python_commands = [["py", "-3.10"], ["py", "-3.12"], ["py", "-3.11"], ["python"], ["pythonw"]]

    required_imports = "import sys; print(sys.executable)"
    diagnostics: list[str] = []

    for py_cmd in python_commands:
        _log(f"probing: {' '.join(py_cmd)}")
        try:
            probe = subprocess.run(
                py_cmd + ["-c", required_imports],
                cwd=str(app_root),
                capture_output=True,
                text=True,
                timeout=25,
            )
        except Exception as exc:  # pragma: no cover
            _log(f"probe failed: {' '.join(py_cmd)} -> {exc}")
            diagnostics.append(f"{' '.join(py_cmd)} -> {exc}")
            continue

        if probe.returncode != 0:
            err = (probe.stderr or probe.stdout or "dependency check failed").strip().splitlines()
            short_err = err[-1] if err else "unknown error"
            _log(f"probe bad: {' '.join(py_cmd)} -> {short_err}")
            diagnostics.append(f"{' '.join(py_cmd)} -> {short_err}")
            continue

        selected_python = (probe.stdout or "").strip().splitlines()
        python_exe = Path(selected_python[-1]).resolve() if selected_python else None
        if not python_exe or not python_exe.exists():
            diagnostics.append(f"{' '.join(py_cmd)} -> could not resolve python executable")
            continue

        pythonw_exe = python_exe.with_name("pythonw.exe")
        run_exe = pythonw_exe if pythonw_exe.exists() else python_exe
        _log(f"resolved interpreter: {run_exe}")

        try:
            if _launch_app(run_exe, app_path, app_root) == 0:
                CACHE_PATH.write_text(str(run_exe), encoding="utf-8")
                return
        except Exception as exc:  # pragma: no cover
            _log(f"launch failed: {' '.join(py_cmd)} -> {exc}")
            diagnostics.append(f"{' '.join(py_cmd)} -> launch failed: {exc}")

    msg = (
        "Could not start the app.\n\n"
        "Most likely cause: required AI dependencies are missing, or Python 3.14 is being used.\n"
        "Use Python 3.10-3.12 and install dependencies:\n\n"
        f"python -m pip install -r {requirements_path}\n\n"
        "Detected errors:\n- "
        + "\n- ".join(diagnostics[:5])
    )
    _show_error(msg)


if __name__ == "__main__":
    _log("--- launcher run ---")
    _log(f"frozen={getattr(sys, 'frozen', False)}")
    _log(f"cwd={Path.cwd()}")
    main()
