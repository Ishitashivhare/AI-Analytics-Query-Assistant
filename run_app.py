"""
run_app.py
----------
Single-command launcher for the AI Analytics Query Assistant.

Starts:
  - FastAPI backend on http://127.0.0.1:8000
  - Static frontend on http://127.0.0.1:5500

Run with:
    python run_app.py
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5500/index.html"


def _start_process(args: list[str], label: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=None,
        stderr=None,
        stdin=None,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        text=True,
    )


def _wait_for_backend_ready(timeout_seconds: float = 30.0) -> bool:
    try:
        import httpx
    except Exception:
        return False

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{BACKEND_URL}/health", timeout=2.0)
            if response.status_code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    backend = _start_process([PYTHON, "-m", "uvicorn", "api_server:app", "--port", "8000"], "backend")
    frontend = _start_process([PYTHON, "-m", "http.server", "5500"], "frontend")

    def shutdown(*_args: object) -> None:
        for process in (frontend, backend):
            if process.poll() is None:
                try:
                    if sys.platform == "win32":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        process.terminate()
                except Exception:
                    process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    print(f"Backend:  {BACKEND_URL}")
    print(f"Frontend: {FRONTEND_URL}")
    print("Press Ctrl+C to stop both servers.")

    threading.Timer(1.5, lambda: webbrowser.open(FRONTEND_URL)).start()

    backend_ready = _wait_for_backend_ready()
    if not backend_ready:
        print("Warning: backend health check did not confirm readiness within the timeout.")

    try:
        while True:
            backend_exit = backend.poll()
            frontend_exit = frontend.poll()
            if backend_exit is not None or frontend_exit is not None:
                if backend_exit is not None:
                    print(f"Backend exited with code {backend_exit}.")
                if frontend_exit is not None:
                    print(f"Frontend exited with code {frontend_exit}.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()

    shutdown()

    backend_wait = backend.wait(timeout=10)
    frontend_wait = frontend.wait(timeout=10)
    return backend_wait or frontend_wait or 0


if __name__ == "__main__":
    raise SystemExit(main())