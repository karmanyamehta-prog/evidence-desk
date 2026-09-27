"""Two supervised processes, one public Streamlit port; API stays loopback."""

import subprocess
import sys
import time
import urllib.request

subprocess.run([sys.executable, "-m", "grounded_qa.cli", "fetch"], check=True)
subprocess.run([sys.executable, "-m", "grounded_qa.cli", "ingest"], check=True)
api = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "grounded_qa.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
)
ui = None
try:
    for _ in range(120):
        if api.poll() is not None:
            raise RuntimeError("API exited")
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("API startup timed out")
    ui = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.address=0.0.0.0",
            "--server.port=7860",
            "--browser.gatherUsageStats=false",
        ]
    )
    while api.poll() is None and ui.poll() is None:
        time.sleep(1)
    raise SystemExit(1)
finally:
    api.terminate()
    if ui:
        ui.terminate()
