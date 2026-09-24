"""Simple poll-until-terminal-state script for the Direction A rationale
kernel. Unlike the training orchestrator, no GPU-retry or epoch-chaining
logic is needed here -- this is a single fast CPU job. Just polls, then
downloads output once it reaches COMPLETE or ERROR, and writes a marker
file so the calling session knows to come look."""
import subprocess
import sys
import time
from pathlib import Path

KAGGLE = r"C:\Users\HP\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts\kaggle.exe"
SLUG = sys.argv[1]
OUT_DIR = Path(sys.argv[2])
POLL_INTERVAL = 60

OUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = OUT_DIR / "poll.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_kaggle(args):
    r = subprocess.run([KAGGLE] + args, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


while True:
    rc, out, err = run_kaggle(["kernels", "status", SLUG])
    status = (out + err).strip()
    log(f"status: {status}")
    if "RUNNING" in status.upper() or "QUEUED" in status.upper():
        time.sleep(POLL_INTERVAL)
        continue
    break

log(f"Terminal state reached: {status}")
rc, out, err = run_kaggle(["kernels", "output", SLUG, "-p", str(OUT_DIR), "-o"])
log(f"output stdout: {out.strip()}")
if err.strip():
    log(f"output stderr: {err.strip()}")

(OUT_DIR / "DONE.txt").write_text(status, encoding="utf-8")
log("Wrote DONE.txt")
