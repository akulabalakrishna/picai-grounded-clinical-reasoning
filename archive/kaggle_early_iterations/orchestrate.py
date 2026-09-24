"""
Local orchestration tool (never uploaded to Kaggle) that drives the Kaggle
Kernels API end-to-end: push -> poll -> on completion, detect whether
training finished or hit MAX_TRAIN_HOURS -> if the latter, push a NEW
kernel (incrementally numbered, referencing the previous one's output via
kernel_sources) and repeat, until POST-TRAINING VALIDATION appears in the
log or a safety cap is hit.

Only ever runs the ONE unchanged script (kaggle_single_cell_train.py) on
the ONE dataset (picai-strict-201pos-220neg) on Kaggle's side -- this
script is local tooling to automate clicking through the web UI, not an
additional Kaggle-side artifact.

Writes progress to orchestrate.log and a final marker file
(FINAL_STATE.json) when it stops (success, error, or safety-cap hit) so
the calling session can pick up from there without re-deriving state.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE = r"C:\Users\HP\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\Scripts\kaggle.exe"
OWNER = "akulabalakrishna143"
DATASET_SLUG = f"{OWNER}/picai-strict-201pos-220neg"
SCRIPT_SRC = Path(r"E:\Cancer_IITH\picai\kaggle_single_cell_train.py")
BASE_DIR = Path(r"E:\Cancer_IITH\picai\kaggle_kernel")
LOG_PATH = BASE_DIR / "orchestrate.log"
FINAL_STATE_PATH = BASE_DIR / "FINAL_STATE.json"

POLL_INTERVAL_SECONDS = 150
MAX_CHAINED_RUNS = 6  # safety cap -- 6 * 8.5h would be an implausible amount
                       # of sessions for this run size; stops runaway chaining
                       # if something is silently wrong rather than genuinely
                       # needing more epochs.


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_kaggle(args, **kwargs):
    result = subprocess.run([KAGGLE] + args, capture_output=True, text=True, **kwargs)
    return result.returncode, result.stdout, result.stderr


def push_kernel(run_number: int, kernel_sources):
    run_dir = BASE_DIR / f"run{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCRIPT_SRC, run_dir / SCRIPT_SRC.name)

    slug = f"picai-strict-201pos-220neg-training-run-{run_number}"
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": f"PICAI Strict Training Run {run_number}",
        "code_file": SCRIPT_SRC.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [DATASET_SLUG],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    with open(run_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log(f"Pushing run {run_number} (kernel_sources={kernel_sources})...")
    rc, out, err = run_kaggle(["kernels", "push", "-p", str(run_dir)])
    log(f"push stdout: {out.strip()}")
    if err.strip():
        log(f"push stderr: {err.strip()}")
    if rc != 0:
        raise RuntimeError(f"kaggle kernels push failed (rc={rc}): {err}")

    # Kaggle may resolve the actual slug from the title rather than the id
    # field verbatim -- parse the real slug from the printed progress URL.
    m = re.search(r"kaggle\.com/code/([\w.-]+)/([\w.-]+)", out)
    if not m:
        m = re.search(r"kaggle\.com/code/([\w.-]+)/([\w.-]+)", err)
    if not m:
        raise RuntimeError(f"Could not parse kernel slug from push output: {out}\n{err}")
    actual_owner, actual_slug = m.group(1), m.group(2)
    full_slug = f"{actual_owner}/{actual_slug}"
    log(f"Resolved actual kernel slug: {full_slug}")
    return full_slug, run_dir


def poll_until_finished(full_slug: str):
    while True:
        rc, out, err = run_kaggle(["kernels", "status", full_slug])
        status_text = (out + err).strip()
        log(f"status: {status_text}")
        if "RUNNING" in status_text or "QUEUED" in status_text:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        return status_text


def download_output(full_slug: str, run_dir: Path):
    out_dir = run_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading output for {full_slug} to {out_dir} ...")
    rc, out, err = run_kaggle(["kernels", "output", full_slug, "-p", str(out_dir), "-o"])
    log(f"output stdout: {out.strip()[-2000:]}")
    if rc != 0:
        log(f"output stderr: {err.strip()}")
    return out_dir


def find_log_file(out_dir: Path) -> Path:
    candidates = list(out_dir.glob("*.log"))
    if not candidates:
        # some kaggle-api versions name it differently; fall back to any text-ish file
        candidates = [p for p in out_dir.iterdir() if p.is_file() and p.suffix in ("", ".txt", ".log")]
    if not candidates:
        raise FileNotFoundError(f"No log file found in {out_dir}: {list(out_dir.iterdir())}")
    return max(candidates, key=lambda p: p.stat().st_size)


GPU_INCOMPATIBLE_SIGNATURE = "no kernel image is available for execution on the device"
MAX_GPU_RETRIES = 4  # each failed attempt costs ~1 minute (fails fast, before training
                      # starts), so retrying for a compatible GPU assignment is cheap


def _get_status_and_log(full_slug, run_dir, pending_status):
    """pending_status: a status string already fetched before this call (to
    avoid a redundant status check right after a push we just polled to
    completion), or None to fetch fresh."""
    if pending_status is not None and not any(
        s in pending_status.upper() for s in ("RUNNING", "QUEUED")
    ):
        final_status = pending_status
    else:
        final_status = poll_until_finished(full_slug)
    out_dir = download_output(full_slug, run_dir)
    log_file = find_log_file(out_dir)
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    log(f"Downloaded log: {log_file} ({len(log_text)} chars)")
    return final_status, log_file, log_text


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")  # fresh log for this orchestration session

    run_number = 6  # runs 1-5 already used (run 1 = original, 2-5 = GPU-incompatibility
                     # retries before the torch==2.4.1 fix); this run's own kernel
                     # (picai-strict-training-gpu-compat-test) is being treated as run 6
                     # so any future chaining continues from 7 without slug collisions
    kernel_sources = []
    gpu_retry_count = 0
    chain_count = 0

    # This kernel was already manually pushed (with the torch==2.4.1 fix) and
    # confirmed running past the previous crash point (15+ min, vs ~1-2.5 min
    # before) -- do NOT push over it. Resume monitoring it directly instead.
    already_slug = f"{OWNER}/picai-strict-training-gpu-compat-test"
    rc, out, err = run_kaggle(["kernels", "status", already_slug])
    initial_status = (out + err).strip()
    if rc == 0 and any(s in initial_status.upper() for s in ("RUNNING", "QUEUED", "COMPLETE", "ERROR")):
        log(f"Run 6 (gpu-compat-test) already running, status={initial_status} -- "
            f"resuming monitoring instead of re-pushing.")
        full_slug = already_slug
        run_dir = BASE_DIR / "run6_gputest"
        run_dir.mkdir(parents=True, exist_ok=True)
        pending_status = initial_status
    else:
        full_slug, run_dir = push_kernel(run_number, kernel_sources)
        pending_status = None

    while True:
        try:
            final_status, log_file, log_text = _get_status_and_log(full_slug, run_dir, pending_status)
        except FileNotFoundError as e:
            log(f"ERROR: {e}")
            _write_final_state("error_no_log", full_slug, run_dir, None, None)
            return
        pending_status = None  # only reuse a pre-fetched status once

        if "ERROR" in final_status.upper():
            if GPU_INCOMPATIBLE_SIGNATURE in log_text:
                gpu_retry_count += 1
                log(f"Run {run_number}: GPU assigned is incompatible with Kaggle's installed PyTorch "
                    f"build (retry {gpu_retry_count}/{MAX_GPU_RETRIES}) -- a GPU-assignment lottery "
                    f"issue, not a script bug (dataset/label/split loading all succeeded before the "
                    f"crash). Pushing a fresh kernel to get reassigned.")
                if gpu_retry_count > MAX_GPU_RETRIES:
                    log("Exceeded max GPU-incompatibility retries. Stopping for manual review.")
                    _write_final_state("error_gpu_incompatible_retries_exhausted",
                                        full_slug, run_dir, log_file, final_status)
                    return
                run_number += 1
                kernel_sources = []  # fresh retry, not a legitimate resume -- no prior state to keep
                full_slug, run_dir = push_kernel(run_number, kernel_sources)
                continue
            log(f"Run {run_number}: kernel errored for a reason OTHER than GPU incompatibility "
                f"(status={final_status}). Stopping for manual diagnosis.")
            _write_final_state("error_other", full_slug, run_dir, log_file, final_status)
            return

        if "COMPLETE" not in final_status.upper():
            log(f"Run {run_number}: kernel did not complete successfully (status={final_status}). Stopping.")
            _write_final_state("error_bad_status", full_slug, run_dir, log_file, final_status)
            return

        if "POST-TRAINING VALIDATION" in log_text:
            log(f"Run {run_number}: training completed and post-training validation ran. DONE.")
            _write_final_state("success", full_slug, run_dir, log_file, final_status)
            return

        if "Hit MAX_TRAIN_HOURS" in log_text:
            chain_count += 1
            if chain_count > MAX_CHAINED_RUNS:
                log(f"Hit MAX_CHAINED_RUNS={MAX_CHAINED_RUNS} without finishing -- stopping as a safety measure.")
                _write_final_state("error_max_chained_runs", full_slug, run_dir, log_file, final_status)
                return
            log(f"Run {run_number}: hit MAX_TRAIN_HOURS mid-training. Chaining to run {run_number + 1}.")
            run_number += 1
            kernel_sources = [full_slug]
            full_slug, run_dir = push_kernel(run_number, kernel_sources)
            continue

        log(f"Run {run_number}: completed but log has neither a clean MAX_TRAIN_HOURS stop nor "
            f"POST-TRAINING VALIDATION -- unexpected state, stopping for manual diagnosis.")
        _write_final_state("error_unexpected_log_state", full_slug, run_dir, log_file, final_status)
        return


def _write_final_state(outcome, full_slug, run_dir, log_file, final_status):
    state = {
        "outcome": outcome,
        "kernel_slug": full_slug,
        "run_dir": str(run_dir) if run_dir else None,
        "log_file": str(log_file) if log_file else None,
        "final_status": final_status,
    }
    with open(FINAL_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    log(f"Wrote final state: {state}")


if __name__ == "__main__":
    main()
