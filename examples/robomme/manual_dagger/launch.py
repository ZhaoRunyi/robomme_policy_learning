"""manual_dagger / launch: shared behavior owned by this layer."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from pathlib import Path
import os
import sys
import time
import urllib.error
import urllib.request


SESSION_NAME = "robottt-manual-dagger"
COMMAND_PATH = Path("/tmp/robottt-manual-dagger-command.json")
SCREEN_LOG_PATH = Path("/tmp/robottt-manual-dagger-screen.log")


def _screen_command(command: list[str]) -> list[str]:
    return [
        "screen",
        "-L",
        "-Logfile",
        str(SCREEN_LOG_PATH),
        "-dmS",
        SESSION_NAME,
        *command,
    ]


def _session_running() -> bool:
    result = subprocess.run(
        ["screen", "-S", SESSION_NAME, "-Q", "select", "."],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def stop_workload() -> None:
    if _session_running():
        subprocess.run(
            ["screen", "-S", SESSION_NAME, "-X", "quit"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def start_workload(command: list[str]) -> None:
    if _session_running():
        raise RuntimeError(f"screen session {SESSION_NAME!r} is already running")
    COMMAND_PATH.write_text(json.dumps(command), encoding="utf-8")
    subprocess.run(_screen_command(command), check=True)


def restart_workload() -> None:
    command = json.loads(COMMAND_PATH.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RuntimeError(f"invalid workload command in {COMMAND_PATH}")
    stop_workload()
    start_workload(command)


def supervise(command: list[str]) -> None:
    start_workload(command)

    def terminate(signum: int, frame: object) -> None:
        del signum, frame
        stop_workload()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    print(f"MANUAL_DAGGER_SLOT_READY={SESSION_NAME}", flush=True)
    while True:
        signal.pause()


def slot_command() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "restart", "stop"))
    arguments = parser.parse_args()
    if arguments.action == "status":
        print("RUNNING" if _session_running() else "STOPPED")
    elif arguments.action == "restart":
        restart_workload()
    else:
        stop_workload()


POLICY_REPO = Path(__file__).resolve().parents[3]
POLICY_PYTHON = POLICY_REPO / ".venv/bin/python"
EVALUATOR_PYTHON = POLICY_REPO / "third_party/robomme_benchmark/.venv/bin/python"
CHECKPOINT = Path(
    "/workspace/ttt_ws/runs/robottt/train_logs/"
    "robottt_frozen_pi05_baseline_10k_fast/robomme_pi05_robottt/"
    "robottt_frozen_pi05_baseline_10k_fast/60000"
)


def _option_path(arguments: list[str], name: str) -> Path | None:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if argument.startswith(f"{name}="):
            return Path(argument.split("=", 1)[1])
    return None


def _ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz",
            timeout=2.0,
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def run_workload(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy-port", type=int, default=8011)
    known, remaining = parser.parse_known_args(arguments)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["NVIDIA_DRIVER_CAPABILITIES"] = "all"
    output_root = _option_path(remaining, "--output-root")
    if output_root is None:
        policy_log_path = Path("/tmp/manual-dagger-policy.log")
        browser_log_path = Path("/tmp/manual-dagger-browser.log")
        service_log_path = Path("/tmp/manual-dagger-service.log")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        policy_log_path = output_root / "policy.log"
        browser_log_path = output_root / "browser.log"
        service_log_path = output_root / "service.log"

    def record(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with service_log_path.open("a", encoding="utf-8") as service_log:
            service_log.write(f"{timestamp} {message}\n")

    record("launcher starting")
    policy_log = policy_log_path.open("w", encoding="utf-8", buffering=1)
    policy = subprocess.Popen(
        [
            str(POLICY_PYTHON),
            str(POLICY_REPO / "scripts/serve_policy.py"),
            "--seed=7",
            f"--port={known.policy_port}",
            "--parallel-clients=2",
            "policy:checkpoint",
            f"--policy.dir={CHECKPOINT}",
            "--policy.config=robomme_pi05_robottt",
        ],
        cwd=POLICY_REPO,
        env=environment,
        stdout=policy_log,
        stderr=subprocess.STDOUT,
    )
    record(f"policy started pid={policy.pid}")
    browser: subprocess.Popen[bytes] | None = None

    def terminate(signum: int, frame: object) -> None:
        del signum, frame
        if browser is not None and browser.poll() is None:
            browser.terminate()
        if policy.poll() is None:
            policy.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    deadline = time.monotonic() + 1200.0
    next_progress_log = time.monotonic()
    while not _ready(known.policy_port):
        if policy.poll() is not None:
            record(f"policy exited returncode={policy.returncode}")
            raise RuntimeError(f"policy server exited with {policy.returncode}; see {policy_log_path}")
        if time.monotonic() >= deadline:
            record("policy readiness timed out")
            raise TimeoutError(f"policy server not ready; see {policy_log_path}")
        if time.monotonic() >= next_progress_log:
            record("waiting for policy health")
            next_progress_log = time.monotonic() + 10.0
        time.sleep(1.0)
    record("policy ready; starting browser service")
    browser_log = browser_log_path.open("w", encoding="utf-8", buffering=1)
    browser = subprocess.Popen(
        [
            str(EVALUATOR_PYTHON),
            "-m",
            "manual_dagger.web.app",
            "--policy-port",
            str(known.policy_port),
            *remaining,
        ],
        cwd=POLICY_REPO,
        env=environment,
        stdout=browser_log,
        stderr=subprocess.STDOUT,
    )
    record(f"browser service started pid={browser.pid}")
    try:
        returncode = browser.wait()
        record(f"browser service exited returncode={returncode}")
        if returncode:
            raise RuntimeError(f"browser service exited with {returncode}; see {browser_log_path}")
    finally:
        if policy.poll() is None:
            policy.terminate()
            try:
                policy.wait(timeout=30)
            except subprocess.TimeoutExpired:
                policy.kill()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("status", "restart", "stop"):
        slot_command()
        return
    arguments = sys.argv[1:]
    child_flag = "--slot-child"
    if child_flag in arguments:
        arguments.remove(child_flag)
        run_workload(arguments)
        return
    command = [sys.executable, "-m", "manual_dagger.launch", child_flag, *arguments]
    supervise(command)

if __name__ == "__main__":
    main()
