"""Run or resume the authorized Burgers campaign, after automatic verification gates."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "all"], default="all")
    parser.add_argument("--out", type=Path, default=HERE / "results16")
    args = parser.parse_args()
    from check_package import check
    check()
    out = args.out.expanduser().resolve()
    environment = os.environ.copy()
    environment.update(DDE_BACKEND="pytorch", MPLCONFIGDIR=str(HERE / ".mplcache"),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    active = [None]
    stopping = [False]

    def stop(sig, frame):
        stopping[0] = True
        print("Stop requested; saving complete checkpoints. Wait for workers to exit.", flush=True)
        if active[0] is not None and active[0].poll() is None:
            active[0].send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, stop)

    def run(*command, script="burgers_formal16/runner.py"):
        if stopping[0]:
            raise SystemExit(130)
        cmd = [sys.executable, "-u", str(HERE / script)]
        if script.endswith("runner.py"):
            cmd += ["--out", str(out)]
        print("RUN:", " ".join(cmd + list(command)), flush=True)
        active[0] = subprocess.Popen(cmd + list(command), env=environment)
        code = active[0].wait()
        active[0] = None
        if code:
            raise SystemExit(code)

    run("init")
    requested = dict(workers=4, threads_per_worker=4, device="cpu")
    resources = json.loads((out / "resources.json").read_text())
    if resources != requested:
        run("resources", "--workers", "4", "--threads", "4")
    run("check")
    run("verify")
    run("smoke")
    budget = json.loads((out / "dry_run_report.json").read_text())["budget"]
    print(f"Measured 200-step projection: base 225 runs ~{budget['total_seconds']/3600:.1f} active hours; LR extensions excluded.", flush=True)
    if args.stage == "smoke":
        print("Smoke-only complete. Formal training has not started.", flush=True)
        return
    if not (out / "execution_approval.json").exists():
        run("approve", "--note", "User selected the 16-CPU Burgers server package; automatic gates passed; run_campaign --stage all authorizes the frozen campaign.")
    for command in ["sweep", "select", "final", "evaluate", "status"]:
        run(command)
    run("--out", str(out), script="report.py")


if __name__ == "__main__":
    main()
