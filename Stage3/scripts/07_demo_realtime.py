"""Realtime demo orchestrator.

Spawns three components on localhost:
    1. FastAPI inference service (uvicorn)  - port 8000
    2. Streamlit UI                          - port 8501
    3. AI-attacker streaming records into the FastAPI /score endpoint

Stops all subprocesses on Ctrl-C.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from diploma_nids.attacker import AttackerRuntime, records_to_dataframe
from diploma_nids.utils import get_logger, load_yaml, setup_logging

logger = get_logger(__name__)


def _start_api(env: dict[str, str], host: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "uvicorn",
        "diploma_nids.inference.service:app",
        "--host", host, "--port", str(port),
        "--log-level", "warning",
    ]
    return subprocess.Popen(cmd, env={**os.environ, **env})


def _start_ui(env: dict[str, str], host: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).resolve().parent.parent / "src" / "diploma_nids" / "ui" / "streamlit_app.py"),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    return subprocess.Popen(cmd, env={**os.environ, **env})


def _wait_api(api_url: str, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{api_url}/health", timeout=1)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline/realtime.yaml")
    parser.add_argument("--policy", default="configs/attacker/policy.yaml")
    parser.add_argument("--drift", default="configs/attacker/drift.yaml")
    parser.add_argument("--model", required=True, help="model YAML")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessor", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-attacker", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO")
    cfg = load_yaml(args.config)
    inf_cfg = cfg["inference"]
    ui_cfg = cfg["ui"]

    api_url = f"http://{inf_cfg['service_host']}:{inf_cfg['service_port']}"

    api_env = {
        "DIPLOMA_MODEL_YAML": args.model,
        "DIPLOMA_CHECKPOINT": args.checkpoint,
        "DIPLOMA_PREPROCESSOR": args.preprocessor,
        "DIPLOMA_THRESHOLD": str(args.threshold),
    }
    api_proc = _start_api(api_env, inf_cfg["service_host"], int(inf_cfg["service_port"]))
    logger.info("started FastAPI -> %s", api_url)

    ui_proc: subprocess.Popen | None = None
    try:
        if not _wait_api(api_url):
            raise RuntimeError("FastAPI did not become ready")

        if not args.no_ui:
            ui_env = {"DIPLOMA_API_URL": api_url}
            ui_proc = _start_ui(ui_env, ui_cfg["host"], int(ui_cfg["port"]))
            logger.info("started Streamlit -> http://%s:%s", ui_cfg["host"], ui_cfg["port"])

        if not args.no_attacker:
            runtime = AttackerRuntime(args.policy, drift_path=args.drift, seed=42)
            logger.info("starting attacker stream (Ctrl-C to stop)")
            for tick in runtime.agent.run(int(cfg["inference"].get("ticks", 600))):
                df = records_to_dataframe(runtime._materialize(tick))
                payload = {"records": df.drop(columns=["label", "attack_cat", "fsm_state", "drift_kind", "tick_idx"]).to_dict(orient="records")}
                try:
                    requests.post(f"{api_url}/score", json=payload, timeout=2)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("score request failed: %s", exc)
                time.sleep(runtime.tick_seconds)
        else:
            logger.info("running without attacker; press Ctrl-C to exit")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Ctrl-C received")
    finally:
        for p in (ui_proc, api_proc):
            if p is not None:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.wait(timeout=5)
                except Exception:
                    p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
