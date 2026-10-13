"""Dedicated Inference and object storage resources for one run, managed with ``cwic``.

The launcher uses this module to create, and at the end delete:

* an inference gateway (the HTTPS endpoint Gym calls),
* a hot-load deployment of the base model behind it,
* a temporary object storage access key for the SUNK trainer.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable

Runner = Callable[[list[str]], str]


def run_cwic(args: list[str]) -> str:
    done = subprocess.run(["cwic", *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"cwic {' '.join(args[:3])} failed: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout


@dataclass(frozen=True)
class DeploymentSettings:
    zone: str = "US-WEST-04A"
    instance_type: str = "gd-8xh100ib-i128"
    engine_version: str = "0.30.0+crwv.5"  # dynamo-vllm with hot-load support, vLLM 0.30
    replicas: int = 2
    gpus_per_replica: int = 1
    concurrency: int = 256  # sequences per replica (vLLM --max-num-seqs)
    max_model_len: int = 3072


def deployment_spec(
    name: str, gateway_id: str, bucket: str, model_path: str, hotload_prefix: str, s: DeploymentSettings
) -> dict[str, Any]:
    """The ``CreateDeploymentRequest`` body. Hot-load can only be enabled at creation."""
    return {
        "name": name,
        "gatewayIds": [gateway_id],
        "runtime": {
            "engine": "dynamo-vllm",
            "version": s.engine_version,
            "engineConfig": {"max-model-len": str(s.max_model_len), "gpu-memory-utilization": "0.90"},
        },
        "resources": {"instanceType": s.instance_type, "gpuCount": s.gpus_per_replica},
        "model": {"name": name, "bucket": bucket, "path": model_path},
        "autoscaling": {"min": s.replicas, "max": s.replicas, "concurrency": s.concurrency},
        "traffic": {"weight": 100},
        "hotLoad": {"bucket": bucket, "pathPrefix": hotload_prefix, "transitionMode": "TRANSITION_MODE_ASYNC"},
    }


class Inference:
    def __init__(self, cwic: Runner = run_cwic, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.cwic = cwic
        self.sleep = sleep

    def json(self, *args: str) -> Any:
        return json.loads(self.cwic([*args, "-o", "json"]))

    # ------------------------------------------------------------------ gateway

    def create_gateway(self, name: str, zone: str, *, timeout: float = 900) -> tuple[str, str]:
        """Create a gateway with CoreWeave auth. Return (id, https endpoint) once READY."""
        gateway_id = self.json("inference", "gateway", "create", "--name", name, "--zones", zone)["spec"]["id"]
        gateway = self.wait("gateway", gateway_id, lambda g: g["status"].get("status") == "STATUS_READY", timeout)
        return gateway_id, gateway["status"]["endpoints"][0]

    def delete_gateway(self, gateway_id: str) -> None:
        self.cwic(["inference", "gateway", "delete", gateway_id])

    # --------------------------------------------------------------- deployment

    def create_deployment(self, spec: dict[str, Any]) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".json") as fh:
            json.dump(spec, fh)
            fh.flush()
            return self.json("inference", "deployment", "create", "-f", fh.name)["spec"]["id"]

    def wait_deployment(self, deployment_id: str, replicas: int, *, timeout: float = 3600) -> dict[str, Any]:
        def ready(d: dict[str, Any]) -> bool:
            status = d["status"]
            if status.get("status") in ("STATUS_FAILED", "STATUS_ERROR"):
                raise RuntimeError(f"deployment {deployment_id} {status.get('status')}: {status.get('conditions')}")
            messages = [c.get("message", "") for c in status.get("conditions", []) if c.get("type") == "Ready"]
            return status.get("status") == "STATUS_READY" and any(f"{replicas}/{replicas}" in m for m in messages)

        return self.wait("deployment", deployment_id, ready, timeout)

    def delete_deployment(self, deployment_id: str, *, timeout: float = 900) -> None:
        """Delete and wait until it is gone, so its gateway can be deleted next."""
        self.cwic(["inference", "deployment", "delete", deployment_id])
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.json("inference", "deployment", "get", deployment_id)
            except RuntimeError:  # not found
                return
            self.sleep(15)

    def wait(self, kind: str, rid: str, done: Callable[[dict[str, Any]], bool], timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            obj = self.json("inference", kind, "get", rid)
            if done(obj):
                return obj
            if time.monotonic() > deadline:
                raise RuntimeError(f"{kind} {rid} not ready after {timeout:.0f}s: {obj.get('status')}")
            self.sleep(15)

    # ----------------------------------------------------------- object storage

    def bucket_zone(self, bucket: str) -> str:
        return self.json("cwobject", "bucket", "get", bucket)["location"]

    def create_access_key(self, name: str, seconds: int) -> dict[str, str]:
        """A temporary access key: ``accessKeyID``, ``secretKey``, ``expiry``."""
        return self.json("cwobject", "access-key", "create", "--name", name, "--duration", str(seconds))

    def delete_access_key(self, key_id: str) -> None:
        self.cwic(["cwobject", "access-key", "delete", key_id, "--yes"])
