"""Client for checkpoint hot-load on a CoreWeave Dedicated Inference deployment.

A hot-load replaces the weights of a running deployment without a restart:

1. Upload a full HF checkpoint (safetensors, index, config.json) to
   ``s3://<bucket>/<pathPrefix>/<identity>/``. The bucket and prefix are the deployment's
   ``hotLoad`` settings.
2. ``POST /v1alpha1/inference/hotLoads`` with the identity.
3. Poll the hot-load until it is COMPLETED, that is, every replica serves the new weights.

Docs: https://docs.coreweave.com/products/inference/rl-rollouts
"""

from __future__ import annotations

import json
import shutil
import struct
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from dapo_nemo_rl import caios

API_URL = "https://api.coreweave.com/v1alpha1/inference"
TERMINAL = {"HOT_LOAD_STATE_COMPLETED", "HOT_LOAD_STATE_FAILED", "HOT_LOAD_STATE_CANCELED"}


class HotLoadError(RuntimeError):
    pass


def checkpoint_dtypes(checkpoint: Path) -> dict[str, str]:
    """Tensor name -> safetensors dtype for every tensor the checkpoint's index lists."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    dtypes: dict[str, str] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(checkpoint / shard, "rb") as fh:
            (size,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(size))
        for name, info in header.items():
            if name != "__metadata__" and name in index["weight_map"]:
                dtypes[name] = info["dtype"]
    missing = set(index["weight_map"]) - set(dtypes)
    if missing:
        raise ValueError(f"{len(missing)} indexed tensors are not in any shard header")
    return dtypes


def add_model_files(base: Path, export: Path) -> None:
    """A full snapshot needs config.json next to the weights; copy the base model's other files."""
    for path in base.iterdir():
        is_weights = path.suffix == ".safetensors" or path.name == "model.safetensors.index.json"
        if path.is_file() and not is_weights and not path.name.startswith("."):
            shutil.copy2(path, export / path.name)


class HotLoadClient:
    def __init__(
        self,
        token: str,
        deployment_id: str,
        bucket: str,
        path_prefix: str,
        *,
        api_url: str = API_URL,
        s3: Any = None,
    ) -> None:
        self.token = token
        self.deployment_id = deployment_id
        self.bucket = bucket
        self.path_prefix = path_prefix.strip("/")
        self.api_url = api_url.rstrip("/")
        self.s3 = s3 if s3 is not None else caios.client()

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.api_url + path,
            data=data,
            method=method,
            # The API's edge rejects urllib's default User-Agent (HTTP 403, error code 1010).
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "dapo-nemo-rl",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise HotLoadError(
                f"{method} {path}: HTTP {exc.code}: {exc.read()[:2000].decode(errors='replace')}"
            ) from exc
        return json.loads(raw) if raw else {}

    def snapshot_prefix(self, identity: str) -> str:
        return f"{self.path_prefix}/{identity}"

    def upload(self, directory: Path, identity: str) -> int:
        return caios.upload_dir(self.s3, directory, self.bucket, self.snapshot_prefix(identity), marker=False)

    def create(self, identity: str, *, timeout: float = 600) -> str:
        """Start a full-snapshot hot-load. Retry while the deployment is busy or not READY."""
        body = {
            "deploymentId": self.deployment_id,
            "hotLoadId": str(uuid.uuid4()),  # client-chosen, so a retry cannot start a second one
            "identity": identity,
            "type": "SNAPSHOT_TYPE_FULL",
            # GRPO is on-policy: drop KV cache computed with the old weights.
            "promptCachePolicy": "PROMPT_CACHE_POLICY_RESET_ALL",
        }
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.request("POST", "/hotLoads", body)["hotLoad"]["spec"]["id"]
            except HotLoadError as exc:
                if "already exists" in str(exc).lower():
                    return body["hotLoadId"]
                text = str(exc).lower()
                retryable = any(s in text for s in ("http 429", "http 5", "in progress", "not ready"))
                if not retryable or time.monotonic() > deadline:
                    raise
            time.sleep(10)

    def wait(self, hot_load_id: str, *, timeout: float = 1800, every: float = 3) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            hot_load = self.request("GET", f"/hotLoads/{hot_load_id}")["hotLoad"]
            state = hot_load.get("status", {}).get("state", "")
            if state == "HOT_LOAD_STATE_COMPLETED":
                return hot_load
            if state in TERMINAL:
                conditions = hot_load.get("status", {}).get("conditions", [])
                raise HotLoadError(f"hot-load {hot_load_id} {state}: {conditions}")
            if time.monotonic() > deadline:
                raise HotLoadError(f"hot-load {hot_load_id} still {state} after {timeout:.0f}s")
            time.sleep(every)

    def publish(self, directory: Path, identity: str) -> dict[str, float]:
        """Upload, hot-load and wait. Return the time of each phase in seconds."""
        started = time.perf_counter()
        size = self.upload(directory, identity)
        uploaded = time.perf_counter()
        self.wait(self.create(identity))
        return {"bytes": size, "upload_s": uploaded - started, "load_s": time.perf_counter() - uploaded}

    def delete(self, identity: str) -> None:
        caios.delete_prefix(self.s3, self.bucket, self.snapshot_prefix(identity))
