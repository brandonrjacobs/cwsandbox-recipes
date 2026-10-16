import argparse
from types import SimpleNamespace

import pytest

import recovery


def test_backup_stops_databases_before_copying_and_keeps_node_identity():
    assert recovery.BACKUP.index("docker stop") < recovery.BACKUP.index("cp -a")
    assert recovery.BACKUP.index("pkill -x headscale") < recovery.BACKUP.index("cp -a")
    assert "/cluster/server /checkpoint/k3s/" in recovery.BACKUP
    assert "/demo/node/. /checkpoint/node/" in recovery.BACKUP
    assert "/demo/tailscaled.state /checkpoint/" in recovery.BACKUP


def test_restore_refuses_changed_worker_membership_before_stopping_source():
    state = {
        "checkpoint": {
            "phase": "ready",
            "snapshot_id": "saved",
            "manifest": {"workers": {"one": "original"}, "source_sandbox_id": "source"},
        },
        "workers": {"one": {"sandbox_id": "replacement"}},
    }
    stopped = []
    api = SimpleNamespace(
        load=lambda: state,
        save=lambda _: None,
        stop_node=lambda node: stopped.append(node),
        Sandbox=SimpleNamespace(
            get_snapshot=lambda _: SimpleNamespace(
                result=lambda: SimpleNamespace(
                    status=SimpleNamespace(value="ready"),
                    source_sandbox_id="source",
                    created_at=SimpleNamespace(timestamp=lambda: 1),
                )
            )
        ),
    )
    with pytest.raises(RuntimeError, match="membership changed"):
        recovery.restore(argparse.Namespace(test=True), api)
    assert not stopped


def test_snapshot_cleanup_recovers_lost_create_response():
    state = {"checkpoint": {"manifest": {"source_sandbox_id": "owned-source"}}}
    deleted = []

    def listing(**kwargs):
        assert kwargs == {"source_sandbox_id": "owned-source"}
        return SimpleNamespace(
            result=lambda: [SimpleNamespace(file_system_snapshot_id="owned-snapshot")]
        )

    def delete(sid, **kwargs):
        assert kwargs == {"missing_ok": True}
        deleted.append(sid)
        return SimpleNamespace(result=lambda: None)

    api = SimpleNamespace(
        Sandbox=SimpleNamespace(list_snapshots=listing, delete_snapshot=delete), save=lambda _: None
    )
    recovery.cleanup_snapshots(state, api)
    assert deleted == ["owned-snapshot"]


@pytest.mark.parametrize(
    ("source", "created", "message"),
    [("another-sandbox", 20, "different source"), ("owned-source", 5, "predates")],
)
def test_restore_rejects_unrelated_or_old_snapshot_before_stopping_source(source, created, message):
    state = {
        "checkpoint": {
            "phase": "prepared",
            "prepared_at": 10,
            "manifest": {"source_sandbox_id": "owned-source"},
        }
    }
    stopped = []
    record = SimpleNamespace(
        status=SimpleNamespace(value="ready"),
        source_sandbox_id=source,
        created_at=SimpleNamespace(timestamp=lambda: created),
    )
    api = SimpleNamespace(
        load=lambda: state,
        stop_node=lambda n: stopped.append(n),
        Sandbox=SimpleNamespace(get_snapshot=lambda _: SimpleNamespace(result=lambda: record)),
    )
    with pytest.raises(RuntimeError, match=message):
        recovery.restore(argparse.Namespace(snapshot="selected", test=True), api)
    assert not stopped


def test_recovery_waits_for_fresh_kubelet_heartbeats(monkeypatch):
    import json

    calls = []

    def leases(*args):
        calls.append(args)
        stamp = "1970-01-01T00:00:10Z" if len(calls) == 1 else "1970-01-01T00:02:00Z"
        return json.dumps(
            {
                "items": [
                    {"metadata": {"name": name}, "spec": {"renewTime": stamp}}
                    for name in ["k3s-server", "k3s-worker-1"]
                ]
            }
        )

    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    api = SimpleNamespace(kube=leases)
    recovery.wait_for_fresh_leases({"workers": {"k3s-worker-1": {}}}, object(), 100, api)
    assert len(calls) == 2


def test_failed_fencing_never_creates_a_second_control_plane():
    import time

    state = {
        "checkpoint": {
            "phase": "ready",
            "snapshot_id": "snapshot",
            "manifest": {"source_sandbox_id": "source", "workers": {}},
        },
        "workers": {},
        "server": {"sandbox_id": "source"},
        "expires_at": time.time() + 3600,
    }
    created = []

    def stop(_):
        raise TimeoutError("termination not confirmed")

    record = SimpleNamespace(
        status=SimpleNamespace(value="ready"),
        source_sandbox_id="source",
        created_at=SimpleNamespace(timestamp=lambda: 1),
    )
    api = SimpleNamespace(
        load=lambda: state,
        save=lambda _: None,
        stop_node=stop,
        new_sandbox=lambda *args: created.append(args),
        Sandbox=SimpleNamespace(get_snapshot=lambda _: SimpleNamespace(result=lambda: record)),
    )
    with pytest.raises(TimeoutError, match="termination not confirmed"):
        recovery.restore(argparse.Namespace(snapshot=None, test=True), api)
    assert not created
