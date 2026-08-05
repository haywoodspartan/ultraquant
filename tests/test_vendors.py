"""Tests for the array control planes, driven against local mock endpoints.

A real FlashArray, 3PAR or Lightbits cluster cannot be reached from a test
suite, so these exercise the parts that are ours: request construction, auth
header placement, response parsing, unit conversion, and failure handling. The
mock speaks each vendor's documented request and response shapes over plain
HTTP; what is *not* proven here is that a real array agrees with the spec.
"""

from __future__ import annotations

import http.server
import json
import threading
import unittest

from ultraquant.storage.vendors import (
    ArrayError,
    HPE3PAR,
    LightbitsCluster,
    PureFlashArray,
    controller_for,
)

_STATE: dict = {}


def _lower(headers) -> dict:
    """Header names lowercased: HTTP is case-insensitive, urllib title-cases."""
    return {k.lower(): v for k, v in headers.items()}


class _MockArray(http.server.BaseHTTPRequestHandler):
    """Answers the three vendors' management endpoints."""

    def log_message(self, *args) -> None:
        """Silence the server."""

    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?")[0]
        _STATE.setdefault("seen", []).append(("GET", path, _lower(self.headers)))

        if path.endswith("/volumes") and "/api/2" in path:
            if self.headers.get("x-auth-token") != "session-abc":
                return self._send({"errors": ["unauthorised"]}, 401)
            return self._send({"items": [
                {"name": "uq-library", "provisioned": 2199023255552,
                 "serial": "PURE123", "space": {"total_physical": 1099511627776}},
            ]})
        if path.endswith("/arrays"):
            return self._send({"items": [
                {"capacity": 10995116277760, "space": {"total_physical": 2199023255552}},
            ]})
        if path.endswith("/volume-snapshots"):
            return self._send({"items": [
                {"name": "uq-library.daily", "source": {"name": "uq-library"},
                 "created": 1700000000, "provisioned": 2199023255552},
            ]})
        if path == "/api/v1/volumes":
            if self.headers.get("X-HP3PAR-WSAPI-SessionKey") != "key-xyz":
                return self._send({"desc": "no session"}, 403)
            return self._send({"members": [
                {"name": "uq-library", "sizeMiB": 2097152, "usedSizeMiB": 1048576, "id": 7},
                {"name": "uq-library.snap", "sizeMiB": 2097152, "copyOf": "uq-library",
                 "creationTimeSec": 1700000000},
            ]})
        if path == "/api/v1/system":
            return self._send({"totalCapacityMiB": 10485760, "allocatedCapacityMiB": 2097152})
        if path == "/api/v2/clusterinfo":
            if self.headers.get("Authorization") != "Bearer jwt-token":
                return self._send({"message": "denied"}, 401)
            return self._send({"statistics": {
                "physicalStorage": 10995116277760, "physicalUsedStorage": 1099511627776,
            }})
        if path == "/api/v2/volumes":
            return self._send({"volumes": [
                {"name": "uq-library", "size": 2199023255552, "UUID": "lb-uuid-1"},
            ]})
        if path == "/api/v2/snapshots":
            return self._send({"snapshots": [
                {"name": "uq-snap", "sourceVolumeName": "uq-library",
                 "creationTime": "2026-01-01T00:00:00Z", "size": 2199023255552},
            ]})
        return self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?")[0]
        body = self._body()
        _STATE.setdefault("posts", []).append((path, body, _lower(self.headers)))

        if path.endswith("/login"):
            if self.headers.get("api-token") != "pure-token":
                return self._send({"errors": ["bad token"]}, 401)
            return self._send({"x-auth-token": "session-abc"})
        if path.endswith("/volumes") and "/api/2" in path:
            return self._send({"items": [
                {"name": "new-vol", "provisioned": body.get("provisioned", 0),
                 "serial": "PURE999"},
            ]})
        if path.endswith("/volume-snapshots"):
            return self._send({"items": [
                {"name": "uq-library.fresh", "created": 1700000001,
                 "provisioned": 2199023255552},
            ]})
        if path == "/api/v1/credentials":
            if body.get("user") == "admin" and body.get("password") == "secret":
                return self._send({"key": "key-xyz"})
            return self._send({"desc": "invalid credentials"}, 403)
        if path == "/api/v1/volumes":
            return self._send({}, 201)
        if path.startswith("/api/v1/volumes/"):
            return self._send({}, 200)
        if path == "/api/v2/volumes":
            return self._send({"name": body.get("name"), "size": int(body.get("size", 0)),
                               "UUID": "lb-new"})
        if path == "/api/v2/snapshots":
            return self._send({"name": body.get("name"),
                               "creationTime": "2026-02-02T00:00:00Z"})
        return self._send({"error": "not found"}, 404)


class VendorTests(unittest.TestCase):
    """Each client builds the right requests and parses the right fields."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockArray)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self) -> None:
        _STATE.clear()

    # -- Pure ------------------------------------------------------------

    def test_pure_login_then_list(self) -> None:
        array = PureFlashArray(self.base, api_token="pure-token")
        array.connect()
        volumes = array.list_volumes()
        self.assertEqual(volumes[0].name, "uq-library")
        self.assertEqual(volumes[0].size_bytes, 2199023255552)
        self.assertEqual(volumes[0].serial, "PURE123")

    def test_pure_bad_token_reports_unavailable(self) -> None:
        array = PureFlashArray(self.base, api_token="wrong")
        self.assertFalse(array.available(), "available() must not raise")
        with self.assertRaises(ArrayError):
            array.connect()

    def test_pure_session_token_is_sent_not_the_api_token(self) -> None:
        array = PureFlashArray(self.base, api_token="pure-token")
        array.connect()
        array.list_volumes()
        get_headers = [h for method, path, h in _STATE["seen"] if method == "GET"][-1]
        self.assertEqual(get_headers.get("x-auth-token"), "session-abc")
        self.assertIsNone(get_headers.get("api-token"),
                          "the long-lived API token must not be resent")

    def test_pure_create_and_snapshot(self) -> None:
        array = PureFlashArray(self.base, api_token="pure-token")
        volume = array.create_volume("new-vol", 1024 ** 3)
        self.assertEqual(volume.name, "new-vol")
        snapshot = array.snapshot("uq-library", "fresh")
        self.assertEqual(snapshot.source, "uq-library")
        self.assertTrue(snapshot.name)

    def test_pure_capacity_math(self) -> None:
        array = PureFlashArray(self.base, api_token="pure-token")
        capacity = array.capacity()
        self.assertEqual(capacity["total_bytes"], 10995116277760)
        self.assertEqual(
            capacity["free_bytes"], capacity["total_bytes"] - capacity["used_bytes"]
        )

    # -- 3PAR ------------------------------------------------------------

    def test_3par_session_and_units(self) -> None:
        array = HPE3PAR(self.base, username="admin", password="secret")
        array.connect()
        volumes = array.list_volumes()
        self.assertEqual(volumes[0].name, "uq-library")
        self.assertEqual(volumes[0].size_bytes, 2097152 * 1024 * 1024,
                         "3PAR reports MiB and must be converted")

    def test_3par_forgets_the_password_after_login(self) -> None:
        array = HPE3PAR(self.base, username="admin", password="secret")
        array.connect()
        self.assertEqual(array._password, "",
                         "the password must not stay resident after the session key")

    def test_3par_bad_credentials(self) -> None:
        array = HPE3PAR(self.base, username="admin", password="nope")
        with self.assertRaises(ArrayError):
            array.connect()

    def test_3par_snapshot_uses_the_copy_action(self) -> None:
        array = HPE3PAR(self.base, username="admin", password="secret")
        snapshot = array.snapshot("uq-library", "nightly")
        self.assertEqual(snapshot.name, "nightly")
        path, body, _headers = _STATE["posts"][-1]
        self.assertTrue(path.startswith("/api/v1/volumes/"))
        self.assertEqual(body["action"], 1)
        self.assertTrue(body["parameters"]["readOnly"])

    def test_3par_snapshots_found_by_copy_of(self) -> None:
        array = HPE3PAR(self.base, username="admin", password="secret")
        snapshots = array.list_snapshots("uq-library")
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].source, "uq-library")

    def test_3par_create_rounds_up_to_mib(self) -> None:
        array = HPE3PAR(self.base, username="admin", password="secret")
        array.create_volume("v", 1024 * 1024 + 1)
        _path, body, _headers = _STATE["posts"][-1]
        self.assertEqual(body["sizeMiB"], 2)
        self.assertTrue(body["tpvv"])

    # -- Lightbits -------------------------------------------------------

    def test_lightbits_bearer_and_volumes(self) -> None:
        cluster = LightbitsCluster(self.base, jwt="jwt-token")
        cluster.connect()
        volumes = cluster.list_volumes()
        self.assertEqual(volumes[0].name, "uq-library")
        self.assertEqual(volumes[0].serial, "lb-uuid-1")

    def test_lightbits_bad_token(self) -> None:
        cluster = LightbitsCluster(self.base, jwt="nope")
        self.assertFalse(cluster.available())

    def test_lightbits_create_sets_replicas(self) -> None:
        cluster = LightbitsCluster(self.base, jwt="jwt-token")
        cluster.create_volume("v", 1024 ** 3, replicas=3)
        _path, body, _headers = _STATE["posts"][-1]
        self.assertEqual(body["replicaCount"], 3)
        self.assertEqual(body["projectName"], "default")

    def test_lightbits_snapshot(self) -> None:
        cluster = LightbitsCluster(self.base, jwt="jwt-token")
        snapshot = cluster.snapshot("uq-library", "s1")
        self.assertEqual(snapshot.name, "s1")
        self.assertEqual(snapshot.source, "uq-library")

    # -- shared behaviour -------------------------------------------------

    def test_snapshot_library_records_the_vault(self) -> None:
        import shutil
        import tempfile
        from pathlib import Path

        from ultraquant.shards.vault import ShardVault

        tmp = Path(tempfile.mkdtemp(prefix="uq_snap_"))
        try:
            vault = ShardVault(tmp / "vault")
            vault.add_shard("expert:a", "a", {"w": [1.0]})
            array = PureFlashArray(self.base, api_token="pure-token")
            info = array.snapshot_library("uq-library", vault=vault, label="fresh")
            self.assertEqual(info["vendor"], "pure-flasharray")
            self.assertEqual(info["source"], "uq-library")
            self.assertEqual(info["vault"]["shards"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_describe_never_leaks_credentials(self) -> None:
        for array in (
            PureFlashArray(self.base, api_token="pure-token"),
            HPE3PAR(self.base, username="admin", password="secret"),
            LightbitsCluster(self.base, jwt="jwt-token"),
        ):
            with self.subTest(vendor=array.vendor):
                blob = json.dumps(array.describe())
                for secret in ("pure-token", "secret", "jwt-token"):
                    self.assertNotIn(secret, blob)

    def test_unreachable_endpoint_is_reported_not_raised(self) -> None:
        array = PureFlashArray("http://127.0.0.1:1", api_token="x", timeout=1.0)
        self.assertFalse(array.available())
        with self.assertRaises(ArrayError):
            array.list_volumes()

    def test_controller_factory(self) -> None:
        self.assertIsInstance(
            controller_for("pure", endpoint=self.base, api_token="t"), PureFlashArray
        )
        self.assertIsInstance(
            controller_for("3par", endpoint=self.base, username="u", password="p"), HPE3PAR
        )
        with self.assertRaises(ArrayError):
            controller_for("netapp", endpoint=self.base)

    def test_endpoint_defaults_to_https(self) -> None:
        array = PureFlashArray("array.example.net", api_token="t")
        self.assertEqual(array.endpoint, "https://array.example.net")
        self.assertTrue(array.describe()["verify_tls"])


if __name__ == "__main__":
    unittest.main()
