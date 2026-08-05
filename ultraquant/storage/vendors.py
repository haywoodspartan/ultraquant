"""Array control planes: provisioning, snapshots and capacity.

The data path to Pure Storage, 3PAR and Lightbits is a block device — that is
:mod:`ultraquant.storage.blockdev`, and it is where shard bytes actually move.
This module is the *other* half, the part that genuinely differs per vendor: the
management REST API that creates the volume the library lives on, snapshots it,
and reports how much room is left.

Why it belongs here at all: a shard library is a model, and a model wants
point-in-time copies. ``snapshot_library()`` takes a crash-consistent array
snapshot of the volume holding a library — which is a far cheaper way to version
a 200 GB model than copying it, and it composes with the Ar(T)chive, which
records *that* the snapshot was taken and under which digest.

Every client speaks HTTPS with the standard library only, verifies certificates
by default, and reports itself unreachable rather than raising at import. None of
them can be exercised without a real array and credentials, so the test suite
drives them against a local mock that mimics each vendor's request and response
shapes; treat the endpoint paths as written-to-spec, not as field-proven.

Credentials are never logged, never placed in a URL, and never stored on the
instance beyond the session token the array itself issues.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from ultraquant.storage.base import StorageError

__all__ = [
    "ArrayError",
    "Volume",
    "Snapshot",
    "ArrayController",
    "PureFlashArray",
    "HPE3PAR",
    "LightbitsCluster",
    "controller_for",
    "VENDORS",
]


class ArrayError(StorageError):
    """Raised when an array management call fails."""


@dataclass(frozen=True)
class Volume:
    """A provisioned volume on an array."""

    name: str
    size_bytes: int
    serial: str = ""
    used_bytes: int = 0
    raw: dict = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Snapshot:
    """A point-in-time copy of a volume."""

    name: str
    source: str
    created: str = ""
    size_bytes: int = 0
    raw: dict = None  # type: ignore[assignment]


class ArrayController:
    """Shared HTTPS plumbing for a storage array's management API."""

    #: Vendor label used in reports.
    vendor = "array"
    #: Storage URI scheme whose data path this array serves.
    data_scheme = "blockdev"

    def __init__(
        self,
        endpoint: str,
        ca_file: str | None = None,
        insecure: bool = False,
        timeout: float = 20.0,
    ) -> None:
        """Configure the client.

        Args:
            endpoint: Management address, e.g. ``https://array.example.net``.
            ca_file: PEM bundle trusting the array's certificate. Arrays
                routinely present a self-signed certificate; exporting it and
                passing it here is the correct fix.
            insecure: Disable certificate verification entirely. Only for a
                trusted management network where the array's certificate cannot
                be exported — it makes the connection interceptable.
            timeout: Per-request timeout in seconds.
        """
        if "://" not in endpoint:
            endpoint = f"https://{endpoint}"
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)
        self.insecure = bool(insecure)
        self._token: str | None = None

        if insecure:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            from ultraquant.interpreter.webaccess import build_ssl_context

            context = build_ssl_context(ca_file)
        self._context = context

    # -- transport ---------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        """Issue one JSON request and decode the reply.

        Raises:
            ArrayError: On transport failure or a non-2xx status.
        """
        url = f"{self.endpoint}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/json")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=self._context
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:  # noqa: BLE001 - best effort
                detail = ""
            raise ArrayError(
                f"{self.vendor}: HTTP {exc.code} {exc.reason} for {method} {path}"
                + (f" — {detail}" if detail else "")
            ) from exc
        except urllib.error.URLError as exc:
            raise ArrayError(f"{self.vendor}: cannot reach {self.endpoint}: {exc.reason}") from exc
        except ssl.SSLError as exc:
            raise ArrayError(
                f"{self.vendor}: TLS error ({exc}); pass ca_file with the array's "
                f"certificate, or insecure=True on a trusted management network"
            ) from exc

        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ArrayError(f"{self.vendor}: unreadable response to {method} {path}") from exc

    # -- interface ---------------------------------------------------------

    def connect(self) -> None:
        """Authenticate. Subclasses override.

        Raises:
            ArrayError: If authentication fails.
        """
        raise NotImplementedError

    def available(self) -> bool:
        """Whether the array answers and accepts our credentials. Never raises."""
        try:
            self.connect()
        except Exception:  # noqa: BLE001 - unreachable is a normal answer here
            return False
        return True

    def list_volumes(self) -> list[Volume]:
        """Volumes on the array."""
        raise NotImplementedError

    def create_volume(self, name: str, size_bytes: int) -> Volume:
        """Provision a volume to hold a shard library."""
        raise NotImplementedError

    def snapshot(self, volume: str, name: str | None = None) -> Snapshot:
        """Take a point-in-time copy of ``volume``."""
        raise NotImplementedError

    def list_snapshots(self, volume: str | None = None) -> list[Snapshot]:
        """Snapshots, optionally for one volume."""
        raise NotImplementedError

    def capacity(self) -> dict:
        """Total, used and free bytes on the array."""
        raise NotImplementedError

    # -- integration -------------------------------------------------------

    def snapshot_library(self, volume: str, vault: Any = None, label: str = "") -> dict:
        """Snapshot the volume holding a shard library, and record it.

        Copying a multi-hundred-gigabyte model to version it is impractical; an
        array snapshot is near-instant and space-efficient. When a vault is
        supplied its catalog summary is returned alongside, so the snapshot can
        be tied to exactly what the library contained at that moment.

        Args:
            volume: Volume backing the library.
            vault: Optional :class:`~ultraquant.shards.vault.ShardVault`.
            label: Optional snapshot name.

        Returns:
            ``{"snapshot": ..., "vault": ...}``.
        """
        snap = self.snapshot(volume, label or None)
        info: dict = {
            "vendor": self.vendor,
            "snapshot": snap.name,
            "source": snap.source,
            "created": snap.created,
        }
        if vault is not None:
            stats = vault.stats()
            info["vault"] = {
                "shards": stats["shards"],
                "total_bytes": stats["total_bytes"],
                "libraries": stats.get("libraries", []),
            }
        return info

    def describe(self) -> dict:
        """A JSON-safe summary; never includes credentials."""
        return {
            "vendor": self.vendor,
            "endpoint": self.endpoint,
            "data_scheme": self.data_scheme,
            "authenticated": self._token is not None,
            "verify_tls": not self.insecure,
        }


class PureFlashArray(ArrayController):
    """Pure Storage FlashArray, via the Purity//FA REST API (2.x).

    Authentication exchanges an API token for a short-lived session token,
    which is then sent as ``x-auth-token``.
    """

    vendor = "pure-flasharray"
    data_scheme = "pure"

    def __init__(self, endpoint: str, api_token: str, version: str = "2.26", **kwargs) -> None:
        """Configure the client.

        Args:
            endpoint: Array management address.
            api_token: Purity API token for the user.
            version: REST API version segment.
            **kwargs: Passed to :class:`ArrayController`.
        """
        super().__init__(endpoint, **kwargs)
        self._api_token = api_token
        self.version = version

    @property
    def _base(self) -> str:
        """API path prefix."""
        return f"/api/{self.version}"

    def _auth(self) -> dict:
        """Auth header for a call.

        Raises:
            ArrayError: If not connected.
        """
        if self._token is None:
            self.connect()
        return {"x-auth-token": self._token or ""}

    def connect(self) -> None:
        """Exchange the API token for a session token."""
        reply = self._request(
            "POST", f"{self._base}/login", headers={"api-token": self._api_token}
        )
        token = None
        if isinstance(reply, dict):
            token = reply.get("x-auth-token") or reply.get("items", [{}])[0].get(
                "x-auth-token"
            ) if reply.get("items") else reply.get("x-auth-token")
        if not token:
            raise ArrayError("pure-flasharray: login returned no session token")
        self._token = token

    def list_volumes(self) -> list[Volume]:
        """Volumes on the array."""
        reply = self._request("GET", f"{self._base}/volumes", headers=self._auth())
        out = []
        for item in (reply.get("items") or []):
            space = item.get("space") or {}
            out.append(Volume(
                name=item.get("name", ""),
                size_bytes=int(item.get("provisioned") or 0),
                serial=item.get("serial", ""),
                used_bytes=int(space.get("total_physical") or 0),
                raw=item,
            ))
        return out

    def create_volume(self, name: str, size_bytes: int) -> Volume:
        """Provision a volume."""
        reply = self._request(
            "POST", f"{self._base}/volumes",
            params={"names": name},
            body={"provisioned": int(size_bytes)},
            headers=self._auth(),
        )
        items = reply.get("items") or [{}]
        item = items[0]
        return Volume(
            name=item.get("name", name),
            size_bytes=int(item.get("provisioned") or size_bytes),
            serial=item.get("serial", ""),
            raw=item,
        )

    def snapshot(self, volume: str, name: str | None = None) -> Snapshot:
        """Take a volume snapshot."""
        params = {"source_names": volume}
        if name:
            params["suffix"] = name
        reply = self._request(
            "POST", f"{self._base}/volume-snapshots", params=params, headers=self._auth()
        )
        items = reply.get("items") or [{}]
        item = items[0]
        return Snapshot(
            name=item.get("name", f"{volume}.{name or 'snap'}"),
            source=volume,
            created=str(item.get("created", "")),
            size_bytes=int(item.get("provisioned") or 0),
            raw=item,
        )

    def list_snapshots(self, volume: str | None = None) -> list[Snapshot]:
        """Volume snapshots."""
        params = {"source_names": volume} if volume else None
        reply = self._request(
            "GET", f"{self._base}/volume-snapshots", params=params, headers=self._auth()
        )
        return [
            Snapshot(
                name=item.get("name", ""),
                source=(item.get("source") or {}).get("name", volume or ""),
                created=str(item.get("created", "")),
                size_bytes=int(item.get("provisioned") or 0),
                raw=item,
            )
            for item in (reply.get("items") or [])
        ]

    def capacity(self) -> dict:
        """Array capacity."""
        reply = self._request("GET", f"{self._base}/arrays", headers=self._auth())
        items = reply.get("items") or [{}]
        space = items[0].get("space") or {}
        total = int(items[0].get("capacity") or 0)
        used = int(space.get("total_physical") or 0)
        return {"total_bytes": total, "used_bytes": used, "free_bytes": max(0, total - used)}


class HPE3PAR(ArrayController):
    """HPE/IBM 3PAR (and Primera) via WSAPI.

    A session key is obtained from ``/api/v1/credentials`` and returned in the
    ``X-HP3PAR-WSAPI-SessionKey`` header on every later call.
    """

    vendor = "3par"
    data_scheme = "3par"

    def __init__(self, endpoint: str, username: str, password: str, **kwargs) -> None:
        """Configure the client.

        Args:
            endpoint: WSAPI address (commonly port 8080).
            username: WSAPI user.
            password: WSAPI password. Held only until the session key is issued.
            **kwargs: Passed to :class:`ArrayController`.
        """
        super().__init__(endpoint, **kwargs)
        self._user = username
        self._password = password

    def _auth(self) -> dict:
        """Auth header for a call."""
        if self._token is None:
            self.connect()
        return {"X-HP3PAR-WSAPI-SessionKey": self._token or ""}

    def connect(self) -> None:
        """Obtain a WSAPI session key."""
        reply = self._request(
            "POST", "/api/v1/credentials",
            body={"user": self._user, "password": self._password},
        )
        key = reply.get("key") if isinstance(reply, dict) else None
        if not key:
            raise ArrayError("3par: credentials call returned no session key")
        self._token = key
        # The password is not needed again; drop it rather than keep it resident.
        self._password = ""

    def list_volumes(self) -> list[Volume]:
        """Virtual volumes."""
        reply = self._request("GET", "/api/v1/volumes", headers=self._auth())
        out = []
        for item in (reply.get("members") or []):
            # 3PAR reports sizes in MiB.
            out.append(Volume(
                name=item.get("name", ""),
                size_bytes=int(item.get("sizeMiB") or 0) * 1024 * 1024,
                serial=str(item.get("id", "")),
                used_bytes=int(item.get("usedSizeMiB") or 0) * 1024 * 1024,
                raw=item,
            ))
        return out

    def create_volume(self, name: str, size_bytes: int, cpg: str = "SSD_r6") -> Volume:
        """Provision a thin virtual volume.

        Args:
            name: Volume name.
            size_bytes: Requested size; rounded up to whole MiB.
            cpg: Common Provisioning Group to carve it from.
        """
        size_mib = max(1, (int(size_bytes) + 1024 * 1024 - 1) // (1024 * 1024))
        self._request(
            "POST", "/api/v1/volumes",
            body={"name": name, "cpg": cpg, "sizeMiB": size_mib, "tpvv": True},
            headers=self._auth(),
        )
        return Volume(name=name, size_bytes=size_mib * 1024 * 1024, raw={"cpg": cpg})

    def snapshot(self, volume: str, name: str | None = None) -> Snapshot:
        """Create a virtual copy (snapshot) of a volume."""
        snap_name = name or f"{volume}.snap"
        self._request(
            "POST", f"/api/v1/volumes/{urllib.parse.quote(volume)}",
            body={
                "action": 1,                       # CREATE_SNAPSHOT
                "parameters": {"name": snap_name, "readOnly": True},
            },
            headers=self._auth(),
        )
        return Snapshot(name=snap_name, source=volume, raw={})

    def list_snapshots(self, volume: str | None = None) -> list[Snapshot]:
        """Snapshots, identified by their copy-of relationship."""
        reply = self._request("GET", "/api/v1/volumes", headers=self._auth())
        out = []
        for item in (reply.get("members") or []):
            parent = item.get("copyOf")
            if not parent:
                continue
            if volume and parent != volume:
                continue
            out.append(Snapshot(
                name=item.get("name", ""),
                source=parent,
                created=str(item.get("creationTimeSec", "")),
                size_bytes=int(item.get("sizeMiB") or 0) * 1024 * 1024,
                raw=item,
            ))
        return out

    def capacity(self) -> dict:
        """System capacity."""
        reply = self._request("GET", "/api/v1/system", headers=self._auth())
        total = int(reply.get("totalCapacityMiB") or 0) * 1024 * 1024
        used = int(reply.get("allocatedCapacityMiB") or 0) * 1024 * 1024
        return {"total_bytes": total, "used_bytes": used, "free_bytes": max(0, total - used)}


class LightbitsCluster(ArrayController):
    """Lightbits NVMe/TCP cluster, via its REST API with a JWT bearer token."""

    vendor = "lightbits"
    data_scheme = "lightbits"

    def __init__(self, endpoint: str, jwt: str, project: str = "default", **kwargs) -> None:
        """Configure the client.

        Args:
            endpoint: Cluster API address.
            jwt: Bearer token issued by the cluster.
            project: Project namespace volumes live in.
            **kwargs: Passed to :class:`ArrayController`.
        """
        super().__init__(endpoint, **kwargs)
        self._jwt = jwt
        self.project = project

    def _auth(self) -> dict:
        """Auth header for a call."""
        return {"Authorization": f"Bearer {self._jwt}"}

    def connect(self) -> None:
        """Confirm the token is accepted."""
        self._request("GET", "/api/v2/clusterinfo", headers=self._auth())
        self._token = "bearer"

    def list_volumes(self) -> list[Volume]:
        """Volumes in the project."""
        reply = self._request(
            "GET", "/api/v2/volumes", params={"project-name": self.project},
            headers=self._auth(),
        )
        items = reply.get("volumes") if isinstance(reply, dict) else reply
        return [
            Volume(
                name=item.get("name", ""),
                size_bytes=int(item.get("size") or 0),
                serial=item.get("UUID", "") or item.get("uuid", ""),
                raw=item,
            )
            for item in (items or [])
        ]

    def create_volume(self, name: str, size_bytes: int, replicas: int = 2) -> Volume:
        """Provision a replicated volume."""
        reply = self._request(
            "POST", "/api/v2/volumes",
            body={
                "name": name,
                "size": str(int(size_bytes)),
                "replicaCount": int(replicas),
                "projectName": self.project,
                "acl": {"values": ["ALLOW_ANY"]},
            },
            headers=self._auth(),
        )
        return Volume(
            name=reply.get("name", name),
            size_bytes=int(reply.get("size") or size_bytes),
            serial=reply.get("UUID", ""),
            raw=reply if isinstance(reply, dict) else {},
        )

    def snapshot(self, volume: str, name: str | None = None) -> Snapshot:
        """Take a volume snapshot."""
        snap_name = name or f"{volume}-snap"
        reply = self._request(
            "POST", "/api/v2/snapshots",
            body={
                "name": snap_name,
                "sourceVolumeName": volume,
                "projectName": self.project,
            },
            headers=self._auth(),
        )
        return Snapshot(
            name=reply.get("name", snap_name) if isinstance(reply, dict) else snap_name,
            source=volume,
            created=str(reply.get("creationTime", "")) if isinstance(reply, dict) else "",
            raw=reply if isinstance(reply, dict) else {},
        )

    def list_snapshots(self, volume: str | None = None) -> list[Snapshot]:
        """Snapshots in the project."""
        reply = self._request(
            "GET", "/api/v2/snapshots", params={"project-name": self.project},
            headers=self._auth(),
        )
        items = reply.get("snapshots") if isinstance(reply, dict) else reply
        out = []
        for item in (items or []):
            source = item.get("sourceVolumeName", "")
            if volume and source != volume:
                continue
            out.append(Snapshot(
                name=item.get("name", ""),
                source=source,
                created=str(item.get("creationTime", "")),
                size_bytes=int(item.get("size") or 0),
                raw=item,
            ))
        return out

    def capacity(self) -> dict:
        """Cluster capacity."""
        reply = self._request("GET", "/api/v2/clusterinfo", headers=self._auth())
        stats = reply.get("statistics") or reply
        total = int(stats.get("physicalStorage") or stats.get("totalPhysicalStorage") or 0)
        used = int(stats.get("physicalUsedStorage") or 0)
        return {"total_bytes": total, "used_bytes": used, "free_bytes": max(0, total - used)}


#: Vendor name -> controller class.
VENDORS: dict[str, type[ArrayController]] = {
    "pure": PureFlashArray,
    "3par": HPE3PAR,
    "lightbits": LightbitsCluster,
}


def controller_for(vendor: str, **kwargs) -> ArrayController:
    """Build a controller by vendor name.

    Args:
        vendor: One of :data:`VENDORS`.
        **kwargs: Passed to the controller's constructor.

    Raises:
        ArrayError: If the vendor is unknown.
    """
    key = vendor.lower()
    if key not in VENDORS:
        raise ArrayError(f"unknown vendor {vendor!r}; expected one of {sorted(VENDORS)}")
    return VENDORS[key](**kwargs)
