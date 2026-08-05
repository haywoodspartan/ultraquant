"""Gated internet access.

Network access is off by default and must be switched on explicitly.  Only URLs
the local user supplies are ever fetched — nothing here crawls, follows links, or
schedules its own requests.  Responses are size-capped and reduced to plain text.

Crucially, this module has **no path into memory**.  Fetched text is returned to
the caller and nothing else; turning a web page into stored knowledge is the job
of :mod:`ultraquant.interpreter.stash`, which quarantines and analyses claims
first.  Nothing off the network becomes a fact by being downloaded.
"""

from __future__ import annotations

import datetime
import ssl
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urlparse

__all__ = ["WebDisabled", "WebAccess", "build_ssl_context"]

_CACHED_CONTEXT: ssl.SSLContext | None = None


class WebDisabled(RuntimeError):
    """Raised when a fetch is attempted while offline."""


def _not_expired(pem: str, now: datetime.datetime) -> bool:
    """Whether a PEM certificate is still within its validity period."""
    try:
        probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        probe.load_verify_locations(cadata=pem)
        info = probe.get_ca_certs()
    except (ssl.SSLError, ValueError):
        return False
    if not info:
        return False
    try:
        expires = datetime.datetime.strptime(
            info[0]["notAfter"], "%b %d %H:%M:%S %Y %Z"
        ).replace(tzinfo=datetime.timezone.utc)
    except (KeyError, ValueError):
        return False
    return expires > now


def build_ssl_context(ca_file: str | None = None) -> ssl.SSLContext:
    """Build a verifying TLS context that ignores expired trust anchors.

    Windows machines accumulate stale roots, and OpenSSL matches a trust anchor
    by subject name: if the store holds an *expired* copy of a root the server
    chains to, verification fails with "certificate has expired" even though the
    server's own chain is perfectly valid.  This is exactly what happens with
    Let's Encrypt's ``ISRG Root X2`` — the old cross-signed copy expired in
    September 2025, and any store still holding it breaks half the web.

    So the roots are loaded individually and the expired ones are left out.
    Verification stays fully on; an expired root is not something that should be
    trusted anyway, and dropping it lets OpenSSL build the valid path instead.

    Args:
        ca_file: Optional PEM bundle to trust *instead* of the system store —
            the correct way to handle a corporate TLS-inspecting proxy.

    Returns:
        A context with hostname checking and certificate verification enabled.
    """
    if ca_file:
        context = ssl.create_default_context(cafile=ca_file)
        return context

    global _CACHED_CONTEXT
    if _CACHED_CONTEXT is not None:
        return _CACHED_CONTEXT

    context = ssl.create_default_context()
    if not hasattr(ssl, "enum_certificates"):  # not Windows: the default is fine
        _CACHED_CONTEXT = context
        return context

    now = datetime.datetime.now(datetime.timezone.utc)
    curated = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    curated.verify_mode = ssl.CERT_REQUIRED
    curated.check_hostname = True
    loaded = 0
    try:
        for store in ("ROOT", "CA"):
            for der, encoding, _trust in ssl.enum_certificates(store):
                if encoding != "x509_asn":
                    continue
                try:
                    pem = ssl.DER_cert_to_PEM_cert(der)
                except ValueError:
                    continue
                if not _not_expired(pem, now):
                    continue
                try:
                    curated.load_verify_locations(cadata=pem)
                    loaded += 1
                except ssl.SSLError:
                    continue
    except OSError:
        _CACHED_CONTEXT = context
        return context

    # Never hand back an empty trust store -- that would fail every request.
    _CACHED_CONTEXT = curated if loaded >= 10 else context
    return _CACHED_CONTEXT


class _TextExtractor(HTMLParser):
    """Collect visible text and the document title, dropping script/style."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._chunks: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = max(0, self._skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title += data.strip()
        else:
            self._chunks.append(data)

    def text(self) -> str:
        """Collapsed visible text of the document."""
        joined = "".join(self._chunks)
        lines = [" ".join(line.split()) for line in joined.splitlines()]
        return "\n".join(line for line in lines if line).strip()


class WebAccess:
    """Fetch web documents on demand, only when explicitly enabled."""

    def __init__(
        self,
        online: bool = False,
        timeout: float = 10.0,
        max_bytes: int = 262_144,
        user_agent: str = "UltraQuant/0.1 (local research tool; +stdlib urllib)",
        ca_file: str | None = None,
    ) -> None:
        """Configure the gate.

        Args:
            online: Whether fetching is permitted right now.
            timeout: Per-request socket timeout in seconds.
            max_bytes: Hard cap on bytes read from any single response.
            user_agent: ``User-Agent`` header sent with requests. A descriptive
                one matters: several large sites (Wikipedia among them) reject
                the bare ``Python-urllib/x.y`` default with 403.
            ca_file: Optional PEM bundle to trust instead of the system store.
        """
        self.online = bool(online)
        self.timeout = float(timeout)
        self.max_bytes = int(max_bytes)
        self.user_agent = user_agent
        self.ca_file = ca_file

    def set_online(self, flag: bool) -> None:
        """Enable or disable network access."""
        self.online = bool(flag)

    def fetch(self, url: str) -> dict:
        """Retrieve one URL and reduce it to text.

        Args:
            url: An ``http://`` or ``https://`` URL supplied by the user.

        Returns:
            ``{"url", "netloc", "status", "title", "text"}``.

        Raises:
            WebDisabled: If access is currently switched off.
            ValueError: If the URL scheme is not http/https.
            RuntimeError: If the request fails.
        """
        if not self.online:
            raise WebDisabled("web access is off — enable it with ':online on'")
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme: {parsed.scheme or '(none)'}")

        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        context = build_ssl_context(self.ca_file)
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout, context=context
            ) as response:
                status = getattr(response, "status", 200)
                content_type = response.headers.get("Content-Type", "")
                charset = response.headers.get_content_charset() or "utf-8"
                raw = response.read(self.max_bytes)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"fetch failed: HTTP {exc.code} {exc.reason}") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, ssl.SSLCertVerificationError):
                raise RuntimeError(
                    f"fetch failed: the TLS certificate could not be verified "
                    f"({reason.verify_message or reason}). If this machine sits "
                    f"behind a TLS-inspecting proxy or VPN, export its root "
                    f"certificate and pass it as ca_file."
                ) from exc
            raise RuntimeError(f"fetch failed: {exc}") from exc

        body = raw.decode(charset, errors="replace")
        if "html" in content_type.lower() or body.lstrip()[:1] == "<":
            parser = _TextExtractor()
            parser.feed(body)
            title, text = parser.title, parser.text()
        else:
            title, text = "", body.strip()

        return {
            "url": url,
            "netloc": parsed.netloc,
            "status": status,
            "title": title,
            "text": text,
        }
