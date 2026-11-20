"""TLS certificate pinning for edge-to-cloud connections (Doc 14 §8.6, F-SEC-07).

Validates the cloud API's TLS certificate SHA-256 fingerprint against
a pinned value injected at provisioning. Connection is refused if the
pin does not match, preventing MITM attacks.
"""

from __future__ import annotations

import hashlib
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class CertificatePinningError(Exception):
    """Raised when TLS certificate fingerprint does not match the pin."""


def verify_cert_pin(
    cert_der: bytes,
    expected_sha256: str,
) -> bool:
    """Verify a DER-encoded certificate against the pinned SHA-256 hash.

    Args:
        cert_der: DER-encoded certificate bytes.
        expected_sha256: Expected SHA-256 hex digest (64 characters).

    Returns:
        True if the fingerprint matches.
    """
    actual = hashlib.sha256(cert_der).hexdigest()
    return actual == expected_sha256.lower()


class PinningTransport(httpx.AsyncHTTPTransport):
    """Custom httpx transport that enforces TLS certificate pinning.

    Usage:
        transport = PinningTransport(
            pin_sha256="abcd1234...",
            verify="/path/to/ca.pem",
        )
        client = httpx.AsyncClient(transport=transport)
    """

    def __init__(
        self,
        pin_sha256: str,
        verify: str | bool = True,
        cert: tuple[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._pin_sha256 = pin_sha256.lower()
        super().__init__(verify=verify, cert=cert, **kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Override to validate certificate pin after TLS handshake."""
        response = await super().handle_async_request(request)

        # Extract peer certificate from the connection
        stream = response.stream
        if hasattr(stream, "_stream") and hasattr(stream._stream, "_ssl_object"):
            ssl_obj = stream._stream._ssl_object
            cert_der = ssl_obj.getpeercert(binary_form=True)
            if cert_der:
                if not verify_cert_pin(cert_der, self._pin_sha256):
                    actual = hashlib.sha256(cert_der).hexdigest()
                    logger.error(
                        "cert_pin_mismatch",
                        expected=self._pin_sha256[:16] + "...",
                        actual=actual[:16] + "...",
                    )
                    raise CertificatePinningError(
                        f"Certificate pin mismatch: "
                        f"expected {self._pin_sha256[:16]}..., "
                        f"got {actual[:16]}..."
                    )

        return response


def create_pinned_client(
    pin_sha256: str,
    cert_path: str,
    key_path: str,
    ca_cert_path: str,
    timeout: float = 30.0,
) -> httpx.AsyncClient:
    """Create an httpx client with certificate pinning and mTLS.

    Args:
        pin_sha256: Expected SHA-256 of the server's TLS certificate.
        cert_path: Path to device client certificate.
        key_path: Path to device private key.
        ca_cert_path: Path to CA certificate for server verification.
        timeout: Request timeout in seconds.

    Returns:
        Configured httpx.AsyncClient with pinning transport.
    """
    if pin_sha256:
        transport = PinningTransport(
            pin_sha256=pin_sha256,
            verify=ca_cert_path,
            cert=(cert_path, key_path),
        )
        return httpx.AsyncClient(transport=transport, timeout=timeout)
    else:
        # No pin configured — use standard mTLS
        logger.warning(
            "cert_pin_not_configured",
            msg="Certificate pinning disabled — using standard TLS verification",
        )
        return httpx.AsyncClient(
            cert=(cert_path, key_path),
            verify=ca_cert_path,
            timeout=timeout,
        )
