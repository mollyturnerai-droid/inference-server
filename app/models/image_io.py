from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

import requests
from PIL import Image


def _is_private_ip(hostname: str) -> bool:
    """Check whether a hostname resolves to a private/link-local address."""
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            addr = ipaddress.ip_address(sockaddr[0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return True
    except (socket.gaierror, ValueError):
        return False
    return False


def load_image_rgb(
    image_input: str,
    *,
    storage_path: str,
    api_base_url: str | None = None,
    timeout_s: float = 30.0,
    allow_private: bool = False,
) -> Image.Image:
    """Load an image from local storage, file://, /v1/files/*, or http(s) and convert to RGB."""
    if not image_input:
        raise ValueError("Missing image input")

    parsed = urlparse(image_input)

    # Bare path: try storage-relative first, then absolute.
    if not parsed.scheme:
        storage_candidate = os.path.join(storage_path, image_input)
        if os.path.exists(storage_candidate):
            return Image.open(storage_candidate).convert("RGB")
        if os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")
        # Fall back to storage-relative (will raise a helpful OS error).
        return Image.open(storage_candidate).convert("RGB")

    if parsed.scheme == "file":
        return Image.open(parsed.path).convert("RGB")

    # API file paths.
    if image_input.startswith("/v1/files/"):
        rel = image_input[len("/v1/files/") :]
        return Image.open(os.path.join(storage_path, rel)).convert("RGB")

    api_base = (api_base_url or "").rstrip("/")
    if api_base:
        api_files_prefix = f"{api_base}/v1/files/"
        if image_input.startswith(api_files_prefix):
            rel = image_input[len(api_files_prefix) :]
            return Image.open(os.path.join(storage_path, rel)).convert("RGB")

    if parsed.scheme in ("http", "https"):
        # SSRF protection: block private/link-local IPs unless explicitly allowed.
        if not allow_private and parsed.hostname and _is_private_ip(parsed.hostname):
            raise ValueError(
                f"Fetching from private/internal addresses is blocked: {parsed.hostname}"
            )
        resp = requests.get(image_input, stream=True, timeout=timeout_s)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")

    raise ValueError(f"Unsupported image input: {image_input}")
