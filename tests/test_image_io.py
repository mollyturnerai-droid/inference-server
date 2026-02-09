"""Tests for the image I/O module (SSRF protection)."""

import pytest

from app.models.image_io import _is_private_ip


def test_private_ip_localhost():
    assert _is_private_ip("127.0.0.1") is True


def test_private_ip_loopback_name():
    assert _is_private_ip("localhost") is True


def test_private_ip_rfc1918():
    assert _is_private_ip("10.0.0.1") is True
    assert _is_private_ip("192.168.1.1") is True


def test_public_ip():
    assert _is_private_ip("8.8.8.8") is False


def test_nonexistent_host():
    # Should return False (not crash) for unresolvable hostnames
    assert _is_private_ip("this-host-does-not-exist.invalid") is False
