"""Unit tests for device selection helpers."""

from __future__ import annotations

import pytest

from videoclean.exceptions import ConfigError
from videoclean.utils.device import describe_device, resolve_device


def test_auto_resolves_to_cpu() -> None:
    assert resolve_device("auto") == "cpu"
    assert resolve_device("CPU") == "cpu"


def test_explicit_devices() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_unknown_device() -> None:
    with pytest.raises(ConfigError, match="Unknown device"):
        resolve_device("tpu")


def test_describe_device() -> None:
    assert "CPU" in describe_device("cpu")
