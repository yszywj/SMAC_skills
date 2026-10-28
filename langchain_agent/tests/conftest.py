"""Shared fixtures: config + mock backend."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from smac_langchain.config import load_config, resolve_path  # noqa: E402


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def skills_dir():
    path = resolve_path("skills/current")
    assert path.is_dir(), f"missing skills package: {path}"
    return path


@pytest.fixture
def mock_backend(skills_dir):
    from smac_langchain.backends.mock import MockBackend
    backend = MockBackend(skills_dir)
    yield backend
    backend.close()
