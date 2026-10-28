"""Configuration loading for smac-langchain.

Shallow per-section merge of default.toml with an optional local.toml.
All paths are pathlib.Path resolved against the SMAC_agent workspace root,
never against the process CWD.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

PKG_ROOT = Path(__file__).resolve().parents[2]      # langchain_agent/
WORKSPACE_ROOT = PKG_ROOT.parent                    # SMAC_agent/


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | os.PathLike | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else Path(
        os.environ.get("SMAC_LANGCHAIN_CONFIG", PKG_ROOT / "configs" / "default.toml"))
    data: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    local = path.parent / "local.toml"
    if local.is_file():
        with local.open("rb") as handle:
            data = _merge(data, tomllib.load(handle))
    data["_config_path"] = str(path)
    data["_pkg_root"] = str(PKG_ROOT)
    return data


def resolve_path(value: str | os.PathLike) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()
