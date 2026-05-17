"""
Central runtime settings for the public template-driven OCR service.

Values can come from:
1. Environment variables
2. config/settings.yml
3. config/settings.example.yml
4. Code defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.example.yml"
DEFAULT_PROMPTS_PATH = CONFIG_DIR / "prompts.yml"


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_settings() -> dict[str, Any]:
    configured_path = os.getenv("APP_CONFIG_PATH", "").strip()
    settings_path = Path(configured_path) if configured_path else (CONFIG_DIR / "settings.yml")

    base = _load_yaml_file(DEFAULT_SETTINGS_PATH)
    override = _load_yaml_file(settings_path)
    return _deep_merge(base, override)


SETTINGS = _load_settings()


def get_nested(path: Iterable[str], default: Any = None) -> Any:
    current: Any = SETTINGS
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def get_env_or_config(env_name: str, path: Iterable[str], default: Any = None) -> Any:
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return env_value
    return get_nested(path, default)


def get_int_env_or_config(env_name: str, path: Iterable[str], default: int) -> int:
    value = get_env_or_config(env_name, path, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float_env_or_config(env_name: str, path: Iterable[str], default: float) -> float:
    value = get_env_or_config(env_name, path, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_settings_path() -> Path:
    configured_path = os.getenv("APP_CONFIG_PATH", "").strip()
    return Path(configured_path) if configured_path else (CONFIG_DIR / "settings.yml")


def get_prompts_path() -> Path:
    configured_path = os.getenv("PROMPTS_CONFIG_PATH", "").strip()
    return Path(configured_path) if configured_path else DEFAULT_PROMPTS_PATH
