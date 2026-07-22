import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILENAME = "config.yaml"
_CONFIG_EXAMPLE_FILENAME = "config.example.yaml"

_cached_config = None


def _find_config_file():
    """Return the path to config.yaml if it exists, else config.example.yaml."""
    user_config = _PROJECT_ROOT / _CONFIG_FILENAME
    if user_config.exists():
        return user_config
    return _PROJECT_ROOT / _CONFIG_EXAMPLE_FILENAME


def load_config(force_reload=False):
    """Load configuration from config.yaml (or config.example.yaml as fallback).

    Results are cached; pass force_reload=True to re-read from disk.

    Returns a dict with the full configuration tree.
    """
    global _cached_config
    if _cached_config is not None and not force_reload:
        return _cached_config

    config_path = _find_config_file()

    if yaml is None:
        raise ImportError(
            "PyYAML is required for configuration. "
            "Install it with: pip install pyyaml"
        )

    with open(config_path, "r") as f:
        _cached_config = yaml.safe_load(f)

    return _cached_config


def get_trading_fee_percent():
    """Return the configured trading fee percentage."""
    return load_config()["backtest"]["trading_fee_percent"]


def get_data_dir():
    """Return the data directory path."""
    return load_config()["paths"]["data_dir"]


def get_results_dir():
    """Return the results directory path."""
    return load_config()["paths"]["results_dir"]


def get_binance_api_url():
    """Return the Binance API URL."""
    return load_config()["data"]["binance_api_url"]


def get_interval():
    """Return the Binance API interval."""
    return load_config()["data"]["interval"]


def get_api_limit():
    """Return the Binance API limit."""
    return load_config()["data"]["api_limit"]
