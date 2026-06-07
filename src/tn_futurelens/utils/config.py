"""Config loading and run-provenance helpers.

Coding standard (briefing §13): every run saves its config, git commit hash,
seed, model name, dataset info, metrics JSON, and parameter count. These helpers
make that cheap and uniform.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit_hash(short: bool = True) -> str | None:
    """Return the current git commit hash, or None if not in a git repo."""
    try:
        args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        out = subprocess.check_output(args, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def git_is_dirty() -> bool | None:
    """Return True if the working tree has uncommitted changes (None if no git)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return len(out.decode().strip()) > 0
    except Exception:
        return None


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, e.g. '2026-06-07T01:50:41+00:00'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion of dataclasses/paths to JSON-serializable values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonify(asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_run_metadata(
    out_dir: str | Path,
    config: dict[str, Any] | Any,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a ``run_metadata.json`` capturing config + provenance + metrics.

    Returns the path to the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit_hash(),
        "git_dirty": git_is_dirty(),
        "config": _jsonify(config),
        "metrics": _jsonify(metrics or {}),
    }
    if extra:
        payload.update(_jsonify(extra))
    path = out_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
