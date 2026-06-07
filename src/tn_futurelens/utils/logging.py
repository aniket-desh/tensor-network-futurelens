"""Lightweight logging + metric persistence.

No wandb in this environment (key unset); we log to stdout, to TensorBoard if
available, and ALWAYS to disk as JSON/CSV so every result is reproducible from
files alone (briefing §0, §13).
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch.nn as nn

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "tn_futurelens", level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters of a module (for parameter-matched comparisons, §9.5)."""
    return sum(
        p.numel()
        for p in module.parameters()
        if (p.requires_grad or not trainable_only)
    )


class CSVMetricWriter:
    """Append rows of metrics to a CSV (header written from the first row's keys)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def append(self, row: dict[str, Any]) -> None:
        write_header = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in self._fieldnames})


def save_json(path: str | Path, obj: Any) -> Path:
    """Dump ``obj`` to JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
