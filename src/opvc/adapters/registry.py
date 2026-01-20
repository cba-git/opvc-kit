from __future__ import annotations
from typing import Any, Dict

from .base import DatasetConfig
from .csv_adapter import CSVDatasetAdapter

def get_adapter(cfg: DatasetConfig):
    fmt = (cfg.format or "").lower()
    if fmt == "csv":
        return CSVDatasetAdapter(cfg)
    raise ValueError(f"Unsupported dataset format: {cfg.format}")
