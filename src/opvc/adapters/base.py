from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

@dataclass
class DatasetConfig:
    name: str
    format: str
    path: str
    columns: Dict[str, str]
    timestamp: Dict[str, Any]
    views: List[Dict[str, str]]
    filters: Dict[str, Any]

class DatasetAdapter:
    """Convert dataset (csv/jsonl/...) into OPVC eventlist JSONL records:
    {
      meta: {...},
      t0: float,
      delta: float,
      T: int,
      E: [ [event,...], [event,...], ... ]   # V views
    }
    Each event minimally has {"ts": float, "parse_ok": True, ...}
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

    def iter_eventlist_records(
        self,
        delta: float,
        T: int,
        t0: Optional[float] = None,
        max_rows: Optional[int] = None,
    ):
        raise NotImplementedError
