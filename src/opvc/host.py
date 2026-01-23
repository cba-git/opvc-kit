"""opvc.host

Host/node inference helpers.

Several parts of this repository need a stable "node"/"host" identifier:

- Step0 / dataset adapters may want to segment samples by host.
- Step1/Step3 artifacts are commonly aligned with label files using node/sample_id.

This module centralizes the small, auditable heuristics that were previously
duplicated across scripts.

Design goals
-----------
- **No heavy deps** (stdlib only).
- **Deterministic** output for a given input.
- **Best-effort** heuristics: if we cannot infer anything, return a configurable
  default ("unknown", "local", ...).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional


IPV4_RE = re.compile(r"\b(?:(?:\d{1,3}\.){3}\d{1,3})\b")


def host_from_node_str(x: str) -> Optional[str]:
    """Try to extract a host identifier from a node string.

    Supported patterns (best-effort):
    - `net:10.20.2.66:5010->...`  -> `10.20.2.66`
    - `net:2001:db8::1:1234->...` -> `2001` (fallback: token after `net:`)
    - `subj:WORKSTATION42|...`    -> `WORKSTATION42`

    Returns None if no pattern matches.
    """
    if not isinstance(x, str) or not x:
        return None

    # net:IP:port->...
    if x.startswith("net:"):
        m = re.search(r"^net:([0-9]{1,3}(?:\.[0-9]{1,3}){3})\b", x)
        if m:
            return m.group(1)
        # IPv6-ish fallback: take the first token after net:
        m = re.search(r"^net:([^:]+)", x)
        if m:
            return m.group(1)

    # subj:Name|...
    if x.startswith("subj:"):
        v = x[len("subj:") :]
        v = v.split("|", 1)[0].strip()
        return v or None

    return None


def _iter_strings(obj: Any) -> Iterable[str]:
    """Yield all strings in a nested container (dict/list/tuple)."""
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _iter_strings(v)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)
        return


def infer_host_from_any(obj: Any, default: str = "unknown") -> str:
    """Infer a host identifier from an arbitrary nested structure.

    Strategy:
    1) Count occurrences of `host_from_node_str` candidates from all strings.
    2) Count occurrences of IPv4 substrings.
    3) Return the most common candidate. If ties, choose lexicographically.

    If nothing is found, return `default`.
    """
    default = str(default)
    counter: Dict[str, int] = {}

    def add(tok: str) -> None:
        if not tok:
            return
        counter[tok] = counter.get(tok, 0) + 1

    for s in _iter_strings(obj):
        h = host_from_node_str(s)
        if h:
            add(h)
        for ip in IPV4_RE.findall(s):
            add(ip)

    if not counter:
        return default

    # Deterministic: (count desc, token asc)
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[0][0]


def infer_ipv4_from_any(obj: Any, default: str = "unknown") -> str:
    """Infer the most common IPv4 address from a nested structure.

    This matches the lightweight heuristic used in some data-fixing scripts:
    count all IPv4 substrings in all strings, return the most frequent.
    If nothing is found, return `default`.
    """
    default = str(default)
    c = Counter()
    for s in _iter_strings(obj):
        for ip in IPV4_RE.findall(s):
            c[ip] += 1
    if not c:
        return default
    items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[0][0]


def infer_host_from_eventlist_E(E: Any, default: str = "unknown") -> str:
    """Infer host from Step0 eventlist field `E`.

    `E` is typically: list[view] where each view is list[edge/event].
    """
    return infer_host_from_any(E, default=default)


def infer_host_from_eventlist_record(record: Mapping[str, Any], default: str = "unknown") -> str:
    """Infer host from a Step0 eventlist record dict."""
    E = record.get("E")
    return infer_host_from_eventlist_E(E, default=default)
