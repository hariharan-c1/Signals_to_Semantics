"""Stable identifiers used across files, tables, and pipeline stages."""

from __future__ import annotations

import hashlib
import re


def canonical_window_key(
    log_id: str,
    t_start_s: float,
    t_end_s: float,
    precision: int = 6,
) -> str:
    """Create the logical database key for one event window."""
    if precision < 0:
        raise ValueError("precision must be non-negative")
    fmt = f"{{:.{precision}f}}"
    return f"{str(log_id).strip()}|{fmt.format(t_start_s)}|{fmt.format(t_end_s)}"


def safe_window_stem(window_key: str) -> str:
    """Create a deterministic filename that works on Linux, macOS, and Windows."""
    raw = str(window_key)
    log_id = raw.split("|", 1)[0]
    log_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", log_id).strip("-._")
    log_slug = (log_slug or "window")[:64]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{log_slug}-{digest}"
