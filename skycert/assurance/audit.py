"""Audit artifact logger.

Each call to :class:`SkyCertPipeline` is expected to produce exactly
one audit record, committed to a JSON-Lines file at the configured
path.  A record captures *enough* information for an offline reviewer
to reconstruct:

* the firing rules (if the symbolic engine is available),
* the conformal prediction set and set size,
* the current martingale value and alert flag,
* the final decision and the reasons that led to it.

The very first line of every log is a *header* record that pins the
calibration state the stream was evaluated against: the conformal
quantile ``q_hat`` and ``alpha``, the martingale betting parameters, the
policy knobs, and SHA-256 digests of the calibration set and of the
materialized rule set.  Together with the per-record ``p_value`` and
probability vector this is sufficient for :mod:`scripts.replay_audit`
to recompute every set, every martingale value and every decision
offline and to flag any divergence from what was logged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import safe_json
from .policy import AssuranceDecision


def sha256_of_arrays(*arrays: np.ndarray) -> str:
    """Deterministic digest of one or more NumPy arrays (dtype+shape+bytes)."""
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def sha256_of_json(obj: Any) -> str:
    """Digest of a JSON-serialisable object (canonical key ordering)."""
    payload = json.dumps(safe_json(obj), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._closed = False
        self._counter = 0
        self._header_written = False

    def write_header(self, header: dict[str, Any]) -> None:
        """Write the calibration-pinning header as the first line."""
        if self._header_written:
            return
        record = {"record_type": "header", **header}
        self._fh.write(json.dumps(safe_json(record), ensure_ascii=False) + "\n")
        self._fh.flush()
        self._header_written = True

    def log(
        self,
        decision: AssuranceDecision,
        *,
        sample_id: int | str,
        probs: list[float] | None = None,
        rule_trace: list[dict] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "record_type": "decision",
            "sample_id": sample_id,
            "index": self._counter,
            "decision": asdict(decision),
            "probs": probs,
            "rule_trace": rule_trace or [],
            "extra": extra or {},
        }
        self._fh.write(json.dumps(safe_json(record), ensure_ascii=False) + "\n")
        self._counter += 1

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()
