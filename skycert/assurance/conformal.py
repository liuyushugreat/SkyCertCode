"""Split-conformal prediction for multiclass UAM risk reasoning.

Two nonconformity scores are supported:

* ``lac`` — Least Ambiguous Classifier  (``s = 1 - p_y``), the classic
  score of Sadinle et al. (2019). Gives small sets under heavy skew but
  under-covers rare classes.
* ``aps`` — Adaptive Prediction Sets (Romano et al., 2020). Gives
  class-conditionally stable coverage at the cost of slightly larger
  sets, which is what we want for safety-critical risk reasoning.

Given a calibration set ``(X_cal, y_cal)`` and miscoverage level
``alpha``, ``ConformalRiskSet`` computes the empirical
``(1 - alpha)(1 + 1/n)``-quantile of calibration scores and, at test
time, returns the *top-1 augmented* prediction set

    C(x) = { y : s(x, y) <= q_hat }  union  { argmax_k p_k(x) }.

The raw APS set can be empty when the top-ranked class alone already has
cumulative mass above ``q_hat``; the union with the top-1 label makes the
set non-empty by construction.  Because the augmented set is a superset
of the raw set, the marginal coverage guarantee is preserved (coverage
can only increase), at the cost of a slightly larger average set size.
Set ``include_top1=False`` to obtain the raw APS/LAC set.
"""

from __future__ import annotations

import numpy as np


class ConformalRiskSet:
    def __init__(
        self,
        alpha: float = 0.1,
        score: str = "aps",
        include_top1: bool = True,
    ) -> None:
        if score not in {"aps", "lac"}:
            raise ValueError(f"unknown score {score!r}")
        self.alpha = float(alpha)
        self.score = score
        self.include_top1 = bool(include_top1)
        self.q_hat: float | None = None
        self.num_classes: int | None = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n, k = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumulative = np.cumsum(sorted_probs, axis=1)
        rank_of_true = np.argmax(order == labels[:, None], axis=1)
        s = cumulative[np.arange(n), rank_of_true]
        return s

    @staticmethod
    def _aps_class_scores(probs: np.ndarray) -> np.ndarray:
        """Return an (n, k) matrix of APS scores for every candidate class."""
        n, k = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumulative = np.cumsum(sorted_probs, axis=1)
        out = np.empty_like(probs)
        out[np.arange(n)[:, None], order] = cumulative
        return out

    @staticmethod
    def _lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return 1.0 - probs[np.arange(probs.shape[0]), labels]

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> float:
        n, k = probs.shape
        self.num_classes = k
        if self.score == "aps":
            s = self._aps_scores(probs, labels)
        else:
            s = self._lac_scores(probs, labels)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = float(min(level, 1.0))
        self.q_hat = float(np.quantile(s, level, method="higher"))
        return self.q_hat

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """Return a boolean mask ``(n, k)`` with ``True`` on included labels."""
        if self.q_hat is None or self.num_classes is None:
            raise RuntimeError("ConformalRiskSet.calibrate must be called first")
        if self.score == "aps":
            class_scores = self._aps_class_scores(probs)
        else:
            class_scores = 1.0 - probs
        mask = class_scores <= self.q_hat
        if self.include_top1:
            # Top-1 augmentation: C(x) := C_raw(x) U {argmax p}. Makes the
            # set non-empty by construction and only enlarges it, so the
            # marginal coverage guarantee of the raw set is preserved.
            top1 = np.argmax(probs, axis=1)
            mask[np.arange(mask.shape[0]), top1] = True
        return mask

    def raw_empty_rate(self, probs: np.ndarray) -> float:
        """Fraction of inputs whose *raw* (un-augmented) set would be empty."""
        if self.q_hat is None:
            raise RuntimeError("ConformalRiskSet.calibrate must be called first")
        if self.score == "aps":
            class_scores = self._aps_class_scores(probs)
        else:
            class_scores = 1.0 - probs
        return float((~(class_scores <= self.q_hat)).all(axis=1).mean())

    def coverage(self, probs: np.ndarray, labels: np.ndarray) -> float:
        mask = self.predict_sets(probs)
        return float(mask[np.arange(labels.shape[0]), labels].mean())

    def average_set_size(self, probs: np.ndarray) -> float:
        mask = self.predict_sets(probs)
        return float(mask.sum(axis=1).mean())


def sequestered_coverage_check(
    probs_val: np.ndarray,
    labels_val: np.ndarray,
    conformal: "ConformalRiskSet",
    *,
    z: float = 2.33,
    critical_class: int | None = None,
) -> dict:
    """Post-hoc calibration-integrity test on a sequestered validation split.

    Under an intact calibration set, empirical coverage on an independent
    exchangeable split concentrates around ``1 - alpha``; a poisoned
    calibration set that shrinks ``q_hat`` shows up as coverage below
    ``1 - alpha - z * sqrt(alpha (1 - alpha) / n_val)`` (one-sided normal
    bound; ``z = 2.33`` is the 99% point).  The same test is applied to
    the class-conditional coverage of ``critical_class`` when given.
    """
    n = int(labels_val.shape[0])
    alpha = conformal.alpha
    mask = conformal.predict_sets(probs_val)
    cov = float(mask[np.arange(n), labels_val].mean())
    tol = z * float(np.sqrt(alpha * (1.0 - alpha) / max(n, 1)))
    flagged = cov < (1.0 - alpha - tol)
    out = {
        "n_val": n,
        "coverage": cov,
        "lower_bound": 1.0 - alpha - tol,
        "flagged": bool(flagged),
    }
    if critical_class is not None:
        sel = labels_val == critical_class
        if sel.any():
            n_c = int(sel.sum())
            cov_c = float(mask[np.arange(n)[sel], critical_class].mean())
            tol_c = z * float(np.sqrt(alpha * (1.0 - alpha) / n_c))
            out["critical_n"] = n_c
            out["critical_coverage"] = cov_c
            out["critical_lower_bound"] = 1.0 - alpha - tol_c
            out["critical_flagged"] = bool(cov_c < (1.0 - alpha - tol_c))
            out["flagged"] = bool(out["flagged"] or out["critical_flagged"])
    return out
