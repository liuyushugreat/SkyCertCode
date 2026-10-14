"""End-to-end SkyCert assurance pipeline.

The pipeline offers two entry points:

* :meth:`SkyCertPipeline.fit` — trains the neural scorer on ``X_train``
  and calibrates the conformal predictor on ``X_calib``.
* :meth:`SkyCertPipeline.step` — consumes a single operation request
  ``x``, produces an :class:`AssuranceDecision`, updates the martingale,
  and (optionally) writes an audit record.

``step`` is deliberately stateful in the martingale so it can be driven
online by the platform's control loop. Call :meth:`reset_monitor` between
independent runs (e.g. when the operator starts a new shift).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .assurance.audit import AuditLogger, sha256_of_arrays, sha256_of_json
from .assurance.conformal import ConformalRiskSet
from .assurance.martingale import MartingaleMonitor, SimpleJumperBetting
from .assurance.policy import AssuranceDecision, AssurancePolicy, DecisionKind
from .base.neuro_symbolic import NeuroSymbolicRiskReasoner
from .config import SkyCertConfig


@dataclass
class _CalibrationState:
    q_hat: float
    alpha: float
    score: str
    reference_scores: np.ndarray  # label-free nonconformity on the calib set
    calib_sha256: str = ""
    rules_sha256: str = ""


class SkyCertPipeline:
    def __init__(
        self,
        reasoner: NeuroSymbolicRiskReasoner,
        config: SkyCertConfig,
        audit_path: str | Path | None = None,
    ) -> None:
        self.reasoner = reasoner
        self.config = config
        self.num_classes = reasoner.symbolic.num_classes

        c = config.assurance
        self.conformal = ConformalRiskSet(
            alpha=c.conformal.alpha, score=c.conformal.score
        )
        self.monitor = MartingaleMonitor(
            threshold=c.martingale.threshold,
            betting=SimpleJumperBetting(epsilon=c.martingale.epsilon),
        )
        self.policy = AssurancePolicy(
            num_classes=self.num_classes,
            max_set_fraction=c.policy.max_set_fraction,
            escalate_on_martingale=c.policy.escalate_on_martingale,
        )
        self._calibrated: _CalibrationState | None = None
        # Feature statistics captured at calibration time. Used to compute a
        # robust OOD-style nonconformity that drives the martingale and
        # stays sensitive to covariate shift and feature manipulation even
        # when the neural scorer remains confidently wrong.
        self._feature_mean: np.ndarray | None = None
        self._feature_scale: np.ndarray | None = None

        self._logger: AuditLogger | None = (
            AuditLogger(audit_path) if audit_path is not None else None
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> None:
        probs = self.reasoner.predict_proba(X_calib)
        q_hat = self.conformal.calibrate(probs, y_calib)
        self._feature_mean = X_calib.mean(axis=0)
        self._feature_scale = X_calib.std(axis=0) + 1e-6
        # Warm-start the martingale with calibration scores so that test
        # p-values are computed against the combined calibration+stream
        # history. This is the conformal test-martingale construction of
        # Vovk, Nouretdinov and Gammerman (2003) and preserves the
        # anytime-valid exchangeability test.
        ref = np.array(
            [self._nonconformity(X_calib[i], probs[i]) for i in range(X_calib.shape[0])]
        )
        self.monitor.warm_start(ref)
        calib_sha = sha256_of_arrays(X_calib, y_calib)
        rules_sha = sha256_of_json(
            [r.__dict__ for r in self.reasoner.symbolic.rules]
        )
        self._calibrated = _CalibrationState(
            q_hat=q_hat,
            alpha=self.conformal.alpha,
            score=self.conformal.score,
            reference_scores=ref,
            calib_sha256=calib_sha,
            rules_sha256=rules_sha,
        )
        if self._logger is not None:
            self._logger.write_header(self.calibration_header())

    def calibration_header(self) -> dict:
        """Calibration-pinning metadata (first line of every audit log)."""
        if self._calibrated is None:
            raise RuntimeError("SkyCertPipeline.calibrate must be called first")
        c = self.config.assurance
        st = self._calibrated
        return {
            "q_hat": st.q_hat,
            "alpha": st.alpha,
            "conformal_score": st.score,
            "num_classes": self.num_classes,
            "n_calib": int(st.reference_scores.shape[0]),
            "calib_sha256": st.calib_sha256,
            "rules_sha256": st.rules_sha256,
            "martingale": {
                "epsilon": c.martingale.epsilon,
                "threshold": c.martingale.threshold,
                "lambda_drift": float(
                    getattr(c.martingale, "lambda_drift", 1.0)
                ),
                "drift_mode": str(getattr(c.martingale, "drift_mode", "rms")),
                "hinge_z": float(getattr(c.martingale, "hinge_z", 2.0)),
                "reset_every": int(getattr(c.martingale, "reset_every", 0) or 0),
                "drift_features": (
                    [int(i) for i in c.martingale.drift_features]
                    if getattr(c.martingale, "drift_features", None) else None
                ),
            },
            "policy": {
                "max_set_fraction": c.policy.max_set_fraction,
                "escalate_on_martingale": c.policy.escalate_on_martingale,
            },
        }

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------
    def reset_monitor(self) -> None:
        self.monitor.reset()

    def _nonconformity(
        self, x: np.ndarray, probs_row: np.ndarray
    ) -> float:
        """Hybrid label-free nonconformity driving the martingale.

        Combines two complementary signals:

        * *Confidence slack*  ``1 - max(p)``  — grows whenever the
          reasoner becomes less certain, e.g. under rule poisoning or
          mild feature perturbation.
        * *Input drift*       standardised L2 distance from the
          calibration feature mean — grows under covariate shift even
          when the reasoner remains confidently wrong.

        The two are combined additively.  Under IID conditions both
        terms are approximately identically distributed between the
        calibration and test streams, so the conformal p-values used
        by the martingale remain uniform (no false alarms in
        expectation).
        """
        conf = float(1.0 - probs_row.max())
        if self._feature_mean is None or self._feature_scale is None:
            return conf
        mc = self.config.assurance.martingale
        z = (x - self._feature_mean) / self._feature_scale
        feats = getattr(mc, "drift_features", None)
        if feats:
            # Restrict the drift term to the monitored (attack-surface)
            # features; slowly varying exogenous context that is not an
            # attack surface can be excluded to avoid harmless-shift alarms.
            z = z[np.asarray(feats, dtype=int)]
        mode = str(getattr(mc, "drift_mode", "rms"))
        if mode == "hinge":
            # Tolerance band: only deviations beyond +-hinge_z calibration
            # standard deviations count as drift. Still a deterministic
            # function of (x, p), hence a valid nonconformity score.
            z = np.maximum(np.abs(z) - float(getattr(mc, "hinge_z", 2.0)), 0.0)
        drift = float(np.sqrt(np.mean(z * z)))
        lam = float(getattr(mc, "lambda_drift", 1.0))
        return conf + lam * drift

    def step(
        self,
        x: np.ndarray,
        sample_id: int | str | None = None,
    ) -> AssuranceDecision:
        if self._calibrated is None:
            raise RuntimeError("SkyCertPipeline.calibrate must be called first")
        x2 = x.reshape(1, -1)
        probs = self.reasoner.predict_proba(x2)[0]
        set_mask = self.conformal.predict_sets(probs.reshape(1, -1))[0]
        score = self._nonconformity(x, probs)
        reset_every = int(getattr(self.config.assurance.martingale, "reset_every", 0) or 0)
        if reset_every > 0 and self.monitor.steps_in_window >= reset_every:
            self.monitor.restart_window()
        update = self.monitor.update(score)
        decision = self.policy.decide(
            probs_row=probs,
            set_mask=set_mask,
            martingale_value=update["martingale"],
            martingale_alert=update["alert"],
        )
        if self._logger is not None:
            trace = self.reasoner.symbolic.trace(x)
            self._logger.log(
                decision,
                sample_id=sample_id if sample_id is not None else "n/a",
                probs=probs.tolist(),
                rule_trace=trace,
                extra={
                    "p_value": update["p_value"],
                    "t": update["t"],
                    "nonconformity": float(score),
                    "q_hat": self._calibrated.q_hat,
                },
            )
        return decision

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------
    def predict_sets(self, X: np.ndarray) -> np.ndarray:
        if self._calibrated is None:
            raise RuntimeError("SkyCertPipeline.calibrate must be called first")
        probs = self.reasoner.predict_proba(X)
        return self.conformal.predict_sets(probs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.reasoner.predict_proba(X)

    def run_batch(
        self,
        X: np.ndarray,
        sample_ids: Optional[list[int | str]] = None,
    ) -> list[AssuranceDecision]:
        if sample_ids is None:
            sample_ids = list(range(X.shape[0]))
        decisions: list[AssuranceDecision] = []
        for i, sid in enumerate(sample_ids):
            decisions.append(self.step(X[i], sample_id=sid))
        return decisions

    def close(self) -> None:
        if self._logger is not None:
            self._logger.close()


__all__ = ["SkyCertPipeline", "AssuranceDecision", "DecisionKind"]
