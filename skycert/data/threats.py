"""Threat injection primitives.

These functions implement the adversary model described in Section 2 of the
paper: the attacker can either corrupt the symbolic rule base (``rule_flip``
or ``rule_inject``), perturb the input features inside a small L-inf ball
(``feature_attack``), or shift the covariate distribution at test time
(``covariate_shift``).  All injections return fresh data / rule structures
and never mutate their arguments.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Sequence

import numpy as np

from ..config import SymbolicRule


def corrupt_rules(
    rules: Sequence[SymbolicRule],
    strength: float,
    rng: np.random.Generator,
) -> list[SymbolicRule]:
    """Flip / perturb rule deltas with probability ~ ``strength``."""

    corrupted: list[SymbolicRule] = []
    for r in rules:
        if rng.random() < strength:
            delta = [-d + 0.1 * rng.standard_normal() for d in r.delta]
            corrupted.append(replace(r, delta=delta))
        else:
            corrupted.append(copy.deepcopy(r))
    return corrupted


def rank_rules_by_critical_impact(
    rules: Sequence[SymbolicRule],
    reasoner_factory,
    X_ref: np.ndarray,
    critical_class: int = 3,
) -> list[dict]:
    """Rank rules by how strongly flipping each one alone suppresses
    CRITICAL-class decisions on a reference sample.

    ``reasoner_factory(rules) -> object with predict(X)`` builds a reasoner
    around an arbitrary rule set.  The impact of rule ``i`` is the net
    number of reference operations whose arg-max moves *away from*
    CRITICAL when only ``delta_i`` is negated (an attacker wants fewer
    CRITICAL labels), normalised by ``len(X_ref)``.  Ties are broken by the
    magnitude of the rule's CRITICAL-class delta.
    """
    base_pred = reasoner_factory(list(rules)).predict(X_ref)
    base_crit = base_pred == critical_class
    ranked: list[dict] = []
    for i, r in enumerate(rules):
        flipped = [copy.deepcopy(rr) for rr in rules]
        flipped[i] = replace(r, delta=[-d for d in r.delta])
        pred = reasoner_factory(flipped).predict(X_ref)
        lost = int(np.sum(base_crit & (pred != critical_class)))
        gained = int(np.sum(~base_crit & (pred == critical_class)))
        ranked.append({
            "rule_id": i,
            "feature": r.feature,
            "impact": (lost - gained) / max(len(X_ref), 1),
            "critical_delta": float(r.delta[critical_class]),
        })
    ranked.sort(key=lambda d: (-d["impact"], -abs(d["critical_delta"])))
    return ranked


def corrupt_rules_targeted(
    rules: Sequence[SymbolicRule],
    k: int,
    ranking: Sequence[dict],
) -> list[SymbolicRule]:
    """Negate exactly the ``k`` highest-impact rule deltas (targeted T1).

    Unlike :func:`corrupt_rules`, which flips each rule independently with
    probability ``strength`` (expected footprint ``strength * m``), this is
    a minimal-footprint attacker who chooses *which* rules to invert.
    """
    victims = {int(d["rule_id"]) for d in list(ranking)[: int(k)]}
    out: list[SymbolicRule] = []
    for i, r in enumerate(rules):
        if i in victims:
            out.append(replace(r, delta=[-d for d in r.delta]))
        else:
            out.append(copy.deepcopy(r))
    return out


def inject_rule_noise(
    rules: Sequence[SymbolicRule],
    strength: float,
    num_features: int,
    num_classes: int,
    rng: np.random.Generator,
) -> list[SymbolicRule]:
    """Add spurious rules that fire on unrelated features."""

    poisoned = [copy.deepcopy(r) for r in rules]
    num_new = max(1, int(round(strength * len(rules))))
    for _ in range(num_new):
        feature = int(rng.integers(0, num_features))
        op = str(rng.choice([">", "<"]))
        thr = float(rng.uniform(0.2, 0.8))
        delta = (rng.standard_normal(num_classes) * 0.4).tolist()
        poisoned.append(
            SymbolicRule(feature=feature, op=op, thr=thr, delta=delta)
        )
    return poisoned


def perturb_features(
    X: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Untargeted L-inf bounded perturbation used for sensor-spoof simulation."""
    noise = rng.uniform(-strength, strength, size=X.shape)
    return np.clip(X + noise, 0.0, 1.0)


def adaptive_feature_attack(
    X: np.ndarray,
    mode: str,
    strength: float,
    rng: np.random.Generator,
    *,
    y_true: np.ndarray | None = None,
    critical_class: int = 3,
    duty_cycle: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """SkyCert-aware (adaptive) variants of the T3 feature attack.

    Returns ``(X_adv, attacked_mask)``.  Modes:

    * ``persistent_sub`` -- constant radius ``strength`` (meant to be used
      below the martingale's empirical sensitivity band, e.g. 0.08).
    * ``ramp`` -- radius grows linearly from 0 to ``strength`` across the
      attacked window, so the stream drifts slowly instead of jumping.
    * ``intermittent`` -- radius ``strength`` applied only to a random
      ``duty_cycle`` fraction of operations, interleaved with clean ones.
    * ``critical_directed`` -- perturbs *only* truly CRITICAL operations
      (requires ``y_true``; a strong attacker with ground-truth knowledge)
      and pushes the five safety features toward the benign regime
      (lower weather/density/congestion/noise, higher battery) by exactly
      ``strength``; all other operations are left untouched so the
      population-level input distribution barely moves.
    """
    X_adv = X.copy()
    n, d = X.shape
    if mode == "persistent_sub":
        noise = rng.uniform(-strength, strength, size=X.shape)
        X_adv = X + noise
        mask = np.ones(n, dtype=bool)
    elif mode == "ramp":
        radii = np.linspace(0.0, strength, n)[:, None]
        noise = rng.uniform(-1.0, 1.0, size=X.shape) * radii
        X_adv = X + noise
        mask = np.ones(n, dtype=bool)
    elif mode == "intermittent":
        mask = rng.random(n) < duty_cycle
        noise = rng.uniform(-strength, strength, size=X.shape)
        X_adv[mask] = X[mask] + noise[mask]
    elif mode == "critical_directed":
        if y_true is None:
            raise ValueError("critical_directed requires y_true")
        mask = y_true == critical_class
        direction = np.zeros(d)
        direction[:5] = np.array([-1.0, -1.0, -1.0, +1.0, -1.0])
        X_adv[mask] = X[mask] + strength * direction[None, :]
    else:
        raise ValueError(f"unknown adaptive mode {mode!r}")
    return np.clip(X_adv, 0.0, 1.0), mask


def poison_calibration_labels(
    y_calib: np.ndarray,
    preds_calib: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration-set *agreement laundering*.

    A fraction of calibration labels is overwritten with the reasoner's own
    top-1 prediction.  Every laundered sample then has a minimal
    nonconformity score, so the conformal quantile ``q_hat`` shrinks and
    the deployed sets silently under-cover.  Returns ``(y_poisoned, mask)``.
    """
    n = y_calib.shape[0]
    disagree = np.flatnonzero(preds_calib != y_calib)
    k = int(round(fraction * n))
    chosen = rng.choice(disagree, size=min(k, disagree.size), replace=False)
    y_p = y_calib.copy()
    y_p[chosen] = preds_calib[chosen]
    mask = np.zeros(n, dtype=bool)
    mask[chosen] = True
    return y_p, mask


def shift_covariates(
    X: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Push feature distribution toward the high-risk regime.

    We simulate a combination of realistic UAM-level stressors: a
    meteorological front (higher weather severity, higher humidity),
    rush-hour traffic (higher airspace density and vertiport congestion),
    fleet-wide battery fatigue (lower state-of-charge), plus a global
    variance inflation that reflects a degraded sensing stack.
    """
    bias = np.zeros(X.shape[1])
    bias[:5] = np.array([+1, +1, +1, -1, +1])
    # Global heteroscedastic noise captures a degraded sensing stack.
    het = 0.2 * strength * rng.standard_normal(X.shape)
    shifted = X + strength * 0.4 * bias[None, :] + het
    # Flip a subset of features into the opposite regime with probability
    # ``strength/2`` to simulate sensor-source substitution / failover.
    mask = rng.random(X.shape) < 0.5 * strength
    shifted = np.where(mask, 1.0 - shifted, shifted)
    return np.clip(shifted, 0.0, 1.0)


def apply_threat(
    kind: str,
    *,
    X: np.ndarray | None = None,
    rules: Sequence[SymbolicRule] | None = None,
    strength: float = 0.0,
    num_features: int | None = None,
    num_classes: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Dispatch to the correct primitive and return a dict with updates.

    Returns a mapping with keys among ``{"X", "rules"}``: only the pieces
    actually changed by the given threat are present, so callers can merge
    the result into the baseline run.
    """
    if rng is None:
        rng = np.random.default_rng()

    if kind == "none":
        return {}
    if kind == "rule_flip":
        assert rules is not None
        return {"rules": corrupt_rules(rules, strength, rng)}
    if kind == "rule_inject":
        assert rules is not None and num_features and num_classes
        return {
            "rules": inject_rule_noise(
                rules, strength, num_features, num_classes, rng
            )
        }
    if kind == "feature_attack":
        assert X is not None
        return {"X": perturb_features(X, strength, rng)}
    if kind == "covariate_shift":
        assert X is not None
        return {"X": shift_covariates(X, strength, rng)}
    raise ValueError(f"unknown threat kind: {kind!r}")
