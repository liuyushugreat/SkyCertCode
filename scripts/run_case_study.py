"""Trajectory-driven UAM case study (paper Sec. "Operational case study").

A vertiport network with diurnal demand, AR(1) weather, battery/separation/
congestion physics and a time-ordered operation stream is simulated by
``skycert.data.trajectory_sim``.  The stream contains two scripted events:

* **Day A, 09:00--11:00** -- a surveillance-feed spoofing episode (ghost
  traffic, corrupted wind telemetry, under-reported weather; T3-like,
  bounded in l_inf by 0.12).
* **Day B, 13:00** -- a weather front (wind +9 m/s, precipitation +0.35
  over a 1.5 h ramp) that persists until the end of the stream.

For every seed we report, per stream segment (benign / spoof / front):
coverage, CRITICAL-class coverage, mean set size, abstention rate, the
CRITICAL miss rate of the base reasoner and after abstention, and the
martingale alert statistics (benign alarm fraction, detection delay).

Several monitor variants are evaluated to study *harmless* versus
*harmful* distribution shift on autocorrelated operational streams:

* ``default``   -- paper monitor (lambda_drift = 1, all features);
* ``no_drift``  -- confidence slack only (lambda_drift = 0);
* ``surface``   -- drift term restricted to attack-surface features;
* ``windowed``  -- windowed martingale (W = 100 ops, threshold 240);

and two calibration windows (2 and 7 days) and two abstention budgets
(gamma = 0.75 and 1.0).

Usage::

    python -m scripts.run_case_study --seeds 20260417 1 2 3 4 \
        --output outputs/case_study.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from skycert.assurance.policy import DecisionKind
from skycert.base import (
    NeuralRiskScorer,
    NeuroSymbolicRiskReasoner,
    SymbolicRuleEngine,
)
from skycert.config import SkyCertConfig
from skycert.data import make_trajectory_case_study
from skycert.pipeline import SkyCertPipeline
from skycert.utils import ensure_dir, safe_json

CRITICAL = 3
# Surveillance / telemetry derived features (the T3 attack surface):
# severity, density, congestion, wind, head-wind, filed-plan conflicts.
ATTACK_SURFACE = [0, 1, 2, 7, 8, 12]

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "no_drift": {"lambda_drift": 0.0},
    "surface": {"drift_features": ATTACK_SURFACE},
    "windowed": {"reset_every": 100, "threshold": 240.0},
}


def _segment_metrics(
    *,
    mask: np.ndarray,
    y: np.ndarray,
    set_mask: np.ndarray,
    preds: np.ndarray,
    abstained: np.ndarray,
    alerts: np.ndarray,
) -> dict[str, Any]:
    if not mask.any():
        return {"n": 0}
    yy = y[mask]
    sm = set_mask[mask]
    pp = preds[mask]
    ab = abstained[mask]
    crit = yy == CRITICAL
    cov = float(sm[np.arange(yy.size), yy].mean())
    out: dict[str, Any] = {
        "n": int(mask.sum()),
        "coverage": cov,
        "critical_coverage": float(sm[crit, CRITICAL].mean()) if crit.any() else None,
        "avg_set_size": float(sm.sum(axis=1).mean()),
        "abstain_rate": float(ab.mean()),
        "alert_rate": float(alerts[mask].mean()),
        "critical_frac": float(crit.mean()),
        "critical_miss_base": float((pp[crit] != CRITICAL).mean()) if crit.any() else None,
        "critical_miss_after": (
            float((pp[crit & ~ab] != CRITICAL).mean()) if (crit & ~ab).any() else 0.0
        ),
        "critical_abstained_frac": float(ab[crit].mean()) if crit.any() else None,
    }
    return out


def run_seed(
    *,
    seed: int,
    base_config: SkyCertConfig,
    calib_window_days: int,
    gammas: tuple[float, ...],
    variants: dict[str, dict[str, Any]],
    trace_dir: Path | None,
) -> dict[str, Any]:
    w = calib_window_days
    cs = make_trajectory_case_study(
        seed=seed,
        num_days=4 + w,
        train_days=(0, 2),
        calib_days=(2, 2 + w),
        stream_days=(2 + w, 4 + w),
        spoof_day=2 + w,
        front_day=3 + w,
    )
    neural = NeuralRiskScorer(
        num_classes=cs.num_classes,
        l2=base_config.base_model.l2,
        max_iter=base_config.base_model.max_iter,
        model_type=base_config.base_model.type,
        hidden=base_config.base_model.hidden,
    ).fit(cs.X_train, cs.y_train)
    reasoner = NeuroSymbolicRiskReasoner(
        neural,
        SymbolicRuleEngine(base_config.symbolic.rules, cs.num_classes),
        base_config.symbolic.lambda_,
    )

    benign = ~cs.spoof_mask & ~cs.front_mask
    segments = {"benign": benign, "spoof": cs.spoof_mask, "front": cs.front_mask}
    t = cs.t_stream_min
    minutes_per_op_front = (
        float(np.mean(np.diff(t[cs.front_start:cs.front_start + 60])))
        if cs.front_start >= 0 else None
    )

    result: dict[str, Any] = {
        "seed": seed,
        "calib_window_days": w,
        "meta": cs.meta,
        "n_train": int(cs.y_train.size),
        "n_calib": int(cs.y_calib.size),
        "n_stream": int(cs.y_stream.size),
        "variants": {},
    }

    probs = reasoner.predict_proba(cs.X_stream)
    preds = probs.argmax(axis=1)
    for vname, overrides in variants.items():
        for gamma in gammas:
            cfg = copy.deepcopy(base_config)
            mc = cfg.assurance.martingale
            for k, v in overrides.items():
                setattr(mc, k, v)
            cfg.assurance.policy.max_set_fraction = gamma
            pipe = SkyCertPipeline(reasoner, cfg)
            pipe.calibrate(cs.X_calib, cs.y_calib)
            t0 = time.perf_counter()
            decisions = pipe.run_batch(cs.X_stream)
            elapsed = time.perf_counter() - t0
            set_mask = pipe.predict_sets(cs.X_stream)
            abstained = np.array(
                [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT,
                            DecisionKind.ESCALATE} for d in decisions]
            )
            alerts = np.array([d.martingale_alert for d in decisions])
            trace = np.asarray(pipe.monitor.trace, dtype=np.float64)

            # Alarm onsets on benign operations (distinct episodes).
            ab = alerts[benign]
            onsets = int((ab & ~np.concatenate([[False], ab[:-1]])).sum())
            spoof_alerts = np.flatnonzero(alerts[cs.spoof_start:cs.spoof_end])
            front_alerts = np.flatnonzero(alerts[cs.front_start:])
            front_delay_ops = int(front_alerts[0]) if front_alerts.size else None
            # alarm already active when the front begins?
            front_pre_alarm = bool(alerts[cs.front_start - 1]) if cs.front_start > 0 else False

            entry: dict[str, Any] = {
                "q_hat": pipe.calibration_header()["q_hat"],
                "segments": {
                    name: _segment_metrics(
                        mask=m, y=cs.y_stream, set_mask=set_mask, preds=preds,
                        abstained=abstained, alerts=alerts,
                    )
                    for name, m in segments.items()
                },
                "benign_alarm_onsets": onsets,
                "benign_martingale_max": float(trace[benign].max()),
                "spoof_detected": bool(spoof_alerts.size > 0),
                "spoof_first_alert_ops": int(spoof_alerts[0]) if spoof_alerts.size else None,
                "spoof_alarm_preexisting": bool(alerts[cs.spoof_start - 1]) if cs.spoof_start > 0 else False,
                "front_detected": bool(front_alerts.size > 0),
                "front_delay_ops": front_delay_ops,
                "front_delay_min": (
                    float(t[cs.front_start + front_delay_ops] - t[cs.front_start])
                    if front_delay_ops is not None else None
                ),
                "front_alarm_preexisting": front_pre_alarm,
                "ops_per_min_at_front": (1.0 / minutes_per_op_front) if minutes_per_op_front else None,
                "avg_decision_ms": 1000.0 * elapsed / cs.y_stream.size,
            }
            result["variants"][f"{vname}@gamma={gamma}"] = entry

            if trace_dir is not None and gamma == gammas[0]:
                cov = set_mask[np.arange(cs.y_stream.size), cs.y_stream].astype(np.int8)
                np.savez_compressed(
                    trace_dir / f"trace_seed{seed}_w{w}_{vname}.npz",
                    t_min=t, martingale=trace, alert=alerts, covered=cov,
                    abstained=abstained, y=cs.y_stream, pred=preds,
                    set_size=set_mask.sum(axis=1), spoof_mask=cs.spoof_mask,
                    front_mask=cs.front_mask, spoof_start=cs.spoof_start,
                    spoof_end=cs.spoof_end, front_start=cs.front_start,
                )
    return result


def _agg(values: list[float | None]) -> dict[str, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_w: dict[int, list[dict[str, Any]]] = {}
    for r in runs:
        by_w.setdefault(r["calib_window_days"], []).append(r)
    for w, rs in by_w.items():
        agg_w: dict[str, Any] = {"n_seeds": len(rs), "variants": {}}
        for vkey in rs[0]["variants"]:
            entries = [r["variants"][vkey] for r in rs]
            a: dict[str, Any] = {"segments": {}}
            for seg in ("benign", "spoof", "front"):
                segd: dict[str, Any] = {}
                for metric in ("coverage", "critical_coverage", "avg_set_size",
                               "abstain_rate", "alert_rate", "critical_frac",
                               "critical_miss_base", "critical_miss_after",
                               "critical_abstained_frac"):
                    segd[metric] = _agg([e["segments"][seg].get(metric) for e in entries])
                a["segments"][seg] = segd
            a["seeds_with_benign_alarm"] = int(sum(e["benign_alarm_onsets"] > 0 for e in entries))
            a["benign_alarm_onsets"] = _agg([e["benign_alarm_onsets"] for e in entries])
            a["spoof_detected_seeds"] = int(sum(e["spoof_detected"] for e in entries))
            a["spoof_detected_fresh_seeds"] = int(
                sum(e["spoof_detected"] and not e["spoof_alarm_preexisting"] for e in entries)
            )
            a["front_detected_seeds"] = int(sum(e["front_detected"] for e in entries))
            a["front_delay_ops"] = _agg([e["front_delay_ops"] for e in entries])
            a["front_delay_min"] = _agg([e["front_delay_min"] for e in entries])
            a["front_alarm_preexisting_seeds"] = int(sum(e["front_alarm_preexisting"] for e in entries))
            a["avg_decision_ms"] = _agg([e["avg_decision_ms"] for e in entries])
            agg_w["variants"][vkey] = a
        out[f"calib_window_{w}d"] = agg_w
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SkyCert trajectory case study")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260417, 1, 2, 3, 4])
    parser.add_argument("--calib-windows", type=int, nargs="+", default=[2, 7])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.75, 1.0])
    parser.add_argument("--output", default="outputs/case_study.json")
    parser.add_argument("--trace-dir", default="outputs/case_study_traces")
    args = parser.parse_args(argv)

    cfg = SkyCertConfig.load(args.config)
    trace_dir = ensure_dir(Path(args.trace_dir)) if args.trace_dir else None
    runs: list[dict[str, Any]] = []
    for w in args.calib_windows:
        for seed in args.seeds:
            t0 = time.perf_counter()
            r = run_seed(
                seed=seed, base_config=cfg, calib_window_days=w,
                gammas=tuple(args.gammas), variants=VARIANTS, trace_dir=trace_dir,
            )
            runs.append(r)
            d = r["variants"][f"default@gamma={args.gammas[0]}"]
            print(
                f"[w={w}d seed={seed}] n_stream={r['n_stream']} "
                f"benign cov={d['segments']['benign']['coverage']:.3f} "
                f"alarm_onsets={d['benign_alarm_onsets']} "
                f"spoof_det={d['spoof_detected']} front_delay_ops={d['front_delay_ops']} "
                f"({time.perf_counter() - t0:.1f}s)"
            )
    out = {
        "config": args.config,
        "seeds": args.seeds,
        "variants": {k: v for k, v in VARIANTS.items()},
        "attack_surface_features": ATTACK_SURFACE,
        "aggregate": aggregate(runs),
        "runs": runs,
    }
    path = Path(args.output)
    ensure_dir(path.parent)
    path.write_text(json.dumps(safe_json(out), indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
