"""Security extensions added for the camera-ready version.

Four analyses on the synthetic benchmark (same data/backbone/config as
``scripts.run_experiment``), each aggregated over several seeds:

1. **Targeted knowledge-graph corruption (T1, top-k).**  Rules are ranked
   by how strongly negating each one alone suppresses CRITICAL decisions
   (oracle ranking on the attacker's own data), and the ``k`` highest-impact
   rules are negated for k in {1, 2, 5}.  The corrupted rule set is pushed
   *mid-stream* (a knowledge-graph update at the change point ``t*``), so
   the conformal predictor was calibrated with the clean rules.  We report
   post-``t*`` coverage, CRITICAL miss before/after abstention, raw-empty
   APS set rate and the martingale detection rate / delay.

2. **Calibration-set poisoning ("agreement laundering").**  A fraction rho
   of the calibration labels is overwritten with the reasoner's top-1
   prediction.  We report q_hat, test coverage, CRITICAL coverage and
   residual miss rate, together with the two integrity defences: the
   SHA-256 calibration pin recorded in the audit header and the
   sequestered-split coverage check.

3. **Adaptive (SkyCert-aware) feature attacks.**  ``persistent_sub``,
   ``ramp``, ``intermittent`` and ``critical_directed`` variants of T3.

4. **Abstention-budget / workload trade-off.**  Sweep ``gamma`` and report
   manual reviews and residual CRITICAL misses per 1000 operations under
   the clean and distribution-shift streams.

Usage::

    python -m scripts.run_security_extensions --seeds 20260417 1 2 3 4 \
        --output outputs/security_extensions.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from skycert.assurance.audit import sha256_of_arrays
from skycert.assurance.conformal import sequestered_coverage_check
from skycert.assurance.policy import DecisionKind
from skycert.base import (
    NeuralRiskScorer,
    NeuroSymbolicRiskReasoner,
    SymbolicRuleEngine,
)
from skycert.config import SkyCertConfig
from skycert.data import (
    adaptive_feature_attack,
    apply_threat,
    corrupt_rules_targeted,
    make_uam_dataset,
    poison_calibration_labels,
    rank_rules_by_critical_impact,
)
from skycert.pipeline import SkyCertPipeline
from skycert.utils import ensure_dir, safe_json

CRITICAL = 3


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------
def _build(config: SkyCertConfig, data, rules=None) -> NeuroSymbolicRiskReasoner:
    neural = NeuralRiskScorer(
        num_classes=data.num_classes,
        l2=config.base_model.l2,
        max_iter=config.base_model.max_iter,
        model_type=config.base_model.type,
        hidden=config.base_model.hidden,
    ).fit(data.X_train, data.y_train)
    return _wrap(config, neural, rules or config.symbolic.rules, data.num_classes)


def _wrap(config, neural, rules, num_classes) -> NeuroSymbolicRiskReasoner:
    return NeuroSymbolicRiskReasoner(
        neural=neural,
        symbolic=SymbolicRuleEngine(rules=rules, num_classes=num_classes),
        lambda_=config.symbolic.lambda_,
    )


def _run_stream(
    pipe: SkyCertPipeline,
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    *,
    change_point: int | None,
    swap_reasoner: NeuroSymbolicRiskReasoner | None = None,
) -> dict[str, Any]:
    """Drive ``X_stream`` through ``pipe``; optionally swap the reasoner at
    ``change_point`` (mid-stream knowledge-graph update)."""
    decisions = []
    if swap_reasoner is None or change_point is None:
        decisions = pipe.run_batch(X_stream)
        probs = pipe.predict_proba(X_stream)
    else:
        decisions += pipe.run_batch(X_stream[:change_point])
        probs_a = pipe.predict_proba(X_stream[:change_point])
        pipe.reasoner = swap_reasoner
        decisions += pipe.run_batch(X_stream[change_point:])
        probs_b = pipe.predict_proba(X_stream[change_point:])
        probs = np.vstack([probs_a, probs_b])
    set_mask = pipe.conformal.predict_sets(probs)
    preds = probs.argmax(axis=1)
    abstained = np.array(
        [d.kind in {DecisionKind.ABSTAIN, DecisionKind.ALERT, DecisionKind.ESCALATE}
         for d in decisions]
    )
    alerts = np.array([d.martingale_alert for d in decisions])
    n = y_stream.size
    idx = np.arange(n)
    post = np.ones(n, dtype=bool) if change_point is None else idx >= change_point
    crit = y_stream == CRITICAL

    def seg(mask: np.ndarray) -> dict[str, Any]:
        yy, sm, pp, ab = y_stream[mask], set_mask[mask], preds[mask], abstained[mask]
        cc = yy == CRITICAL
        return {
            "n": int(mask.sum()),
            "coverage": float(sm[np.arange(yy.size), yy].mean()),
            "critical_coverage": float(sm[cc, CRITICAL].mean()) if cc.any() else None,
            "avg_set_size": float(sm.sum(axis=1).mean()),
            "raw_empty_rate": pipe.conformal.raw_empty_rate(probs[mask]),
            "abstain_rate": float(ab.mean()),
            "critical_miss_base": float((pp[cc] != CRITICAL).mean()) if cc.any() else None,
            "critical_miss_after": (
                float((pp[cc & ~ab] != CRITICAL).mean()) if (cc & ~ab).any() else 0.0
            ),
            "residual_critical_misses_per_1000": float(
                1000.0 * np.sum(cc & ~ab & (pp != CRITICAL)) / max(yy.size, 1)
            ),
            "reviews_per_1000": float(1000.0 * ab.mean()),
        }

    out: dict[str, Any] = {"all": seg(np.ones(n, dtype=bool)), "post": seg(post)}
    if change_point is not None:
        pre_alerts = alerts[:change_point]
        post_alerts = alerts[change_point:]
        out["detection"] = {
            "false_alarm_rate": float(pre_alerts.mean()) if pre_alerts.size else 0.0,
            "detected": bool(post_alerts.any()),
            "delay": int(np.argmax(post_alerts)) if post_alerts.any() else None,
            "post_alert_rate": float(post_alerts.mean()),
        }
    out["martingale_max"] = float(pipe.monitor.max_value())
    return out


# -----------------------------------------------------------------------
# 1. targeted T1
# -----------------------------------------------------------------------
def targeted_t1(config, data, reasoner, ks=(1, 2, 5)) -> dict[str, Any]:
    rules = config.symbolic.rules
    factory = lambda rs: _wrap(config, reasoner.neural, rs, data.num_classes)  # noqa: E731
    ranking = rank_rules_by_critical_impact(rules, factory, data.X_train, CRITICAL)
    half = data.X_test.shape[0] // 2
    out: dict[str, Any] = {"ranking": ranking, "change_point": half, "k": {}}
    for k in ks:
        corrupted = factory(corrupt_rules_targeted(rules, k, ranking))
        pipe = SkyCertPipeline(reasoner=reasoner, config=config, audit_path=None)
        pipe.calibrate(data.X_calib, data.y_calib)
        res = _run_stream(pipe, data.X_test, data.y_test, change_point=half,
                          swap_reasoner=corrupted)
        # Pre-deployment variant: corrupted rules already present when the
        # conformal predictor is calibrated (supply-chain compromise).
        pipe2 = SkyCertPipeline(reasoner=corrupted, config=config, audit_path=None)
        pipe2.calibrate(data.X_calib, data.y_calib)
        res2 = _run_stream(pipe2, data.X_test, data.y_test, change_point=None)
        res["pre_deployment"] = {
            "coverage": res2["all"]["coverage"],
            "critical_coverage": res2["all"]["critical_coverage"],
            "critical_miss_base": res2["all"]["critical_miss_base"],
            "critical_miss_after": res2["all"]["critical_miss_after"],
            "abstain_rate": res2["all"]["abstain_rate"],
            "alert_any": bool(res2["martingale_max"] >= config.assurance.martingale.threshold),
        }
        out["k"][str(k)] = res
    return out


# -----------------------------------------------------------------------
# 2. calibration poisoning
# -----------------------------------------------------------------------
def calibration_poisoning(config, data, reasoner, rng, rhos=(0.0, 0.05, 0.1, 0.2, 0.3),
                          n_sequestered=300) -> dict[str, Any]:
    Xc, yc = data.X_calib, data.y_calib
    Xs, ys = Xc[-n_sequestered:], yc[-n_sequestered:]      # sequestered (not poisonable)
    Xp, yp = Xc[:-n_sequestered], yc[:-n_sequestered]      # operational calibration set
    pinned_sha = sha256_of_arrays(Xp, yp)                  # hash registered at deployment
    preds_calib = reasoner.predict(Xp)
    probs_val = reasoner.predict_proba(Xs)
    out: dict[str, Any] = {"n_calib": int(yp.size), "n_sequestered": int(ys.size), "rho": {}}
    for rho in rhos:
        y_poison, mask = poison_calibration_labels(yp, preds_calib, rho, rng)
        pipe = SkyCertPipeline(reasoner=reasoner, config=config, audit_path=None)
        pipe.calibrate(Xp, y_poison)
        hdr = pipe.calibration_header()
        res = _run_stream(pipe, data.X_test, data.y_test, change_point=None)
        check = sequestered_coverage_check(probs_val, ys, pipe.conformal,
                                           critical_class=CRITICAL)
        out["rho"][str(rho)] = {
            "n_poisoned": int(mask.sum()),
            "q_hat": hdr["q_hat"],
            "pin_match": hdr["calib_sha256"] == pinned_sha,
            "sequestered_check": check,
            "coverage": res["all"]["coverage"],
            "critical_coverage": res["all"]["critical_coverage"],
            "avg_set_size": res["all"]["avg_set_size"],
            "abstain_rate": res["all"]["abstain_rate"],
            "critical_miss_after": res["all"]["critical_miss_after"],
            "residual_critical_misses_per_1000": res["all"]["residual_critical_misses_per_1000"],
            "martingale_max": res["martingale_max"],
        }
    return out


# -----------------------------------------------------------------------
# 3. adaptive adversary
# -----------------------------------------------------------------------
ADAPTIVE_MODES = {
    "persistent_sub": {"strength": 0.08},
    "ramp": {"strength": 0.12},
    "intermittent": {"strength": 0.12, "duty_cycle": 0.3},
    "critical_directed": {"strength": 0.12},
}


def adaptive_adversary(config, data, reasoner, rng) -> dict[str, Any]:
    half = data.X_test.shape[0] // 2
    out: dict[str, Any] = {"change_point": half, "modes": {}}
    # reference: the paper's untargeted T3 at 0.12
    X_ref = data.X_test.copy()
    upd = apply_threat("feature_attack", X=X_ref[half:], strength=0.12,
                       num_features=config.data.num_features,
                       num_classes=config.data.num_classes, rng=rng)
    X_ref[half:] = upd["X"]
    pipe = SkyCertPipeline(reasoner=reasoner, config=config, audit_path=None)
    pipe.calibrate(data.X_calib, data.y_calib)
    out["modes"]["untargeted_0.12"] = _run_stream(pipe, X_ref, data.y_test, change_point=half)
    for mode, kw in ADAPTIVE_MODES.items():
        X_adv = data.X_test.copy()
        X_part, mask = adaptive_feature_attack(
            data.X_test[half:], mode, kw["strength"], rng,
            y_true=data.y_test[half:], critical_class=CRITICAL,
            duty_cycle=kw.get("duty_cycle", 0.3),
        )
        X_adv[half:] = X_part
        pipe = SkyCertPipeline(reasoner=reasoner, config=config, audit_path=None)
        pipe.calibrate(data.X_calib, data.y_calib)
        res = _run_stream(pipe, X_adv, data.y_test, change_point=half)
        res["attacked_fraction_post"] = float(mask.mean())
        res["params"] = kw
        out["modes"][mode] = res
    return out


# -----------------------------------------------------------------------
# 4. gamma workload
# -----------------------------------------------------------------------
def gamma_workload(config, data, reasoner, rng,
                   gammas=(0.5, 0.75, 1.0, 1.01)) -> dict[str, Any]:
    half = data.X_test.shape[0] // 2
    X_shift = data.X_test.copy()
    upd = apply_threat("covariate_shift", X=X_shift[half:], strength=0.8,
                       num_features=config.data.num_features,
                       num_classes=config.data.num_classes, rng=rng)
    X_shift[half:] = upd["X"]
    out: dict[str, Any] = {"gamma": {}}
    for g in gammas:
        cfg = copy.deepcopy(config)
        cfg.assurance.policy.max_set_fraction = g
        entry: dict[str, Any] = {}
        for name, X in (("clean", data.X_test), ("distribution_shift", X_shift)):
            pipe = SkyCertPipeline(reasoner=reasoner, config=cfg, audit_path=None)
            pipe.calibrate(data.X_calib, data.y_calib)
            res = _run_stream(pipe, X, data.y_test, change_point=half if name != "clean" else None)
            seg = res["post"] if name != "clean" else res["all"]
            entry[name] = {
                "reviews_per_1000": seg["reviews_per_1000"],
                "residual_critical_misses_per_1000": seg["residual_critical_misses_per_1000"],
                "critical_miss_after": seg["critical_miss_after"],
                "critical_miss_base": seg["critical_miss_base"],
                "coverage": seg["coverage"],
                "abstain_rate": seg["abstain_rate"],
            }
        out["gamma"][str(g)] = entry
    return out


# -----------------------------------------------------------------------
# aggregation
# -----------------------------------------------------------------------
def _walk_mean(objs: list[Any]) -> Any:
    """Recursively aggregate a list of equally-structured JSON objects:
    numbers -> {mean, std}; bools -> fraction true; dicts -> recurse."""
    first = objs[0]
    if isinstance(first, dict):
        return {k: _walk_mean([o[k] for o in objs if o is not None and k in o])
                for k in first}
    if isinstance(first, bool):
        return {"frac_true": float(np.mean([bool(o) for o in objs])), "n": len(objs)}
    if isinstance(first, (int, float)) or first is None:
        vals = [float(o) for o in objs if o is not None and np.isfinite(o)]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return first


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SkyCert security extensions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260417, 1, 2, 3, 4])
    parser.add_argument("--output", default="outputs/security_extensions.json")
    args = parser.parse_args(argv)

    base = SkyCertConfig.load(args.config)
    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        config = copy.deepcopy(base)
        config.seed = int(seed)
        data = make_uam_dataset(
            num_train=config.data.num_train, num_calib=config.data.num_calib,
            num_test=config.data.num_test, num_features=config.data.num_features,
            num_classes=config.data.num_classes, class_prior=config.data.class_prior,
            seed=config.seed,
        )
        reasoner = _build(config, data)
        rng = np.random.default_rng(seed + 104729)
        res = {
            "seed": seed,
            "targeted_t1": targeted_t1(config, data, reasoner),
            "calibration_poisoning": calibration_poisoning(config, data, reasoner, rng),
            "adaptive_adversary": adaptive_adversary(config, data, reasoner, rng),
            "gamma_workload": gamma_workload(config, data, reasoner, rng),
        }
        per_seed.append(res)
        t1 = res["targeted_t1"]["k"]
        print(f"[seed {seed}] ranking={[d['rule_id'] for d in res['targeted_t1']['ranking']]} "
              f"k=1 post-cov={t1['1']['post']['coverage']:.3f} miss={t1['1']['post']['critical_miss_base']:.3f} "
              f"det={t1['1']['detection']['detected']} | "
              f"k=5 post-cov={t1['5']['post']['coverage']:.3f} miss={t1['5']['post']['critical_miss_base']:.3f} "
              f"det={t1['5']['detection']['detected']} ({time.perf_counter() - t0:.1f}s)")

    agg = {
        "targeted_t1": {
            "k": _walk_mean([r["targeted_t1"]["k"] for r in per_seed]),
            "ranking_top_rule_ids": [[d["rule_id"] for d in r["targeted_t1"]["ranking"]] for r in per_seed],
        },
        "calibration_poisoning": _walk_mean([r["calibration_poisoning"] for r in per_seed]),
        "adaptive_adversary": _walk_mean([r["adaptive_adversary"]["modes"] for r in per_seed]),
        "gamma_workload": _walk_mean([r["gamma_workload"]["gamma"] for r in per_seed]),
    }
    out = {"config": args.config, "seeds": args.seeds, "aggregate": agg, "per_seed": per_seed}
    path = Path(args.output)
    ensure_dir(path.parent)
    path.write_text(json.dumps(safe_json(out), indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
