"""Tests for the camera-ready extensions: audit pinning / replay, empty-set
handling, sequestered calibration check, targeted / adaptive threats,
windowed monitoring and the trajectory case study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from skycert.assurance.audit import sha256_of_arrays, sha256_of_json
from skycert.assurance.conformal import ConformalRiskSet, sequestered_coverage_check
from skycert.assurance.martingale import MartingaleMonitor, SimpleJumperBetting
from skycert.base import NeuralRiskScorer, NeuroSymbolicRiskReasoner, SymbolicRuleEngine
from skycert.config import SkyCertConfig
from skycert.data import (
    adaptive_feature_attack,
    corrupt_rules_targeted,
    make_trajectory_case_study,
    make_uam_dataset,
    poison_calibration_labels,
    rank_rules_by_critical_impact,
)
from skycert.pipeline import SkyCertPipeline
from scripts.replay_audit import replay

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _small_setup(seed: int = 0):
    config = SkyCertConfig.load(CFG)
    config.seed = seed
    data = make_uam_dataset(num_train=800, num_calib=400, num_test=300,
                            num_features=16, num_classes=4,
                            class_prior=config.data.class_prior, seed=seed)
    neural = NeuralRiskScorer(num_classes=4, l2=1e-3, max_iter=200,
                              model_type="logistic", hidden=16).fit(data.X_train, data.y_train)
    reasoner = NeuroSymbolicRiskReasoner(
        neural, SymbolicRuleEngine(config.symbolic.rules, 4), config.symbolic.lambda_)
    return config, data, reasoner


def test_raw_aps_sets_can_be_empty_and_top1_augmentation_fixes_it():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4) * 0.3, size=2000)   # peaked, over-confident
    y = np.array([rng.choice(4, p=p) for p in probs])
    cp_raw = ConformalRiskSet(alpha=0.1, score="aps", include_top1=False)
    cp_raw.calibrate(probs[:1000], y[:1000])
    raw_empty = cp_raw.raw_empty_rate(probs[1000:])
    assert raw_empty > 0.0                                 # Eq. (3) alone can return {}
    cp = ConformalRiskSet(alpha=0.1, score="aps", include_top1=True)
    cp.calibrate(probs[:1000], y[:1000])
    mask = cp.predict_sets(probs[1000:])
    assert (mask.sum(axis=1) >= 1).all()                   # augmented sets never empty
    assert cp.coverage(probs[1000:], y[1000:]) >= cp_raw.coverage(probs[1000:], y[1000:])


def test_sha256_helpers_are_deterministic_and_sensitive():
    a = np.arange(10, dtype=np.float64)
    b = a.copy(); b[3] += 1e-9
    assert sha256_of_arrays(a) == sha256_of_arrays(a.copy())
    assert sha256_of_arrays(a) != sha256_of_arrays(b)
    assert sha256_of_json({"x": 1, "y": [1, 2]}) == sha256_of_json({"y": [1, 2], "x": 1})


def test_audit_header_pin_and_replay(tmp_path: Path):
    config, data, reasoner = _small_setup()
    log = tmp_path / "audit.jsonl"
    pipe = SkyCertPipeline(reasoner, config, audit_path=log)
    pipe.calibrate(data.X_calib, data.y_calib)
    pipe.run_batch(data.X_test[:50])
    pipe.close()
    first = json.loads(log.read_text().splitlines()[0])
    assert first["record_type"] == "header"
    assert first["calib_sha256"] == sha256_of_arrays(data.X_calib, data.y_calib)
    npz = tmp_path / "calib.npz"
    np.savez(npz, X_calib=data.X_calib, y_calib=data.y_calib)
    rep = replay(log, calib_npz=npz)
    assert rep["n_records"] == 50
    assert sum(rep["mismatches"].values()) == 0 and rep["verified"]
    assert rep["calibration_pin"]["match"] is True
    # tamper with the calibration snapshot -> pin mismatch
    np.savez(npz, X_calib=data.X_calib, y_calib=np.roll(data.y_calib, 1))
    assert replay(log, calib_npz=npz)["calibration_pin"]["match"] is False


def test_sequestered_check_flags_poisoned_calibration():
    config, data, reasoner = _small_setup(seed=3)
    rng = np.random.default_rng(1)
    Xc, yc = data.X_calib[:300], data.y_calib[:300]
    Xs, ys = data.X_calib[300:], data.y_calib[300:]
    preds = reasoner.predict(Xc)
    y_p, mask = poison_calibration_labels(yc, preds, 0.4, rng)
    assert 0 < mask.sum() <= int(round(0.4 * yc.size))
    assert (y_p[mask] == preds[mask]).all()
    cp_clean = ConformalRiskSet(alpha=0.1); cp_clean.calibrate(reasoner.predict_proba(Xc), yc)
    cp_pois = ConformalRiskSet(alpha=0.1); cp_pois.calibrate(reasoner.predict_proba(Xc), y_p)
    assert cp_pois.q_hat < cp_clean.q_hat                  # laundering shrinks q_hat
    chk = sequestered_coverage_check(reasoner.predict_proba(Xs), ys, cp_pois, z=1.0)
    assert chk["coverage"] < 1.0 - 0.1 + 0.05 or chk["flagged"]


def test_targeted_rule_corruption_negates_top_k():
    config, data, reasoner = _small_setup(seed=5)
    rules = config.symbolic.rules
    factory = lambda rs: NeuroSymbolicRiskReasoner(  # noqa: E731
        reasoner.neural, SymbolicRuleEngine(rs, 4), config.symbolic.lambda_)
    ranking = rank_rules_by_critical_impact(rules, factory, data.X_train)
    impacts = [d["impact"] for d in ranking]
    assert impacts == sorted(impacts, reverse=True)
    assert sorted(d["rule_id"] for d in ranking) == list(range(len(rules)))
    k = 2
    corrupted = corrupt_rules_targeted(rules, k, ranking)
    victims = {d["rule_id"] for d in ranking[:k]}
    for i, (r, c) in enumerate(zip(rules, corrupted)):
        if i in victims:
            assert np.allclose(c.delta, -np.asarray(r.delta))
        else:
            assert np.allclose(c.delta, r.delta)


def test_adaptive_attacks_respect_budget_and_masks():
    rng = np.random.default_rng(0)
    X = rng.random((200, 16))
    y = rng.integers(0, 4, size=200)
    for mode in ("persistent_sub", "ramp", "intermittent", "critical_directed"):
        X_adv, mask = adaptive_feature_attack(X, mode, 0.1, rng, y_true=y, duty_cycle=0.3)
        assert X_adv.shape == X.shape and mask.shape == (200,)
        assert np.abs(X_adv - X).max() <= 0.1 + 1e-12
        assert np.allclose(X_adv[~mask], X[~mask])
    _, m = adaptive_feature_attack(X, "critical_directed", 0.1, rng, y_true=y)
    assert (m == (y == 3)).all()


def test_windowed_monitor_restarts_against_reference():
    mon = MartingaleMonitor(threshold=20.0, betting=SimpleJumperBetting(0.92))
    ref = np.random.default_rng(0).random(200)
    mon.warm_start(ref)
    for s in np.full(50, 2.0):                             # extreme scores
        mon.update(s)
    assert mon.steps_in_window == 50 and mon.max_value() > 1.0
    mon.restart_window()
    assert mon.steps_in_window == 0 and mon.martingale == 1.0
    assert len(mon.trace) == 50                            # traces are kept


def test_trajectory_case_study_shapes_and_events():
    cs = make_trajectory_case_study(num_days=6, flights_per_day=300, seed=1)
    assert cs.X_train.shape[1] == 16 and cs.X_stream.shape == cs.X_stream_true.shape
    assert cs.t_calib_min.shape[0] == cs.y_calib.shape[0]
    assert cs.spoof_mask.sum() > 0 and cs.front_mask.sum() > 0
    assert np.all(np.diff(cs.t_stream_min) >= 0)           # time ordered
    diff = np.abs(cs.X_stream - cs.X_stream_true).max(axis=1)
    assert (diff[~cs.spoof_mask] == 0).all() and diff[cs.spoof_mask].max() <= 0.12 + 1e-12
    assert cs.front_start > cs.spoof_end
