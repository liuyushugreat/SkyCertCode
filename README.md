# SkyCertCode

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Runtime: CPU-only](https://img.shields.io/badge/runtime-CPU--only-lightgrey.svg)]()

SkyCertCode is the open-source artifact for **SkyCert: Conformal and Martingale-Based Runtime Security Assurance for Neuro-Symbolic Risk Reasoning in Urban Air Mobility**. It packages the Python implementation, default experiment configuration, unit tests, and reproduction scripts used by the paper.

SkyCert wraps a neuro-symbolic UAM risk reasoner with three runtime assurance mechanisms:

- **Conformal risk sets** for calibrated set-valued risk predictions.
- **Conformal test-martingale monitoring** for online evidence of distribution shift or feature tampering.
- **Decision and audit policy** that maps uncertainty and drift evidence to `ACCEPT`, `ABSTAIN`, `ALERT`, or `ESCALATE` and records machine-checkable JSONL audit traces whose header pins the calibration quantile, SHA-256 digests of the calibration set and rule set, and the monitor/policy parameters, so every log can be re-verified offline by `scripts/replay_audit.py`.

Beyond the five baseline threat scenarios, the artifact contains the **extended security evaluation** (targeted top-k rule flips ranked by CRITICAL impact, calibration-set poisoning with a sequestered-set check, adaptive feature-tampering adversaries, and throughput/manual-workload analysis versus the abstention threshold) and a **trajectory-driven operational case study** (an 8-vertiport, 40x40 km UAM network with ~1200 flights/day, AR(1) weather, battery/separation/congestion/noise dynamics, a scripted GNSS-spoofing episode and a weather-front day).

The code is CPU-only and uses synthetic, seed-controlled UAM benchmark data so the experiments can be reproduced without private datasets, API keys, or GPU hardware.

## Quick Start

```bash
git clone git@github.com:liuyushugreat/SkyCertCode.git
cd SkyCertCode

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
bash run.sh
```

On Windows PowerShell:

```powershell
git clone git@github.com:liuyushugreat/SkyCertCode.git
cd SkyCertCode
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
.\run.ps1
```

`run.sh`/`run.ps1` execute the test suite (17 tests), main threat experiments, ablations, baseline comparisons, extension experiments, multi-seed aggregation, the extended security evaluation, the trajectory-driven case study, an audit-log replay check, and figure generation (about 3-4 minutes on one CPU core). Generated artifacts are written under `outputs/`, which is ignored by git because the files are reproducible.

## Repository Structure

```text
SkyCertCode/
├── configs/
│   └── default.yaml          # Seeded experiment and policy configuration
├── scripts/
│   ├── run_experiment.py     # Main 5-scenario threat experiment
│   ├── run_ablation.py       # Component ablations
│   ├── run_baselines.py      # MSP, entropy, conformal-only, backbone variants
│   ├── run_extensions.py     # Sensitivity sweeps and appendix experiments
│   ├── run_multi_seed.py     # Five-seed aggregation
│   ├── run_security_extensions.py  # Targeted T1, calibration poisoning, adaptive T3, gamma workload
│   ├── run_case_study.py     # Trajectory-driven operational case study
│   ├── replay_audit.py       # Offline audit-log verification (hash pins + recomputation)
│   ├── plot_results.py       # Paper figure generation
│   ├── plot_case_study.py    # Case-study timeline and security-extension figures
│   └── print_summary.py      # Human-readable summary tables
├── skycert/
│   ├── assurance/            # Conformal, martingale, policy, audit logger
│   ├── base/                 # Neural, symbolic, neuro-symbolic reasoners
│   ├── data/                 # Synthetic UAM data, threat injection, trajectory simulator
│   ├── config.py             # YAML config loader and dataclasses
│   ├── metrics.py            # Coverage, ECE, critical-error, delay metrics
│   └── pipeline.py           # End-to-end SkyCert pipeline
├── tests/                    # Pytest suite
├── pyproject.toml
├── requirements.txt
├── run.sh
└── run.ps1
```

## Paper-to-Code Mapping

| Paper component | Code location |
| --- | --- |
| Neuro-symbolic reasoner | `skycert/base/neural.py`, `skycert/base/symbolic.py`, `skycert/base/neuro_symbolic.py` |
| Synthetic UAM benchmark | `skycert/data/synthetic.py` |
| T1-T4 threat injection (incl. targeted top-k rule flips, calibration poisoning, adaptive T3 modes) | `skycert/data/threats.py` |
| Trajectory-driven UAM simulator (case study) | `skycert/data/trajectory_sim.py` |
| Conformal risk sets (raw APS set + top-1 augmentation, Eq. augset) | `skycert/assurance/conformal.py` |
| Martingale monitor (hybrid score, `drift_mode`, `reset_every`, `drift_features`) | `skycert/assurance/martingale.py`, `skycert/pipeline.py` |
| Decision policy (Algorithm 1) | `skycert/assurance/policy.py` |
| Audit artifacts: pinned header + per-decision records | `skycert/assurance/audit.py` |
| Audit replay / integrity verification | `scripts/replay_audit.py` |
| End-to-end orchestration | `skycert/pipeline.py` |
| Main experiments (Table I) | `scripts/run_experiment.py`, `scripts/run_multi_seed.py` |
| Ablations and baselines (Tables II-III) | `scripts/run_ablation.py`, `scripts/run_baselines.py` |
| Extended security evaluation (targeted T1, poisoning, adaptive, gamma workload) | `scripts/run_security_extensions.py` |
| Operational case study (timeline figure, per-segment table) | `scripts/run_case_study.py`, `scripts/plot_case_study.py` |
| Appendix/sensitivity experiments | `scripts/run_extensions.py` |
| Paper figures | `scripts/plot_results.py`, `scripts/plot_case_study.py` |

## Reproduction Commands

```bash
# Unit tests
python -m pytest tests -q

# Main experiment: clean, KG corruption, rule poisoning, feature manipulation, covariate shift
python -m scripts.run_experiment --config configs/default.yaml

# Ablation under covariate shift
python -m scripts.run_ablation --config configs/default.yaml

# Baselines and backbone variants
python -m scripts.run_baselines --config configs/default.yaml

# Extension experiments: drift-weight sweep, attack-strength sweeps, MLP backbone, examples
python -m scripts.run_extensions --config configs/default.yaml

# Five-seed aggregation for reported means/stds
python -m scripts.run_multi_seed --config configs/default.yaml

# Extended security evaluation (5 seeds): targeted top-k rule flips (k=1,2,5),
# calibration poisoning (rho in {0,.05,.1,.2,.3}) with sequestered-set check,
# adaptive T3 adversaries, and reviews-vs-residual-misses per gamma
python -m scripts.run_security_extensions --config configs/default.yaml

# Trajectory-driven operational case study (5 seeds x {2,7}-day calibration x
# 4 monitor variants: default, no_drift, surface, windowed)
python -m scripts.run_case_study --config configs/default.yaml

# Verify an audit log offline: recompute every decision from the logged
# probabilities and header parameters, check the SHA-256 pins, report mismatches
python -m scripts.replay_audit outputs/audit/audit_distribution_shift.jsonl --show 1629

# Generate figures in outputs/figs/
python -m scripts.plot_results --config configs/default.yaml
python -m scripts.plot_case_study

# Print summary tables
python scripts/print_summary.py
```

Typical generated files include:

- `outputs/metrics.json`
- `outputs/ablation.json`
- `outputs/baselines.json`
- `outputs/extensions.json`
- `outputs/multi_seed.json`
- `outputs/security_extensions.json`
- `outputs/case_study.json`, `outputs/case_study_traces/*.npz`
- `outputs/audit/*.jsonl`
- `outputs/figs/*.pdf` (incl. `case_study_timeline.pdf`, `security_extensions.pdf`)

### Monitor variants

The martingale configuration in `configs/default.yaml` exposes the options studied in the paper:

| Key | Default | Meaning |
| --- | --- | --- |
| `lambda_drift` | `1.0` | weight of the label-free input-drift term in the hybrid nonconformity score |
| `drift_mode` | `rms` | `rms` aggregates standardised deviations; `hinge` ignores deviations below `hinge_z` sigma |
| `reset_every` | `0` | `>0` restarts the martingale every N operations (windowed monitor; raise `threshold` accordingly) |
| `drift_features` | `null` | optional index list restricting the drift term to attack-surface features |

### Audit-log integrity

Each JSONL log starts with a header record containing `q_hat`, `alpha`, `n_calib`, `calib_sha256`, `rules_sha256`, and the full martingale/policy parameters. `scripts/replay_audit.py` recomputes the prediction set and decision of every record from the logged probabilities and the header, compares them with what was logged, and (optionally, via `--calib-npz`) re-hashes a provided calibration set against the pin. Any post-hoc change to the calibration set, the rule set, or a decision record is reported as a mismatch.

## Minimal Python Usage

```python
from skycert.config import SkyCertConfig
from skycert.pipeline import SkyCertPipeline

cfg = SkyCertConfig.load("configs/default.yaml")
pipeline = SkyCertPipeline(cfg)

pipeline.fit(X_train, y_train)
pipeline.calibrate(X_calib, y_calib)

decision = pipeline.step(x_operation, sample_id="op-001")
print(decision.verdict, decision.conformal_set, decision.martingale)
```

## Threat Scenarios

The default configuration evaluates five scenarios:

- `clean`: benign exchangeable stream.
- `kg_corrupt`: sign flips in selected symbolic rule deltas.
- `rule_poison`: injection of spurious symbolic rules.
- `input_manipulate`: bounded feature perturbation on normalized sensor features.
- `distribution_shift`: weather, traffic, battery, and noise changes that break exchangeability.

`scripts/run_security_extensions.py` adds the stronger variants used in the extended evaluation:

- targeted rule flips: the `k` rules with the largest CRITICAL impact (`rank_rules_by_critical_impact`) are flipped, either before calibration (supply chain) or mid-stream;
- calibration poisoning: a fraction `rho` of calibration labels is replaced by the reasoner's own top-1 prediction ("agreement laundering"), which inflates `q_hat`; detected by the SHA-256 pin and by a sequestered-set coverage check;
- adaptive feature tampering: `persistent_sub` (below the stealth band), `ramp`, `intermittent`, and `critical_directed` (tampering only true-CRITICAL operations);
- gamma workload: manual reviews and residual CRITICAL misses per 1000 operations as the abstention threshold varies.

`scripts/run_case_study.py` runs the trajectory-driven simulator with a scripted GNSS-spoofing episode (day A, 09:00-11:00) and a weather-front day (day B, from 13:00), and reports per-segment coverage, CRITICAL coverage, miss rates, abstention, alarm onsets, and detection delays for several monitor variants and calibration windows.

## Development

```bash
pip install -e .[dev]
python -m pytest -q
```

The package is intentionally small and dependency-light: `numpy`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, and `pytest` for tests.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
