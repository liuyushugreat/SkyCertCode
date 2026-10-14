# SkyCertCode

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Runtime: CPU-only](https://img.shields.io/badge/runtime-CPU--only-lightgrey.svg)]()

SkyCertCode is the open-source artifact for **SkyCert: Conformal and Martingale-Based Runtime Security Assurance for Neuro-Symbolic Risk Reasoning in Urban Air Mobility**. It packages the Python implementation, default experiment configuration, unit tests, and reproduction scripts used by the paper.

SkyCert wraps a neuro-symbolic UAM risk reasoner with three runtime assurance mechanisms:

- **Conformal risk sets** for calibrated set-valued risk predictions.
- **Conformal test-martingale monitoring** for online evidence of distribution shift or feature tampering.
- **Decision and audit policy** that maps uncertainty and drift evidence to `ACCEPT`, `ABSTAIN`, `ALERT`, or `ESCALATE` and records machine-checkable JSONL audit traces.

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

`run.sh`/`run.ps1` execute the test suite, main threat experiments, ablations, baseline comparisons, extension experiments, multi-seed aggregation, and figure generation. Generated artifacts are written under `outputs/`, which is ignored by git because the files are reproducible.

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
│   ├── plot_results.py       # Paper figure generation
│   └── print_summary.py      # Human-readable summary tables
├── skycert/
│   ├── assurance/            # Conformal, martingale, policy, audit logger
│   ├── base/                 # Neural, symbolic, neuro-symbolic reasoners
│   ├── data/                 # Synthetic UAM data and threat injection
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
| T1-T4 threat injection | `skycert/data/threats.py` |
| Conformal risk sets | `skycert/assurance/conformal.py` |
| Martingale monitor | `skycert/assurance/martingale.py` |
| Decision policy | `skycert/assurance/policy.py` |
| Audit artifacts | `skycert/assurance/audit.py` |
| End-to-end orchestration | `skycert/pipeline.py` |
| Main experiments | `scripts/run_experiment.py` |
| Ablations and baselines | `scripts/run_ablation.py`, `scripts/run_baselines.py` |
| Appendix/sensitivity experiments | `scripts/run_extensions.py` |
| Multi-seed paper tables | `scripts/run_multi_seed.py` |
| Paper figures | `scripts/plot_results.py` |

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

# Generate figures in outputs/figs/
python -m scripts.plot_results --config configs/default.yaml

# Print summary tables
python scripts/print_summary.py
```

Typical generated files include:

- `outputs/metrics.json`
- `outputs/ablation.json`
- `outputs/baselines.json`
- `outputs/extensions.json`
- `outputs/multi_seed.json`
- `outputs/audit/*.jsonl`
- `outputs/figs/*.pdf`

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

## Development

```bash
pip install -e .[dev]
python -m pytest -q
```

The package is intentionally small and dependency-light: `numpy`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, and `pytest` for tests.

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
