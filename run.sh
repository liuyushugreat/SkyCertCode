#!/usr/bin/env bash
# SkyCert one-click reproduction for IEEE CSCloud 2026 reviewers.
#
# What it does (runs in order, ~3-4 minutes on a single CPU core):
#   1. install dependencies,
#   2. run the unit test suite (17 tests),
#   3. run the main threat-scenario experiment     -> outputs/metrics.json + outputs/audit/,
#   4. run the ablation / baseline / extension studies,
#   5. run the 5-seed aggregation (Tables I-III),
#   6. run the extended security evaluation        -> outputs/security_extensions.json
#      (targeted top-k rule flips, calibration poisoning, adaptive adversary,
#       gamma workload; Section "Extended Security Evaluation"),
#   7. run the trajectory-driven operational case study -> outputs/case_study.json
#      (Section "Operational Case Study"),
#   8. verify an audit log by replay (hash pins + recomputed decisions),
#   9. render all paper figures                    -> outputs/figs/,
#  10. print a short summary of the key metrics.
#
# No GPU, API key, or network access is required.

set -euo pipefail

cd "$(dirname "$0")"

echo "[SkyCert] Installing dependencies ..."
python -m pip install --quiet -r requirements.txt

echo "[SkyCert] Running unit tests ..."
python -m pytest tests -q

echo "[SkyCert] Running main threat-scenario experiment ..."
python -m scripts.run_experiment --config configs/default.yaml

echo "[SkyCert] Running ablation study ..."
python -m scripts.run_ablation   --config configs/default.yaml

echo "[SkyCert] Running baseline comparison ..."
python -m scripts.run_baselines  --config configs/default.yaml

echo "[SkyCert] Running extension experiments (lambda sweep, attack-strength sweep, failure cases, MLP) ..."
python -m scripts.run_extensions --config configs/default.yaml

echo "[SkyCert] Running 5-seed aggregation (Tables I/II/III mean+-std) ..."
python -m scripts.run_multi_seed --config configs/default.yaml

echo "[SkyCert] Running extended security evaluation (targeted T1, calibration poisoning, adaptive T3, gamma workload) ..."
python -m scripts.run_security_extensions --config configs/default.yaml

echo "[SkyCert] Running trajectory-driven operational case study (5 seeds x {2,7}-day calibration x 4 monitor variants) ..."
python -m scripts.run_case_study --config configs/default.yaml

echo "[SkyCert] Replaying the distribution_shift audit log (hash pins + decision recomputation) ..."
python -m scripts.replay_audit outputs/audit/audit_distribution_shift.jsonl --show 1629

echo "[SkyCert] Rendering paper figures ..."
python -m scripts.plot_results    --config configs/default.yaml
python -m scripts.plot_case_study

echo "[SkyCert] Summary (see outputs/metrics.json for full data):"
python - <<'PY'
import json, pathlib
p = pathlib.Path("outputs/metrics.json")
data = json.loads(p.read_text())
print(f"{'scenario':<22}{'coverage':>10}{'abstain':>10}{'crit_base':>12}{'crit_after':>12}{'M_max':>14}")
for r in data["runs"]:
    name = r["threat"]["name"]
    print(f"{name:<22}"
          f"{r['coverage']:>10.3f}"
          f"{r['abstain_rate']:>10.3f}"
          f"{r['critical_error_rate_base']:>12.3f}"
          f"{r['critical_error_rate_after_abstain']:>12.3f}"
          f"{r['martingale_max']:>14.2e}")
lat = [r["avg_decision_ms"] for r in data["runs"]]
print(f"\nmean end-to-end decision latency: {sum(lat)/len(lat):.3f} ms")
PY

echo "[SkyCert] Done. Artifacts are in ./outputs/"
