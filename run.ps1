# SkyCert one-click reproduction for IEEE CSCloud 2026 reviewers (PowerShell).
#
# What it does (runs in order, ~3-4 minutes on a single CPU core):
#   1. install dependencies,
#   2. run the unit test suite (17 tests),
#   3. run the main threat-scenario experiment (+ audit logs),
#   4. run the ablation / baseline / extension studies,
#   5. run the 5-seed aggregation (Tables I-III),
#   6. run the extended security evaluation (targeted top-k rule flips,
#      calibration poisoning, adaptive adversary, gamma workload),
#   7. run the trajectory-driven operational case study,
#   8. verify an audit log by replay (hash pins + recomputed decisions),
#   9. render all paper figures,
#  10. print a short summary of the key metrics.
#
# No GPU, API key, or network access is required.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "[SkyCert] Installing dependencies ..." -ForegroundColor Cyan
python -m pip install --quiet -r requirements.txt

Write-Host "[SkyCert] Running unit tests ..." -ForegroundColor Cyan
python -m pytest tests -q

Write-Host "[SkyCert] Running main threat-scenario experiment ..." -ForegroundColor Cyan
python -m scripts.run_experiment --config configs/default.yaml

Write-Host "[SkyCert] Running ablation study ..." -ForegroundColor Cyan
python -m scripts.run_ablation   --config configs/default.yaml

Write-Host "[SkyCert] Running baseline comparison ..." -ForegroundColor Cyan
python -m scripts.run_baselines  --config configs/default.yaml

Write-Host "[SkyCert] Running extension experiments (lambda sweep, attack-strength sweep, failure cases, MLP) ..." -ForegroundColor Cyan
python -m scripts.run_extensions --config configs/default.yaml

Write-Host "[SkyCert] Running 5-seed aggregation (Tables I/II/III mean+-std) ..." -ForegroundColor Cyan
python -m scripts.run_multi_seed --config configs/default.yaml

Write-Host "[SkyCert] Running extended security evaluation (targeted T1, calibration poisoning, adaptive T3, gamma workload) ..." -ForegroundColor Cyan
python -m scripts.run_security_extensions --config configs/default.yaml

Write-Host "[SkyCert] Running trajectory-driven operational case study ..." -ForegroundColor Cyan
python -m scripts.run_case_study --config configs/default.yaml

Write-Host "[SkyCert] Replaying the distribution_shift audit log (hash pins + decision recomputation) ..." -ForegroundColor Cyan
python -m scripts.replay_audit outputs/audit/audit_distribution_shift.jsonl --show 1629

Write-Host "[SkyCert] Rendering paper figures ..." -ForegroundColor Cyan
python -m scripts.plot_results    --config configs/default.yaml
python -m scripts.plot_case_study

Write-Host "[SkyCert] Summary (see outputs/metrics.json for full data):" -ForegroundColor Cyan
$py = @'
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
'@
python -c $py

Write-Host "[SkyCert] Done. Artifacts are in .\outputs\" -ForegroundColor Green
