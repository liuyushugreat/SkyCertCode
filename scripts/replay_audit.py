"""Offline replay of a SkyCert audit log (assurance objective O4).

Given a JSON-Lines audit log written by :class:`skycert.pipeline.SkyCertPipeline`,
this script *independently* reconstructs every decision from the raw
record contents and reports any divergence:

1. **Conformal set** -- recomputed from the logged probability vector and
   the header ``q_hat`` (APS or LAC score, plus top-1 augmentation).
2. **Martingale** -- recomputed from the logged per-step ``p_value``
   sequence with the header betting parameter ``epsilon``:
   ``M_t = prod_{i<=t} eps * p_i^(eps-1)``.
3. **Decision** -- recomputed by re-applying the two-parameter policy
   (``max_set_fraction``, ``escalate_on_martingale``, ``threshold``) to
   the reconstructed set size and martingale value.
4. **Calibration pin** -- when ``--calib-npz`` is given, the SHA-256 of
   the provided calibration arrays is compared with the header digest.

Usage:
    python -m scripts.replay_audit outputs/audit/audit_distribution_shift.jsonl
    python -m scripts.replay_audit LOG --calib-npz calib.npz --show 1629
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from skycert.assurance.audit import sha256_of_arrays
from skycert.assurance.conformal import ConformalRiskSet
from skycert.assurance.martingale import SimpleJumperBetting
from skycert.assurance.policy import AssurancePolicy


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") == "header":
                header = obj
            else:
                records.append(obj)
    if header is None:
        raise ValueError(
            "audit log has no header record; regenerate it with the current "
            "SkyCert version (scripts.run_experiment)"
        )
    return header, records


def replay(path: str | Path, calib_npz: str | Path | None = None) -> dict[str, Any]:
    header, records = _load(Path(path))

    conformal = ConformalRiskSet(
        alpha=header["alpha"], score=header["conformal_score"], include_top1=True
    )
    conformal.q_hat = float(header["q_hat"])
    conformal.num_classes = int(header["num_classes"])
    betting = SimpleJumperBetting(epsilon=float(header["martingale"]["epsilon"]))
    threshold = float(header["martingale"]["threshold"])
    policy = AssurancePolicy(
        num_classes=int(header["num_classes"]),
        max_set_fraction=float(header["policy"]["max_set_fraction"]),
        escalate_on_martingale=bool(header["policy"]["escalate_on_martingale"]),
    )

    m = 1.0
    mismatches: dict[str, list[int]] = {"set": [], "martingale": [], "decision": []}
    raw_empty = 0
    for rec in records:
        probs = np.asarray(rec["probs"], dtype=np.float64).reshape(1, -1)
        logged = rec["decision"]
        # (1) conformal set
        mask = conformal.predict_sets(probs)[0]
        rebuilt_set = [int(i) for i, on in enumerate(mask) if on]
        if rebuilt_set != [int(i) for i in logged["prediction_set"]]:
            mismatches["set"].append(rec["index"])
        if conformal.raw_empty_rate(probs) > 0:
            raw_empty += 1
        # (2) martingale from logged p-values
        m *= betting(float(rec["extra"]["p_value"]))
        if not np.isclose(m, float(logged["martingale"]), rtol=1e-6, atol=1e-9):
            mismatches["martingale"].append(rec["index"])
        # (3) decision
        d = policy.decide(
            probs_row=probs[0],
            set_mask=mask,
            martingale_value=m,
            martingale_alert=(m >= threshold),
        )
        if d.kind.value != logged["kind"]:
            mismatches["decision"].append(rec["index"])

    pin: dict[str, Any] = {"checked": False}
    if calib_npz is not None:
        data = np.load(calib_npz)
        digest = sha256_of_arrays(data["X_calib"], data["y_calib"])
        pin = {
            "checked": True,
            "header_sha256": header["calib_sha256"],
            "provided_sha256": digest,
            "match": digest == header["calib_sha256"],
        }

    return {
        "log": str(path),
        "n_records": len(records),
        "header": header,
        "mismatches": {k: len(v) for k, v in mismatches.items()},
        "mismatch_indices": {k: v[:20] for k, v in mismatches.items()},
        "raw_empty_sets_without_top1": raw_empty,
        "final_martingale": m,
        "calibration_pin": pin,
        "verified": all(len(v) == 0 for v in mismatches.values())
        and (pin.get("match", True)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay and verify a SkyCert audit log")
    parser.add_argument("log", help="path to a JSONL audit log")
    parser.add_argument("--calib-npz", default=None,
                        help="npz with X_calib / y_calib to verify the calibration pin")
    parser.add_argument("--show", type=int, default=None,
                        help="print the raw JSON of the record with this index")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    report = replay(args.log, args.calib_npz)
    if args.show is not None:
        _, records = _load(Path(args.log))
        for rec in records:
            if rec["index"] == args.show:
                print(json.dumps(rec, indent=2))
                break
    if args.json:
        print(json.dumps(report, indent=2))
        return
    h = report["header"]
    print(f"[replay] {report['log']}: {report['n_records']} decision records")
    print(f"[replay] q_hat={h['q_hat']:.4f} alpha={h['alpha']} "
          f"eps={h['martingale']['epsilon']} tau={h['martingale']['threshold']} "
          f"gamma={h['policy']['max_set_fraction']}")
    print(f"[replay] calib_sha256={h['calib_sha256'][:16]}... "
          f"rules_sha256={h['rules_sha256'][:16]}...")
    mm = report["mismatches"]
    print(f"[replay] mismatches: set={mm['set']} martingale={mm['martingale']} "
          f"decision={mm['decision']}")
    print(f"[replay] raw APS sets that would be empty without top-1 augmentation: "
          f"{report['raw_empty_sets_without_top1']}")
    if report["calibration_pin"]["checked"]:
        print(f"[replay] calibration pin match: {report['calibration_pin']['match']}")
    print(f"[replay] VERIFIED={report['verified']}")


if __name__ == "__main__":
    main()
