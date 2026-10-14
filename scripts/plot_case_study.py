"""Figures for the operational case study and the security extensions.

Outputs (TrueType-embedded PDFs, IEEE PDF eXpress compliant):

    outputs/figs/case_study_timeline.pdf
        Martingale (log10) and rolling coverage along the two-day stream
        for two seeds: one whose benign day lies inside the calibration
        weather envelope and one whose benign day lies outside it
        (harmless-shift alarm).
    outputs/figs/security_extensions.pdf
        (a) calibration poisoning: test coverage, sequestered coverage and
            the one-sided detection bound versus the poisoned fraction;
        (b) abstention budget: manual reviews versus residual CRITICAL
            misses per 1000 operations for several gamma.

Usage::

    python -m scripts.plot_case_study --seeds 4 3 --calib-window 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 6.5
import matplotlib.pyplot as plt
import numpy as np

from skycert.utils import ensure_dir


def _rolling(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x.astype(np.float64), 0, 0.0))
    out = np.full(x.shape[0], np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def plot_timeline(trace_dir: Path, seeds: list[int], w: int, variant: str,
                  threshold: float, out: Path) -> None:
    fig, axes = plt.subplots(len(seeds), 1, figsize=(3.5, 1.35 * len(seeds)), sharex=False)
    axes = np.atleast_1d(axes)
    cap = 8.0
    handles = None
    for ax, seed in zip(axes, seeds):
        d = np.load(trace_dir / f"trace_seed{seed}_w{w}_{variant}.npz")
        t_h = (d["t_min"] - d["t_min"].min()) / 60.0
        M = np.maximum(d["martingale"], 1e-2)
        logM = np.minimum(np.log10(M), cap)
        ax.plot(t_h, logM, color="tab:red", lw=0.9, label=r"$\log_{10} M_t$ (clipped at 8)")
        ax.axhline(np.log10(threshold), color="tab:red", ls="--", lw=0.7,
                   label=r"threshold $\tau{=}20$")
        ax.set_ylabel(r"$\log_{10} M_t$", color="tab:red")
        ax.set_ylim(-1.2, cap + 0.4)
        ax2 = ax.twinx()
        cov = _rolling(d["covered"], 150)
        crit = _rolling((d["y"] == 3).astype(float), 150)
        ax2.plot(t_h, cov, color="tab:blue", lw=0.9, label="rolling coverage (150 ops)")
        ax2.plot(t_h, crit, color="tab:green", lw=0.9, label="rolling CRITICAL fraction")
        ax2.axhline(0.9, color="tab:blue", ls=":", lw=0.7)
        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("cov. / CRIT frac.")
        sp = d["spoof_mask"]; fr = d["front_mask"]
        if sp.any():
            ax.axvspan(t_h[sp].min(), t_h[sp].max(), color="tab:orange", alpha=0.25, lw=0)
        if fr.any():
            ax.axvspan(t_h[fr].min(), t_h[fr].max(), color="tab:gray", alpha=0.25, lw=0)
        ax.set_title(f"seed {seed}: calibration window {w} d", fontsize=6.5, loc="left")
        ax.set_xlim(0, 48)
        ax.set_xticks(range(0, 49, 6))
        ax.grid(alpha=0.25)
        if handles is None:
            h1, _ = ax.get_legend_handles_labels()
            h2, _ = ax2.get_legend_handles_labels()
            handles = h1 + h2
    axes[-1].set_xlabel("hours since start of stream (day A: 0--24 h, day B: 24--48 h)")
    from matplotlib.patches import Patch
    handles = handles + [Patch(color="tab:orange", alpha=0.4, label="spoofing episode"),
                         Patch(color="tab:gray", alpha=0.4, label="weather front")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=5.5, frameon=False,
               bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_security(sec_json: Path, out: Path) -> None:
    agg = json.loads(sec_json.read_text())["aggregate"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.05))
    # (a) poisoning
    rhos = sorted(float(k) for k in agg["calibration_poisoning"]["rho"])
    P = agg["calibration_poisoning"]["rho"]
    cov = [P[str(r) if str(r) in P else f"{r}"]["coverage"]["mean"] for r in rhos]
    cov_s = [P[str(r)]["coverage"]["std"] for r in rhos]
    seq = [P[str(r)]["sequestered_check"]["coverage"]["mean"] for r in rhos]
    seq_s = [P[str(r)]["sequestered_check"]["coverage"]["std"] for r in rhos]
    lb = P[str(rhos[0])]["sequestered_check"]["lower_bound"]["mean"]
    abst = [P[str(r)]["abstain_rate"]["mean"] for r in rhos]
    a.errorbar(rhos, cov, yerr=cov_s, marker="o", ms=3, lw=1, color="tab:blue", label="test coverage")
    a.errorbar(rhos, seq, yerr=seq_s, marker="s", ms=3, lw=1, color="tab:purple", label="sequestered coverage")
    a.axhline(lb, color="tab:red", ls="--", lw=0.8, label=r"detection bound ($n_{\rm seq}{=}300$)")
    a.axhline(0.9, color="gray", ls=":", lw=0.7)
    a.set_xlabel(r"poisoned fraction $\rho$")
    a.set_ylabel("coverage")
    a2 = a.twinx()
    a2.plot(rhos, abst, marker="^", ms=3, lw=1, color="tab:green", label="abstention rate")
    a2.set_ylabel("abstention", color="tab:green")
    a2.set_ylim(0, 0.4)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, handlelength=1.5, columnspacing=0.8)
    a.set_title("(a) calibration poisoning", fontsize=6.5)
    a.grid(alpha=0.25)
    # (b) gamma
    G = agg["gamma_workload"]
    gammas = [k for k in G if float(k) <= 1.0]
    for stream, color, marker in (("clean", "tab:blue", "o"), ("distribution_shift", "tab:red", "s")):
        xs = [G[g][stream]["reviews_per_1000"]["mean"] for g in gammas]
        ys = [G[g][stream]["residual_critical_misses_per_1000"]["mean"] for g in gammas]
        ye = [G[g][stream]["residual_critical_misses_per_1000"]["std"] for g in gammas]
        b.errorbar(xs, ys, yerr=ye, marker=marker, ms=3, lw=1, color=color, label=stream.replace("_", " "))
        if stream == "clean":
            for g, x, y in zip(gammas, xs, ys):
                b.annotate(rf"$\gamma{{=}}{float(g):g}$", (x, y), textcoords="offset points",
                           xytext=(3, 2), fontsize=5, color=color)
        else:
            b.annotate("all $\\gamma$\n(alerts force\nabstention)", (xs[0], ys[0]),
                       textcoords="offset points", xytext=(-38, 10), fontsize=5, color=color)
    b.set_xlabel("manual reviews / 1000 ops")
    b.set_ylabel("residual CRIT misses / 1000 ops")
    b.set_title("(b) abstention budget", fontsize=6.5)
    b.grid(alpha=0.25)
    b.legend(fontsize=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", default="outputs/case_study_traces")
    p.add_argument("--security-json", default="outputs/security_extensions.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[4, 3])
    p.add_argument("--calib-window", type=int, default=7)
    p.add_argument("--variant", default="default")
    p.add_argument("--threshold", type=float, default=20.0)
    p.add_argument("--out-dir", default="outputs/figs")
    args = p.parse_args(argv)
    out = ensure_dir(Path(args.out_dir))
    plot_timeline(Path(args.trace_dir), args.seeds, args.calib_window, args.variant,
                  args.threshold, out / "case_study_timeline.pdf")
    if Path(args.security_json).exists():
        plot_security(Path(args.security_json), out / "security_extensions.pdf")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
