"""Trajectory-driven UAM case-study generator.

Unlike :mod:`skycert.data.synthetic` (i.i.d. feature draws with a latent
linear label), this module simulates a small vertiport network with
*kinematics and time*: scheduled eVTOL flights, a diurnal demand profile,
an autoregressive wind/weather process, battery discharge as a function
of route length and head-wind, traffic density and vertiport congestion
computed from the aircraft that are actually airborne, and noise
exposure computed from the ground cells each route overflies.  The risk
label is derived from *physical* quantities of the simulated flight
(minimum separation to other traffic, battery reserve on arrival,
weather exceedance, congestion), not from the feature vector.

The resulting operation stream is time-ordered and auto-correlated, which
is the realistic regime in which the exchangeability assumption behind
conformal calibration is actually stressed.  Two events are scripted on
the streaming days:

* a **GPS/ADS-B spoofing episode** (threat T3) that corrupts the
  surveillance- and weather-derived features of every operation filed
  during a two-hour window, without changing the physical risk; and
* a **weather front** (threat T4) that raises wind and precipitation
  for the rest of the horizon and *does* change the physical risk.

The feature schema is identical to :mod:`synthetic` (16 features, same
semantics for f0--f6) so that the same reasoner, rule set and SkyCert
configuration can be used unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AREA_KM = 24.0
CRUISE_KMH = 160.0
NOMINAL_RANGE_KM = 60.0
MIN_PER_DAY = 24 * 60

_VERTIPORTS_KM = np.array([
    [3.0, 3.0], [12.0, 2.5], [21.0, 4.0], [4.0, 12.5],
    [13.0, 12.0], [20.5, 13.5], [6.0, 21.0], [17.0, 20.0],
])
_VERTIPORT_POPULARITY = np.array([1.4, 1.0, 0.8, 1.2, 1.8, 0.9, 0.7, 1.0])


@dataclass
class TrajectoryCaseStudy:
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_stream: np.ndarray          # features as *seen by the reasoner* (spoofed)
    X_stream_true: np.ndarray     # physical ground-truth features
    y_stream: np.ndarray
    t_stream_min: np.ndarray      # departure time (minutes since start)
    t_calib_min: np.ndarray       # departure time of calibration operations
    spoof_mask: np.ndarray        # True on spoofed stream operations
    front_mask: np.ndarray        # True on operations filed after the front
    spoof_start: int              # first stream index inside the spoof window
    spoof_end: int                # first stream index after the spoof window
    front_start: int              # first stream index after front onset
    num_classes: int = 4
    meta: dict = field(default_factory=dict)


def _demand_profile(hour: np.ndarray) -> np.ndarray:
    """Bimodal diurnal demand (relative flights per minute)."""
    morning = np.exp(-0.5 * ((hour - 8.5) / 1.6) ** 2)
    evening = np.exp(-0.5 * ((hour - 18.0) / 1.9) ** 2)
    midday = 0.45 * np.exp(-0.5 * ((hour - 13.0) / 3.0) ** 2)
    night = 0.05
    return night + morning + 0.9 * evening + midday


def _ar1(n: int, phi: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n)
    out[0] = rng.normal(0.0, sd / np.sqrt(1 - phi ** 2))
    innov = rng.normal(0.0, sd, size=n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innov[i]
    return out


def make_trajectory_case_study(
    *,
    num_days: int = 6,
    flights_per_day: int = 1200,
    train_days: tuple[int, int] = (0, 2),
    calib_days: tuple[int, int] = (2, 4),
    stream_days: tuple[int, int] = (4, 6),
    spoof_day: int = 4,
    spoof_hours: tuple[float, float] = (9.0, 11.0),
    spoof_radius: float = 0.12,
    front_day: int = 5,
    front_hour: float = 13.0,
    front_strength: float = 1.0,
    class_prior: tuple[float, ...] = (0.55, 0.25, 0.15, 0.05),
    seed: int = 0,
) -> TrajectoryCaseStudy:
    rng = np.random.default_rng(seed)
    n_minutes = num_days * MIN_PER_DAY
    n_hours = num_days * 24

    # ------------------------------------------------------------------
    # 1. Weather processes (hourly, AR(1)), with a scripted front.
    # ------------------------------------------------------------------
    wind = 6.0 + _ar1(n_hours, phi=0.85, sd=1.6, rng=rng)           # m/s
    wind_dir = np.cumsum(rng.normal(0.0, 0.25, size=n_hours)) % (2 * np.pi)
    precip = 0.25 + _ar1(n_hours, phi=0.9, sd=0.08, rng=rng)           # [0,1]
    front_t = front_day * 24 + front_hour
    hours = np.arange(n_hours)
    ramp = np.clip((hours - front_t + 1.0) / 1.5, 0.0, 1.0)            # 1.5 h ramp
    wind = wind + front_strength * 9.0 * ramp
    precip = precip + front_strength * 0.35 * ramp
    wind = np.clip(wind, 0.0, 28.0)
    precip = np.clip(precip, 0.0, 1.0)
    severity = np.clip(0.55 * wind / 22.0 + 0.45 * precip, 0.0, 1.0)
    temp = 18.0 + 6.0 * np.sin((hours % 24 - 9.0) / 24.0 * 2 * np.pi) \
        + _ar1(n_hours, 0.9, 0.6, rng)
    humidity = np.clip(0.5 + 0.4 * (precip - 0.25) + _ar1(n_hours, 0.9, 0.05, rng), 0, 1)

    # Residential ground cells for noise exposure (12 x 12 grid).
    cells = rng.random((12, 12)) < 0.45

    # ------------------------------------------------------------------
    # 2. Schedule flights from the diurnal demand profile.
    # ------------------------------------------------------------------
    minute = np.arange(n_minutes)
    hod = (minute % MIN_PER_DAY) / 60.0
    prof = _demand_profile(hod)
    rate = prof / prof[:MIN_PER_DAY].sum() * flights_per_day
    counts = rng.poisson(rate)
    dep_min = np.repeat(minute, counts).astype(np.float64)
    dep_min += rng.uniform(0.0, 1.0, size=dep_min.shape[0])
    n = dep_min.shape[0]
    order = np.argsort(dep_min)
    dep_min = dep_min[order]

    pop = _VERTIPORT_POPULARITY / _VERTIPORT_POPULARITY.sum()
    origin = rng.choice(8, size=n, p=pop)
    dest = rng.choice(8, size=n, p=pop)
    same = dest == origin
    while same.any():
        dest[same] = rng.choice(8, size=int(same.sum()), p=pop)
        same = dest == origin
    p0 = _VERTIPORTS_KM[origin]
    p1 = _VERTIPORTS_KM[dest]
    length_km = np.linalg.norm(p1 - p0, axis=1)
    duration_min = length_km / CRUISE_KMH * 60.0 + 3.0        # +3 min hover/transition
    arr_min = dep_min + duration_min
    heading = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])

    hour_idx = np.clip((dep_min // 60).astype(int), 0, n_hours - 1)
    w = wind[hour_idx]
    headwind = w * np.cos(wind_dir[hour_idx] - heading)     # + = headwind
    soc_dep = rng.beta(5.0, 2.0, size=n)
    payload = rng.beta(2.0, 2.0, size=n)
    consumption = (length_km / NOMINAL_RANGE_KM) * (
        1.0 + 0.035 * np.maximum(headwind, 0.0) + 0.012 * w + 0.15 * payload
    ) + 0.04
    reserve = soc_dep - consumption

    # ------------------------------------------------------------------
    # 3. Traffic-derived quantities from actually airborne aircraft.
    # ------------------------------------------------------------------
    def position(i: int, t: float) -> np.ndarray:
        frac = np.clip((t - dep_min[i]) / duration_min[i], 0.0, 1.0)
        return p0[i] + frac * (p1[i] - p0[i])

    density = np.zeros(n)
    congestion = np.zeros(n)
    min_sep = np.full(n, 10.0)
    conflicts = np.zeros(n)      # filed-plan conflicts (planned CPA < 1.5 km)
    mid = 0.5 * (p0 + p1)
    # sliding window over departures sorted in time
    j_lo = 0
    for i in range(n):
        t = dep_min[i]
        while j_lo < n and arr_min[j_lo] < t - 20.0:
            j_lo += 1
        # candidates: flights airborne during [dep_i, arr_i]
        j = j_lo
        airborne_now: list[int] = []
        overlapping: list[int] = []
        while j < n and dep_min[j] <= arr_min[i]:
            if j != i:
                if dep_min[j] <= t <= arr_min[j]:
                    airborne_now.append(j)
                if arr_min[j] >= t:
                    overlapping.append(j)
            j += 1
        if airborne_now:
            pos = np.array([position(k, t) for k in airborne_now])
            density[i] = np.sum(np.linalg.norm(pos - mid[i], axis=1) < 6.0)
        if overlapping:
            ts = np.arange(t, arr_min[i] + 1e-9, 0.5)
            mine = np.array([position(i, tt) for tt in ts])
            best = 10.0
            for k in overlapping:
                lo, hi = max(t, dep_min[k]), min(arr_min[i], arr_min[k])
                if hi <= lo:
                    continue
                sel = (ts >= lo) & (ts <= hi)
                if not sel.any():
                    continue
                other = np.array([position(k, tt) for tt in ts[sel]])
                dist = np.linalg.norm(mine[sel] - other, axis=1)
                dmin = float(dist.min())
                best = min(best, dmin)
                if dmin < 1.5:
                    conflicts[i] += 1
            min_sep[i] = best
        # arrivals at my destination within +-5 min of my arrival
        lo_t, hi_t = arr_min[i] - 5.0, arr_min[i] + 5.0
        k0 = np.searchsorted(dep_min, lo_t - 30.0)
        k1 = np.searchsorted(dep_min, hi_t)
        seg = slice(k0, k1)
        congestion[i] = np.sum(
            (dest[seg] == dest[i]) & (arr_min[seg] >= lo_t) & (arr_min[seg] <= hi_t)
        ) - 1

    # noise exposure: fraction of route samples over residential cells
    noise = np.zeros(n)
    for i in range(n):
        k = max(int(length_km[i] / 0.5), 2)
        pts = p0[i] + np.linspace(0, 1, k)[:, None] * (p1[i] - p0[i])
        cx = np.clip((pts[:, 0] / AREA_KM * 12).astype(int), 0, 11)
        cy = np.clip((pts[:, 1] / AREA_KM * 12).astype(int), 0, 11)
        noise[i] = cells[cx, cy].mean()

    # ------------------------------------------------------------------
    # 4. Feature matrix (same schema as the synthetic benchmark).
    # ------------------------------------------------------------------
    hod_f = (dep_min % MIN_PER_DAY) / 60.0
    X = np.zeros((n, 16))
    X[:, 0] = severity[hour_idx] + rng.normal(0, 0.03, n)
    X[:, 1] = density / 8.0 + rng.normal(0, 0.03, n)
    X[:, 2] = congestion / 6.0 + rng.normal(0, 0.03, n)
    X[:, 3] = soc_dep
    X[:, 4] = noise + rng.normal(0, 0.02, n)
    X[:, 5] = 0.5 * (np.sin(hod_f / 24 * 2 * np.pi) + 1)
    X[:, 6] = 0.5 * (np.cos(hod_f / 24 * 2 * np.pi) + 1)
    X[:, 7] = w / 28.0 + rng.normal(0, 0.03, n)
    X[:, 8] = (headwind + 28.0) / 56.0 + rng.normal(0, 0.03, n)
    X[:, 9] = length_km / (AREA_KM * np.sqrt(2))
    X[:, 10] = precip[hour_idx] + rng.normal(0, 0.03, n)
    X[:, 11] = payload
    # Strategic-deconfliction proxy available at filing time: number of
    # already-filed plans whose closest point of approach to this route is
    # below 1.5 km (noisy, because plans are executed with timing jitter).
    X[:, 12] = conflicts / 4.0 + rng.normal(0, 0.05, n)
    X[:, 13] = (temp[hour_idx] - 5.0) / 30.0 + rng.normal(0, 0.02, n)
    X[:, 14] = humidity[hour_idx] + rng.normal(0, 0.03, n)
    X[:, 15] = rng.beta(2.0, 2.0, n)                       # planned vertical rate
    X = np.clip(X, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 5. Physical risk score -> ordinal label (thresholds fixed on train days).
    # ------------------------------------------------------------------
    risk = (
        1.1 * np.clip((1.2 - min_sep) / 1.2, 0.0, 1.0)      # loss-of-separation hazard
        + 1.0 * np.clip((0.18 - reserve) / 0.18, 0.0, 1.0)  # battery reserve shortfall
        + 0.9 * severity[hour_idx]
        + 0.35 * np.clip(congestion / 6.0, 0, 1)
        + 0.3 * np.clip(density / 8.0, 0, 1)
        + 0.25 * noise
        + 0.15 * payload
        + rng.normal(0.0, 0.08, n)
    )
    day_of = (dep_min // MIN_PER_DAY).astype(int)
    train_sel = (day_of >= train_days[0]) & (day_of < train_days[1])
    cum = np.cumsum(np.asarray(class_prior) / np.sum(class_prior))[:-1]
    thresholds = np.quantile(risk[train_sel], cum)
    y = np.digitize(risk, thresholds)

    calib_sel = (day_of >= calib_days[0]) & (day_of < calib_days[1])
    stream_sel = (day_of >= stream_days[0]) & (day_of < stream_days[1])
    X_stream_true = X[stream_sel].copy()
    y_stream = y[stream_sel]
    t_stream = dep_min[stream_sel]

    # ------------------------------------------------------------------
    # 6. Scripted events on the stream.
    # ------------------------------------------------------------------
    hod_s = (t_stream % MIN_PER_DAY) / 60.0
    day_s = (t_stream // MIN_PER_DAY).astype(int)
    spoof_mask = (day_s == spoof_day) & (hod_s >= spoof_hours[0]) & (hod_s < spoof_hours[1])
    front_mask = (t_stream / 60.0) >= front_t

    X_stream = X_stream_true.copy()
    srng = np.random.default_rng(seed + 7919)
    # Spoofed surveillance feed: ghost traffic & congestion, corrupted wind
    # and head-wind telemetry, bounded by ``spoof_radius`` in l_inf.
    k = int(spoof_mask.sum())
    if k:
        pert = np.zeros((k, 16))
        pert[:, 1] = srng.uniform(0.0, spoof_radius, k)          # ghost aircraft
        pert[:, 2] = srng.uniform(0.0, spoof_radius, k)
        pert[:, 7] = srng.uniform(-spoof_radius, spoof_radius, k)
        pert[:, 8] = srng.uniform(-spoof_radius, spoof_radius, k)
        pert[:, 0] = srng.uniform(-spoof_radius, 0.0, k)         # under-reported weather
        X_stream[spoof_mask] = np.clip(X_stream_true[spoof_mask] + pert, 0.0, 1.0)

    idx = np.arange(X_stream.shape[0])
    spoof_idx = idx[spoof_mask]
    front_idx = idx[front_mask]
    meta = {
        "num_flights": int(n),
        "flights_per_day": flights_per_day,
        "num_days": num_days,
        "thresholds": thresholds.tolist(),
        "class_prior_train": np.bincount(y[train_sel], minlength=4).tolist(),
        "class_counts_stream": np.bincount(y_stream, minlength=4).tolist(),
        "class_counts_front": np.bincount(y_stream[front_mask], minlength=4).tolist(),
        "spoof_window_ops": int(k),
        "front_ops": int(front_mask.sum()),
        "mean_min_sep_km": float(min_sep.mean()),
        "mean_reserve": float(reserve.mean()),
    }
    return TrajectoryCaseStudy(
        X_train=X[train_sel], y_train=y[train_sel],
        X_calib=X[calib_sel], y_calib=y[calib_sel],
        X_stream=X_stream, X_stream_true=X_stream_true, y_stream=y_stream,
        t_stream_min=t_stream, t_calib_min=dep_min[calib_sel],
        spoof_mask=spoof_mask, front_mask=front_mask,
        spoof_start=int(spoof_idx[0]) if k else -1,
        spoof_end=int(spoof_idx[-1]) + 1 if k else -1,
        front_start=int(front_idx[0]) if front_idx.size else -1,
        num_classes=4, meta=meta,
    )
