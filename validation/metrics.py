"""Quantitative metrics for model-to-data validation.

The functions are deliberately NumPy-only so they can score digitised
experimental traces, model outputs, or both without importing JAX. Every
metric is paired (truth, prediction) unless stated otherwise, and every
function validates shapes and finiteness up front.

Terminology follows the verification/validation protocol:

- rate metrics quantify continuous firing-rate agreement;
- spike metrics quantify point-process agreement;
- decoding metrics ask whether the information content is preserved;
- acceptance helpers turn pre-registered thresholds into pass/fail records.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _vector(name: str, value: ArrayLike, *, allow_empty: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    if arr.size == 0 and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _paired(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    truth = _vector("y_true", y_true)
    pred = _vector("y_pred", y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}.")
    return truth, pred


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator / denominator)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root-mean-square error."""
    truth, pred = _paired(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    truth, pred = _paired(y_true, y_pred)
    return float(np.mean(np.abs(pred - truth)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error (prediction minus truth)."""
    truth, pred = _paired(y_true, y_pred)
    return float(np.mean(pred - truth))


def nrmse(y_true: ArrayLike, y_pred: ArrayLike, normalization: str = "range") -> float:
    """Normalised RMSE.

    ``normalization`` is one of ``"range"`` (default; robust to offset),
    ``"mean"`` (absolute mean of the truth), or ``"std"``.
    """
    truth, pred = _paired(y_true, y_pred)
    error = rmse(truth, pred)
    if normalization == "range":
        denom = float(np.ptp(truth))
    elif normalization == "mean":
        denom = float(abs(np.mean(truth)))
    elif normalization == "std":
        denom = float(np.std(truth))
    else:
        raise ValueError("normalization must be 'range', 'mean', or 'std'.")
    return _safe_ratio(error, denom)


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination (1 is perfect, 0 is mean-predictor)."""
    truth, pred = _paired(y_true, y_pred)
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def pearson_r(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Pearson correlation; NaN when either signal is constant."""
    truth, pred = _paired(y_true, y_pred)
    if float(np.std(truth)) == 0.0 or float(np.std(pred)) == 0.0:
        return float("nan")
    return float(np.corrcoef(truth, pred)[0, 1])


def window_mean(y: ArrayLike, dt: float, start_s: float, end_s: float) -> float:
    """Mean of samples whose time lies in ``[start_s, end_s)``."""
    arr = _vector("y", y)
    if not dt > 0.0:
        raise ValueError("dt must be positive.")
    if end_s <= start_s:
        raise ValueError("end_s must exceed start_s.")
    times = np.arange(arr.size) * dt
    mask = (times >= start_s) & (times < end_s)
    if not np.any(mask):
        raise ValueError(f"No samples in window [{start_s}, {end_s}) with dt={dt}.")
    return float(np.mean(arr[mask]))


def peak_amplitude(y: ArrayLike, baseline: float | None = None) -> float:
    """Peak value, optionally relative to a supplied baseline."""
    arr = _vector("y", y)
    base = 0.0 if baseline is None else float(baseline)
    return float(np.max(arr) - base)


def peak_latency(y: ArrayLike, dt: float) -> float:
    """Time of the maximum in seconds (first occurrence)."""
    arr = _vector("y", y)
    if not dt > 0.0:
        raise ValueError("dt must be positive.")
    return float(np.argmax(arr) * dt)


def adaptation_index(
    y: ArrayLike,
    dt: float,
    *,
    baseline_window: tuple[float, float],
    peak_window: tuple[float, float],
    hold_window: tuple[float, float],
) -> float:
    """Adaptation index ``(peak - hold) / (peak - baseline)``.

    Values near 0 indicate a tonic response; values near 1 indicate complete
    adaptation from the peak to the hold level. Returns 0 when peak equals
    baseline (no response).
    """
    arr = _vector("y", y)
    base = window_mean(arr, dt, *baseline_window)
    peak = float(np.max(arr[_window_mask(arr.size, dt, *peak_window)]))
    hold = window_mean(arr, dt, *hold_window)
    denom = peak - base
    if denom == 0.0:
        return 0.0
    return float((peak - hold) / denom)


def _window_mask(n: int, dt: float, start_s: float, end_s: float) -> np.ndarray:
    if not dt > 0.0:
        raise ValueError("dt must be positive.")
    if end_s <= start_s:
        raise ValueError("end_s must exceed start_s.")
    times = np.arange(n) * dt
    mask = (times >= start_s) & (times < end_s)
    if not np.any(mask):
        raise ValueError(f"No samples in window [{start_s}, {end_s}) with dt={dt}.")
    return mask


def cross_correlation_lag(
    reference: ArrayLike,
    signal: ArrayLike,
    dt: float,
    *,
    max_lag_s: float | None = None,
) -> float:
    """Lag of ``signal`` relative to ``reference`` at maximum correlation.

    A positive return value means the signal lags the reference. Both signals
    are mean-centred before correlation.
    """
    ref, sig = _paired(reference, signal)
    if not dt > 0.0:
        raise ValueError("dt must be positive.")
    ref = ref - np.mean(ref)
    sig = sig - np.mean(sig)
    corr = np.correlate(sig, ref, mode="full")
    lags = np.arange(-ref.size + 1, ref.size)
    if max_lag_s is not None:
        keep = np.abs(lags) * dt <= max_lag_s
        if not np.any(keep):
            raise ValueError("max_lag_s excludes every lag.")
        corr = corr[keep]
        lags = lags[keep]
    return float(lags[int(np.argmax(corr))] * dt)


def isi_from_spike_times(times_s: ArrayLike) -> np.ndarray:
    """Inter-spike intervals from sorted spike times."""
    times = _vector("times_s", times_s, allow_empty=True)
    if times.size < 2:
        return np.empty(0, dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("spike times must be strictly increasing.")
    return np.diff(times)


def isi_from_binary(spikes: ArrayLike, dt: float) -> np.ndarray:
    """Inter-spike intervals from a boolean/0-1 spike train."""
    arr = np.asarray(spikes)
    if arr.ndim != 1:
        raise ValueError(f"spikes must be one-dimensional, got shape {arr.shape}.")
    if not dt > 0.0:
        raise ValueError("dt must be positive.")
    indices = np.flatnonzero(arr.astype(bool))
    if indices.size < 2:
        return np.empty(0, dtype=float)
    return np.diff(indices) * dt


def fano_factor(counts: ArrayLike) -> float:
    """Fano factor (variance / mean) of spike counts across trials/windows."""
    arr = _vector("counts", counts)
    mean = float(np.mean(arr))
    if mean == 0.0:
        return float("nan")
    return float(np.var(arr, ddof=1) / mean) if arr.size > 1 else float("nan")


def cv_isi(intervals: ArrayLike) -> float:
    """Coefficient of variation of inter-spike intervals."""
    arr = _vector("intervals", intervals, allow_empty=True)
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if mean == 0.0:
        return float("nan")
    return float(np.std(arr, ddof=1) / mean) if arr.size > 1 else float("nan")


def ks_statistic(sample_a: ArrayLike, sample_b: ArrayLike) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no SciPy dependency)."""
    a = np.sort(_vector("sample_a", sample_a, allow_empty=True))
    b = np.sort(_vector("sample_b", sample_b, allow_empty=True))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def van_rossum_distance(
    times_a: ArrayLike,
    times_b: ArrayLike,
    *,
    tau: float = 0.01,
    normalize: bool = True,
) -> float:
    """Van Rossum spike-train distance with an exponential kernel.

    ``D^2 = sum k(a_i, a_j) + sum k(b_i, b_j) - 2 sum k(a_i, b_j)`` with
    ``k(t) = exp(-|t|/tau)/tau``. When ``normalize`` is true the distance is
    divided by ``sqrt(0.5 * (K(a,a) + K(b,b)))`` so it is bounded in ``[0, 2]``
    for non-identical trains (0 for identical trains).
    """
    a = _vector("times_a", times_a, allow_empty=True)
    b = _vector("times_b", times_b, allow_empty=True)
    if not tau > 0.0:
        raise ValueError("tau must be positive.")
    if a.size == 0 and b.size == 0:
        return 0.0

    def kernel(x: np.ndarray, y: np.ndarray) -> float:
        if x.size == 0 or y.size == 0:
            return 0.0
        return float(np.sum(np.exp(-np.abs(x[:, None] - y[None, :]) / tau)) / tau)

    self_a = kernel(a, a)
    self_b = kernel(b, b)
    cross = kernel(a, b)
    d2 = max(self_a + self_b - 2.0 * cross, 0.0)
    if not normalize:
        return float(np.sqrt(d2))
    denom = np.sqrt(max(0.5 * (self_a + self_b), 1e-12))
    return float(np.sqrt(d2) / denom)


def victor_purpura_distance(times_a: ArrayLike, times_b: ArrayLike, *, q: float = 1.0) -> float:
    """Victor-Purpura spike-train distance with cost ``q * |dt|`` per move.

    Insertion/deletion costs 1. The dynamic program is ``O(n*m)`` and is exact
    for the standard VP metric.
    """
    a = _vector("times_a", times_a, allow_empty=True)
    b = _vector("times_b", times_b, allow_empty=True)
    if not q >= 0.0:
        raise ValueError("q must be non-negative.")
    if a.size == 0:
        return float(b.size)
    if b.size == 0:
        return float(a.size)
    n, m = a.size, b.size
    table = np.zeros((n + 1, m + 1), dtype=float)
    table[:, 0] = np.arange(n + 1)
    table[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i, j] = min(
                table[i - 1, j] + 1.0,
                table[i, j - 1] + 1.0,
                table[i - 1, j - 1] + q * abs(float(a[i - 1]) - float(b[j - 1])),
            )
    return float(table[n, m])


def rate_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    dt: float,
    *,
    baseline_s: float = 0.2,
    adaptation_windows: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    max_lag_s: float | None = None,
) -> dict[str, float]:
    """Standard continuous-rate agreement bundle for one trial/channel."""
    truth, pred = _paired(y_true, y_pred)
    baseline = float(np.mean(truth[: max(1, int(round(baseline_s / dt)))])) if baseline_s > 0 else 0.0
    metrics: dict[str, float] = {
        "rmse": rmse(truth, pred),
        "mae": mae(truth, pred),
        "bias": bias(truth, pred),
        "nrmse_range": nrmse(truth, pred, "range"),
        "nrmse_mean": nrmse(truth, pred, "mean"),
        "r2": r2_score(truth, pred),
        "pearson_r": pearson_r(truth, pred),
        "peak_amplitude_true": peak_amplitude(truth, baseline),
        "peak_amplitude_pred": peak_amplitude(pred, baseline),
        "peak_amplitude_error": peak_amplitude(pred, baseline) - peak_amplitude(truth, baseline),
        "peak_latency_true_s": peak_latency(truth, dt),
        "peak_latency_pred_s": peak_latency(pred, dt),
        "peak_latency_error_s": peak_latency(pred, dt) - peak_latency(truth, dt),
        "lag_s": cross_correlation_lag(truth, pred, dt, max_lag_s=max_lag_s),
    }
    metrics["abs_lag_s"] = abs(metrics["lag_s"])
    if adaptation_windows is not None:
        baseline_window, peak_window, hold_window = adaptation_windows
        metrics["adaptation_index_true"] = adaptation_index(
            truth, dt, baseline_window=baseline_window, peak_window=peak_window, hold_window=hold_window
        )
        metrics["adaptation_index_pred"] = adaptation_index(
            pred, dt, baseline_window=baseline_window, peak_window=peak_window, hold_window=hold_window
        )
        metrics["adaptation_index_error"] = metrics["adaptation_index_pred"] - metrics["adaptation_index_true"]
    return metrics


def spike_metrics(
    times_true: ArrayLike,
    times_pred: ArrayLike,
    *,
    tau: float = 0.01,
    q: float = 1.0,
    duration_s: float | None = None,
) -> dict[str, float]:
    """Point-process agreement bundle for one afferent channel."""
    truth = _vector("times_true", times_true, allow_empty=True)
    pred = _vector("times_pred", times_pred, allow_empty=True)
    isi_true = isi_from_spike_times(truth)
    isi_pred = isi_from_spike_times(pred)
    out = {
        "vp_distance": victor_purpura_distance(truth, pred, q=q),
        "van_rossum_distance": van_rossum_distance(truth, pred, tau=tau),
        "ks_isi": ks_statistic(isi_true, isi_pred),
        "cv_isi_true": cv_isi(isi_true),
        "cv_isi_pred": cv_isi(isi_pred),
        "n_spikes_true": float(truth.size),
        "n_spikes_pred": float(pred.size),
    }
    if duration_s is not None:
        if not duration_s > 0.0:
            raise ValueError("duration_s must be positive.")
        out["rate_true_hz"] = float(truth.size / duration_s)
        out["rate_pred_hz"] = float(pred.size / duration_s)
        out["rate_error_hz"] = out["rate_pred_hz"] - out["rate_true_hz"]
    return out


def ridge_decode(
    features: np.ndarray,
    targets: ArrayLike,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    alpha: float = 1e-3,
) -> dict[str, float]:
    """Ridge-regression decoding of a scalar variable from population rates.

    Features are standardised using training statistics only. Returns held-out
    R^2 and NRMSE (range-normalised) so decoding can be compared across
    encodings.
    """
    x = np.asarray(features, dtype=float)
    y = _vector("targets", targets)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {x.shape}.")
    if x.shape[0] != y.size:
        raise ValueError("features and targets must have the same number of samples.")
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("train_idx and test_idx must be non-empty.")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("train_idx and test_idx overlap.")
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    x_train = np.column_stack([np.ones(x_train.shape[0]), x_train])
    x_test = np.column_stack([np.ones(x_test.shape[0]), x_test])
    gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
    gram[0, 0] -= alpha  # do not regularise the intercept
    coef = np.linalg.solve(gram, x_train.T @ y_train)
    pred = x_test @ coef
    return {
        "r2": r2_score(y_test, pred),
        "nrmse_range": nrmse(y_test, pred, "range"),
        "rmse": rmse(y_test, pred),
    }


def bootstrap_ci(
    values: ArrayLike,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval.

    Returns ``(point_estimate, lower, upper)``.
    """
    arr = _vector("values", values)
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n_boot, arr.size), replace=True)
    stats = np.apply_along_axis(statistic, 1, draws)
    return float(statistic(arr)), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def check_acceptance(
    metrics: Mapping[str, float],
    criteria: Mapping[str, Mapping[str, Any] | tuple[str, float]],
) -> dict[str, Any]:
    """Apply pre-registered thresholds to a metrics mapping.

    ``criteria`` maps metric name to either ``{"op": "<=", "threshold": 0.15}``
    or a ``(op, threshold)`` tuple. Supported operators are ``<``, ``<=``,
    ``>``, ``>=``, ``==``. Returns a JSON-serialisable record with one entry per
    criterion and an overall ``passed`` flag.
    """
    ops: dict[str, Callable[[float, float], bool]] = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
    }
    checks: dict[str, dict[str, Any]] = {}
    for name, rule in criteria.items():
        if name not in metrics:
            raise KeyError(f"Metric {name!r} is not present in the metrics mapping.")
        if isinstance(rule, tuple):
            op, threshold = rule
        else:
            op = str(rule["op"])
            threshold = float(rule["threshold"])
        if op not in ops:
            raise ValueError(f"Unsupported operator {op!r}.")
        value = float(metrics[name])
        passed = bool(ops[op](value, float(threshold)))
        checks[name] = {
            "value": value,
            "op": op,
            "threshold": float(threshold),
            "passed": passed,
        }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def aggregate_metrics(records: Sequence[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-trial metric records into mean/std/n fields."""
    if not records:
        raise ValueError("records must be non-empty.")
    keys = sorted(set().union(*(record.keys() for record in records)))
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([record[key] for record in records if key in record], dtype=float)
        finite = values[np.isfinite(values)]
        out[key] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
            "n": float(finite.size),
        }
    return out
