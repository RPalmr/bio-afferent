"""Unit tests for the validation metrics."""

import numpy as np
import pytest

from validation.metrics import (
    adaptation_index,
    aggregate_metrics,
    bias,
    bootstrap_ci,
    check_acceptance,
    cross_correlation_lag,
    cv_isi,
    fano_factor,
    isi_from_binary,
    isi_from_spike_times,
    ks_statistic,
    mae,
    nrmse,
    peak_latency,
    r2_score,
    rate_metrics,
    ridge_decode,
    rmse,
    spike_metrics,
    van_rossum_distance,
    victor_purpura_distance,
)


def test_basic_error_metrics():
    truth = np.array([0.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0])
    assert rmse(truth, pred) == pytest.approx(1.0)
    assert mae(truth, pred) == pytest.approx(1.0)
    assert bias(truth, pred) == pytest.approx(1.0)
    assert nrmse(truth, pred, "mean") == float("inf")
    assert r2_score(truth, pred) == pytest.approx(0.0)


def test_perfect_prediction():
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    assert r2_score(truth, truth) == pytest.approx(1.0)
    assert nrmse(truth, truth) == pytest.approx(0.0)
    assert rmse(truth, truth) == pytest.approx(0.0)


def test_peak_latency():
    assert peak_latency([0.0, 1.0, 3.0, 2.0], 0.1) == pytest.approx(0.2)


def test_adaptation_index():
    dt = 0.1
    t = np.arange(30) * dt
    y = np.where(t < 1.0, 0.0, np.where(t < 2.0, 10.0, 5.0))
    ai = adaptation_index(
        y,
        dt,
        baseline_window=(0.0, 1.0),
        peak_window=(1.0, 2.0),
        hold_window=(2.0, 3.0),
    )
    assert ai == pytest.approx(0.5)


def test_cross_correlation_lag_positive_for_delayed_signal():
    dt = 0.01
    reference = np.zeros(100)
    reference[40] = 1.0
    signal = np.zeros(100)
    signal[42] = 1.0
    assert cross_correlation_lag(reference, signal, dt) == pytest.approx(0.02)


def test_isi_functions():
    assert isi_from_spike_times([0.1, 0.3, 0.6]).tolist() == pytest.approx([0.2, 0.3])
    assert isi_from_binary([1, 0, 1, 0, 0, 1], 0.1).tolist() == pytest.approx([0.2, 0.3])
    assert isi_from_spike_times([]).size == 0


def test_fano_and_cv():
    assert fano_factor([1, 1, 1, 1]) == pytest.approx(0.0)
    assert fano_factor([0, 2, 0, 2]) == pytest.approx(4.0 / 3.0)
    assert cv_isi([0.1, 0.1, 0.1]) == pytest.approx(0.0)
    assert cv_isi([0.1, 0.2, 0.3]) > 0.0


def test_ks_statistic():
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert ks_statistic([1, 2, 3], [4, 5, 6]) == pytest.approx(1.0)


def test_spike_train_distances():
    assert van_rossum_distance([0.1, 0.2], [0.1, 0.2]) == pytest.approx(0.0)
    assert van_rossum_distance([0.1], [0.2]) > 0.0
    assert victor_purpura_distance([0.1, 0.2], [0.1, 0.2]) == pytest.approx(0.0)
    assert victor_purpura_distance([], [0.1, 0.2]) == pytest.approx(2.0)
    assert victor_purpura_distance([0.1], [0.2], q=1.0) == pytest.approx(0.1)


def test_rate_metrics_bundle_and_acceptance():
    dt = 0.01
    t = np.arange(200) * dt
    truth = 10.0 + 5.0 * np.sin(2.0 * np.pi * t)
    pred = truth + 0.5
    metrics = rate_metrics(truth, pred, dt)
    assert set(metrics) >= {"rmse", "nrmse_range", "r2", "lag_s", "abs_lag_s"}
    result = check_acceptance(metrics, {"nrmse_range": ("<=", 1.0), "r2": (">=", 0.9)})
    assert result["passed"] is True
    failing = check_acceptance(metrics, {"nrmse_range": ("<=", 0.0)})
    assert failing["passed"] is False


def test_spike_metrics_bundle():
    metrics = spike_metrics([0.1, 0.2, 0.3], [0.1, 0.21, 0.31], duration_s=1.0)
    assert metrics["n_spikes_true"] == 3.0
    assert metrics["rate_error_hz"] == pytest.approx(0.0)
    assert metrics["vp_distance"] > 0.0


def test_ridge_decode_recovers_linear_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=200)
    train = np.arange(150)
    test = np.arange(150, 200)
    result = ridge_decode(x, y, train, test, alpha=1e-6)
    assert result["r2"] > 0.9


def test_aggregate_and_bootstrap():
    aggregate = aggregate_metrics([{"a": 1.0}, {"a": 3.0}])
    assert aggregate["a"]["mean"] == pytest.approx(2.0)
    point, lower, upper = bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_boot=200, seed=0)
    assert point == pytest.approx(2.5)
    assert lower <= point <= upper


def test_input_validation():
    with pytest.raises(ValueError):
        rmse([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        rmse([np.nan], [1.0])
    with pytest.raises(ValueError):
        nrmse([1.0], [1.0], "bogus")
