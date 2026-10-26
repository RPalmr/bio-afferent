"""Parameter calibration and held-out evaluation.

The calibration harness fits only explicitly selected parameters, on the
calibration split, using the independent NumPy reference implementation. The
fitted configuration is then evaluated unchanged on the validation split. No
validation trial is ever used in the objective.

This is deliberately conservative: published constants should be frozen and
cited; only genuinely free/engineering parameters should appear in
``fields``. The default field lists exclude published rise times and numerical
settings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bioafferent.models.gto_parameters import GTOConfig
from bioafferent.models.spindle_parameters import SpindleConfig
from validation.io import Trial, dataset_hash, discover_datasets
from validation.metrics import aggregate_metrics, rate_metrics
from validation.reference import GTOReference, SpindleReference
from validation.splits import trials_for_split

SPINDLE_FIT_FIELDS = (
    "chain_static_scale",
    "stiffness_rest",
    "stiffness_gain",
    "damping_rest",
    "damping_gain",
    "tau_fiber_rest",
    "tau_fiber_gain",
    "length_threshold",
    "prestretch_gain",
    "receptor_gain",
    "receptor_exponent",
    "receptor_threshold",
    "ia_weights",
    "occlusion",
    "baseline_ia",
    "baseline_ii",
)
GTO_FIT_FIELDS = ("tau_force", "threshold", "gain", "saturation_force", "baseline")

_POSITIVE_FIELDS = {
    "stiffness_rest",
    "damping_rest",
    "tau_fiber_rest",
    "receptor_gain",
    "receptor_exponent",
    "tau_force",
    "saturation_force",
}


@dataclass(frozen=True)
class FitResult:
    """Outcome of a calibration run, with the exact data hash it was fitted to."""

    model: str
    config: dict[str, Any]
    fields: tuple[str, ...]
    params: np.ndarray
    cost: float
    success: bool
    message: str
    nfev: int
    data_hash: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "config": self.config,
            "fields": list(self.fields),
            "params": self.params.tolist(),
            "cost": self.cost,
            "success": self.success,
            "message": self.message,
            "nfev": self.nfev,
            "data_hash": self.data_hash,
            "metrics": self.metrics,
        }


def flatten_config(config: SpindleConfig | GTOConfig, fields: Sequence[str]) -> np.ndarray:
    """Flatten selected dataclass fields into a parameter vector."""
    values: list[float] = []
    for name in fields:
        value = getattr(config, name)
        if isinstance(value, tuple):
            values.extend(float(item) for item in value)
        else:
            values.append(float(value))
    return np.asarray(values, dtype=float)


def apply_params(
    config: SpindleConfig | GTOConfig,
    fields: Sequence[str],
    params: Sequence[float],
) -> SpindleConfig | GTOConfig:
    """Rebuild a config with selected fields replaced by a parameter vector."""
    vector = np.asarray(params, dtype=float)
    if vector.ndim != 1:
        raise ValueError("params must be one-dimensional.")
    updates: dict[str, Any] = {}
    cursor = 0
    for name in fields:
        current = getattr(config, name)
        if isinstance(current, tuple):
            size = len(current)
            updates[name] = tuple(float(item) for item in vector[cursor : cursor + size])
            cursor += size
        else:
            updates[name] = float(vector[cursor])
            cursor += 1
    if cursor != vector.size:
        raise ValueError(f"Parameter vector has {vector.size} values but fields require {cursor}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Parameter vector contains non-finite values.")
    return dataclasses.replace(config, **updates)


def default_bounds(
    config: SpindleConfig | GTOConfig,
    fields: Sequence[str],
    *,
    factor: float = 5.0,
) -> list[tuple[float, float]]:
    """Wide but physiologically plausible bounds derived from the defaults."""
    reference = type(config)()
    bounds: list[tuple[float, float]] = []
    for name in fields:
        items = getattr(config, name)
        reference_items = getattr(reference, name)
        items = items if isinstance(items, tuple) else (items,)
        reference_items = reference_items if isinstance(reference_items, tuple) else (reference_items,)
        for _item, reference_item in zip(items, reference_items, strict=True):
            default = float(reference_item) if float(reference_item) > 0.0 else 1.0
            lower = 1e-6 if name in _POSITIVE_FIELDS else 0.0
            upper = max(default * factor, lower * 10.0, 1e-6)
            if name == "chain_static_scale":
                upper = 1.0
            bounds.append((lower, upper))
    return bounds


def _trial_inputs(trial: Trial, *, model: str) -> tuple[np.ndarray, ...]:
    if model == "spindle":
        length = trial.inputs.get("length_norm")
        velocity = trial.inputs.get("velocity_norm_s")
        if length is None or velocity is None:
            raise ValueError(f"Trial {trial.trial_id!r} needs length_norm and velocity_norm_s.")
        gamma_dynamic = trial.inputs.get("gamma_dynamic_hz", np.zeros_like(length))
        gamma_static = trial.inputs.get("gamma_static_hz", np.zeros_like(length))
        return length, velocity, gamma_dynamic, gamma_static
    force = trial.inputs.get("force_n")
    if force is None:
        raise ValueError(f"Trial {trial.trial_id!r} needs force_n.")
    return (force,)


def _target_names(model: str) -> tuple[str, ...]:
    return ("ia_rate_hz", "ii_rate_hz") if model == "spindle" else ("ib_rate_hz",)


def _simulate(model: str, config: SpindleConfig | GTOConfig, trial: Trial) -> dict[str, np.ndarray]:
    if model == "spindle":
        length, velocity, gamma_dynamic, gamma_static = _trial_inputs(trial, model=model)
        reference = SpindleReference(config)  # type: ignore[arg-type]
        state = reference.init_state(())
        _, out = reference.rollout(state, length, velocity, gamma_dynamic, gamma_static, trial.dt)
        return {"ia_rate_hz": out["ia_rate"], "ii_rate_hz": out["ii_rate"]}
    force = _trial_inputs(trial, model=model)[0]
    reference = GTOReference(config)  # type: ignore[arg-type]
    state = reference.init_state(())
    _, out = reference.rollout(state, force, trial.dt)
    return {"ib_rate_hz": out["ib_rate"]}


def _interior(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Move initial values away from bounds to avoid degenerate trust regions."""
    span = upper - lower
    margin = np.minimum(np.maximum(1e-6, 1e-3 * span), 0.25 * span)
    return np.clip(values, lower + margin, upper - margin)


def fit_model(
    model: str,
    config: SpindleConfig | GTOConfig,
    trials: Sequence[Trial],
    *,
    fields: Sequence[str] | None = None,
    bounds: Sequence[tuple[float, float]] | None = None,
    max_nfev: int = 2000,
    global_search: bool = False,
    seed: int = 0,
    data_hash: str = "",
) -> FitResult:
    """Fit selected parameters with bounded least squares (optional global seed)."""
    if model not in ("spindle", "gto"):
        raise ValueError("model must be 'spindle' or 'gto'.")
    if fields is None:
        fields = SPINDLE_FIT_FIELDS if model == "spindle" else GTO_FIT_FIELDS
    fields = tuple(fields)
    if bounds is None:
        bounds = default_bounds(config, fields)
    if len(bounds) != flatten_config(config, fields).size:
        raise ValueError("bounds must have one entry per scalar parameter.")
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    if np.any(lower >= upper):
        raise ValueError("Every bound must satisfy lower < upper.")

    try:
        from scipy.optimize import differential_evolution, least_squares
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Calibration requires SciPy: pip install 'bioafferent[validation]'.") from exc

    x0 = _interior(flatten_config(config, fields), lower, upper)

    def objective(vector: np.ndarray) -> np.ndarray:
        return _residuals_with_params(model, config, fields, trials, vector)

    if global_search:
        seed_result = differential_evolution(
            lambda vector: float(np.sum(objective(vector) ** 2)),
            list(zip(lower, upper, strict=True)),
            seed=seed,
            maxiter=100,
            polish=False,
            workers=1,
        )
        x0 = _interior(seed_result.x, lower, upper)

    result = least_squares(objective, x0, bounds=(lower, upper), max_nfev=max_nfev, x_scale="jac")
    fitted = apply_params(config, fields, result.x)
    return FitResult(
        model=model,
        config=dataclasses.asdict(fitted),
        fields=fields,
        params=np.asarray(result.x, dtype=float),
        cost=float(result.cost),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        data_hash=data_hash,
    )


def _residuals_with_params(
    model: str,
    base_config: SpindleConfig | GTOConfig,
    fields: Sequence[str],
    trials: Sequence[Trial],
    params: np.ndarray,
) -> np.ndarray:
    config = apply_params(base_config, fields, params)
    residuals: list[np.ndarray] = []
    for trial in trials:
        simulated = _simulate(model, config, trial)
        for target in _target_names(model):
            if target not in trial.targets:
                continue
            truth = np.asarray(trial.targets[target], dtype=float)
            prediction = np.asarray(simulated[target], dtype=float)
            scale = max(float(np.std(truth)), float(np.ptp(truth)), 1.0)
            residuals.append((prediction - truth) / scale)
    if not residuals:
        raise ValueError("No target channels available for calibration.")
    return np.concatenate(residuals)


def fit_spindle(
    config: SpindleConfig,
    trials: Sequence[Trial],
    **kwargs: Any,
) -> FitResult:
    return fit_model("spindle", config, trials, **kwargs)


def fit_gto(config: GTOConfig, trials: Sequence[Trial], **kwargs: Any) -> FitResult:
    return fit_model("gto", config, trials, **kwargs)


def evaluate_model(
    model: str,
    config: SpindleConfig | GTOConfig,
    trials: Sequence[Trial],
    *,
    adaptation_windows: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Per-trial and aggregate rate metrics on a held-out split."""
    records: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []
    for trial in trials:
        simulated = _simulate(model, config, trial)
        for target in _target_names(model):
            if target not in trial.targets:
                continue
            record = rate_metrics(
                trial.targets[target],
                simulated[target],
                trial.dt,
                adaptation_windows=adaptation_windows,
            )
            records.append(record)
            details.append({"trial_id": trial.trial_id, "channel": target, **record})
    if not records:
        raise ValueError("No target channels available for evaluation.")
    return {"per_trial": details, "aggregate": aggregate_metrics(records)}


def save_fit(result: FitResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_fit(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fit bioafferent parameters on a calibration split.")
    parser.add_argument("--data-root", required=True, help="Directory containing dataset subdirectories.")
    parser.add_argument("--model", choices=("spindle", "gto"), default="spindle")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--fields", nargs="*", default=None, help="Fields to fit (default: engineering set).")
    parser.add_argument("--global-search", action="store_true", help="Seed with differential evolution.")
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = discover_datasets(args.data_root)
    calibration = trials_for_split(datasets, "calibration")
    validation = trials_for_split(datasets, "validation")
    if not calibration:
        raise SystemExit("No calibration trials found.")
    config: SpindleConfig | GTOConfig = SpindleConfig() if args.model == "spindle" else GTOConfig()
    combined_hash = "+".join(dataset_hash(dataset) for dataset in datasets)
    result = fit_model(
        args.model,
        config,
        calibration,
        fields=args.fields,
        max_nfev=args.max_nfev,
        global_search=args.global_search,
        seed=args.seed,
        data_hash=combined_hash,
    )
    fitted_config = SpindleConfig(**result.config) if args.model == "spindle" else GTOConfig(**result.config)
    if validation:
        result.metrics["validation"] = evaluate_model(args.model, fitted_config, validation)
    result.metrics["calibration"] = evaluate_model(args.model, fitted_config, calibration)
    save_fit(result, args.output)
    print(json.dumps({"output": str(args.output), "success": result.success, "cost": result.cost}, indent=2))


if __name__ == "__main__":
    _main()
