"""Condition-level data splits and leakage checks.

Splits must be assigned by experimental condition, not by random time samples:
neighbouring samples are highly correlated, so random sample splits leak
information. The canonical split plan is stored in each dataset's metadata;
this module enforces that no condition appears in more than one split.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from validation.io import Dataset, DatasetError, Trial

SPLITS = ("calibration", "validation", "blind")


@dataclass(frozen=True)
class SplitPlan:
    """Mapping from condition name to split name."""

    assignments: Mapping[str, str]

    def __post_init__(self) -> None:
        for condition, split in self.assignments.items():
            if not condition:
                raise DatasetError("Condition names must be non-empty.")
            if split not in SPLITS:
                raise DatasetError(f"Unknown split {split!r} for condition {condition!r}.")

    def split_for(self, condition: str) -> str:
        try:
            return self.assignments[condition]
        except KeyError as exc:
            raise DatasetError(f"Condition {condition!r} has no pre-registered split.") from exc


def plan_from_datasets(datasets: Iterable[Dataset]) -> SplitPlan:
    """Build a split plan from dataset metadata and reject contradictions."""
    assignments: dict[str, str] = {}
    for dataset in datasets:
        split = dataset.split
        for condition in dataset.conditions:
            if condition in assignments and assignments[condition] != split:
                raise DatasetError(
                    f"Condition {condition!r} appears in both {assignments[condition]!r} and {split!r} splits."
                )
            assignments[condition] = split
    if not assignments:
        raise DatasetError("No conditions found.")
    return SplitPlan(assignments=assignments)


def assert_no_leakage(datasets: Iterable[Dataset]) -> SplitPlan:
    """Raise when a condition spans multiple splits; return the validated plan."""
    return plan_from_datasets(datasets)


def trials_for_split(datasets: Iterable[Dataset], split: str) -> tuple[Trial, ...]:
    """Collect trials belonging to a pre-registered split."""
    if split not in SPLITS:
        raise DatasetError(f"Unknown split {split!r}.")
    trials: list[Trial] = []
    for dataset in datasets:
        if dataset.split == split:
            trials.extend(dataset.trials)
    return tuple(trials)


def group_conditions_by_split(datasets: Iterable[Dataset]) -> dict[str, tuple[str, ...]]:
    """Return condition names grouped by split."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for dataset in datasets:
        grouped[dataset.split].update(dataset.conditions)
    return {split: tuple(sorted(conditions)) for split, conditions in sorted(grouped.items())}


def assert_blind_inputs_only(dataset: Dataset) -> None:
    """Raise if a blind dataset carries any target channel or spikes."""
    if dataset.split != "blind":
        raise DatasetError(f"Dataset {dataset.dataset_id!r} is not a blind dataset.")
    for trial in dataset.trials:
        if trial.targets:
            raise DatasetError(f"Blind trial {trial.trial_id!r} exposes targets {sorted(trial.targets)}.")
        if trial.spikes:
            raise DatasetError(f"Blind trial {trial.trial_id!r} exposes spikes {sorted(trial.spikes)}.")


def assert_calibration_has_targets(datasets: Iterable[Dataset]) -> None:
    """Raise if a calibration dataset has no observed target channels."""
    for dataset in datasets:
        if dataset.split != "calibration":
            continue
        for trial in dataset.trials:
            if not trial.targets:
                raise DatasetError(f"Calibration trial {trial.trial_id!r} has no targets.")
