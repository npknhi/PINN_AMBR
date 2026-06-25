from __future__ import annotations

"""Dataset split strategies shared by benchmark experiments."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.pinn import INPUT_COLUMNS
from src.utils.dataloader import (
    OBSERVABLE_TARGET_COLUMNS,
    PseudomonasDataset,
    _build_bioreactor_datasets,
    _build_dataset,
    _ensure_columns,
    _filter_experiment_ids,
    _load_processed_frame,
    _unique_experiment_ids,
    _validate_experiment_ids,
)


def load_pseudomonas_bioreactor_split(
    processed_csv: str | Path = "data/processed/ambr_preprocessed.csv",
    input_columns: tuple[str, ...] = tuple(INPUT_COLUMNS),
    target_columns: tuple[str, ...] = OBSERVABLE_TARGET_COLUMNS,
    train_experiment_ids: Sequence[str] = (),
    test_experiment_ids: Sequence[str] = (),
    validation_experiment_ids: Sequence[str] = (),
) -> tuple[PseudomonasDataset, PseudomonasDataset | None, PseudomonasDataset, dict[str, object]]:
    """Load an explicit benchmark split by bioreactor id."""

    train_ids = tuple(str(value) for value in train_experiment_ids)
    test_ids = tuple(str(value) for value in test_experiment_ids)
    val_ids = tuple(str(value) for value in validation_experiment_ids)
    if not train_ids or not test_ids:
        raise ValueError("train_experiment_ids and test_experiment_ids must not be empty.")
    overlap = sorted((set(train_ids) & set(test_ids)) | (set(train_ids) & set(val_ids)) | (set(val_ids) & set(test_ids)))
    if overlap:
        raise ValueError(f"Train, validation, and test bioreactors must be disjoint: {overlap}")

    frame = _ensure_columns(_load_processed_frame(processed_csv), input_columns, target_columns)
    _validate_experiment_ids(frame, (*train_ids, *val_ids, *test_ids))
    train, validation, test = _build_bioreactor_datasets(
        frame, input_columns, target_columns, train_ids, test_ids, val_ids
    )
    metadata = {
        "split_strategy": "bioreactor_id",
        "train_experiment_ids": list(train_ids),
        "validation_experiment_ids": list(val_ids),
        "test_experiment_ids": list(test_ids),
        "all_experiment_ids": [*train_ids, *val_ids, *test_ids],
    }
    return train, validation, test, metadata


def make_leave_one_bioreactor_out_folds(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    experiment_ids = _unique_experiment_ids(frame)
    if len(experiment_ids) < 2:
        raise ValueError("Leave-one-bioreactor-out requires at least 2 selected bioreactors.")
    return [(experiment_id,) for experiment_id in experiment_ids]


def load_pseudomonas_leave_one_bioreactor_out_split(
    processed_csv: str | Path = "data/processed/ambr_preprocessed.csv",
    input_columns: tuple[str, ...] = tuple(INPUT_COLUMNS),
    target_columns: tuple[str, ...] = OBSERVABLE_TARGET_COLUMNS,
    experiment_ids: Sequence[str] | None = None,
    fold_index: int = 0,
    validation_strategy: str = "rotate",
    validation_seed: int | None = None,
) -> tuple[PseudomonasDataset, PseudomonasDataset, PseudomonasDataset, dict[str, object]]:
    frame = _ensure_columns(_load_processed_frame(processed_csv), input_columns, target_columns)
    frame = _filter_experiment_ids(frame, experiment_ids)
    folds = make_leave_one_bioreactor_out_folds(frame)
    if not 0 <= fold_index < len(folds):
        raise ValueError(f"fold_index must be in [0, {len(folds) - 1}].")

    test_ids = tuple(folds[fold_index])
    if validation_strategy == "rotate":
        val_fold_index = (fold_index + 1) % len(folds)
    elif validation_strategy == "random":
        candidates = [index for index in range(len(folds)) if index != fold_index]
        rng = np.random.default_rng((0 if validation_seed is None else int(validation_seed)) + int(fold_index))
        val_fold_index = int(rng.choice(candidates))
    else:
        raise ValueError("validation_strategy must be 'rotate' or 'random'.")
    val_ids = tuple(folds[val_fold_index])
    train_ids = tuple(
        experiment_id
        for current_index, fold in enumerate(folds)
        if current_index not in {fold_index, val_fold_index}
        for experiment_id in fold
    )
    train, validation, test = _build_bioreactor_datasets(
        frame, input_columns, target_columns, train_ids, test_ids, val_ids
    )
    metadata = {
        "split_strategy": "leave_one_bioreactor_out",
        "validation_strategy": validation_strategy,
        "validation_seed": None if validation_seed is None else int(validation_seed),
        "n_splits": len(folds),
        "fold_index": int(fold_index),
        "train_experiment_ids": list(train_ids),
        "validation_experiment_ids": list(val_ids),
        "test_experiment_ids": list(test_ids),
        "all_experiment_ids": [value for fold in folds for value in fold],
    }
    return train, validation, test, metadata


def load_pseudomonas_forecasting_split(
    processed_csv: str | Path = "data/processed/ambr_preprocessed.csv",
    input_columns: tuple[str, ...] = tuple(INPUT_COLUMNS),
    target_columns: tuple[str, ...] = OBSERVABLE_TARGET_COLUMNS,
    experiment_ids: Sequence[str] | None = None,
    observation_fraction: float = 0.5,
) -> tuple[PseudomonasDataset, None, PseudomonasDataset, dict[str, object]]:
    fraction = float(observation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("observation_fraction must be strictly between 0 and 1.")

    frame = _ensure_columns(_load_processed_frame(processed_csv), input_columns, target_columns)
    frame = _filter_experiment_ids(frame, experiment_ids)
    prefix_mask = np.zeros((len(frame),), dtype=bool)
    cutoff_time_min: dict[str, float] = {}
    for experiment_id, group in frame.groupby("Experiment_id", sort=False):
        times = pd.to_numeric(group["time_min"], errors="coerce")
        finite_times = times[np.isfinite(times)]
        if finite_times.empty:
            raise ValueError(f"Experiment {experiment_id} has no finite time_min values.")
        cutoff = float(finite_times.min() + fraction * (finite_times.max() - finite_times.min()))
        cutoff_time_min[str(experiment_id)] = cutoff
        prefix_mask[group.index.to_numpy()] = times.to_numpy(dtype=float) <= cutoff

    prefix_frame = frame.loc[prefix_mask].reset_index(drop=True)
    if prefix_frame.empty:
        raise ValueError("Forecasting split produced an empty prefix dataset.")
    train = _build_dataset(prefix_frame, input_columns, target_columns)
    validation = None
    forecast = _build_dataset(frame, input_columns, target_columns, scalers=train)
    metadata = {
        "split_strategy": "independent_temporal_forecasting",
        "observation_fraction": fraction,
        "validation_strategy": "none",
        "experiment_ids": _unique_experiment_ids(frame),
        "cutoff_time_min": cutoff_time_min,
    }
    return train, validation, forecast, metadata


__all__ = [
    "load_pseudomonas_bioreactor_split",
    "load_pseudomonas_forecasting_split",
    "load_pseudomonas_leave_one_bioreactor_out_split",
    "make_leave_one_bioreactor_out_folds",
]
