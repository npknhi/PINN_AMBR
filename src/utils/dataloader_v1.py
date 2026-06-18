#src/utils/dataloader_v1.py
from __future__ import annotations

"""Data loading helpers for the processed AMBR data."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.pinn_v1 import INPUT_COLUMNS, OBSERVABLE_COLUMNS, PseudomonasBIOSODE, STATE_INDEX, STATE_NAMES

OBSERVABLE_TARGET_COLUMNS = tuple(OBSERVABLE_COLUMNS.keys())
GAS_TARGET_COLUMNS = ("OUR_mol_min", "CER_mol_min", "CO2_offgas_fraction", "RQ")
WATER_ION_PRODUCT = 1e-14
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUXILIARY_CACHE_VERSION = "v7_observed_output_boundaries"
ONE_HOT_INPUT_COLUMNS = ("antifoam_absent", "antifoam_present")


@dataclass(frozen=True)
class PseudomonasDataset:
    frame: pd.DataFrame
    experiment_ids: np.ndarray
    experiment_codes: np.ndarray
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    x_raw: np.ndarray
    x: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y: np.ndarray
    y_mask: np.ndarray
    gas_columns: tuple[str, ...]
    gas_y: np.ndarray
    gas_mask: np.ndarray
    gas_scale: np.ndarray
    time_min: np.ndarray
    volume_l: np.ndarray
    air_flow_l_min: np.ndarray
    sampling_rate_l_min: np.ndarray
    acid_rate_l_min: np.ndarray
    base_rate_l_min: np.ndarray
    boundary_x: np.ndarray
    boundary_y: np.ndarray
    boundary_mask: np.ndarray
    state_scale: np.ndarray
    target_scale: np.ndarray
    output_dir: str | None = None


def load_pseudomonas_bioreactor_split(
    processed_csv: str | Path = "data/processed/ambr_preprocessed.csv",
    input_columns: tuple[str, ...] = tuple(INPUT_COLUMNS),
    target_columns: tuple[str, ...] = OBSERVABLE_TARGET_COLUMNS,
    train_experiment_ids: Sequence[str] = (),
    test_experiment_ids: Sequence[str] = (),
    validation_experiment_ids: Sequence[str] = (),
) -> tuple[PseudomonasDataset, PseudomonasDataset | None, PseudomonasDataset, dict[str, object]]:
    """Load an explicit benchmark split by bioreactor id."""

    train_ids = tuple(str(experiment_id) for experiment_id in train_experiment_ids)
    test_ids = tuple(str(experiment_id) for experiment_id in test_experiment_ids)
    val_ids = tuple(str(experiment_id) for experiment_id in validation_experiment_ids)
    if not train_ids:
        raise ValueError("train_experiment_ids must not be empty.")
    if not test_ids:
        raise ValueError("test_experiment_ids must not be empty.")
    overlap = sorted((set(train_ids) & set(test_ids)) | (set(train_ids) & set(val_ids)) | (set(val_ids) & set(test_ids)))
    if overlap:
        raise ValueError(f"Train, validation, and test bioreactors must be disjoint: {overlap}")

    frame = _ensure_columns(_load_processed_frame(processed_csv), input_columns, target_columns)
    _validate_experiment_ids(frame, (*train_ids, *val_ids, *test_ids))
    train_dataset, val_dataset, test_dataset = _build_bioreactor_datasets(
        frame,
        input_columns,
        target_columns,
        train_ids,
        test_ids,
        val_ids,
    )
    metadata: dict[str, object] = {
        "split_strategy": "bioreactor_id",
        "train_experiment_ids": list(train_ids),
        "validation_experiment_ids": list(val_ids),
        "test_experiment_ids": list(test_ids),
        "all_experiment_ids": [*train_ids, *val_ids, *test_ids],
    }
    return train_dataset, val_dataset, test_dataset, metadata


def load_pseudomonas_leave_one_bioreactor_out_split(
    processed_csv: str | Path = "data/processed/ambr_preprocessed.csv",
    input_columns: tuple[str, ...] = tuple(INPUT_COLUMNS),
    target_columns: tuple[str, ...] = OBSERVABLE_TARGET_COLUMNS,
    experiment_ids: Sequence[str] | None = None,
    fold_index: int = 0,
    validation_strategy: str = "rotate",
    validation_seed: int | None = None,
) -> tuple[PseudomonasDataset, PseudomonasDataset, PseudomonasDataset, dict[str, object]]:
    """Load one leave-one-bioreactor-out benchmark fold.

    Splits are made exclusively by ``Experiment_id`` so no time points from a
    held-out bioreactor leak into training. Normalization and scaling
    statistics are fitted on the training bioreactors only.
    """

    frame = _ensure_columns(_load_processed_frame(processed_csv), input_columns, target_columns)
    frame = _filter_experiment_ids(frame, experiment_ids)
    folds = make_leave_one_bioreactor_out_folds(frame)
    if not 0 <= fold_index < len(folds):
        raise ValueError(f"fold_index must be in [0, {len(folds) - 1}].")

    test_ids = tuple(folds[fold_index])
    if validation_strategy == "rotate":
        val_fold_index = (fold_index + 1) % len(folds)
    elif validation_strategy == "random":
        candidate_indices = [index for index in range(len(folds)) if index != fold_index]
        rng_seed = int(validation_seed) if validation_seed is not None else 0
        rng = np.random.default_rng(rng_seed + int(fold_index))
        val_fold_index = int(rng.choice(candidate_indices))
    else:
        raise ValueError("validation_strategy must be 'rotate' or 'random'.")
    val_ids = tuple(folds[val_fold_index])
    train_ids = tuple(
        experiment_id
        for current_index, fold in enumerate(folds)
        if current_index not in {fold_index, val_fold_index}
        for experiment_id in fold
    )
    train_dataset, val_dataset, test_dataset = _build_bioreactor_datasets(
        frame,
        input_columns,
        target_columns,
        train_ids,
        test_ids,
        val_ids,
    )
    metadata: dict[str, object] = {
        "split_strategy": "leave_one_bioreactor_out",
        "validation_strategy": validation_strategy,
        "validation_seed": None if validation_seed is None else int(validation_seed),
        "n_splits": len(folds),
        "fold_index": int(fold_index),
        "train_experiment_ids": list(train_ids),
        "validation_experiment_ids": list(val_ids),
        "test_experiment_ids": list(test_ids),
        "all_experiment_ids": [experiment_id for fold in folds for experiment_id in fold],
    }
    return train_dataset, val_dataset, test_dataset, metadata


def make_leave_one_bioreactor_out_folds(
    frame: pd.DataFrame,
) -> list[tuple[str, ...]]:
    """Create one deterministic fold per unique bioreactor id."""

    experiment_ids = _unique_experiment_ids(frame)
    if len(experiment_ids) < 2:
        raise ValueError("Leave-one-bioreactor-out requires at least 2 selected bioreactors.")
    return [(experiment_id,) for experiment_id in experiment_ids]


def _build_bioreactor_datasets(
    frame: pd.DataFrame,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    val_ids: Sequence[str] = (),
) -> tuple[PseudomonasDataset, PseudomonasDataset | None, PseudomonasDataset]:
    train_frame = frame[frame["Experiment_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    val_frame = frame[frame["Experiment_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    test_frame = frame[frame["Experiment_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Bioreactor split produced an empty dataset.")

    train_dataset = _build_dataset(
        train_frame,
        input_columns,
        target_columns,
        scaler_frame=train_frame,
        boundary_frame=train_frame,
    )
    val_dataset = None
    if val_ids:
        if val_frame.empty:
            raise ValueError("Validation split produced an empty dataset.")
        val_dataset = _build_dataset(
            val_frame,
            input_columns,
            target_columns,
            scalers=train_dataset,
            boundary_frame=val_frame,
        )
    test_dataset = _build_dataset(
        test_frame,
        input_columns,
        target_columns,
        scalers=train_dataset,
        boundary_frame=test_frame,
    )
    return train_dataset, val_dataset, test_dataset


def _load_processed_frame(processed_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(processed_csv, low_memory=False)
    if "Experiment_id" not in frame.columns:
        raise ValueError("Processed data must contain an Experiment_id column.")

    frame = frame.sort_values(["Experiment_id", "time_min"]).reset_index(drop=True)
    return frame


def _filter_experiment_ids(
    frame: pd.DataFrame,
    experiment_ids: Sequence[str] | None,
) -> pd.DataFrame:
    if experiment_ids is None:
        return frame
    selected = tuple(str(experiment_id) for experiment_id in experiment_ids)
    if not selected:
        raise ValueError("experiment_ids must not be empty.")
    _validate_experiment_ids(frame, selected)
    return frame[frame["Experiment_id"].astype(str).isin(selected)].reset_index(drop=True)


def _validate_experiment_ids(frame: pd.DataFrame, experiment_ids: Sequence[str]) -> None:
    available = set(frame["Experiment_id"].astype(str).unique())
    missing = sorted(set(experiment_ids) - available)
    if missing:
        raise ValueError(f"Unknown Experiment_id values: {missing}")


def _unique_experiment_ids(frame: pd.DataFrame) -> list[str]:
    values = frame["Experiment_id"].dropna().astype(str).drop_duplicates().tolist()
    return sorted(values, key=_experiment_id_sort_key)


def _experiment_id_sort_key(experiment_id: object) -> tuple[int, int, str]:
    text = str(experiment_id)
    match = re.match(r"^AMBR(\d+)_(\d+)$", text)
    if match:
        return int(match.group(1)), int(match.group(2)), text
    return 10**9, 10**9, text


def _build_dataset(
    frame: pd.DataFrame,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    scalers: PseudomonasDataset | None = None,
    boundary_frame: pd.DataFrame | None = None,
    scaler_frame: pd.DataFrame | None = None,
) -> PseudomonasDataset:
    frame = _ensure_columns(frame, input_columns, target_columns)
    scale_source = frame if scaler_frame is None else _ensure_columns(scaler_frame, input_columns, target_columns)

    features = frame.loc[:, input_columns].apply(pd.to_numeric, errors="coerce")
    features = _fill_feature_values(features)
    x_raw = features.to_numpy(dtype=np.float32)
    if scalers is None:
        scale_features = scale_source.loc[:, input_columns].apply(pd.to_numeric, errors="coerce")
        scale_features = _fill_feature_values(scale_features)
        scale_x = scale_features.to_numpy(dtype=np.float32)
        x_mean = np.nanmin(scale_x, axis=0).astype(np.float32)
        x_std = (np.nanmax(scale_x, axis=0) - x_mean).astype(np.float32)
        scale_targets = scale_source.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
        scale_y_mask = scale_targets.notna().to_numpy(dtype=bool)
        scale_y = scale_targets.fillna(0.0).to_numpy(dtype=np.float32)
        target_scale = _scale_from_targets(scale_y, scale_y_mask)
    else:
        x_mean = scalers.x_mean
        x_std = scalers.x_std
        target_scale = scalers.target_scale
    x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
    for column in ONE_HOT_INPUT_COLUMNS:
        if column in input_columns:
            index = input_columns.index(column)
            x_mean[index] = 0.0
            x_std[index] = 1.0
    x = ((x_raw - x_mean) / x_std).astype(np.float32)

    targets = frame.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    y_mask = targets.notna().to_numpy(dtype=bool)
    y = targets.fillna(0.0).to_numpy(dtype=np.float32)
    gas_targets = frame.loc[:, GAS_TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    gas_mask = gas_targets.notna().to_numpy(dtype=bool)
    gas_y = gas_targets.fillna(0.0).to_numpy(dtype=np.float32)
    if scalers is None:
        scale_gas_targets = scale_source.loc[:, GAS_TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
        scale_gas_mask = scale_gas_targets.notna().to_numpy(dtype=bool)
        scale_gas_y = scale_gas_targets.fillna(0.0).to_numpy(dtype=np.float32)
        gas_scale = _scale_from_targets(scale_gas_y, scale_gas_mask)
    else:
        gas_scale = scalers.gas_scale

    boundary_source = frame if boundary_frame is None else boundary_frame
    boundary_source = boundary_source[
        boundary_source["Experiment_id"].astype(str).isin(frame["Experiment_id"].astype(str).unique())
    ].copy()
    boundary_x, boundary_y, boundary_mask = _build_boundary_conditions(boundary_source, input_columns, x_mean, x_std)
    state_scale = _scale_from_boundary(boundary_y, boundary_mask) if scalers is None else scalers.state_scale
    experiment_ids = frame["Experiment_id"].astype(str).to_numpy()
    _, experiment_codes = np.unique(experiment_ids, return_inverse=True)

    return PseudomonasDataset(
        frame=frame,
        experiment_ids=experiment_ids,
        experiment_codes=experiment_codes.astype(np.int32),
        input_columns=tuple(input_columns),
        target_columns=tuple(target_columns),
        x_raw=x_raw,
        x=x,
        x_mean=x_mean,
        x_std=x_std,
        y=y,
        y_mask=y_mask,
        gas_columns=GAS_TARGET_COLUMNS,
        gas_y=gas_y,
        gas_mask=gas_mask,
        gas_scale=gas_scale,
        time_min=_numeric(frame["time_min"]).to_numpy(dtype=np.float32),
        volume_l=_numeric(frame["volume_l"]).to_numpy(dtype=np.float32),
        air_flow_l_min=_numeric(frame["air_flow_l_min"]).to_numpy(dtype=np.float32),
        sampling_rate_l_min=_numeric(frame["sampling_rate_l_min"]).to_numpy(dtype=np.float32),
        acid_rate_l_min=_numeric(frame["acid_rate_l_min"]).to_numpy(dtype=np.float32),
        base_rate_l_min=_numeric(frame["base_rate_l_min"]).to_numpy(dtype=np.float32),
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        boundary_mask=boundary_mask,
        state_scale=state_scale,
        target_scale=target_scale,
    )


def _ensure_columns(
    frame: pd.DataFrame,
    input_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = (*input_columns, *target_columns, *GAS_TARGET_COLUMNS, "Experiment_id", "time_min", "volume_l")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame


def _fill_feature_values(features: pd.DataFrame) -> pd.DataFrame:
    return features.fillna(0.0)


def _build_boundary_conditions(
    frame: pd.DataFrame,
    input_columns: tuple[str, ...],
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = _auxiliary_cache_path(frame, input_columns)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        boundary_features = cached["boundary_features"].astype(np.float32)
        boundary_y = cached["boundary_y"].astype(np.float32)
        boundary_mask = cached["boundary_mask"].astype(bool)
        boundary_x = ((boundary_features - x_mean) / x_std).astype(np.float32)
        return boundary_x, boundary_y, boundary_mask

    boundary_frames = []
    boundary_states = []
    boundary_masks = []
    for _, group in frame.sort_values(["Experiment_id", "time_min"]).groupby("Experiment_id", sort=False):
        first_row = group.iloc[[0]]
        final_row = group.iloc[[-1]]
        initial_state, initial_mask = _observed_boundary_state(first_row.iloc[0], use_initial_metadata=True)
        final_state, final_mask = _observed_boundary_state(final_row.iloc[0], use_initial_metadata=False)

        boundary_frames.extend([first_row, final_row])
        boundary_states.extend([initial_state, final_state])
        boundary_masks.extend([initial_mask, final_mask])

    boundary_frame = pd.concat(boundary_frames, ignore_index=True)
    features = boundary_frame.loc[:, input_columns].apply(pd.to_numeric, errors="coerce")
    features = _fill_feature_values(features)
    boundary_features = features.to_numpy(dtype=np.float32)
    boundary_x = ((boundary_features - x_mean) / x_std).astype(np.float32)
    boundary_y = np.asarray(boundary_states, dtype=np.float32)
    boundary_mask = np.asarray(boundary_masks, dtype=bool)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        boundary_features=boundary_features,
        boundary_y=boundary_y,
        boundary_mask=boundary_mask,
    )
    return boundary_x, boundary_y, boundary_mask


def _auxiliary_cache_path(frame: pd.DataFrame, input_columns: tuple[str, ...]) -> Path:
    relevant_columns = [
        "Experiment_id",
        "time_min",
        *input_columns,
        "initial_volume_l",
        "initial_glucose_mol_l",
        "initial_biomass_g_l",
        "initial_pH",
        "glucose_mol_l",
        "biomass_g_l",
        "O2_l_mol",
        "pH",
        "volume_l",
        "air_flow_l_min",
        "sampling_rate_l_min",
        "acid_rate_l_min",
        "base_rate_l_min",
    ]
    relevant_columns = [column for column in dict.fromkeys(relevant_columns) if column in frame.columns]
    payload = "\n".join(
        [
            AUXILIARY_CACHE_VERSION,
            repr(tuple(input_columns)),
            repr(tuple(sorted(PseudomonasBIOSODE.default_parameters.items()))),
            frame.loc[:, relevant_columns].sort_values(["Experiment_id", "time_min"]).to_csv(index=False),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return PROJECT_ROOT / "data" / "processed" / "auxiliary_cache" / f"boundaries_{digest}.npz"


def _observed_boundary_state(row: pd.Series, *, use_initial_metadata: bool) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((len(STATE_NAMES),), dtype=np.float32)
    mask = np.zeros((len(STATE_NAMES),), dtype=bool)
    volume = _finite_or_default(
        row.get("initial_volume_l" if use_initial_metadata else "volume_l"),
        PseudomonasBIOSODE.default_parameters["Vl"],
    )
    glucose = _finite_or_none(row.get("initial_glucose_mol_l" if use_initial_metadata else "glucose_mol_l"))
    biomass = _finite_or_none(row.get("initial_biomass_g_l" if use_initial_metadata else "biomass_g_l"))
    oxygen = _finite_or_none(row.get("O2_l_mol"))
    ph = _finite_or_none(row.get("initial_pH" if use_initial_metadata else "pH"))

    values = {
        "Substrate": glucose * volume if glucose is not None else None,
        "Biomass": biomass * volume if biomass is not None else None,
        "O2_l": oxygen,
        "H": (10.0 ** (-ph)) * volume if ph is not None else None,
    }
    for state_name, value in values.items():
        if value is not None and np.isfinite(value) and value >= 0.0:
            idx = STATE_INDEX[state_name]
            state[idx] = np.float32(value)
            mask[idx] = True
    return state, mask


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _finite_or_default(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _scale_from_boundary(boundary_y: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    scales = np.ones((boundary_y.shape[1],), dtype=np.float32)
    for idx in range(boundary_y.shape[1]):
        values = np.abs(boundary_y[boundary_mask[:, idx], idx])
        if values.size:
            scales[idx] = max(float(np.nanmax(values)), 1e-12)
    return scales


def _scale_from_targets(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scales = np.ones((y.shape[1],), dtype=np.float32)
    for idx in range(y.shape[1]):
        values = y[mask[:, idx], idx]
        if values.size:
            scales[idx] = max(float(np.nanmax(values) - np.nanmin(values)), 1e-8)
    return scales


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


__all__ = [
    "OBSERVABLE_TARGET_COLUMNS",
    "GAS_TARGET_COLUMNS",
    "PseudomonasDataset",
    "load_pseudomonas_bioreactor_split",
    "load_pseudomonas_leave_one_bioreactor_out_split",
    "make_leave_one_bioreactor_out_folds",
]
