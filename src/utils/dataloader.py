#src/utils/dataloader.py
from __future__ import annotations

"""Data loading helpers for the processed AMBR data."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from data.parameters import initial_conditions_dict
from src.models.pinn import INPUT_COLUMNS, OBSERVABLE_COLUMNS, PseudomonasBIOSODE, STATE_INDEX, STATE_NAMES

OBSERVABLE_TARGET_COLUMNS = tuple(OBSERVABLE_COLUMNS.keys())
GAS_TARGET_COLUMNS = ("OUR_mol_min", "CER_mol_min", "CO2_offgas_fraction", "RQ")
WATER_ION_PRODUCT = 1e-14
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUXILIARY_CACHE_VERSION = "v6_odemodelputida_cardinal_ph"


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


def load_pseudomonas_splits(
    processed_csv: str | Path = "data/processed/ambr_preprocessed.csv",
    input_columns: tuple[str, ...] = tuple(INPUT_COLUMNS),
    target_columns: tuple[str, ...] = OBSERVABLE_TARGET_COLUMNS,
    experiment_id: str | None = None,
    test_fraction: float = 0.2,
    split_strategy: str = "random",
    random_seed: int = 42,
) -> tuple[PseudomonasDataset, PseudomonasDataset]:
    """Load train/test datasets for each selected experiment.

    The normalization statistics are fitted on the full selected trajectory so
    train and test rows share the same 0-1 time scaling. The default random
    split is stratified by glucose availability so train keeps glucose rows.
    """

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if split_strategy not in {"random", "time"}:
        raise ValueError("split_strategy must be either 'random' or 'time'.")

    frame = _ensure_columns(_load_processed_frame(processed_csv, experiment_id), input_columns, target_columns)
    if split_strategy == "time":
        train_mask, test_mask = _time_split_mask(frame, target_columns, test_fraction)
    else:
        train_mask, test_mask = _random_stratified_split_mask(frame, test_fraction, random_seed)
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Train/test split produced an empty dataset.")

    train_dataset = _build_dataset(
        train_frame,
        input_columns,
        target_columns,
        scaler_frame=frame,
        boundary_frame=frame,
    )
    test_dataset = _build_dataset(
        test_frame,
        input_columns,
        target_columns,
        scalers=train_dataset,
        boundary_frame=frame,
    )
    return train_dataset, test_dataset


def _load_processed_frame(
    processed_csv: str | Path,
    experiment_id: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(processed_csv, low_memory=False)
    if "Experiment_id" not in frame.columns:
        raise ValueError("Processed data must contain an Experiment_id column.")
    if experiment_id is not None:
        frame = frame[frame["Experiment_id"].astype(str) == experiment_id].copy()
        if frame.empty:
            raise ValueError(f"No rows found for Experiment_id={experiment_id!r}.")

    frame = frame.sort_values(["Experiment_id", "time_min"]).reset_index(drop=True)
    return frame


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


def _time_split_mask(
    frame: pd.DataFrame,
    target_columns: tuple[str, ...],
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_mask = np.zeros(len(frame), dtype=bool)
    test_mask = np.zeros(len(frame), dtype=bool)
    for _, group in frame.groupby("Experiment_id", sort=False):
        observed = group[group.loc[:, target_columns].notna().any(axis=1)].sort_values("time_min")
        if len(observed) < 2:
            raise ValueError(f"Experiment {group['Experiment_id'].iloc[0]} needs at least two observed rows.")
        n_test = min(max(1, int(np.ceil(len(observed) * test_fraction))), len(observed) - 1)
        cutoff = float(observed.iloc[-n_test]["time_min"])
        train_indices = group[group["time_min"] < cutoff].index.to_numpy()
        test_indices = group[group["time_min"] >= cutoff].index.to_numpy()
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(f"Experiment {group['Experiment_id'].iloc[0]} produced an empty split.")
        train_mask[train_indices] = True
        test_mask[test_indices] = True
    return train_mask, test_mask


def _random_stratified_split_mask(
    frame: pd.DataFrame,
    test_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_mask = np.zeros(len(frame), dtype=bool)
    test_mask = np.zeros(len(frame), dtype=bool)
    rng = np.random.default_rng(random_seed)

    for _, group in frame.groupby("Experiment_id", sort=False):
        if len(group) < 2:
            raise ValueError(f"Experiment {group['Experiment_id'].iloc[0]} needs at least two rows.")

        has_glucose = _numeric(group["glucose_mol_l"]).notna()
        for glucose_present in (True, False):
            stratum_indices = group.loc[has_glucose == glucose_present].index.to_numpy()
            if len(stratum_indices) == 0:
                continue
            if len(stratum_indices) == 1:
                train_mask[stratum_indices] = True
                continue

            shuffled = rng.permutation(stratum_indices)
            n_test = min(max(1, int(np.ceil(len(shuffled) * test_fraction))), len(shuffled) - 1)
            test_mask[shuffled[:n_test]] = True
            train_mask[shuffled[n_test:]] = True

        group_indices = group.index.to_numpy()
        if not test_mask[group_indices].any():
            train_indices = group_indices[train_mask[group_indices]]
            fallback_candidates = train_indices[_numeric(frame.loc[train_indices, "glucose_mol_l"]).isna().to_numpy()]
            if len(fallback_candidates) == 0:
                fallback_candidates = train_indices
            test_index = rng.choice(fallback_candidates)
            train_mask[test_index] = False
            test_mask[test_index] = True
        if not train_mask[group_indices].any():
            raise ValueError(f"Experiment {group['Experiment_id'].iloc[0]} produced an empty train split.")

    return train_mask, test_mask


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
        initial_state = _initial_state_from_rows(first_row)[0]
        initial_mask = np.isfinite(initial_state) & (initial_state >= 0.0)

        final_state = _solve_final_state(group, initial_state)
        final_mask = np.isfinite(final_state) & (final_state >= 0.0)
        final_state, final_mask = _overwrite_final_observed_states(group, final_state, final_mask)

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


def _initial_state_from_rows(first_rows: pd.DataFrame) -> np.ndarray:
    p = PseudomonasBIOSODE.default_parameters
    initial_y = np.zeros((len(first_rows), len(STATE_NAMES)), dtype=np.float32)
    initial_volume = _numeric(first_rows["initial_volume_l"])
    initial_glucose = _numeric(first_rows["initial_glucose_mol_l"])
    initial_biomass = _numeric(first_rows["initial_biomass_g_l"])
    initial_ph = _numeric(first_rows["initial_pH"])
    for state_name, value in initial_conditions_dict.items():
        if state_name in STATE_INDEX:
            initial_y[:, STATE_INDEX[state_name]] = float(value)

    initial_y[:, STATE_INDEX["Substrate"]] = (initial_glucose * initial_volume).to_numpy(dtype=np.float32)
    initial_y[:, STATE_INDEX["Biomass"]] = (initial_biomass * initial_volume).to_numpy(dtype=np.float32)

    gas_volume = (p["Vtotal"] - initial_volume).clip(lower=1e-9)
    total_moles_gas = p["Pr"] * (gas_volume / 1000.0) / (p["R"] * p["TKelvin"])
    initial_y[:, STATE_INDEX["CO2_l"]] = (p["CSatCO2"] * initial_volume).to_numpy(dtype=np.float32)
    initial_y[:, STATE_INDEX["O2_l"]] = (p["CSatO2"] * initial_volume).to_numpy(dtype=np.float32)
    initial_y[:, STATE_INDEX["CO2_g"]] = (
        p["FractionCO2"] * total_moles_gas
    ).to_numpy(dtype=np.float32)
    initial_y[:, STATE_INDEX["O2_g"]] = (
        p["FractionO2"] * total_moles_gas
    ).to_numpy(dtype=np.float32)

    h_conc = 10.0 ** (-initial_ph)
    oh_conc = WATER_ION_PRODUCT / h_conc
    initial_y[:, STATE_INDEX["H"]] = (h_conc * initial_volume).to_numpy(dtype=np.float32)
    initial_y[:, STATE_INDEX["OH"]] = (oh_conc * initial_volume).to_numpy(dtype=np.float32)
    return initial_y


def _solve_final_state(group: pd.DataFrame, initial_state: np.ndarray) -> np.ndarray:
    rows = group.sort_values("time_min").reset_index(drop=True)
    y = np.asarray(initial_state, dtype=np.float64)
    times = _numeric(rows["time_min"]).to_numpy(dtype=np.float64)
    for index in range(len(rows) - 1):
        t0 = float(times[index])
        t1 = float(times[index + 1])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            continue
        controls = _ode_controls_from_row(rows.iloc[index])

        params = {**PseudomonasBIOSODE.default_parameters, **controls}

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            return PseudomonasBIOSODE.ode_func_numpy(t, state, params)

        solution = solve_ivp(
            rhs,
            (t0, t1),
            y,
            method="BDF",
            rtol=1e-5,
            atol=1e-10,
        )
        if solution.success:
            y = solution.y[:, -1]
        else:
            y = y + (t1 - t0) * rhs(t0, y)
        y = np.where(np.isfinite(y), y, np.nan)
        y = np.maximum(y, 0.0)
    return y.astype(np.float32)


def _ode_controls_from_row(row: pd.Series) -> dict[str, float]:
    volume_l = _finite_or_default(row.get("volume_l"), PseudomonasBIOSODE.default_parameters["Vl"])
    air_flow_l_min = _finite_or_default(row.get("air_flow_l_min"), PseudomonasBIOSODE.default_parameters["Vg"])
    sampling_rate_l_min = _finite_or_default(row.get("sampling_rate_l_min"), 0.0)
    acid_rate_l_min = _finite_or_default(row.get("acid_rate_l_min"), 0.0)
    base_rate_l_min = _finite_or_default(row.get("base_rate_l_min"), 0.0)
    return {
        "Vl": volume_l,
        "Vg": air_flow_l_min,
        "VOffGas": air_flow_l_min,
        "Vs": sampling_rate_l_min,
        "Va": acid_rate_l_min,
        "Vb": base_rate_l_min,
    }


def _overwrite_final_observed_states(
    group: pd.DataFrame,
    final_state: np.ndarray,
    final_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    volume = _finite_or_default(group.iloc[-1].get("volume_l"), PseudomonasBIOSODE.default_parameters["Vl"])
    replacements = {
        "Substrate": _last_finite(group, "glucose_mol_l", multiply_by=volume),
        "Biomass": _last_finite(group, "biomass_g_l", multiply_by=volume),
        "O2_l": _last_finite(group, "O2_l_mol"),
        "H": _last_ph_as_h_amount(group, volume),
    }
    for state_name, value in replacements.items():
        idx = STATE_INDEX[state_name]
        if value is not None and np.isfinite(value) and value >= 0.0:
            final_state[idx] = value
            final_mask[idx] = True
    return final_state, final_mask


def _last_finite(group: pd.DataFrame, column: str, multiply_by: float | None = None) -> float | None:
    if column not in group.columns:
        return None
    values = _numeric(group[column]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    value = float(values.iloc[-1])
    return value * multiply_by if multiply_by is not None else value


def _last_ph_as_h_amount(group: pd.DataFrame, volume_l: float) -> float | None:
    ph = _last_finite(group, "pH")
    if ph is None:
        return None
    return (10.0 ** (-ph)) * volume_l


def _finite_or_default(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _scale_from_boundary(boundary_y: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    masked = np.where(boundary_mask, np.abs(boundary_y), np.nan)
    scales = np.nanmax(masked, axis=0)
    scales = np.where(np.isfinite(scales), scales, 1.0)
    return np.maximum(scales, 1e-12).astype(np.float32)


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
    "load_pseudomonas_splits",
]
