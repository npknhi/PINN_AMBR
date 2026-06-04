#src/utils/infer_ode_parameters.py
from __future__ import annotations

"""Estimate ODE parameter seeds from processed AMBR data.

This script fits the mechanistic ODE directly to measured observables and saves
the estimated parameters for use as initial values in later PINN training runs.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.models.pinn import OBSERVABLE_COLUMNS, PseudomonasBIOSODE, STATE_INDEX
from src.utils.dataloader import _initial_state_from_rows, _ode_controls_from_row


PARAMETER_NAMES = ("mu_max", "Ksubs", "KLaO2", "KLaCO2")
TARGET_COLUMNS = tuple(OBSERVABLE_COLUMNS.keys())
PENALTY_RESIDUAL = 1e6


def main() -> None:
    args = _parse_args()
    frame = _load_data(args.processed_csv, args.experiment_id, args.global_fit, args.exclude_experiments)
    if args.train_fraction < 1.0:
        frame = _keep_initial_fraction_by_experiment(frame, args.train_fraction)

    result = infer_parameters(
        frame,
        parameter_names=tuple(args.parameters),
        target_columns=tuple(args.targets),
        max_nfev=args.max_nfev,
        loss=args.loss,
        f_scale=args.f_scale,
        n_starts=args.n_starts,
        random_seed=args.random_seed,
        ftol=args.ftol,
        xtol=args.xtol,
        gtol=args.gtol,
    )

    print("\nEstimated parameter seeds")
    for name, value in result["estimated_parameters"].items():
        default = PseudomonasBIOSODE.default_parameters[name]
        print(f"{name:10s} estimated={value:.8g}  default={default:.8g}")

    print("\nSuggested data/parameters.py seed values")
    for name, value in result["estimated_parameters"].items():
        print(f'    "{name}": {value:.8g},')

    print("\nCost by target")
    for name, cost in result["target_costs"].items():
        print(f"{name:14s} cost={cost:.8g}")

    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {output_json}")

    if args.output_csv is not None:
        output_csv = Path(args.output_csv.format(experiment_id=result["experiment_id"]))
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "parameter": name,
                "estimated_value": value,
                "default_value": PseudomonasBIOSODE.default_parameters[name],
                "range_low": result["parameter_ranges"][name][0],
                "range_high": result["parameter_ranges"][name][1],
            }
            for name, value in result["estimated_parameters"].items()
        ]
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"Saved CSV: {output_csv}")


def infer_parameters(
    frame: pd.DataFrame,
    parameter_names: tuple[str, ...] = PARAMETER_NAMES,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
    max_nfev: int = 40,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    n_starts: int = 5,
    random_seed: int = 0,
    ftol: float | None = 1e-8,
    xtol: float | None = None,
    gtol: float | None = None,
) -> dict:
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")
    if ftol is None and xtol is None and gtol is None:
        raise ValueError("At least one of ftol, xtol, or gtol must be set.")
    x0 = np.asarray([PseudomonasBIOSODE.default_parameters[name] for name in parameter_names], dtype=float)
    bounds = _parameter_bounds(parameter_names)
    starts = _initial_guesses(x0, bounds, n_starts=n_starts, random_seed=random_seed)
    target_scale = _target_scale(frame, target_columns)

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            pieces = _residual_pieces(frame, parameter_names, target_columns, values, target_scale)
        except (FloatingPointError, OverflowError, ValueError):
            return _penalty_residual(frame, target_columns)
        if not pieces:
            raise ValueError("No measured observable targets are available for least_squares.")
        return np.concatenate(pieces)

    fits = []
    for start_idx, start in enumerate(starts, start=1):
        print(f"\nLeast-squares start {start_idx}/{len(starts)}")
        fits.append(
            least_squares(
                residual,
                x0=start,
                bounds=(bounds[:, 0], bounds[:, 1]),
                loss=loss,
                f_scale=f_scale,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                verbose=1,
            )
        )
    fit = min(fits, key=lambda item: item.cost)
    estimated = {name: float(value) for name, value in zip(parameter_names, fit.x)}
    target_costs = _target_costs(frame, parameter_names, target_columns, fit.x, target_scale)
    experiment_ids = frame["Experiment_id"].dropna().astype(str).drop_duplicates().tolist()
    experiment_label = "GLOBAL" if len(experiment_ids) > 1 else experiment_ids[0]
    return {
        "experiment_id": experiment_label,
        "experiment_ids": experiment_ids,
        "n_experiments": int(len(experiment_ids)),
        "n_rows": int(len(frame)),
        "parameter_names": list(parameter_names),
        "target_columns": list(target_columns),
        "estimated_parameters": estimated,
        "default_parameters": {name: float(PseudomonasBIOSODE.default_parameters[name]) for name in parameter_names},
        "parameter_ranges": {name: [float(v) for v in bounds[idx]] for idx, name in enumerate(parameter_names)},
        "range_source": "PseudomonasBIOSODE.parameter_ranges",
        "n_starts": int(n_starts),
        "random_seed": int(random_seed),
        "target_costs": target_costs,
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
        "nfev": int(fit.nfev),
        "total_nfev": int(sum(item.nfev for item in fits)),
        "success": bool(fit.success),
        "message": str(fit.message),
    }


def _parameter_bounds(parameter_names: Sequence[str]) -> np.ndarray:
    bounds = np.asarray([PseudomonasBIOSODE.parameter_ranges[name] for name in parameter_names], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Parameter ranges must be finite.")
    if not np.all(bounds[:, 0] > 0):
        raise ValueError("Parameter lower bounds must be positive for log-space starts.")
    if not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError("Parameter lower bounds must be smaller than upper bounds.")
    return bounds


def _initial_guesses(
    default_values: np.ndarray,
    bounds: np.ndarray,
    n_starts: int = 5,
    random_seed: int = 0,
) -> list[np.ndarray]:
    starts = [np.clip(default_values, bounds[:, 0], bounds[:, 1])]
    if n_starts == 1:
        return starts
    rng = np.random.default_rng(random_seed)
    log_low = np.log(bounds[:, 0])
    log_high = np.log(bounds[:, 1])
    random_starts = np.exp(rng.uniform(log_low, log_high, size=(n_starts - 1, len(default_values))))
    starts.extend(random_starts)
    return starts


def _residual_pieces(
    frame: pd.DataFrame,
    parameter_names: Sequence[str],
    target_columns: Sequence[str],
    values: np.ndarray,
    target_scale: np.ndarray,
) -> list[np.ndarray]:
    params = dict(zip(parameter_names, values))
    pieces = []
    for _, experiment_frame in frame.groupby("Experiment_id", sort=False):
        simulation = _simulate_observables(experiment_frame, params)
        for idx, column in enumerate(target_columns):
            observed = pd.to_numeric(experiment_frame[column], errors="coerce").to_numpy(dtype=float)
            predicted = simulation[column]
            mask = np.isfinite(observed)
            if np.any(mask):
                residual = (predicted[mask] - observed[mask]) / target_scale[idx]
                residual = np.where(np.isfinite(residual), residual, PENALTY_RESIDUAL)
                pieces.append(residual)
    return pieces


def _target_costs(
    frame: pd.DataFrame,
    parameter_names: Sequence[str],
    target_columns: Sequence[str],
    values: np.ndarray,
    target_scale: np.ndarray,
) -> dict[str, float]:
    params = dict(zip(parameter_names, values))
    costs = {column: 0.0 for column in target_columns}
    counts = {column: 0 for column in target_columns}
    for _, experiment_frame in frame.groupby("Experiment_id", sort=False):
        try:
            simulation = _simulate_observables(experiment_frame, params)
        except (FloatingPointError, OverflowError, ValueError):
            return {column: float("inf") for column in target_columns}
        for idx, column in enumerate(target_columns):
            observed = pd.to_numeric(experiment_frame[column], errors="coerce").to_numpy(dtype=float)
            predicted = simulation[column]
            mask = np.isfinite(observed)
            if np.any(mask):
                residual = (predicted[mask] - observed[mask]) / target_scale[idx]
                residual = np.where(np.isfinite(residual), residual, PENALTY_RESIDUAL)
                costs[column] += float(0.5 * np.sum(residual**2))
                counts[column] += int(mask.sum())
    for column in target_columns:
        if counts[column] == 0:
            costs[column] = float("nan")
    return costs


def _penalty_residual(frame: pd.DataFrame, target_columns: Sequence[str]) -> np.ndarray:
    count = 0
    for column in target_columns:
        observed = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        count += int(np.isfinite(observed).sum())
    return np.full(max(count, 1), PENALTY_RESIDUAL, dtype=float)


def _simulate_observables(frame: pd.DataFrame, parameter_values: dict[str, float]) -> dict[str, np.ndarray]:
    rows = frame.sort_values("time_min").reset_index(drop=True)
    times = pd.to_numeric(rows["time_min"], errors="coerce").to_numpy(dtype=float)
    state = _initial_state_from_rows(rows.iloc[[0]])[0].astype(float)
    states = np.zeros((len(rows), len(PseudomonasBIOSODE.state_names)), dtype=float)
    states[0] = state

    for idx in range(len(rows) - 1):
        t0 = float(times[idx])
        t1 = float(times[idx + 1])
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            states[idx + 1] = state
            continue
        controls = _ode_controls_from_row(rows.iloc[idx])
        params = {**PseudomonasBIOSODE.default_parameters, **parameter_values, **controls}

        solution = solve_ivp(
            lambda t, y: PseudomonasBIOSODE.ode_func_numpy(t, y, params),
            (t0, t1),
            state,
            method="BDF",
            rtol=1e-5,
            atol=1e-10,
        )
        if not solution.success:
            raise FloatingPointError(f"ODE solver failed: {solution.message}")
        state = solution.y[:, -1]
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("ODE simulation produced non-finite states.")
        state = np.maximum(state, 0.0)
        states[idx + 1] = state

    volume = np.maximum(pd.to_numeric(rows["volume_l"], errors="coerce").to_numpy(dtype=float), 1e-12)
    h_mol_l = states[:, STATE_INDEX["H"]] / volume
    return {
        "glucose_mol_l": states[:, STATE_INDEX["Substrate"]] / volume,
        "biomass_g_l": states[:, STATE_INDEX["Biomass"]] / volume,
        "O2_l_mol": states[:, STATE_INDEX["O2_l"]],
        "pH": -np.log10(np.maximum(h_mol_l, 1e-14)),
    }


def _target_scale(frame: pd.DataFrame, target_columns: Sequence[str]) -> np.ndarray:
    scales = []
    for column in target_columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            scales.append(1.0)
        else:
            scales.append(max(float(values.max() - values.min()), 1e-8))
    return np.asarray(scales, dtype=float)


def _load_data(
    processed_csv: str,
    experiment_id: str,
    global_fit: bool,
    exclude_experiments: Sequence[str],
) -> pd.DataFrame:
    frame = pd.read_csv(processed_csv)
    frame["Experiment_id"] = frame["Experiment_id"].astype(str)
    if exclude_experiments:
        frame = frame[~frame["Experiment_id"].isin(exclude_experiments)]
    if not global_fit:
        frame = frame[frame["Experiment_id"] == experiment_id].copy()
    if frame.empty:
        raise ValueError("No rows found for the selected experiment data.")
    return frame.sort_values(["Experiment_id", "time_min"]).reset_index(drop=True)


def _keep_initial_fraction_by_experiment(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1].")
    if fraction >= 1.0:
        return frame
    pieces = []
    for _, experiment_frame in frame.groupby("Experiment_id", sort=False):
        cutoff_index = max(2, int(np.ceil(len(experiment_frame) * fraction)))
        pieces.append(experiment_frame.iloc[:cutoff_index])
    return pd.concat(pieces, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-csv", default="data/processed/ambr_preprocessed.csv")
    parser.add_argument("--experiment-id", default="AMBR1_15")
    parser.add_argument("--global-fit", action="store_true", help="Fit one shared parameter set across all experiments.")
    parser.add_argument("--exclude-experiments", nargs="*", default=[], help="Experiment ids to exclude from fitting.")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=40)
    parser.add_argument("--loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--f-scale", type=float, default=1.0)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--ftol", type=_optional_float, default=1e-8)
    parser.add_argument("--xtol", type=_optional_float, default=None)
    parser.add_argument("--gtol", type=_optional_float, default=None)
    parser.add_argument("--parameters", nargs="+", default=list(PARAMETER_NAMES), choices=list(PARAMETER_NAMES))
    parser.add_argument("--targets", nargs="+", default=list(TARGET_COLUMNS), choices=list(TARGET_COLUMNS))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default="results/parameters_infer/{experiment_id}.csv")
    return parser.parse_args()


def _optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


if __name__ == "__main__":
    main()
