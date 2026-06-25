from __future__ import annotations

"""Train and score the temporal forecasting benchmark."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.utils.evaluation import (
    evaluate_observables,
    forecasting_metric_table,
    plot_forecasting_curves,
    plot_saved_forecasting_predictions,
    plot_saved_history,
    parameter_report,
    regression_metric_values,
    write_forecasting_table,
    write_hierarchical_forecasting_table,
)
from src.utils.splits import load_pseudomonas_forecasting_split
from src.utils.training import (
    GLUCOSE_MOLAR_MASS_G_MOL,
    TrainingConfig,
    predict_dataset,
    train_pinn,
)


DEFAULT_OBSERVATION_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
FORECAST_WINDOW = "forecast_window"
FORECAST_WINDOW_BENCHMARK = "Forecasting Scan"
FORECAST_OBSERVABLES = ("glucose_g_l", "biomass_g_l", "DO_percent", "pH")
FORECAST_METRIC_COLUMNS = ("Experiment_id", "observable", "n", "nrmse", "mae", "r2")
FORECAST_PREDICTION_COLUMNS = (
    "Experiment_id",
    "time_h",
    "time_min",
    "glucose_g_l",
    "glucose_g_l_pred",
    "biomass_g_l",
    "biomass_g_l_pred",
    "DO_percent",
    "DO_percent_pred",
    "pH",
    "pH_pred",
)

METRICS_FILENAME = "forecasting_metrics_long.csv"
TABLE_FILENAME = "forecasting_table.csv"
FOLD_METRICS_FILENAME = "metrics_test.csv"


def run_independent_forecasting_experiment(
    config: TrainingConfig,
    *,
    observation_fractions: tuple[float, ...] = DEFAULT_OBSERVATION_FRACTIONS,
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2026),
    keep_results: bool = False,
) -> dict[str, Any]:
    """Train one model per fraction and seed, then score the forecast window."""

    if not config.experiment_ids:
        raise ValueError("Independent forecasting requires the selected experiment_ids.")
    if config.use_early_stopping:
        raise ValueError("dAMN-style forecasting uses fixed-epoch training without validation or early stopping.")
    fractions = tuple(float(value) for value in observation_fractions)
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("observation_fractions must contain values strictly between 0 and 1.")
    seed_values = tuple(int(value) for value in seeds)
    if not seed_values:
        raise ValueError("seeds must not be empty.")

    base_name = (config.experiment_name or "FCT").strip()
    output_dir = Path(config.results_dir) / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_results = []
    run_dirs = []
    fold_ids = _selected_experiment_ids(config.processed_csv, config.experiment_ids)
    fold_lookup = {experiment_id: index for index, experiment_id in enumerate(fold_ids)}

    for seed in seed_values:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = seed_dir / "forecasting_runtime.csv"
        for fraction in fractions:
            fraction_label = int(round(100.0 * fraction))
            fraction_dir_name = f"fraction_{fraction_label:02d}"
            run_config = replace(
                config,
                results_dir=str(seed_dir),
                experiment_name=fraction_dir_name,
                seed=seed,
                forecast_observation_fraction=fraction,
                validation_strategy="none",
            )
            train_dataset, val_dataset, test_dataset, split_metadata = load_pseudomonas_forecasting_split(
                run_config.processed_csv,
                experiment_ids=run_config.experiment_ids,
                observation_fraction=fraction,
            )
            train_started = perf_counter()
            result = train_pinn(
                run_config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                split_metadata=split_metadata,
            )
            training_time_s = perf_counter() - train_started
            run_output = Path(result["output_dir"])
            run_dirs.append(run_output)
            parameter_report(result).to_csv(run_output / "parameter_report.csv", index=False)
            evaluate_observables(result["model"], result["train_dataset"]).to_csv(
                run_output / "metrics_train.csv", index=False
            )

            forecast_started = perf_counter()
            metrics, predictions = _score_trained_forecast(result)
            forecasting_time_s = perf_counter() - forecast_started
            _write_forecasting_outputs(
                metrics,
                predictions,
                run_output=run_output,
                fold_lookup=fold_lookup,
            )
            _remove_fraction_seed(runtime_path, fraction, seed)
            runtime = pd.DataFrame(
                [{
                    "observation_fraction": fraction,
                    "seed": seed,
                    "training_time_s": float(training_time_s),
                    "forecasting_time_s": float(forecasting_time_s),
                }]
            )
            _append_frame(runtime_path, runtime)
            if keep_results:
                kept_results.append(result)

    seed_dirs = [output_dir / f"seed_{seed}" for seed in seed_values]
    for seed_dir in seed_dirs:
        _write_seed_forecasting_summary(seed_dir)
    metric_frames = [
        pd.read_csv(path / METRICS_FILENAME)
        for path in run_dirs
    ]
    runtime_frames = [
        pd.read_csv(seed_dir / "forecasting_runtime.csv")
        for seed_dir in seed_dirs
        if (seed_dir / "forecasting_runtime.csv").exists()
    ]
    return {
        "output_dir": str(output_dir),
        "metrics_long": pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame(),
        "runtime": pd.concat(runtime_frames, ignore_index=True) if runtime_frames else pd.DataFrame(),
        "run_dirs": [str(path) for path in run_dirs],
        "results": kept_results,
    }


def _score_trained_forecast(
    result: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict once and score the forecast window."""

    dataset = result["test_dataset"]
    cutoff_by_experiment = result.get("split_metadata", {}).get("cutoff_time_min", {})
    full_predictions = predict_dataset(result["model"], dataset, batch_size=512)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for experiment_id, trajectory in dataset.frame.groupby("Experiment_id", sort=False):
        experiment_id = str(experiment_id)
        if experiment_id not in cutoff_by_experiment:
            raise KeyError(f"Missing forecast cutoff for {experiment_id}.")
        trajectory = trajectory.sort_values("time_min")
        sorted_indices = trajectory.index.to_numpy(dtype=int)
        predictions = {
            observable: np.asarray(full_predictions[observable], dtype=float)[sorted_indices]
            for observable in FORECAST_OBSERVABLES
        }
        trajectory = trajectory.reset_index(drop=True)
        metric_rows.extend(
            _forecast_metric_rows(
                trajectory,
                predictions,
                dataset,
                experiment_id=experiment_id,
                cutoff_time_min=float(cutoff_by_experiment[experiment_id]),
            )
        )
        prediction_rows.extend(
            _forecast_prediction_rows(
                trajectory,
                predictions,
                experiment_id=experiment_id,
            )
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def summarize_forecasting(
    results_dir: str | Path,
    *,
    history_plots: tuple[str, ...] = ("loss", "r2", "r2_by_target"),
) -> dict[str, Any]:
    """Build tables and figures for the forecast-window benchmark."""

    results_path = Path(results_dir)
    table_path = write_hierarchical_forecasting_table(
        results_path,
        metrics_filename=METRICS_FILENAME,
        table_filename=TABLE_FILENAME,
        benchmark=FORECAST_WINDOW_BENCHMARK,
        include_r2=True,
    )
    curve_paths = plot_forecasting_curves(
        pd.read_csv(table_path),
        results_path,
    )
    history_figures = []
    for seed_dir in sorted(results_path.glob("seed_*")):
        for fraction_dir in sorted(seed_dir.glob("fraction_*")):
            history_figures.extend(plot_saved_history(fraction_dir, plots=history_plots))
    prediction_figures = plot_saved_forecasting_predictions(
        results_path,
    )
    return {
        "table": table_path,
        "curves": curve_paths,
        "history_figures": history_figures,
        "prediction_figures": prediction_figures,
    }


def _forecast_metric_rows(
    trajectory: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    dataset,
    *,
    experiment_id: str,
    cutoff_time_min: float | None = None,
    include_r2: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for observable in FORECAST_OBSERVABLES:
        y_true = _true_observable(trajectory, observable)
        y_pred = np.asarray(predictions[observable], dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        if cutoff_time_min is not None:
            time_min = pd.to_numeric(trajectory["time_min"], errors="coerce").to_numpy(dtype=float)
            valid &= np.isfinite(time_min) & (time_min > float(cutoff_time_min))
        metric = regression_metric_values(y_true[valid], y_pred[valid], _target_scale(dataset, observable))
        row = {
            "Experiment_id": experiment_id,
            "observable": observable,
            "n": int(valid.sum()),
            "nrmse": metric["nrmse"],
            "mae": metric["mae"],
        }
        if include_r2:
            row["r2"] = metric["r2"]
        rows.append(row)
    return rows


def _forecast_prediction_rows(
    trajectory: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    experiment_id: str,
) -> list[dict[str, Any]]:
    rows = []
    for full_index, row in trajectory.reset_index(drop=True).iterrows():
        output = {
            "Experiment_id": experiment_id,
            "time_h": row.get("time_h"),
            "time_min": row.get("time_min"),
        }
        for observable in FORECAST_OBSERVABLES:
            true_values = _true_observable(trajectory.iloc[[full_index]], observable)
            output[observable] = true_values[0] if true_values.size else np.nan
            output[f"{observable}_pred"] = float(predictions[observable][full_index])
        rows.append(output)
    return rows


def _target_scale(dataset, observable: str) -> float:
    if observable == "glucose_g_l":
        idx = dataset.target_columns.index("glucose_mol_l")
        return float(dataset.target_scale[idx]) * GLUCOSE_MOLAR_MASS_G_MOL
    idx = dataset.target_columns.index(observable)
    return float(dataset.target_scale[idx])


def _true_observable(frame: pd.DataFrame, observable: str) -> np.ndarray:
    if observable in frame.columns:
        return pd.to_numeric(frame[observable], errors="coerce").to_numpy(dtype=float)
    if observable == "glucose_g_l" and "glucose_mol_l" in frame.columns:
        return pd.to_numeric(frame["glucose_mol_l"], errors="coerce").to_numpy(dtype=float) * GLUCOSE_MOLAR_MASS_G_MOL
    return np.full((len(frame),), np.nan, dtype=float)


def _write_forecasting_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    run_output: Path,
    fold_lookup: dict[str, int],
) -> None:
    """Save forecast-window metrics and shared predictions."""

    metric_columns = list(FORECAST_METRIC_COLUMNS)
    metrics.loc[:, metric_columns].to_csv(
        run_output / METRICS_FILENAME,
        index=False,
    )
    write_forecasting_table(
        run_output,
        metrics_filename=METRICS_FILENAME,
        table_filename=TABLE_FILENAME,
        benchmark=FORECAST_WINDOW_BENCHMARK,
        include_r2=True,
    )
    _write_fold_outputs(
        metrics,
        predictions,
        run_output=run_output,
        fold_lookup=fold_lookup,
        metrics_filename=FOLD_METRICS_FILENAME,
        metric_columns=metric_columns[1:],
        save_predictions=True,
    )


def _selected_experiment_ids(processed_csv: str | Path, experiment_ids: tuple[str, ...] | None) -> tuple[str, ...]:
    frame = pd.read_csv(processed_csv, usecols=["Experiment_id"])
    frame["Experiment_id"] = frame["Experiment_id"].astype(str)
    if experiment_ids is None:
        return tuple(frame["Experiment_id"].drop_duplicates().tolist())
    selected = tuple(str(value) for value in experiment_ids)
    available = set(frame["Experiment_id"].unique())
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Unknown Experiment_id values: {missing}")
    return tuple(frame[frame["Experiment_id"].isin(selected)]["Experiment_id"].drop_duplicates().tolist())


def _write_seed_forecasting_summary(seed_dir: Path) -> None:
    metric_paths = sorted(seed_dir.glob(f"fraction_*/{METRICS_FILENAME}"), key=_fraction_sort_key)
    if not metric_paths:
        return
    frames = []
    for path in metric_paths:
        frame = pd.read_csv(path)
        frame["observation_fraction"] = _forecast_fraction_from_path(path.parent)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    table = forecasting_metric_table(
        metrics,
        benchmark=FORECAST_WINDOW_BENCHMARK,
        include_r2=True,
    )
    table.to_csv(seed_dir / TABLE_FILENAME, index=False)


def _forecast_fraction_from_path(path: Path) -> float:
    for part in (path, *path.parents):
        if part.name.startswith("fraction_"):
            try:
                return float(part.name.split("_", 1)[1]) / 100.0
            except ValueError as exc:
                raise ValueError(f"Invalid forecasting fraction directory: {part}") from exc
    raise ValueError(f"No fraction directory found in path: {path}")


def _fraction_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.split("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 10**9, path.name


def _append_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _remove_fraction_seed(path: Path, observation_fraction: float, seed: int) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty or not {"observation_fraction", "seed"}.issubset(frame.columns):
        return
    keep = ~(
        pd.to_numeric(frame["observation_fraction"], errors="coerce").eq(float(observation_fraction))
        & pd.to_numeric(frame["seed"], errors="coerce").eq(int(seed))
    )
    frame.loc[keep].to_csv(path, index=False)


def _write_fold_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    run_output: Path,
    fold_lookup: dict[str, int],
    metrics_filename: str,
    metric_columns: list[str],
    save_predictions: bool,
) -> None:
    if metrics.empty and predictions.empty:
        return

    experiment_ids = sorted(
        {
            *metrics.get("Experiment_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
            *predictions.get("Experiment_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(),
        },
        key=lambda experiment_id: fold_lookup.get(experiment_id, 10**9),
    )
    for experiment_id in experiment_ids:
        fold_index = fold_lookup.get(experiment_id, len(fold_lookup))
        fold_dir = run_output / f"fold_{fold_index}_{experiment_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_metrics = metrics[metrics["Experiment_id"].astype(str).eq(experiment_id)].copy()
        fold_predictions = predictions[predictions["Experiment_id"].astype(str).eq(experiment_id)].copy()
        fold_metrics.loc[:, metric_columns].to_csv(
            fold_dir / metrics_filename, index=False
        )
        if save_predictions:
            fold_predictions.loc[:, list(FORECAST_PREDICTION_COLUMNS)].to_csv(
                fold_dir / "predictions_test.csv", index=False
            )


__all__ = [
    "DEFAULT_OBSERVATION_FRACTIONS",
    "FORECAST_WINDOW",
    "run_independent_forecasting_experiment",
    "summarize_forecasting",
]
