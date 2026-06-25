from __future__ import annotations

"""Re-score saved forecasting outputs with dense interpolated glucose labels."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.evaluation import (
    GLUCOSE_MOLAR_MASS_G_MOL,
    forecasting_metric_table,
    plot_forecasting_curves,
    plot_saved_forecast_predictions,
    regression_metric_values,
    write_forecasting_table,
    write_hierarchical_forecasting_table,
)


FORECAST_WINDOW_BENCHMARK = "Forecasting Scan"
FORECAST_OBSERVABLES = ("glucose_g_l", "biomass_g_l", "DO_percent", "pH")
METRIC_COLUMNS = ("observable", "n", "nrmse", "mae", "r2")
SUFFIX = "_interpolation"


def rescore_forecasting_with_glucose_interpolation(
    results_dir: str | Path,
    *,
    suffix: str = SUFFIX,
    generate_figures: bool = True,
) -> dict[str, Any]:
    """Recompute forecast-window metrics using linearly interpolated glucose truth.

    The helper reads saved ``predictions_test.csv`` files, fills glucose ground-truth
    values between measured glucose samples, and leaves other observables unchanged.
    It writes new metric, table, prediction, and figure artifacts with ``suffix``.
    """

    results_path = Path(results_dir)
    metrics_filename = f"forecasting_metrics_long{suffix}.csv"
    table_filename = f"forecasting_table{suffix}.csv"
    fold_metrics_filename = f"metrics_test{suffix}.csv"
    prediction_filename = f"predictions_test{suffix}.csv"

    fraction_dirs = sorted(results_path.glob("seed_*/fraction_*"), key=_fraction_sort_key)
    if not fraction_dirs:
        raise FileNotFoundError(f"No forecasting fraction directories found under: {results_path}")

    prediction_paths: list[Path] = []
    for fraction_dir in fraction_dirs:
        metrics = _rescore_fraction(
            fraction_dir,
            fold_metrics_filename=fold_metrics_filename,
            prediction_filename=prediction_filename,
        )
        metrics.to_csv(fraction_dir / metrics_filename, index=False)
        write_forecasting_table(
            fraction_dir,
            metrics_filename=metrics_filename,
            table_filename=table_filename,
            benchmark=FORECAST_WINDOW_BENCHMARK,
            include_r2=True,
        )
        prediction_paths.extend(sorted(fraction_dir.glob(f"fold_*/{prediction_filename}"), key=_fold_sort_key))

    seed_dirs = sorted({fraction_dir.parent for fraction_dir in fraction_dirs})
    for seed_dir in seed_dirs:
        _write_seed_summary(
            seed_dir,
            metrics_filename=metrics_filename,
            table_filename=table_filename,
        )

    table_path = write_hierarchical_forecasting_table(
        results_path,
        metrics_filename=metrics_filename,
        table_filename=table_filename,
        benchmark=FORECAST_WINDOW_BENCHMARK,
        include_r2=True,
    )
    curves = plot_forecasting_curves(pd.read_csv(table_path), results_path, filename_suffix=suffix)

    prediction_figures = []
    if generate_figures:
        for prediction_path in prediction_paths:
            prediction_figures.append(
                plot_saved_forecast_predictions(
                    prediction_path,
                    filename_suffix=suffix,
                )
            )
            plt.close("all")

    return {
        "table": table_path,
        "curves": curves,
        "prediction_figures": prediction_figures,
        "metrics_filename": metrics_filename,
        "fold_metrics_filename": fold_metrics_filename,
        "prediction_filename": prediction_filename,
    }


def _rescore_fraction(
    fraction_dir: Path,
    *,
    fold_metrics_filename: str,
    prediction_filename: str,
) -> pd.DataFrame:
    config = _load_config(fraction_dir)
    cutoff_by_experiment = {
        str(key): float(value)
        for key, value in config.get("data", {}).get("cutoff_time_min", {}).items()
    }
    scales = _load_observable_scales(fraction_dir)
    rows = []
    for predictions_path in sorted(fraction_dir.glob("fold_*/predictions_test.csv"), key=_fold_sort_key):
        frame = pd.read_csv(predictions_path)
        if frame.empty:
            continue
        experiment_ids = frame["Experiment_id"].dropna().astype(str).unique().tolist()
        if len(experiment_ids) != 1:
            raise ValueError(f"Expected one Experiment_id in {predictions_path}, found {experiment_ids}")
        experiment_id = experiment_ids[0]
        if experiment_id not in cutoff_by_experiment:
            raise KeyError(f"Missing forecast cutoff for {experiment_id} in {fraction_dir / 'config.json'}")

        dense_frame = _with_interpolated_glucose(frame)
        dense_frame.to_csv(predictions_path.with_name(prediction_filename), index=False)
        fold_metrics = _metric_rows(
            dense_frame,
            experiment_id=experiment_id,
            cutoff_time_min=cutoff_by_experiment[experiment_id],
            scales=scales,
        )
        pd.DataFrame(fold_metrics).loc[:, METRIC_COLUMNS].to_csv(
            predictions_path.with_name(fold_metrics_filename),
            index=False,
        )
        rows.extend(fold_metrics)
    return pd.DataFrame(rows)


def _write_seed_summary(
    seed_dir: Path,
    *,
    metrics_filename: str,
    table_filename: str,
) -> None:
    metric_paths = sorted(seed_dir.glob(f"fraction_*/{metrics_filename}"), key=_fraction_sort_key)
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
    table.to_csv(seed_dir / table_filename, index=False)


def _with_interpolated_glucose(frame: pd.DataFrame) -> pd.DataFrame:
    dense = frame.copy()
    if "glucose_g_l" not in dense.columns:
        return dense

    time_min = pd.to_numeric(dense["time_min"], errors="coerce").to_numpy(dtype=float)
    glucose = pd.to_numeric(dense["glucose_g_l"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_min) & np.isfinite(glucose)
    if valid.sum() < 2:
        dense["glucose_g_l"] = np.nan
        return dense

    points = (
        pd.DataFrame({"time_min": time_min[valid], "glucose_g_l": glucose[valid]})
        .groupby("time_min", as_index=False)["glucose_g_l"]
        .mean()
        .sort_values("time_min")
    )
    xp = points["time_min"].to_numpy(dtype=float)
    fp = points["glucose_g_l"].to_numpy(dtype=float)
    interpolated = np.interp(time_min, xp, fp)
    interpolated[(time_min < xp[0]) | (time_min > xp[-1]) | ~np.isfinite(time_min)] = np.nan
    dense["glucose_g_l"] = interpolated
    return dense


def _metric_rows(
    frame: pd.DataFrame,
    *,
    experiment_id: str,
    cutoff_time_min: float,
    scales: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    time_min = pd.to_numeric(frame["time_min"], errors="coerce").to_numpy(dtype=float)
    in_window = np.isfinite(time_min) & (time_min > float(cutoff_time_min))
    for observable in FORECAST_OBSERVABLES:
        y_true = pd.to_numeric(frame[observable], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(frame[f"{observable}_pred"], errors="coerce").to_numpy(dtype=float)
        valid = in_window & np.isfinite(y_true) & np.isfinite(y_pred)
        metric = regression_metric_values(y_true[valid], y_pred[valid], scales[observable])
        rows.append(
            {
                "Experiment_id": experiment_id,
                "observable": observable,
                "n": int(valid.sum()),
                "nrmse": metric["nrmse"],
                "mae": metric["mae"],
                "r2": metric["r2"],
            }
        )
    return rows


def _load_observable_scales(fraction_dir: Path) -> dict[str, float]:
    columns_path = fraction_dir / "checkpoints" / "dataset_columns.json"
    scalers_path = fraction_dir / "checkpoints" / "scalers.npz"
    if not columns_path.exists() or not scalers_path.exists():
        raise FileNotFoundError(f"Missing saved dataset columns or scalers under: {fraction_dir / 'checkpoints'}")

    columns = json.loads(columns_path.read_text(encoding="utf-8"))
    target_columns = list(columns["target_columns"])
    with np.load(scalers_path) as saved:
        target_scale = np.asarray(saved["target_scale"], dtype=float)

    scales = {
        column: float(target_scale[index])
        for index, column in enumerate(target_columns)
    }
    return {
        "glucose_g_l": scales["glucose_mol_l"] * GLUCOSE_MOLAR_MASS_G_MOL,
        "biomass_g_l": scales["biomass_g_l"],
        "DO_percent": scales["DO_percent"],
        "pH": scales["pH"],
    }


def _load_config(fraction_dir: Path) -> dict[str, Any]:
    config_path = fraction_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing forecasting config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _forecast_fraction_from_path(path: Path) -> float:
    for part in (path, *path.parents):
        if part.name.startswith("fraction_"):
            try:
                return float(part.name.split("_", 1)[1]) / 100.0
            except ValueError as exc:
                raise ValueError(f"Invalid forecasting fraction directory: {part}") from exc
    raise ValueError(f"No fraction directory found in path: {path}")


def _fraction_sort_key(path: Path) -> tuple[str, int, str]:
    seed = path.parent.name
    try:
        fraction = int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        fraction = 10**9
    return seed, fraction, path.name


def _fold_sort_key(path: Path) -> tuple[int, str]:
    fold_name = path.parent.name if path.name.startswith("predictions_") else path.name
    parts = fold_name.split("_")
    try:
        return int(parts[1]), fold_name
    except (IndexError, ValueError):
        return 10**9, fold_name


__all__ = ["rescore_forecasting_with_glucose_interpolation"]
