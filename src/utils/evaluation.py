#src/utils/evaluation.py
from __future__ import annotations

"""Evaluation helpers for the AMBR Pseudomonas PINN flow."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.dataloader import PseudomonasDataset
from src.utils.training import predict_dataset


GLUCOSE_MOLAR_MASS_G_MOL = 180.156

OBSERVABLE_PLOT_TITLES = {
    "glucose_g_l": "Glucose",
    "glucose_mol_l": "Glucose",
    "biomass_g_l": "Biomass",
    "DO_percent": "Dissolved O2",
    "O2_l_mol": "Dissolved O2",
    "pH": "pH",
}

OBSERVABLE_PLOT_YLABELS = {
    "glucose_g_l": "g/L",
    "glucose_mol_l": "mol/L",
    "biomass_g_l": "g/L",
    "DO_percent": "%",
    "O2_l_mol": "mol",
    "pH": "pH",
}

OBSERVABLE_TABLE_LABELS = {
    "biomass_g_l": "Biomass / OD",
    "glucose_g_l": "Glucose",
    "glucose_mol_l": "Glucose",
    "DO_percent": "DO",
    "O2_l_mol": "DO",
    "pH": "pH",
}

FORECAST_TABLE_ORDER = ("Biomass / OD", "Glucose", "DO", "pH")
FORECAST_PLOT_ORDER = ("Glucose", "Biomass / OD", "DO", "pH")

PLOT_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 10


def evaluate_observables(
    model,
    dataset: PseudomonasDataset,
    predictions: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Compute benchmark metrics where each observable has measured data.

    NRMSE uses the training-split variable range carried by the dataset.
    """

    predictions = predictions or predict_dataset(model, dataset)
    rows = []
    for idx, column in enumerate(dataset.target_columns):
        mask = dataset.y_mask[:, idx]
        if not np.any(mask):
            rows.append(_empty_metric_row(_reported_observable_name(column)))
            continue
        y_true, y_pred, scale, reported_column = _reported_observable_values(
            dataset,
            predictions,
            column,
            idx,
            mask,
        )
        if len(y_true) == 0:
            rows.append(_empty_metric_row(reported_column))
            continue
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
        rows.append(
            {
                "observable": reported_column,
                "n": int(mask.sum()),
                "nrmse": _safe_nrmse(rmse, scale),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": _safe_r2(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(
    model,
    dataset: PseudomonasDataset,
    predictions: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Return measured columns plus model predictions for inspection/plotting."""

    predictions = predictions or predict_dataset(model, dataset)
    observable_columns = [_reported_observable_name(column) for column in dataset.target_columns]
    if "pH" in dataset.frame.columns and "pH" in predictions and "pH" not in observable_columns:
        observable_columns.append("pH")
    frame = dataset.frame[["Experiment_id", "time_h", "time_min"]].copy()
    for column in observable_columns:
        if column in dataset.frame.columns:
            frame[column] = dataset.frame[column].to_numpy()
        elif column == "glucose_g_l" and "glucose_mol_l" in dataset.frame.columns:
            frame[column] = pd.to_numeric(dataset.frame["glucose_mol_l"], errors="coerce") * GLUCOSE_MOLAR_MASS_G_MOL
        else:
            frame[column] = np.nan
        frame[f"{column}_pred"] = predictions[column]
    return frame


def load_prediction_results(predictions_csv: str | Path) -> pd.DataFrame:
    """Read a saved predictions CSV produced by ``save_reports``."""

    frame = pd.read_csv(predictions_csv)
    required = {"Experiment_id", "time_h", "time_min"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Saved prediction file is missing columns: {missing}")
    return frame


def parameter_report(result: dict[str, Any]) -> pd.DataFrame:
    """Compare learned BIOS parameters against defaults/ranges."""

    from src.models.pinn import PseudomonasBIOSODE

    learned = result.get("learned_params", {})
    rows = []
    for name in PseudomonasBIOSODE.learnable_parameters:
        default = float(PseudomonasBIOSODE.default_parameters[name])
        value = float(learned.get(name, np.nan))
        low, high = PseudomonasBIOSODE.parameter_ranges[name]
        rows.append(
            {
                "parameter": name,
                "default_value": default,
                "learned_value": value,
                "range_low": low,
                "range_high": high,
                "relative_change": float((value - default) / default) if default else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_loss(
    result: dict[str, Any],
    output_path: str | Path | None = None,
) -> None:
    history = result.get("history", {})
    if not history:
        raise ValueError("Result has no history.")

    loss_colors = {
        "loss": "tab:olive",
        "data_loss": "tab:orange",
        "residual_loss": "tab:purple",
        "auxiliary_loss": "tab:gray",
        "regularization_loss": "tab:blue",
    }
    loss_labels = {
        "loss": "Total",
        "data_loss": "Variable",
        "residual_loss": "Residual",
        "auxiliary_loss": "Auxiliary",
        "regularization_loss": "Regularization",
    }

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    loss_keys = tuple(loss_colors)
    if "loss" not in history:
        ax.text(0.5, 0.5, "no losses", ha="center", va="center", transform=ax.transAxes)
    else:
        for key in loss_keys:
            values = history.get(key, [])
            if values:
                ax.plot(values, color=loss_colors[key], linewidth=1.4, label=loss_labels[key], linestyle="-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epochs", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Loss", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.set_title(_result_title(result), fontsize=PLOT_TITLE_FONTSIZE, fontweight="normal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path or _default_figure_path(result, "loss.png"))


def plot_r2(
    result: dict[str, Any],
    output_path: str | Path | None = None,
) -> None:
    """Plot mean train/test R2 score per epoch."""

    history = result.get("history", {})
    r2_train = np.asarray(history.get("r2_scores_train", []), dtype=float)
    r2_val = np.asarray(history.get("r2_scores_val", []), dtype=float)
    has_train = r2_train.size > 0 and np.isfinite(r2_train).any()
    has_val = r2_val.size > 0 and np.isfinite(r2_val).any()

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    if not has_train and not has_val:
        ax.text(0.5, 0.5, "no R2 history", ha="center", va="center", transform=ax.transAxes)
    else:
        if has_train:
            train_epochs = np.arange(1, len(r2_train) + 1)
            ax.plot(train_epochs, r2_train, color="tab:blue", linewidth=1.4, label="R2 train")
        if has_val:
            val_epochs = np.arange(1, len(r2_val) + 1)
            ax.plot(val_epochs, r2_val, color="tab:green", linewidth=1.4, label="R2 val")
        ax.legend(fontsize=LEGEND_FONTSIZE)
        ax.set_xscale("log")
        config = result.get("config")
        num_epochs = int(getattr(config, "num_epochs", max(len(r2_train), len(r2_val))))
        ax.set_xlim(1, num_epochs)

    ax.set_title(_result_title(result), fontsize=PLOT_TITLE_FONTSIZE, fontweight="normal")
    ax.set_xlabel("Epochs", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("R2 Score", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path or _default_figure_path(result, "r2_mean.png"))


def plot_r2_by_target(
    result: dict[str, Any],
    output_path: str | Path | None = None,
) -> None:
    """Plot train/test R2 score per target across epochs."""

    history = result.get("history", {})
    columns = [
        column
        for column in OBSERVABLE_PLOT_TITLES
        if f"r2_train_{column}" in history or f"r2_val_{column}" in history
    ]
    ncols = 2 if len(columns) > 1 else 1
    nrows = int(np.ceil(len(columns) / ncols)) if columns else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3333 * ncols, 3 * nrows), sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    fig.suptitle(f"R2 Evolution - {_result_title(result)}", fontsize=PLOT_TITLE_FONTSIZE, fontweight="normal")

    if not columns:
        axes_array[0].text(0.5, 0.5, "no target R2 history", ha="center", va="center", transform=axes_array[0].transAxes)
    for idx, (ax, column) in enumerate(zip(axes_array, columns)):
        train_values = np.asarray(history.get(f"r2_train_{column}", []), dtype=float)
        val_values = np.asarray(history.get(f"r2_val_{column}", []), dtype=float)
        has_train = train_values.size > 0 and np.isfinite(train_values).any()
        has_val = val_values.size > 0 and np.isfinite(val_values).any()

        if has_train:
            train_epochs = np.arange(1, len(train_values) + 1)
            ax.plot(train_epochs, train_values, color="tab:blue", linewidth=1.4, label="R2 train")
        if has_val:
            val_epochs = np.arange(1, len(val_values) + 1)
            ax.plot(val_epochs, val_values, color="tab:green", linewidth=1.4, label="R2 val")

        ax.set_title(OBSERVABLE_PLOT_TITLES.get(column, column), fontweight="normal", fontsize=PLOT_TITLE_FONTSIZE)
        ax.set_ylabel("R2 Score", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xscale("log")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Epochs", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=LEGEND_FONTSIZE)

    for ax in axes_array[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, output_path or _default_figure_path(result, "r2_by_target.png"))


def plot_saved_observable_predictions(
    predictions_csv: str | Path,
    experiment_id: str | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Plot measured vs saved PINN predictions from a predictions CSV."""

    frame = load_prediction_results(predictions_csv)
    if experiment_id is not None:
        frame = frame[frame["Experiment_id"].astype(str) == str(experiment_id)]
    if frame.empty:
        raise ValueError("No rows available for plotting.")

    title_suffix = experiment_id or _prediction_file_label(predictions_csv)
    output = output_path or Path(predictions_csv).with_name(f"observable_predictions_{title_suffix}.png")
    return _plot_prediction_frame(frame, title=title or title_suffix, output_path=output)


def plot_saved_prediction_splits(
    fold_dir: str | Path,
    splits: tuple[str, ...] = ("test", "val"),
) -> list[Path]:
    """Create observable prediction figures from saved per-split prediction CSVs."""

    fold_path = Path(fold_dir)
    plot_paths = []
    for split in splits:
        predictions_csv = fold_path / f"predictions_{split}.csv"
        if not predictions_csv.exists():
            continue
        frame = pd.read_csv(predictions_csv, usecols=["Experiment_id"])
        experiment_ids = frame["Experiment_id"].dropna().astype(str).drop_duplicates().tolist()
        if split == "train" and len(experiment_ids) > 1:
            continue
        experiment_id = experiment_ids[0] if len(experiment_ids) == 1 else None
        label = experiment_id or split
        output_path = fold_path / f"{split}_{label}.png"
        plot_paths.append(
            plot_saved_observable_predictions(
                predictions_csv,
                experiment_id=experiment_id,
                output_path=output_path,
                title=fold_path.name,
            )
        )
        plt.close("all")
    return plot_paths


def plot_saved_history(
    fold_dir: str | Path,
    plots: tuple[str, ...] = ("loss", "r2", "r2_by_target"),
) -> list[Path]:
    """Create history figures from a saved checkpoints/history.csv file."""

    fold_path = Path(fold_dir)
    history_result = _saved_history_result(fold_path)
    if history_result is None:
        return []

    plot_paths = []
    if "loss" in plots:
        output_path = fold_path / "loss.png"
        plot_loss(history_result, output_path=output_path)
        plt.close("all")
        plot_paths.append(output_path)
    if "r2" in plots:
        output_path = fold_path / "r2_mean.png"
        plot_r2(history_result, output_path=output_path)
        plt.close("all")
        plot_paths.append(output_path)
    if "r2_by_target" in plots:
        output_path = fold_path / "r2_by_target.png"
        plot_r2_by_target(history_result, output_path=output_path)
        plt.close("all")
        plot_paths.append(output_path)
    return plot_paths


def write_leave_one_out_table(
    results_dir: str | Path,
    *,
    metrics_filename: str = "leave_one_out_metrics_long.csv",
    table_filename: str = "leave_one_out_table.csv",
    benchmark: str = "Leave-One-Bioreactor-Out",
) -> Path:
    """Create the LOO summary table from the saved long-form metrics CSV."""

    results_path = Path(results_dir)
    metrics_path = results_path / metrics_filename
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    table = leave_one_out_metric_table(
        metrics,
        benchmark=benchmark,
        observable_labels=OBSERVABLE_TABLE_LABELS,
    )
    table_path = results_path / table_filename
    table.to_csv(table_path, index=False)
    return table_path


def write_hierarchical_leave_one_out_table(
    results_dir: str | Path,
    *,
    table_filename: str = "leave_one_out_table.csv",
    benchmark: str = "Leave-One-Bioreactor-Out",
) -> Path:
    """Aggregate seeds within each fold, then summarize across fold means."""

    results_path = Path(results_dir)
    metric_paths = sorted(results_path.glob("seed_*/leave_one_out_metrics_long.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No seed metrics found under: {results_path}")
    metrics = pd.concat((pd.read_csv(path) for path in metric_paths), ignore_index=True)
    required = {"fold_index", "observable", "r2", "mae", "nrmse"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"LOO seed metrics are missing columns: {missing}")

    test_metrics = metrics[metrics["split"].eq("test")].copy() if "split" in metrics.columns else metrics.copy()
    fold_means = (
        test_metrics.groupby(["fold_index", "observable"], dropna=False)[["r2", "mae", "nrmse"]]
        .mean()
        .reset_index()
    )
    table = leave_one_out_metric_table(
        fold_means,
        benchmark=benchmark,
        observable_labels=OBSERVABLE_TABLE_LABELS,
    )
    table_path = results_path / table_filename
    table.to_csv(table_path, index=False)
    return table_path


def leave_one_out_metric_table(
    metrics: pd.DataFrame,
    *,
    benchmark: str,
    observable_labels: dict[str, str],
) -> pd.DataFrame:
    """Build the compact LOO table from long-form test metrics."""

    columns = [
        "Benchmark",
        "Observable",
        "R2 mean",
        "R2 median",
        "R2 std",
        "MAE mean",
        "MAE median",
        "MAE std",
        "NRMSE mean",
        "NRMSE median",
        "NRMSE std",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    test_metrics = metrics[metrics["split"].eq("test")].copy() if "split" in metrics.columns else metrics.copy()
    grouped = (
        test_metrics.groupby("observable", dropna=False)[["r2", "mae", "nrmse"]]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    grouped.columns = _flatten_columns(grouped.columns)
    grouped["Benchmark"] = benchmark
    grouped["Observable"] = grouped["observable"].map(lambda value: observable_labels.get(str(value), str(value)))
    grouped = grouped.rename(
        columns={
            "r2_mean": "R2 mean",
            "r2_median": "R2 median",
            "r2_std": "R2 std",
            "mae_mean": "MAE mean",
            "mae_median": "MAE median",
            "mae_std": "MAE std",
            "nrmse_mean": "NRMSE mean",
            "nrmse_median": "NRMSE median",
            "nrmse_std": "NRMSE std",
        }
    )
    observable_order = {"Biomass / OD": 0, "Glucose": 1, "DO": 2, "pH": 3}
    grouped["_observable_order"] = grouped["Observable"].map(observable_order).fillna(len(observable_order))
    grouped = grouped.sort_values("_observable_order").drop(columns="_observable_order")
    return grouped.loc[:, columns].reset_index(drop=True)


def _flatten_columns(columns: Any) -> list[str]:
    flattened = []
    for column in columns:
        if isinstance(column, tuple):
            flattened.append("_".join(str(part) for part in column if part))
        else:
            flattened.append(str(column))
    return flattened


def _plot_prediction_frame(
    frame: pd.DataFrame,
    *,
    title: str,
    output_path: str | Path,
    forecast_start_h: float | None = None,
    metric_start_h: float | None = None,
) -> Path:
    if frame.empty:
        raise ValueError("No rows available for plotting.")
    columns = [column[:-5] for column in frame.columns if column.endswith("_pred")]
    ncols = 2 if len(columns) > 1 else 1
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3333 * ncols, 3 * nrows), sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=PLOT_TITLE_FONTSIZE, fontweight="normal")
    data_color = "tab:blue"
    pinn_color = "tab:orange"

    for idx, (ax, column) in enumerate(zip(axes_array, columns)):
        measured = frame[column].notna()
        if measured.any():
            measured_frame = frame.loc[measured].sort_values("time_h")
            ax.plot(
                measured_frame["time_h"],
                measured_frame[column],
                "o",
                label="Data",
                color=data_color,
                linestyle="None",
                markersize=2.4,
                alpha=0.9,
            )
        ax.plot(
            frame["time_h"],
            frame[f"{column}_pred"],
            "--",
            label="PINN",
            color=pinn_color,
            linewidth=1.4,
            alpha=0.95,
        )
        if forecast_start_h is not None and np.isfinite(forecast_start_h):
            ax.axvline(
                forecast_start_h,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                label="Forecast start",
            )
        metric_rows = measured & frame[f"{column}_pred"].notna()
        if metric_start_h is not None and np.isfinite(metric_start_h):
            metric_rows &= pd.to_numeric(frame["time_h"], errors="coerce") > metric_start_h
        if metric_rows.any():
            measured_frame = frame.loc[metric_rows].sort_values("time_h")
            r2_value = _safe_r2(
                measured_frame[column].to_numpy(),
                measured_frame[f"{column}_pred"].to_numpy(),
            )
            if np.isfinite(r2_value):
                ax.text(
                    0.62,
                    0.48,
                    f"R2 = {r2_value:.4f}",
                    transform=ax.transAxes,
                    fontsize=AXIS_LABEL_FONTSIZE,
                    ha="left",
                    va="center",
                )
        ax.set_title(OBSERVABLE_PLOT_TITLES.get(column, column), fontweight="normal", fontsize=PLOT_TITLE_FONTSIZE)
        ax.set_ylabel(OBSERVABLE_PLOT_YLABELS.get(column, column), fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Time (h)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        legend_map = dict(zip(labels, handles))
        ordered_labels = [
            "PINN",
            "Data",
            "Forecast start",
        ]
        ordered_labels = [name for name in ordered_labels if name in legend_map]
        if ordered_labels:
            ordered_handles = [legend_map[name] for name in ordered_labels]
            ax.legend(ordered_handles, ordered_labels, loc="best", fontsize=LEGEND_FONTSIZE)
        else:
            ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)

    for ax in axes_array[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _result_title(result: dict[str, Any]) -> str:
    config = result.get("config")
    experiment_name = getattr(config, "experiment_name", None)
    if experiment_name:
        return f"{experiment_name}"
    split_metadata = result.get("split_metadata", {})
    fold_index = split_metadata.get("fold_index") if isinstance(split_metadata, dict) else None
    if fold_index is not None:
        return f"Leave-One-Bioreactor-Out fold {fold_index}"
    return "PINN"


def _save_figure(fig, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


def _default_figure_path(result: dict[str, Any], filename: str) -> Path:
    output_dir = Path(result.get("output_dir", "results/pseudomonas_pinn"))
    return output_dir / filename


def _prediction_file_label(predictions_csv: str | Path) -> str:
    path = Path(predictions_csv)
    return path.parent.name or path.stem


def _saved_history_result(fold_dir: Path) -> dict | None:
    history_path = fold_dir / "checkpoints" / "history.csv"
    if not history_path.exists():
        return None
    history_frame = pd.read_csv(history_path)
    history = {
        column: history_frame[column].dropna().tolist()
        for column in history_frame.columns
    }
    config = SimpleNamespace(
        experiment_name=fold_dir.name,
        num_epochs=len(history_frame),
    )
    return {
        "history": history,
        "config": config,
        "output_dir": fold_dir,
    }


def save_reports(result: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_dir or result.get("output_dir", "results/pseudomonas_pinn"))
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = result["train_dataset"]
    val_dataset = result.get("val_dataset")
    test_dataset = result["test_dataset"]
    train_predictions = result.get("train_predictions")
    val_predictions = result.get("val_predictions")
    test_predictions = result.get("test_predictions")
    metrics = evaluate_observables(result["model"], train_dataset, train_predictions)
    params = parameter_report(result)
    train_metrics_path = output / "metrics_train.csv"
    test_metrics_path = output / "metrics_test.csv"
    params_path = output / "parameter_report.csv"
    predictions_train_path = output / "predictions_train.csv"
    predictions_test_path = output / "predictions_test.csv"
    metrics.to_csv(train_metrics_path, index=False)
    evaluate_observables(result["model"], test_dataset, test_predictions).to_csv(test_metrics_path, index=False)
    params.to_csv(params_path, index=False)
    prediction_frame(result["model"], train_dataset, train_predictions).to_csv(predictions_train_path, index=False)
    prediction_frame(result["model"], test_dataset, test_predictions).to_csv(predictions_test_path, index=False)
    paths = {
        "metrics_train": train_metrics_path,
        "metrics_test": test_metrics_path,
        "parameters": params_path,
        "predictions_train": predictions_train_path,
        "predictions_test": predictions_test_path,
    }
    if val_dataset is not None:
        metrics_val_path = output / "metrics_val.csv"
        predictions_val_path = output / "predictions_val.csv"
        evaluate_observables(result["model"], val_dataset, val_predictions).to_csv(metrics_val_path, index=False)
        prediction_frame(result["model"], val_dataset, val_predictions).to_csv(predictions_val_path, index=False)
        paths.update(
            {
                "metrics_val": metrics_val_path,
                "predictions_val": predictions_val_path,
            }
        )
    return paths


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) < 1e-12:
        return float("nan")
    return float(max(0.0, r2_score(y_true, y_pred)))


def _reported_observable_name(column: str) -> str:
    return "glucose_g_l" if column == "glucose_mol_l" else column


def _reported_observable_values(
    dataset: PseudomonasDataset,
    predictions: dict[str, np.ndarray],
    column: str,
    idx: int,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    if column != "glucose_mol_l":
        return (
            dataset.y[mask, idx],
            np.asarray(predictions[column])[mask],
            float(dataset.target_scale[idx]),
            column,
        )

    reported_column = "glucose_g_l"
    if reported_column in dataset.frame.columns:
        y_true = pd.to_numeric(dataset.frame.loc[mask, reported_column], errors="coerce").to_numpy(dtype=float)
    else:
        y_true = dataset.y[mask, idx] * GLUCOSE_MOLAR_MASS_G_MOL
    y_pred = np.asarray(predictions[reported_column])[mask]
    scale = float(dataset.target_scale[idx]) * GLUCOSE_MOLAR_MASS_G_MOL
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[valid], y_pred[valid], scale, reported_column


def _safe_nrmse(rmse: float, scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 1e-12:
        return float("nan")
    return float(rmse / scale)


def _empty_metric_row(column: str) -> dict[str, float | str | int]:
    return {"observable": column, "n": 0, "nrmse": np.nan, "mae": np.nan, "r2": np.nan}


def regression_metric_values(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> dict[str, float]:
    """Compute the same R2, MAE and train-range NRMSE used by LOO."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return {"r2": np.nan, "mae": np.nan, "nrmse": np.nan}
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return {
        "r2": _safe_r2(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "nrmse": _safe_nrmse(rmse, scale),
    }


def forecasting_metric_table(
    metrics: pd.DataFrame,
    *,
    benchmark: str = "Forecasting Scan",
    include_observation_fraction: bool = True,
    include_r2: bool = True,
) -> pd.DataFrame:
    """Average seeds per reactor first, then summarize across reactor means."""

    columns = [
        "Benchmark",
        "Observable",
        "MAE mean",
        "MAE median",
        "MAE std",
        "NRMSE mean",
        "NRMSE median",
        "NRMSE std",
    ]
    if include_r2:
        columns[2:2] = ["R2 mean", "R2 median", "R2 std"]
    if include_observation_fraction:
        columns.insert(1, "Observation fraction")
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    metric_columns = ["mae", "nrmse"]
    if include_r2:
        metric_columns.insert(0, "r2")
    required = {"observation_fraction", "Experiment_id", "observable", *metric_columns}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Forecasting metrics are missing columns: {missing}")
    reactor_means = (
        metrics.groupby(["observation_fraction", "Experiment_id", "observable"], dropna=False)[
            metric_columns
        ]
        .mean()
        .reset_index()
    )
    grouped = (
        reactor_means.groupby(["observation_fraction", "observable"], dropna=False)[metric_columns]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    grouped.columns = _flatten_columns(grouped.columns)
    grouped["Benchmark"] = benchmark
    grouped["Observation fraction"] = grouped["observation_fraction"]
    grouped["Observable"] = grouped["observable"].map(
        lambda value: OBSERVABLE_TABLE_LABELS.get(str(value), str(value))
    )
    grouped = grouped.rename(
        columns={
            "r2_mean": "R2 mean",
            "r2_median": "R2 median",
            "r2_std": "R2 std",
            "mae_mean": "MAE mean",
            "mae_median": "MAE median",
            "mae_std": "MAE std",
            "nrmse_mean": "NRMSE mean",
            "nrmse_median": "NRMSE median",
            "nrmse_std": "NRMSE std",
        }
    )
    order = {name: index for index, name in enumerate(FORECAST_TABLE_ORDER)}
    grouped["_observable_order"] = grouped["Observable"].map(order).fillna(len(order))
    grouped = grouped.sort_values(["Observation fraction", "_observable_order"]).drop(columns="_observable_order")
    return grouped.loc[:, columns].reset_index(drop=True)


def write_forecasting_table(
    results_dir: str | Path,
    *,
    metrics_filename: str = "forecasting_metrics_long.csv",
    table_filename: str = "forecasting_table.csv",
    benchmark: str = "Forecasting Scan",
    include_r2: bool = True,
) -> Path:
    results_path = Path(results_dir)
    metrics_path = results_path / metrics_filename
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing forecasting metrics: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    metrics["observation_fraction"] = _forecast_fraction_from_path(results_path)
    table = forecasting_metric_table(
        metrics,
        benchmark=benchmark,
        include_observation_fraction=False,
        include_r2=include_r2,
    )
    table_path = results_path / table_filename
    table.to_csv(table_path, index=False)
    return table_path


def write_hierarchical_forecasting_table(
    results_dir: str | Path,
    *,
    metrics_filename: str = "forecasting_metrics_long.csv",
    table_filename: str = "forecasting_table.csv",
    benchmark: str = "Forecasting Scan",
    include_r2: bool = True,
) -> Path:
    results_path = Path(results_dir)
    metric_paths = sorted(results_path.glob(f"seed_*/fraction_*/{metrics_filename}"))
    if not metric_paths:
        raise FileNotFoundError(f"No seed forecasting metrics found under: {results_path}")
    frames = []
    for path in metric_paths:
        frame = pd.read_csv(path)
        frame["observation_fraction"] = _forecast_fraction_from_path(path.parent)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    table = forecasting_metric_table(metrics, benchmark=benchmark, include_r2=include_r2)
    table_path = results_path / table_filename
    table.to_csv(table_path, index=False)
    return table_path


def plot_forecasting_curves(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename_suffix: str = "",
) -> dict[str, Path]:
    """Plot forecasting metric curves."""

    output_path = Path(output_dir)
    outputs = {}
    specs = {
        "r2": ("R2 mean", "Forecast R2", f"forecast_r2_curve{filename_suffix}.png"),
        "mae": ("MAE mean", "Forecast MAE", f"forecast_mae_curve{filename_suffix}.png"),
        "nrmse": ("NRMSE mean", "Forecast NRMSE", f"forecast_nrmse_curve{filename_suffix}.png"),
    }
    for key, (column, ylabel, filename) in specs.items():
        if column not in table.columns:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10.6666, 6), sharex=True)
        for ax, observable in zip(axes.ravel(), FORECAST_PLOT_ORDER):
            data = table[table["Observable"].eq(observable)].sort_values("Observation fraction")
            if data.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.plot(data["Observation fraction"], data[column], color="tab:blue", marker="o", linewidth=1.4)
            ax.set_title(observable, fontsize=PLOT_TITLE_FONTSIZE, fontweight="normal")
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, alpha=0.3)
        for ax in axes.ravel()[2:]:
            ax.set_xlabel("Observed trajectory fraction", fontsize=AXIS_LABEL_FONTSIZE)
        fig.tight_layout()
        path = output_path / filename
        _save_figure(fig, path)
        plt.close(fig)
        outputs[key] = path
    return outputs


def plot_saved_forecast_predictions(
    predictions_csv: str | Path,
    *,
    output_path: str | Path | None = None,
    forecast_start_h: float | None = None,
    filename_suffix: str = "",
) -> Path:
    """Plot one saved reactor forecast."""

    predictions_path = Path(predictions_csv)
    frame = pd.read_csv(predictions_path)
    if frame.empty:
        raise ValueError(f"No forecast predictions in {predictions_path}")
    plot_frame = frame.copy()
    for column in tuple(plot_frame.columns):
        if column.endswith("_forecast"):
            plot_frame[column.removesuffix("_forecast") + "_pred"] = plot_frame[column]
    experiment_ids = plot_frame["Experiment_id"].dropna().astype(str).unique().tolist()
    experiment_label = experiment_ids[0] if len(experiment_ids) == 1 else predictions_path.parent.name
    fraction_label = predictions_path.parent.parent.name
    title = f"{fraction_label} - {experiment_label}"
    time_h = pd.to_numeric(plot_frame["time_h"], errors="coerce").to_numpy(dtype=float)
    finite_time = time_h[np.isfinite(time_h)]
    if forecast_start_h is None and finite_time.size:
        fraction = _forecast_fraction_from_path(predictions_path)
        forecast_start_h = float(finite_time.min() + fraction * (finite_time.max() - finite_time.min()))
    elif forecast_start_h is None:
        forecast_start_h = None
    output = (
        Path(output_path)
        if output_path is not None
        else predictions_path.with_name(f"test_{experiment_label}{filename_suffix}.png")
    )
    return _plot_prediction_frame(
        plot_frame,
        title=title,
        output_path=output,
        forecast_start_h=forecast_start_h,
        metric_start_h=forecast_start_h,
    )


def _forecast_fraction_from_path(path: Path) -> float:
    for part in (path, *path.parents):
        if part.name.startswith("fraction_"):
            try:
                return float(part.name.split("_", 1)[1]) / 100.0
            except ValueError as exc:
                raise ValueError(f"Invalid forecasting fraction directory: {part}") from exc
    raise ValueError(f"No fraction directory found in path: {path}")


def plot_saved_forecasting_predictions(
    results_dir: str | Path,
    *,
    filename_suffix: str = "",
) -> list[Path]:
    """Plot all saved reactor forecasts."""

    paths = []
    for predictions_path in sorted(Path(results_dir).glob("seed_*/fraction_*/fold_*/predictions_test.csv")):
        paths.append(
            plot_saved_forecast_predictions(
                predictions_path,
                filename_suffix=filename_suffix,
            )
        )
        plt.close("all")
    return paths


__all__ = [
    "evaluate_observables",
    "forecasting_metric_table",
    "leave_one_out_metric_table",
    "load_prediction_results",
    "parameter_report",
    "plot_saved_history",
    "plot_saved_forecast_predictions",
    "plot_saved_forecasting_predictions",
    "plot_saved_observable_predictions",
    "plot_saved_prediction_splits",
    "plot_loss",
    "plot_forecasting_curves",
    "plot_r2",
    "plot_r2_by_target",
    "prediction_frame",
    "save_reports",
    "regression_metric_values",
    "write_forecasting_table",
    "write_hierarchical_forecasting_table",
    "write_leave_one_out_table",
    "write_hierarchical_leave_one_out_table",
]
