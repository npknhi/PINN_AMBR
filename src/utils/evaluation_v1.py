#src/utils/evaluation_v1.py
from __future__ import annotations

"""Evaluation helpers for the AMBR Pseudomonas PINN flow."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.dataloader_v1 import PseudomonasDataset
from src.utils.training_v1 import predict_dataset


GLUCOSE_MOLAR_MASS_G_MOL = 180.156

OBSERVABLE_PLOT_COLORS = {
    "glucose_g_l": "tab:blue",
    "glucose_mol_l": "tab:blue",
    "biomass_g_l": "tab:green",
    "DO_percent": "tab:pink",
    "O2_l_mol": "tab:pink",
    "pH": "tab:orange",
}

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


def evaluate_observables(model, dataset: PseudomonasDataset) -> pd.DataFrame:
    """Compute benchmark metrics where each observable has measured data.

    NRMSE uses the training-split variable range carried by the dataset.
    """

    predictions = predict_dataset(model, dataset)
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


def prediction_frame(model, dataset: PseudomonasDataset) -> pd.DataFrame:
    """Return measured columns plus model predictions for inspection/plotting."""

    predictions = predict_dataset(model, dataset)
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

    from src.models.pinn_v1 import PseudomonasBIOSODE

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
                ax.plot(values, color=loss_colors[key], linewidth=1.8, label=loss_labels[key], linestyle="-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epochs", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8)
    ax.set_title(_result_title(result), fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path or _default_figure_path(result, "loss.png"))


def plot_r2(
    result: dict[str, Any],
    output_path: str | Path | None = None,
) -> None:
    """Plot mean train/test R2 score per epoch."""

    history = result.get("history", {})
    r2_train = history.get("r2_scores_train", [])
    r2_val = history.get("r2_scores_val", [])

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    if not r2_train or not r2_val:
        ax.text(0.5, 0.5, "no R2 history", ha="center", va="center", transform=ax.transAxes)
    else:
        train_epochs = np.arange(1, len(r2_train) + 1)
        val_epochs = np.arange(1, len(r2_val) + 1)
        ax.plot(train_epochs, r2_train, color="tab:blue", linewidth=2, label="R2 train")
        ax.plot(val_epochs, r2_val, color="tab:green", linewidth=2, label="R2 val")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        config = result.get("config")
        num_epochs = int(getattr(config, "num_epochs", max(len(r2_train), len(r2_val))))
        ax.set_xlim(1, num_epochs)

    ax.set_title(_result_title(result), fontsize=10, fontweight="bold")
    ax.set_xlabel("Epochs", fontsize=9)
    ax.set_ylabel("R2 Score", fontsize=9)
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
    fig.suptitle(f"R2 Evolution - {_result_title(result)}", fontsize=16, fontweight="bold")

    if not columns:
        axes_array[0].text(0.5, 0.5, "no target R2 history", ha="center", va="center", transform=axes_array[0].transAxes)
    for idx, (ax, column) in enumerate(zip(axes_array, columns)):
        train_values = history.get(f"r2_train_{column}", [])
        val_values = history.get(f"r2_val_{column}", [])

        if train_values:
            train_epochs = np.arange(1, len(train_values) + 1)
            ax.plot(train_epochs, train_values, color="tab:blue", linewidth=2, label="R2 train")
        if val_values:
            val_epochs = np.arange(1, len(val_values) + 1)
            ax.plot(val_epochs, val_values, color="tab:green", linewidth=2, label="R2 val")

        ax.set_title(OBSERVABLE_PLOT_TITLES.get(column, column), fontweight="bold", fontsize=12)
        ax.set_ylabel("R2 Score", fontsize=10)
        ax.set_xscale("log")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Epochs", fontsize=10)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=9)

    for ax in axes_array[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, output_path or _default_figure_path(result, "r2_by_target.png"))


def plot_observable_predictions(
    model,
    dataset: PseudomonasDataset,
    experiment_id: str | None = None,
    output_path: str | Path | None = None,
) -> None:
    frame = prediction_frame(model, dataset)
    if experiment_id is not None:
        frame = frame[frame["Experiment_id"] == experiment_id]
    if frame.empty:
        raise ValueError("No rows available for plotting.")

    title_suffix = experiment_id or _dataset_label(dataset)
    _plot_prediction_frame(
        frame,
        title_suffix=title_suffix,
        output_path=output_path or _default_prediction_figure_path(dataset, experiment_id),
    )


def plot_saved_observable_predictions(
    predictions_csv: str | Path,
    experiment_id: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Plot measured vs saved PINN predictions from a predictions CSV."""

    frame = load_prediction_results(predictions_csv)
    if experiment_id is not None:
        frame = frame[frame["Experiment_id"].astype(str) == str(experiment_id)]
    if frame.empty:
        raise ValueError("No rows available for plotting.")

    title_suffix = experiment_id or _prediction_file_label(predictions_csv)
    output = output_path or Path(predictions_csv).with_name(f"observable_predictions_{title_suffix}.png")
    return _plot_prediction_frame(frame, title_suffix=title_suffix, output_path=output)


def plot_loo_test_predictions(
    results_dir: str | Path = "results/LOO",
    experiment_ids: Sequence[str] = (),
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Plot saved test predictions for selected LOO held-out bioreactors."""

    results_path = Path(results_dir)
    output_path = Path(output_dir) if output_dir is not None else results_path / "diagnostic_plots"
    paths: dict[str, Path] = {}
    for experiment_id in experiment_ids:
        prediction_csv = _loo_prediction_csv(results_path, experiment_id, seed=seed)
        target_path = output_path / f"observable_predictions_{experiment_id}.png"
        paths[str(experiment_id)] = plot_saved_observable_predictions(
            prediction_csv,
            experiment_id=experiment_id,
            output_path=target_path,
        )
    return paths


def _plot_prediction_frame(frame: pd.DataFrame, *, title_suffix: str, output_path: str | Path) -> Path:
    if frame.empty:
        raise ValueError("No rows available for plotting.")
    columns = [column[:-5] for column in frame.columns if column.endswith("_pred")]
    ncols = 2 if len(columns) > 1 else 1
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3333 * ncols, 3 * nrows), sharex=True)
    axes_array = np.atleast_1d(axes).ravel()
    fig.suptitle(f"PINN vs Data Comparison - {title_suffix}", fontsize=16, fontweight="bold")
    data_color = "tab:green"
    pinn_color = "tab:blue"

    for idx, (ax, column) in enumerate(zip(axes_array, columns)):
        measured = frame[column].notna()
        if measured.any():
            measured_frame = frame.loc[measured].sort_values("time_h")
            ax.plot(
                measured_frame["time_h"],
                measured_frame[column],
                "--o",
                label="Data",
                color=data_color,
                linewidth=2.5,
                markersize=3.5,
                alpha=0.75,
            )
        ax.plot(
            frame["time_h"],
            frame[f"{column}_pred"],
            "--",
            label="PINN",
            color=pinn_color,
            linewidth=2.5,
            alpha=0.85,
        )
        ax.set_title(OBSERVABLE_PLOT_TITLES.get(column, column), fontweight="bold", fontsize=12)
        ax.set_ylabel(OBSERVABLE_PLOT_YLABELS.get(column, column), fontsize=10)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Time (h)", fontsize=10)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        legend_map = dict(zip(labels, handles))
        ordered_labels = [
            "PINN",
            "Data",
        ]
        ordered_labels = [name for name in ordered_labels if name in legend_map]
        if ordered_labels:
            ordered_handles = [legend_map[name] for name in ordered_labels]
            ax.legend(ordered_handles, ordered_labels, loc="best", fontsize=9)
        else:
            ax.legend(loc="best", fontsize=9)

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


def _dataset_label(dataset: PseudomonasDataset) -> str:
    unique_ids = pd.unique(dataset.frame["Experiment_id"].astype(str))
    if len(unique_ids) == 1:
        return str(unique_ids[0])
    return "selected experiments"


def _save_figure(fig, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


def _default_figure_path(result: dict[str, Any], filename: str) -> Path:
    output_dir = Path(result.get("output_dir", "results/pseudomonas_pinn"))
    return output_dir / filename


def _default_prediction_figure_path(dataset: PseudomonasDataset, experiment_id: str | None) -> Path:
    output_dir = Path(dataset.output_dir or Path("results") / (experiment_id or _dataset_label(dataset).replace(" ", "_")))
    suffix = experiment_id or _dataset_label(dataset).replace(" ", "_")
    return output_dir / f"observable_predictions_{suffix}.png"


def _prediction_file_label(predictions_csv: str | Path) -> str:
    path = Path(predictions_csv)
    return path.parent.name or path.stem


def _loo_prediction_csv(results_dir: Path, experiment_id: str, seed: int | None = None) -> Path:
    seed_pattern = "*" if seed is None else str(seed)
    matches = sorted(results_dir.glob(f"fold_*_{experiment_id}_seed_{seed_pattern}/predictions_test.csv"))
    if not matches:
        raise FileNotFoundError(f"No predictions_test.csv found for {experiment_id!r} under {results_dir}")
    if len(matches) > 1 and seed is None:
        raise ValueError(f"Multiple prediction files found for {experiment_id!r}; pass seed=... to disambiguate.")
    return matches[0]


def save_reports(result: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Path]:
    output = Path(output_dir or result.get("output_dir", "results/pseudomonas_pinn"))
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = result["train_dataset"]
    val_dataset = result.get("val_dataset")
    test_dataset = result["test_dataset"]
    metrics = evaluate_observables(result["model"], train_dataset)
    params = parameter_report(result)
    train_metrics_path = output / "metrics_train.csv"
    test_metrics_path = output / "metrics_test.csv"
    params_path = output / "parameter_report.csv"
    predictions_train_path = output / "predictions_train.csv"
    predictions_test_path = output / "predictions_test.csv"
    metrics.to_csv(train_metrics_path, index=False)
    evaluate_observables(result["model"], test_dataset).to_csv(test_metrics_path, index=False)
    params.to_csv(params_path, index=False)
    prediction_frame(result["model"], train_dataset).to_csv(predictions_train_path, index=False)
    prediction_frame(result["model"], test_dataset).to_csv(predictions_test_path, index=False)
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
        evaluate_observables(result["model"], val_dataset).to_csv(metrics_val_path, index=False)
        prediction_frame(result["model"], val_dataset).to_csv(predictions_val_path, index=False)
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


__all__ = [
    "evaluate_observables",
    "load_prediction_results",
    "parameter_report",
    "plot_loo_test_predictions",
    "plot_observable_predictions",
    "plot_saved_observable_predictions",
    "plot_loss",
    "plot_r2",
    "plot_r2_by_target",
    "prediction_frame",
    "save_reports",
]
