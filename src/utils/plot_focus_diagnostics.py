from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OBSERVABLES = [
    ("glucose_g_l", "Glucose (g/L)"),
    ("biomass_g_l", "Biomass (g/L)"),
    ("DO_percent", "DO (%)"),
    ("pH", "pH"),
]


def _fold_experiment_id(fold_dir: Path) -> str:
    match = re.match(r"fold_\d+_(.+)_seed_", fold_dir.name)
    if not match:
        raise ValueError(f"Cannot infer experiment id from {fold_dir.name!r}.")
    return match.group(1)


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def plot_fold_diagnostics(
    fold_dir: Path,
    raw: pd.DataFrame,
    metadata: pd.DataFrame,
    focus_dir: Path,
) -> str:
    exp_id = _fold_experiment_id(fold_dir)
    pred_path = fold_dir / "predictions_test.csv"
    metrics_path = fold_dir / "metrics_test.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    pred = pd.read_csv(pred_path)
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    exp_meta = metadata.loc[metadata["Experiment_id"].astype(str) == exp_id]
    condition = exp_meta["condition_label"].iloc[0] if len(exp_meta) else "unknown"
    same_condition_ids: list[str] = []
    if len(exp_meta):
        same_condition_ids = metadata.loc[
            metadata["condition_label"] == condition, "Experiment_id"
        ].astype(str).tolist()
    same_condition_ids = [experiment_id for experiment_id in same_condition_ids if experiment_id != exp_id]

    fig, axes = plt.subplots(len(OBSERVABLES), 2, figsize=(13, 13), sharex="col")
    fig.suptitle(f"{exp_id} test diagnostics | {condition}", fontsize=14, y=0.995)
    for row, (observable, label) in enumerate(OBSERVABLES):
        ax = axes[row, 0]
        residual_ax = axes[row, 1]
        pred_col = f"{observable}_pred"
        plotted = pred[["time_h", observable, pred_col]].copy()
        plotted = _to_numeric(plotted, ["time_h", observable, pred_col])

        for index, replicate_id in enumerate(same_condition_ids):
            replicate = raw.loc[
                raw["Experiment_id"].astype(str) == replicate_id, ["time_h", observable]
            ].copy()
            replicate = _to_numeric(replicate, ["time_h", observable]).dropna(
                subset=["time_h", observable]
            )
            if not replicate.empty:
                ax.plot(
                    replicate["time_h"],
                    replicate[observable],
                    color="0.75",
                    lw=1.2,
                    alpha=0.7,
                    label="same-condition replicate" if index == 0 else None,
                )

        observed = plotted.dropna(subset=["time_h", observable])
        predicted = plotted.dropna(subset=["time_h", pred_col])
        ax.plot(predicted["time_h"], predicted[pred_col], color="#d62728", lw=2.0, label="prediction")
        ax.scatter(observed["time_h"], observed[observable], color="black", s=14, label="observed", zorder=3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(loc="best", fontsize=8)

        residual = plotted.dropna(subset=["time_h", observable, pred_col]).copy()
        residual["residual"] = residual[pred_col] - residual[observable]
        residual_ax.axhline(0, color="black", lw=0.8, alpha=0.7)
        residual_ax.plot(residual["time_h"], residual["residual"], color="#1f77b4", lw=1.4)
        residual_ax.scatter(residual["time_h"], residual["residual"], color="#1f77b4", s=10)
        residual_ax.set_ylabel("Pred - obs")
        residual_ax.grid(alpha=0.25)

        if not metrics.empty and observable in set(metrics["observable"]):
            metric = metrics.loc[metrics["observable"] == observable].iloc[0]
            ax.set_title(
                f"{label}: R2={float(metric.r2):.3f}, MAE={float(metric.mae):.3g}, "
                f"NRMSE={float(metric.nrmse):.3f}",
                fontsize=10,
            )
        else:
            ax.set_title(label, fontsize=10)

    axes[-1, 0].set_xlabel("Time (h)")
    axes[-1, 1].set_xlabel("Time (h)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    detailed_path = fold_dir / f"detailed_test_diagnostics_{exp_id}.png"
    fig.savefig(detailed_path, dpi=180)
    plt.close(fig)
    shutil.copy2(detailed_path, focus_dir / detailed_path.name)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(f"{exp_id} predicted vs observed | {condition}", fontsize=14)
    for ax, (observable, label) in zip(axes.ravel(), OBSERVABLES):
        pred_col = f"{observable}_pred"
        plotted = pred[[observable, pred_col]].copy()
        plotted = _to_numeric(plotted, [observable, pred_col]).dropna(subset=[observable, pred_col])
        ax.scatter(plotted[observable], plotted[pred_col], s=16, alpha=0.75)
        if not plotted.empty:
            lower = min(plotted[observable].min(), plotted[pred_col].min())
            upper = max(plotted[observable].max(), plotted[pred_col].max())
            pad = (upper - lower) * 0.05 if upper > lower else 1.0
            ax.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color="black", lw=1, ls="--")
            ax.set_xlim(lower - pad, upper + pad)
            ax.set_ylim(lower - pad, upper + pad)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    scatter_path = fold_dir / f"predicted_vs_observed_{exp_id}.png"
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    shutil.copy2(scatter_path, focus_dir / scatter_path.name)

    exp_raw = raw.loc[raw["Experiment_id"].astype(str) == exp_id].copy()
    exp_raw = _to_numeric(
        exp_raw,
        ["time_h", "DO_percent", "air_flow_l_min", "pH", "glucose_g_l", "biomass_g_l"],
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(exp_raw["time_h"], exp_raw["DO_percent"], color="#1f77b4", lw=1.6)
    if len(exp_meta):
        do_setpoint = float(exp_meta["DO_setpoint"].iloc[0])
        lower_bound = max(0.0, do_setpoint - max(5.0, 0.1 * do_setpoint))
        axes[0].axhline(do_setpoint, color="black", lw=1, ls="--", label=f"DO setpoint {do_setpoint:g}")
        axes[0].axhline(lower_bound, color="0.45", lw=1, ls=":", label="hard lower bound")
    axes[0].set_ylabel("DO (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(exp_raw["time_h"], exp_raw["air_flow_l_min"], color="#2ca02c", lw=1.6)
    if len(exp_meta):
        afr_setpoint_l_min = float(exp_meta["AFR_setpoint"].iloc[0]) / 1000.0
        axes[1].axhline(
            afr_setpoint_l_min,
            color="black",
            lw=1,
            ls="--",
            label=f"AFR setpoint {afr_setpoint_l_min:g} L/min",
        )
    axes[1].set_ylabel("Air flow (L/min)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(exp_raw["time_h"], exp_raw["pH"], label="pH", color="#9467bd", lw=1.5)
    axes[2].scatter(exp_raw["time_h"], exp_raw["glucose_g_l"], label="glucose", color="#ff7f0e", s=12)
    axes[2].plot(exp_raw["time_h"], exp_raw["biomass_g_l"], label="biomass", color="#8c564b", lw=1.3)
    axes[2].set_ylabel("Raw values")
    axes[2].set_xlabel("Time (h)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"{exp_id} raw trajectory/control signals | {condition}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    controls_path = fold_dir / f"raw_controls_{exp_id}.png"
    fig.savefig(controls_path, dpi=180)
    plt.close(fig)
    shutil.copy2(controls_path, focus_dir / controls_path.name)

    return (
        f"{exp_id}: {condition}; same-condition replicates={same_condition_ids or 'none'}; "
        f"saved {detailed_path.name}, {scatter_path.name}, {controls_path.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/LOO_3_10000")
    parser.add_argument("--processed-csv", default="data/processed/ambr_preprocessed.csv")
    parser.add_argument("--metadata-csv", default="data/processed/ambr_metadata.csv")
    parser.add_argument("--fold", action="append", required=True, help="Fold directory name.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    focus_dir = results_dir / "diagnostic_plots_focus"
    focus_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.processed_csv)
    metadata = pd.read_csv(args.metadata_csv)

    summaries = [
        plot_fold_diagnostics(results_dir / fold_name, raw, metadata, focus_dir)
        for fold_name in args.fold
    ]
    summary_path = focus_dir / "README_focus_plots.txt"
    summary_path.write_text("\n".join(summaries) + "\n", encoding="utf-8")
    print("\n".join(summaries))
    print(f"focus_dir={focus_dir}")


if __name__ == "__main__":
    main()
