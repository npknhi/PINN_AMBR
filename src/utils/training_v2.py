#src/utils/training_v1.py
from __future__ import annotations

"""Training utilities for the AMBR Pseudomonas PINN flow with pH range loss."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jinns
import numpy as np
import optax

from src.models.pinn_v1 import (
    INPUT_COLUMNS,
    OBSERVABLE_COLUMNS,
    PseudomonasBIOSODE,
    STATE_INDEX,
    cardinal_pH,
    create_equinox_network,
)
from src.utils.dataloader_v1 import (
    PseudomonasDataset,
    load_pseudomonas_bioreactor_split,
    load_pseudomonas_leave_one_bioreactor_out_split,
    make_leave_one_bioreactor_out_folds,
)


GLUCOSE_MOLAR_MASS_G_MOL = 180.156


@dataclass
class TrainingConfig:
    processed_csv: str = "data/processed/ambr_preprocessed.csv"
    train_experiment_ids: tuple[str, ...] | None = None
    test_experiment_ids: tuple[str, ...] | None = None
    validation_experiment_ids: tuple[str, ...] | None = None
    experiment_ids: tuple[str, ...] | None = None
    results_dir: str = "results"
    experiment_name: str | None = None
    seed: int = 42
    n_neurons: int = 20
    n_hidden_layers: int = 7
    num_epochs: int = 1000
    learning_rate: float = 1e-3
    data_loss_weight: float = 1.0
    residual_loss_weight: float = 1e-4
    auxiliary_loss_weight: float = 1.0
    regularization_loss_weight: float = 1e-6
    use_auxiliary_loss: bool = True
    use_regularization_loss: bool = True
    use_softadapt: bool = True
    pH_range_margin: float = 0.1
    pH_range_loss_weight: float = 1.0
    DO_initial_target_percent: float = 100.0
    DO_initial_margin_percent: float = 2.0
    DO_initial_loss_weight: float = 1.0
    DO_end_target_percent: float = 100.0
    DO_end_margin_percent: float = 3.0
    DO_end_loss_weight: float = 1.0
    glucose_upper_loss_weight: float = 1.0
    glucose_monotonic_loss_weight: float = 1.0
    exclude_do_above_percent: float | None = 105.0
    use_early_stopping: bool = False
    early_stopping_patience: int = 1000
    early_stopping_min_delta: float = 1e-4
    restore_best_weights: bool = True
    track_epoch_r2: bool = True
    obs_fit_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(OBSERVABLE_COLUMNS))
    aux_fit_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(PseudomonasBIOSODE.state_names))
    res_eq_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(PseudomonasBIOSODE.state_names))
    reg_eq_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(PseudomonasBIOSODE.state_names))
    loo_fold_index: int = 0
    trainable_parameters: tuple[str, ...] = tuple(PseudomonasBIOSODE.learnable_parameters)
    frozen_parameters: dict[str, float] = field(default_factory=dict)
    save_config: bool = True
    save_checkpoint: bool = True
    validation_strategy: str = "rotate"
    validation_seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.loo_fold_index < 0:
            raise ValueError("loo_fold_index must be non-negative.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative.")
        if self.validation_strategy not in {"rotate", "random"}:
            raise ValueError("validation_strategy must be 'rotate' or 'random'.")
        if self.pH_range_margin < 0:
            raise ValueError("pH_range_margin must be non-negative.")
        if self.pH_range_loss_weight < 0:
            raise ValueError("pH_range_loss_weight must be non-negative.")
        if self.DO_initial_margin_percent < 0:
            raise ValueError("DO_initial_margin_percent must be non-negative.")
        if self.DO_initial_loss_weight < 0:
            raise ValueError("DO_initial_loss_weight must be non-negative.")
        if self.DO_end_margin_percent < 0:
            raise ValueError("DO_end_margin_percent must be non-negative.")
        if self.DO_end_loss_weight < 0:
            raise ValueError("DO_end_loss_weight must be non-negative.")
        if self.glucose_upper_loss_weight < 0:
            raise ValueError("glucose_upper_loss_weight must be non-negative.")
        if self.glucose_monotonic_loss_weight < 0:
            raise ValueError("glucose_monotonic_loss_weight must be non-negative.")
        if self.exclude_do_above_percent is not None and self.exclude_do_above_percent <= 0:
            raise ValueError("exclude_do_above_percent must be positive or None.")
        has_train_ids = self.train_experiment_ids is not None
        has_test_ids = self.test_experiment_ids is not None
        if has_train_ids != has_test_ids:
            raise ValueError("Set both train_experiment_ids and test_experiment_ids, or neither.")
        if has_train_ids:
            overlap = (
                (set(self.train_experiment_ids or ()) & set(self.test_experiment_ids or ()))
                | (set(self.train_experiment_ids or ()) & set(self.validation_experiment_ids or ()))
                | (set(self.validation_experiment_ids or ()) & set(self.test_experiment_ids or ()))
            )
            if overlap:
                raise ValueError(f"Train, validation, and test bioreactors must be disjoint: {sorted(overlap)}")
        unknown = set(self.trainable_parameters) - set(PseudomonasBIOSODE.learnable_parameters)
        if unknown:
            raise ValueError(f"Unknown trainable parameters: {sorted(unknown)}")
        _validate_weight_vector("obs_fit_weights", self.obs_fit_weights, len(OBSERVABLE_COLUMNS))
        _validate_weight_vector("aux_fit_weights", self.aux_fit_weights, len(PseudomonasBIOSODE.state_names))
        _validate_weight_vector("res_eq_weights", self.res_eq_weights, len(PseudomonasBIOSODE.state_names))
        _validate_weight_vector("reg_eq_weights", self.reg_eq_weights, len(PseudomonasBIOSODE.state_names))


def train_pinn(config: TrainingConfig | None = None) -> dict[str, Any]:
    """Train the Pseudomonas PINN on the processed AMBR table."""

    config = config or TrainingConfig()
    train_dataset, val_dataset, test_dataset, split_metadata = _load_train_test_datasets(config)
    score_dataset = val_dataset or test_dataset
    if config.use_early_stopping and val_dataset is None:
        raise ValueError("Early stopping requires a validation split.")

    key = jax.random.PRNGKey(config.seed)
    model = create_equinox_network(
        key,
        n_neurons=config.n_neurons,
        n_hidden_layers=config.n_hidden_layers,
        input_columns=train_dataset.input_columns,
        output_scale=train_dataset.state_scale,
    )

    active_names = tuple(name for name in config.trainable_parameters if name not in config.frozen_parameters)
    theta_raw = _initial_theta_raw(active_names)

    arrays = _device_arrays(train_dataset, config)
    val_arrays = _device_arrays(score_dataset, config)
    opt = optax.adam(config.learning_rate)
    params = (model, theta_raw)
    opt_state = opt.init(eqx.filter(params, eqx.is_array))

    @eqx.filter_jit
    def train_step(params, opt_state, softadapt_weights):
        (loss_value, components), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(
            params,
            arrays,
            active_names,
            config.frozen_parameters,
            softadapt_weights,
            config,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_value, components

    @eqx.filter_jit
    def validation_data_loss(params):
        model_current, _ = params
        states = _constrained_states(model_current, val_arrays["x"], val_arrays["volume_l"], val_arrays)
        return _data_loss(states, val_arrays)

    history: dict[str, list[float]] = {
        "loss": [],
        "data_loss": [],
        "val_data_loss": [],
        "residual_loss": [],
        "auxiliary_loss": [],
        "regularization_loss": [],
        "r2_scores_train": [],
        "r2_scores_val": [],
        "softadapt_data_weight": [],
        "softadapt_residual_weight": [],
        "softadapt_auxiliary_weight": [],
        "softadapt_regularization_weight": [],
    }
    for column in train_dataset.target_columns:
        history[f"r2_train_{column}"] = []
        history[f"r2_val_{column}"] = []
    softadapt_weights = jnp.asarray(
        [
            config.data_loss_weight,
            config.residual_loss_weight,
            config.auxiliary_loss_weight,
            config.regularization_loss_weight,
        ],
        dtype=jnp.float32,
    )
    stored_data_loss = jnp.zeros((config.num_epochs,), dtype=jnp.float32)
    stored_residual_loss = jnp.zeros((config.num_epochs,), dtype=jnp.float32)
    stored_auxiliary_loss = jnp.zeros((config.num_epochs,), dtype=jnp.float32)
    stored_regularization_loss = jnp.zeros((config.num_epochs,), dtype=jnp.float32)
    best_params = params
    best_val_data_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        params, opt_state, loss_value, components = train_step(params, opt_state, softadapt_weights)
        val_data_loss = validation_data_loss(params)
        history["loss"].append(float(loss_value))
        history["val_data_loss"].append(float(val_data_loss))
        for name, value in components.items():
            history[name].append(float(value))
        if config.track_epoch_r2:
            model_current, _ = params
            train_r2 = _observable_r2_by_target(model_current, arrays, train_dataset.target_columns)
            val_r2 = _observable_r2_by_target(model_current, val_arrays, score_dataset.target_columns)
            history["r2_scores_train"].append(_mean_r2(train_r2.values()))
            history["r2_scores_val"].append(_mean_r2(val_r2.values()))
            for column in train_dataset.target_columns:
                history[f"r2_train_{column}"].append(train_r2[column])
                history[f"r2_val_{column}"].append(val_r2[column])
        else:
            history["r2_scores_train"].append(float("nan"))
            history["r2_scores_val"].append(float("nan"))
            for column in train_dataset.target_columns:
                history[f"r2_train_{column}"].append(float("nan"))
                history[f"r2_val_{column}"].append(float("nan"))
        history["softadapt_data_weight"].append(float(softadapt_weights[0]))
        history["softadapt_residual_weight"].append(float(softadapt_weights[1]))
        history["softadapt_auxiliary_weight"].append(float(softadapt_weights[2]))
        history["softadapt_regularization_weight"].append(float(softadapt_weights[3]))

        stored_data_loss = stored_data_loss.at[epoch].set(components["data_loss"])
        stored_residual_loss = stored_residual_loss.at[epoch].set(components["residual_loss"])
        stored_auxiliary_loss = stored_auxiliary_loss.at[epoch].set(components["auxiliary_loss"])
        stored_regularization_loss = stored_regularization_loss.at[epoch].set(components["regularization_loss"])
        if config.use_softadapt:
            softadapt_weights = jinns.loss.soft_adapt(
                loss_weights=tuple(softadapt_weights),
                iteration_nb=jnp.asarray(epoch, dtype=jnp.int32),
                loss_terms=(
                    components["data_loss"],
                    components["residual_loss"],
                    components["auxiliary_loss"],
                    components["regularization_loss"],
                ),
                stored_loss_terms=(
                    stored_data_loss,
                    stored_residual_loss,
                    stored_auxiliary_loss,
                    stored_regularization_loss,
                ),
            )
            softadapt_weights = jnp.asarray(softadapt_weights, dtype=jnp.float32)

        if config.use_early_stopping:
            current_val_data_loss = float(val_data_loss)
            if current_val_data_loss < best_val_data_loss - config.early_stopping_min_delta:
                best_val_data_loss = current_val_data_loss
                best_params = params
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    break

    if config.use_early_stopping:
        history["best_epoch"] = [float(best_epoch)] * len(history["loss"])
        history["best_val_data_loss"] = [float(best_val_data_loss)] * len(history["loss"])
        if config.restore_best_weights:
            params = best_params

    model, theta_raw = params
    ode_params = _build_ode_params(theta_raw, active_names, config.frozen_parameters)
    learned_params = {name: ode_params[name] for name in active_names}
    learned_params.update({name: float(value) for name, value in config.frozen_parameters.items()})
    train_predictions = predict_dataset(model, train_dataset)
    test_predictions = predict_dataset(model, test_dataset)
    output_dir = _output_dir_from_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = replace(train_dataset, output_dir=str(output_dir))
    if val_dataset is not None:
        val_dataset = replace(val_dataset, output_dir=str(output_dir))
    test_dataset = replace(test_dataset, output_dir=str(output_dir))
    result = {
        "config": config,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "model": model,
        "theta_raw": np.asarray(theta_raw),
        "learned_params": {name: float(value) for name, value in learned_params.items()},
        "history": history,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "split_metadata": split_metadata,
        "output_dir": str(output_dir),
    }
    if config.save_config:
        save_training_config(config, output_dir / "config.json", split_metadata=split_metadata)
    if config.save_checkpoint:
        save_checkpoint(result, output_dir / "checkpoints")
    return result


def _load_train_test_datasets(
    config: TrainingConfig,
) -> tuple[PseudomonasDataset, PseudomonasDataset | None, PseudomonasDataset, dict[str, object]]:
    if config.train_experiment_ids is not None and config.test_experiment_ids is not None:
        return load_pseudomonas_bioreactor_split(
            config.processed_csv,
            input_columns=tuple(INPUT_COLUMNS),
            train_experiment_ids=config.train_experiment_ids,
            validation_experiment_ids=config.validation_experiment_ids or (),
            test_experiment_ids=config.test_experiment_ids,
        )

    return load_pseudomonas_leave_one_bioreactor_out_split(
        config.processed_csv,
        input_columns=tuple(INPUT_COLUMNS),
        experiment_ids=config.experiment_ids,
        fold_index=config.loo_fold_index,
        validation_strategy=config.validation_strategy,
        validation_seed=config.seed if config.validation_seed is None else config.validation_seed,
    )


def _append_csv_frame(path: Path, frame) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_csv_row(path: Path, row: dict[str, object]) -> None:
    import pandas as pd

    _append_csv_frame(path, pd.DataFrame([row]))


def _remove_existing_fold_seed_rows(path: Path, fold_index: int, seed: int) -> None:
    if not path.exists():
        return

    import pandas as pd

    frame = pd.read_csv(path)
    if frame.empty or "fold_index" not in frame.columns or "seed" not in frame.columns:
        return
    keep = ~(
        pd.to_numeric(frame["fold_index"], errors="coerce").eq(int(fold_index))
        & pd.to_numeric(frame["seed"], errors="coerce").eq(int(seed))
    )
    frame.loc[keep].to_csv(path, index=False)


def run_leave_one_bioreactor_out(
    config: TrainingConfig | None = None,
    seeds: Sequence[int] | None = None,
    fold_indices: Sequence[int] | None = None,
    keep_results: bool = False,
) -> dict[str, Any]:
    """Run leave-one-bioreactor-out benchmarking for PINN.

    Each fold holds out one complete bioreactor, repeated across random seeds,
    with metrics saved as long-form and mean/std summary CSV files.
    """

    import pandas as pd

    from src.utils.evaluation_v1 import (
        OBSERVABLE_TABLE_LABELS,
        evaluate_observables,
        save_reports,
    )

    base_config = config or TrainingConfig()
    if base_config.train_experiment_ids is not None or base_config.test_experiment_ids is not None:
        raise ValueError("run_leave_one_bioreactor_out expects experiment_ids, not explicit train/test ids.")
    seed_values = tuple(int(seed) for seed in (seeds or (base_config.seed,)))
    base_name = (base_config.experiment_name or "leave_one_bioreactor_out").strip()
    output_dir = Path(base_config.results_dir) / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_long_path = output_dir / "leave_one_out_metrics_long.csv"
    runtime_path = output_dir / "leave_one_out_runtime.csv"
    table_path = output_dir / "leave_one_out_table.csv"
    fold_frame = _selected_experiment_frame(base_config.processed_csv, base_config.experiment_ids)
    folds = make_leave_one_bioreactor_out_folds(fold_frame)
    all_experiment_ids = [experiment_id for fold in folds for experiment_id in fold]
    selected_fold_indices = None if fold_indices is None else {int(index) for index in fold_indices}
    if selected_fold_indices is not None:
        invalid_fold_indices = sorted(index for index in selected_fold_indices if index < 0 or index >= len(folds))
        if invalid_fold_indices:
            raise ValueError(
                f"fold_indices contains invalid fold(s): {invalid_fold_indices}. "
                f"Valid range is 0 to {len(folds) - 1}."
            )

    metric_frames = []
    runtime_rows = []
    kept_results = []
    for fold_index, test_ids in enumerate(folds):
        if selected_fold_indices is not None and fold_index not in selected_fold_indices:
            continue
        test_label = "_".join(test_ids)
        for seed in seed_values:
            run_config = replace(
                base_config,
                seed=seed,
                results_dir=str(output_dir),
                experiment_ids=tuple(all_experiment_ids),
                loo_fold_index=fold_index,
                experiment_name=f"fold_{fold_index}_{test_label}_seed_{seed}",
            )
            train_start = perf_counter()
            result = train_pinn(run_config)
            training_time_s = perf_counter() - train_start
            inference_start = perf_counter()
            predict_dataset(result["model"], result["test_dataset"])
            inference_time_s = perf_counter() - inference_start
            save_reports(result)
            _remove_existing_fold_seed_rows(runtime_path, fold_index, seed)
            _remove_existing_fold_seed_rows(metrics_long_path, fold_index, seed)
            runtime_row = {
                "fold_index": fold_index,
                "seed": seed,
                "training_time_s": float(training_time_s),
                "inference_time_s": float(inference_time_s),
            }
            runtime_rows.append(runtime_row)
            _append_csv_row(runtime_path, runtime_row)
            datasets = [("train", result["train_dataset"])]
            if result.get("val_dataset") is not None:
                datasets.append(("val", result["val_dataset"]))
            fold_metric_frames = []
            datasets.append(("test", result["test_dataset"]))
            for split_name, dataset in datasets:
                metrics = evaluate_observables(result["model"], dataset)
                metrics.insert(0, "split", split_name)
                metrics.insert(0, "seed", seed)
                metrics.insert(0, "fold_index", fold_index)
                fold_metric_frames.append(metrics)
                metric_frames.append(metrics)
            _append_csv_frame(metrics_long_path, pd.concat(fold_metric_frames, ignore_index=True))
            if keep_results:
                kept_results.append(result)
    if metrics_long_path.exists():
        metrics_long = pd.read_csv(metrics_long_path)
    else:
        metrics_long = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    if runtime_path.exists():
        runtime = pd.read_csv(runtime_path)
    else:
        runtime = pd.DataFrame(runtime_rows)
    table = _v1_metric_table(
        metrics_long,
        benchmark="Leave-One-Bioreactor-Out",
        observable_labels=OBSERVABLE_TABLE_LABELS,
    )
    table.to_csv(table_path, index=False)

    return {
        "output_dir": str(output_dir),
        "metrics_long": metrics_long,
        "runtime": runtime,
        "table": table,
        "paths": {
            "metrics_long": metrics_long_path,
            "runtime": runtime_path,
            "table": table_path,
        },
        "results": kept_results,
    }


def _selected_experiment_frame(processed_csv: str | Path, experiment_ids: Sequence[str] | None) -> "pd.DataFrame":
    import pandas as pd

    frame = pd.read_csv(processed_csv, usecols=["Experiment_id"])
    frame["Experiment_id"] = frame["Experiment_id"].astype(str)
    if experiment_ids is None:
        return frame.drop_duplicates().reset_index(drop=True)

    selected = tuple(str(experiment_id) for experiment_id in experiment_ids)
    if not selected:
        raise ValueError("experiment_ids must not be empty.")
    available = set(frame["Experiment_id"].unique())
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"Unknown Experiment_id values: {missing}")
    return frame[frame["Experiment_id"].isin(selected)].drop_duplicates().reset_index(drop=True)


def _v1_metric_table(
    metrics: "pd.DataFrame",
    *,
    benchmark: str,
    observable_labels: dict[str, str],
) -> "pd.DataFrame":
    import pandas as pd

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

    test_metrics = metrics[metrics["split"].eq("test")].copy()
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
    return grouped.loc[:, columns]


def _flatten_columns(columns: Any) -> list[str]:
    flattened = []
    for column in columns:
        if isinstance(column, tuple):
            flattened.append("_".join(str(part) for part in column if part))
        else:
            flattened.append(str(column))
    return flattened


def _output_dir_from_config(config: TrainingConfig) -> Path:
    experiment_name = (config.experiment_name or "").strip()
    if not experiment_name:
        if config.train_experiment_ids is not None and config.test_experiment_ids is not None:
            train_label = "_".join(config.train_experiment_ids)
            test_label = "_".join(config.test_experiment_ids)
            experiment_name = f"{train_label}_to_{test_label}"
        else:
            experiment_name = f"leave_one_bioreactor_out_fold_{config.loo_fold_index}"

    return Path(config.results_dir) / experiment_name


def save_training_config(
    config: TrainingConfig,
    output_path: str | Path,
    split_metadata: dict[str, object] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_config_json_payload(config, split_metadata), indent=2), encoding="utf-8")
    return output_path


def save_checkpoint(result: dict[str, Any], checkpoint_dir: str | Path | None = None) -> dict[str, Path]:
    output_dir = Path(result["output_dir"])
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "model.eqx"
    theta_path = checkpoint_dir / "theta_raw.npy"
    learned_params_path = checkpoint_dir / "learned_params.json"
    history_path = checkpoint_dir / "history.csv"
    scaler_path = checkpoint_dir / "scalers.npz"
    columns_path = checkpoint_dir / "dataset_columns.json"

    eqx.tree_serialise_leaves(model_path, result["model"])
    np.save(theta_path, np.asarray(result["theta_raw"]))
    learned_params_path.write_text(json.dumps(result["learned_params"], indent=2), encoding="utf-8")
    _history_frame(result["history"]).to_csv(history_path, index=False)
    train_dataset = result["train_dataset"]
    np.savez_compressed(
        scaler_path,
        x_mean=np.asarray(train_dataset.x_mean),
        x_std=np.asarray(train_dataset.x_std),
        target_scale=np.asarray(train_dataset.target_scale),
        state_scale=np.asarray(train_dataset.state_scale),
        gas_scale=np.asarray(train_dataset.gas_scale),
    )
    columns_path.write_text(
        json.dumps(
            {
                "input_columns": list(train_dataset.input_columns),
                "target_columns": list(train_dataset.target_columns),
                "gas_columns": list(train_dataset.gas_columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "model": model_path,
        "theta_raw": theta_path,
        "learned_params": learned_params_path,
        "history": history_path,
        "scalers": scaler_path,
        "dataset_columns": columns_path,
    }


def _config_json_payload(config: TrainingConfig, split_metadata: dict[str, object] | None = None) -> dict[str, Any]:
    payload = {
        "data": {
            "processed_csv": config.processed_csv,
            "train_experiment_ids": list(config.train_experiment_ids) if config.train_experiment_ids is not None else None,
            "validation_experiment_ids": list(config.validation_experiment_ids)
            if config.validation_experiment_ids is not None
            else None,
            "test_experiment_ids": list(config.test_experiment_ids) if config.test_experiment_ids is not None else None,
            "split_strategy": "bioreactor_id"
            if config.train_experiment_ids is not None and config.test_experiment_ids is not None
            else "leave_one_bioreactor_out",
            "validation_strategy": config.validation_strategy,
            "validation_seed": config.seed if config.validation_seed is None else int(config.validation_seed),
            "fold_index": int(config.loo_fold_index),
        },
        "model": {
            "n_neurons": int(config.n_neurons),
            "n_hidden_layers": int(config.n_hidden_layers),
            "trainable_parameters": list(config.trainable_parameters),
        },
        "training": {
            "seed": int(config.seed),
            "num_epochs": int(config.num_epochs),
            "learning_rate": float(config.learning_rate),
            "use_auxiliary_loss": bool(config.use_auxiliary_loss),
            "use_regularization_loss": bool(config.use_regularization_loss),
            "use_softadapt": bool(config.use_softadapt),
            "use_early_stopping": bool(config.use_early_stopping),
            "early_stopping_patience": int(config.early_stopping_patience),
            "early_stopping_min_delta": float(config.early_stopping_min_delta),
            "restore_best_weights": bool(config.restore_best_weights),
            "track_epoch_r2": bool(config.track_epoch_r2),
            "save_config": bool(config.save_config),
            "save_checkpoint": bool(config.save_checkpoint),
        },
        "loss_weights": {
            "data_loss_weight": float(config.data_loss_weight),
            "residual_loss_weight": float(config.residual_loss_weight),
            "auxiliary_loss_weight": float(config.auxiliary_loss_weight),
            "regularization_loss_weight": float(config.regularization_loss_weight),
            "pH_range_margin": float(config.pH_range_margin),
            "pH_range_loss_weight": float(config.pH_range_loss_weight),
            "DO_initial_target_percent": float(config.DO_initial_target_percent),
            "DO_initial_margin_percent": float(config.DO_initial_margin_percent),
            "DO_initial_loss_weight": float(config.DO_initial_loss_weight),
            "DO_end_target_percent": float(config.DO_end_target_percent),
            "DO_end_margin_percent": float(config.DO_end_margin_percent),
            "DO_end_loss_weight": float(config.DO_end_loss_weight),
            "glucose_upper_loss_weight": float(config.glucose_upper_loss_weight),
            "glucose_monotonic_loss_weight": float(config.glucose_monotonic_loss_weight),
            "exclude_do_above_percent": (
                None if config.exclude_do_above_percent is None else float(config.exclude_do_above_percent)
            ),
            "obs_fit_weights": [float(value) for value in config.obs_fit_weights],
            "aux_fit_weights": [float(value) for value in config.aux_fit_weights],
            "res_eq_weights": [float(value) for value in config.res_eq_weights],
            "reg_eq_weights": [float(value) for value in config.reg_eq_weights],
        },
    }
    if config.frozen_parameters:
        payload["model"]["frozen_parameters"] = {
            name: float(value) for name, value in config.frozen_parameters.items()
        }
    if split_metadata is not None:
        data = payload["data"]
        for key in (
            "split_strategy",
            "n_splits",
            "fold_index",
            "validation_strategy",
            "validation_seed",
            "train_experiment_ids",
            "validation_experiment_ids",
            "test_experiment_ids",
        ):
            if key in split_metadata:
                data[key] = split_metadata[key]
    return payload


def _history_frame(history: dict[str, list[float]]) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame({name: list(values) for name, values in history.items()})


def predict_dataset(model, dataset: PseudomonasDataset) -> dict[str, np.ndarray]:
    arrays = _device_arrays(dataset)
    states = _constrained_states(model, arrays["x"], arrays["volume_l"], arrays)
    states_np = np.asarray(states)
    predictions = {name: states_np[:, idx] for name, idx in PseudomonasBIOSODE.state_index.items()}
    volume = np.maximum(dataset.volume_l, 1e-12)
    predictions["glucose_mol_l"] = predictions["Substrate"] / volume
    predictions["glucose_g_l"] = predictions["glucose_mol_l"] * GLUCOSE_MOLAR_MASS_G_MOL
    predictions["biomass_g_l"] = predictions["Biomass"] / volume
    predictions["O2_l_mol"] = predictions["O2_l"]
    predictions["DO_percent"] = np.asarray(_do_percent_from_o2_l(jnp.asarray(predictions["O2_l"]), jnp.asarray(volume)))
    h_mol_l = predictions["H"] / volume
    predictions["pH"] = -np.log10(np.maximum(h_mol_l, 1e-14))
    return predictions


def _loss(
    params,
    arrays: dict[str, jnp.ndarray],
    active_names: tuple[str, ...],
    frozen_parameters: dict[str, float],
    softadapt_weights: jnp.ndarray,
    config: TrainingConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    model, theta_raw = params
    ode_params = _build_ode_params(theta_raw, active_names, frozen_parameters)
    states = _constrained_states(model, arrays["x"], arrays["volume_l"], arrays)

    data_loss = _data_loss(states, arrays)
    residual_loss = _residual_loss(model, ode_params, arrays)
    auxiliary_loss = (
        _auxiliary_loss(model, arrays, config)
        if config.use_auxiliary_loss
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    regularization_loss = (
        _regularization_loss(model, ode_params, arrays)
        if config.use_regularization_loss
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    active_loss_mask = jnp.asarray(
        [True, True, config.use_auxiliary_loss, config.use_regularization_loss],
        dtype=jnp.float32,
    )

    total = jnp.sum(
        softadapt_weights
        * active_loss_mask
        * jnp.asarray(
            [
                data_loss,
                residual_loss,
                auxiliary_loss,
                regularization_loss,
            ],
            dtype=jnp.float32,
        )
    )
    return total, {
        "data_loss": data_loss,
        "residual_loss": residual_loss,
        "auxiliary_loss": auxiliary_loss,
        "regularization_loss": regularization_loss,
    }


def _data_loss(states: jnp.ndarray, arrays: dict[str, jnp.ndarray]) -> jnp.ndarray:
    pred = _observable_matrix(states, arrays["volume_l"])
    residual = (pred - arrays["y"]) / arrays["target_scale"]
    residual = jnp.where(arrays["y_mask"], residual, 0.0)
    denom = jnp.maximum(jnp.sum(arrays["y_mask"], axis=0), 1)
    mse_per_observable = jnp.sum(residual**2, axis=0) / denom
    return jnp.sum(arrays["obs_fit_weights"] * mse_per_observable)


def _observable_r2_by_target(
    model,
    arrays: dict[str, jnp.ndarray],
    target_columns: tuple[str, ...],
) -> dict[str, float]:
    states = _constrained_states(model, arrays["x"], arrays["volume_l"], arrays)
    pred = _observable_matrix(states, arrays["volume_l"])
    y_true = np.asarray(arrays["y"])
    y_pred = np.asarray(pred)
    mask = np.asarray(arrays["y_mask"], dtype=bool)
    scores = {}
    for idx, column in enumerate(target_columns):
        valid = mask[:, idx]
        if valid.sum() < 2:
            scores[column] = float("nan")
            continue
        true_values = y_true[valid, idx]
        pred_values = y_pred[valid, idx]
        denom = np.sum((true_values - np.mean(true_values)) ** 2)
        if denom < 1e-12:
            scores[column] = float("nan")
        else:
            r2 = 1.0 - np.sum((true_values - pred_values) ** 2) / denom
            scores[column] = float(max(0.0, r2))
    return scores


def _mean_r2(scores: Any) -> float:
    values = np.asarray(list(scores), dtype=float)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _observable_matrix(states: jnp.ndarray, volume_l: jnp.ndarray) -> jnp.ndarray:
    volume = jnp.maximum(volume_l, 1e-12)
    h_mol_l = states[:, STATE_INDEX["H"]] / volume
    return jnp.stack(
        [
            states[:, STATE_INDEX["Substrate"]] / volume,
            states[:, STATE_INDEX["Biomass"]] / volume,
            _do_percent_from_o2_l(states[:, STATE_INDEX["O2_l"]], volume),
            -jnp.log10(jnp.maximum(h_mol_l, 1e-14)),
        ],
        axis=1,
    )


def _constrained_states(
    model,
    x: jnp.ndarray,
    volume_l: jnp.ndarray,
    arrays: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    raw_states = jax.vmap(model)(x)
    return _apply_hard_initial_constraints(raw_states, x, volume_l, arrays)


def _constrained_state(
    model,
    xi: jnp.ndarray,
    volume_l: jnp.ndarray,
    arrays: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    raw_state = model(xi)
    constrained = _apply_hard_initial_constraints(
        raw_state[jnp.newaxis, :],
        xi[jnp.newaxis, :],
        jnp.asarray([volume_l], dtype=raw_state.dtype),
        arrays,
    )
    return constrained[0]


def _apply_hard_initial_constraints(
    states: jnp.ndarray,
    x: jnp.ndarray,
    volume_l: jnp.ndarray,
    arrays: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Apply hard condition constraints from experiment inputs."""

    time_idx = arrays["time_column_index"]
    glucose_idx = arrays["initial_glucose_column_index"]
    ph_idx = arrays["initial_ph_column_index"]
    do_idx = arrays["do_setpoint_column_index"]

    time_phys = x[:, time_idx] * arrays["x_std"][time_idx] + arrays["x_mean"][time_idx]
    is_initial_time = time_phys <= arrays["time_x_mean"] + 1e-6

    initial_glucose_g_l = x[:, glucose_idx] * arrays["x_std"][glucose_idx] + arrays["x_mean"][glucose_idx]
    initial_ph = x[:, ph_idx] * arrays["x_std"][ph_idx] + arrays["x_mean"][ph_idx]
    do_setpoint = x[:, do_idx] * arrays["x_std"][do_idx] + arrays["x_mean"][do_idx]
    volume = jnp.maximum(volume_l, 1e-12)
    initial_substrate = (initial_glucose_g_l / GLUCOSE_MOLAR_MASS_G_MOL) * volume
    initial_h = (10.0 ** (-initial_ph)) * volume
    # DO setpoint is treated as a lower operating target.
    do_margin = 0.0
    do_lower_percent = jnp.maximum(do_setpoint - do_margin, 0.0)
    p = PseudomonasBIOSODE.default_parameters
    saturation_o2_l_mol = p["FractionO2"] * p["Pr"] * p["HenryConstantO2"] * volume
    do_lower_o2_l = (do_lower_percent / 100.0) * saturation_o2_l_mol
    do_smooth_scale = jnp.maximum(0.01 * saturation_o2_l_mol, 1e-12)

    substrate_idx = STATE_INDEX["Substrate"]
    h_idx = STATE_INDEX["H"]
    o2_idx = STATE_INDEX["O2_l"]
    substrate = jnp.where(is_initial_time, initial_substrate, states[:, substrate_idx])
    h = jnp.where(is_initial_time, initial_h, states[:, h_idx])
    o2_l = do_lower_o2_l + do_smooth_scale * jax.nn.softplus((states[:, o2_idx] - do_lower_o2_l) / do_smooth_scale)
    states = states.at[:, substrate_idx].set(jnp.maximum(substrate, 0.0))
    states = states.at[:, h_idx].set(jnp.maximum(h, 1e-14 * volume))
    states = states.at[:, o2_idx].set(o2_l)
    return states


def _residual_loss(
    model,
    ode_params: dict[str, jnp.ndarray],
    arrays: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    residual_raw = jax.vmap(
        lambda xi, vl, vg, vs, va, vb: _raw_ode_residual(model, ode_params, arrays, xi, vl, vg, vs, va, vb)
    )(
        arrays["x"],
        arrays["volume_l"],
        arrays["air_flow_l_min"],
        arrays["sampling_rate_l_min"],
        arrays["acid_rate_l_min"],
        arrays["base_rate_l_min"],
    )
    residual = residual_raw / arrays["state_scale"]
    mse_per_state = jnp.mean(residual**2, axis=0)
    return jnp.sum(arrays["res_eq_weights"] * mse_per_state)


def _gas_predictions(
    states: jnp.ndarray,
    ode_params: dict[str, jnp.ndarray],
    arrays: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    p = {**PseudomonasBIOSODE.default_parameters, **ode_params}
    substrate = jnp.maximum(states[:, STATE_INDEX["Substrate"]], 0.0)
    biomass = states[:, STATE_INDEX["Biomass"]]
    co2_g = states[:, STATE_INDEX["CO2_g"]]
    h = states[:, STATE_INDEX["H"]]
    volume = jnp.maximum(arrays["volume_l"], 1e-12)
    ph = -jnp.log10(jnp.maximum(h / volume, 1e-14))
    ph_factor = cardinal_pH(ph, p["pH_min"], p["pH_opt"], p["pH_max"])
    gas_volume = jnp.maximum(p["Vtotal"] - volume, 1e-9)
    total_gas_moles = p["Pr"] * (gas_volume / 1000.0) / (p["R"] * p["TKelvin"])

    mu = p["mu_max"] * substrate / (p["Ksubs"] + substrate + 1e-12) * ph_factor
    growth = mu * biomass
    our = p["YO2"] * growth
    cer = p["YCO2"] * growth
    co2_offgas_fraction = co2_g / jnp.maximum(total_gas_moles, 1e-12)
    rq = cer / jnp.maximum(our, 1e-12)
    return jnp.stack([our, cer, co2_offgas_fraction, rq], axis=1)


def _auxiliary_loss(model, arrays: dict[str, jnp.ndarray], config: TrainingConfig) -> jnp.ndarray:
    """Auxiliary condition loss from reliable condition constraints only.
    """

    boundary_volume_l = jnp.full(
        (arrays["boundary_x"].shape[0],),
        PseudomonasBIOSODE.default_parameters["Vl"],
        dtype=arrays["boundary_x"].dtype,
    )
    boundary_pred = _constrained_states(model, arrays["boundary_x"], boundary_volume_l, arrays)
    initial_pred = boundary_pred[0::2]
    final_pred = boundary_pred[1::2]
    initial_target = arrays["boundary_y"][0::2]
    initial_mask = arrays["boundary_mask"][0::2]

    substrate_idx = STATE_INDEX["Substrate"]
    h_idx = STATE_INDEX["H"]
    o2_idx = STATE_INDEX["O2_l"]

    glucose_residual = (
        (initial_pred[:, substrate_idx] - initial_target[:, substrate_idx])
        / jnp.maximum(arrays["state_scale"][substrate_idx], 1e-12)
    )
    glucose_mask = initial_mask[:, substrate_idx]
    glucose0_loss = _masked_scalar_mse(glucose_residual, glucose_mask)

    initial_volume_l = jnp.asarray(PseudomonasBIOSODE.default_parameters["Vl"], dtype=initial_pred.dtype)
    pred_ph0 = -jnp.log10(jnp.maximum(initial_pred[:, h_idx] / initial_volume_l, 1e-14))
    target_ph0 = -jnp.log10(jnp.maximum(initial_target[:, h_idx] / initial_volume_l, 1e-14))
    ph_residual = pred_ph0 - target_ph0
    ph_mask = initial_mask[:, h_idx]
    ph0_loss = _masked_scalar_mse(ph_residual, ph_mask)

    states = _constrained_states(model, arrays["x"], arrays["volume_l"], arrays)
    observables = _observable_matrix(states, arrays["volume_l"])
    observable_columns = tuple(OBSERVABLE_COLUMNS)
    pred_ph = observables[:, observable_columns.index("pH")]
    pred_glucose_g_l = observables[:, observable_columns.index("glucose_mol_l")] * GLUCOSE_MOLAR_MASS_G_MOL
    ph_idx = arrays["initial_ph_column_index"]
    glucose_idx = arrays["initial_glucose_column_index"]
    initial_ph = arrays["x"][:, ph_idx] * arrays["x_std"][ph_idx] + arrays["x_mean"][ph_idx]
    initial_glucose_g_l = arrays["x"][:, glucose_idx] * arrays["x_std"][glucose_idx] + arrays["x_mean"][glucose_idx]
    lower = initial_ph - config.pH_range_margin
    upper = initial_ph + config.pH_range_margin
    ph_range_residual = jax.nn.relu(lower - pred_ph) + jax.nn.relu(pred_ph - upper)
    ph_range_loss = jnp.mean(ph_range_residual**2)

    glucose_scale = jnp.maximum(initial_glucose_g_l, 1.0)
    glucose_upper_violation = jax.nn.relu(pred_glucose_g_l - initial_glucose_g_l)
    glucose_upper_loss = jnp.mean((glucose_upper_violation / glucose_scale) ** 2)

    consecutive_same_experiment = arrays["experiment_codes"][1:] == arrays["experiment_codes"][:-1]
    increasing_glucose = jax.nn.relu(pred_glucose_g_l[1:] - pred_glucose_g_l[:-1])
    monotonic_scale = jnp.maximum(initial_glucose_g_l[:-1], 1.0)
    glucose_monotonic_loss = _masked_scalar_mse(
        increasing_glucose / monotonic_scale,
        consecutive_same_experiment,
    )

    initial_do_percent = _do_percent_from_o2_l(initial_pred[:, o2_idx], boundary_volume_l[0::2])
    final_do_percent = _do_percent_from_o2_l(final_pred[:, o2_idx], boundary_volume_l[1::2])
    do_initial_lower = config.DO_initial_target_percent - config.DO_initial_margin_percent
    do_initial_upper = config.DO_initial_target_percent + config.DO_initial_margin_percent
    do_end_lower = config.DO_end_target_percent - config.DO_end_margin_percent
    do_end_upper = config.DO_end_target_percent + config.DO_end_margin_percent
    do_initial_violation = (
        jax.nn.relu(do_initial_lower - initial_do_percent)
        + jax.nn.relu(initial_do_percent - do_initial_upper)
    )
    do_end_violation = (
        jax.nn.relu(do_end_lower - final_do_percent)
        + jax.nn.relu(final_do_percent - do_end_upper)
    )
    do_initial_loss = jnp.mean((do_initial_violation / 100.0) ** 2)
    do_end_loss = jnp.mean((do_end_violation / 100.0) ** 2)

    return (
        arrays["aux_fit_weights"][substrate_idx] * glucose0_loss
        + arrays["aux_fit_weights"][h_idx] * ph0_loss
        + config.pH_range_loss_weight * ph_range_loss
        + config.DO_initial_loss_weight * do_initial_loss
        + config.DO_end_loss_weight * do_end_loss
        + config.glucose_upper_loss_weight * glucose_upper_loss
        + config.glucose_monotonic_loss_weight * glucose_monotonic_loss
    )


def _do_percent_from_o2_l(o2_l_mol: jnp.ndarray, volume_l: jnp.ndarray) -> jnp.ndarray:
    p = PseudomonasBIOSODE.default_parameters
    saturation_o2_l_mol = p["FractionO2"] * p["Pr"] * p["HenryConstantO2"] * jnp.maximum(volume_l, 1e-12)
    return 100.0 * o2_l_mol / jnp.maximum(saturation_o2_l_mol, 1e-12)


def _regularization_loss(
    model,
    ode_params: dict[str, jnp.ndarray],
    arrays: dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """JVP regularization on the ODE residual, matching training_v1_1."""

    unit_tangent = jnp.zeros((arrays["x"].shape[1],), dtype=jnp.float32).at[arrays["time_column_index"]].set(1.0)

    def residual_derivative(xi, vl, vg, vs, va, vb):
        def residual_inner(x_inner):
            return _raw_ode_residual(model, ode_params, arrays, x_inner, vl, vg, vs, va, vb)

        _, residual_deriv_norm = jax.jvp(residual_inner, (xi,), (unit_tangent,))
        return residual_deriv_norm / arrays["time_x_std"]

    residual_dt_raw = jax.vmap(residual_derivative)(
        arrays["x"],
        arrays["volume_l"],
        arrays["air_flow_l_min"],
        arrays["sampling_rate_l_min"],
        arrays["acid_rate_l_min"],
        arrays["base_rate_l_min"],
    )
    mse_per_state = jnp.mean((residual_dt_raw / arrays["state_scale"]) ** 2, axis=0)
    return jnp.sum(arrays["reg_eq_weights"] * mse_per_state)


def _raw_ode_residual(
    model,
    ode_params: dict[str, jnp.ndarray],
    arrays: dict[str, jnp.ndarray],
    xi: jnp.ndarray,
    volume_l: jnp.ndarray,
    air_flow_l_min: jnp.ndarray,
    sampling_rate_l_min: jnp.ndarray,
    acid_rate_l_min: jnp.ndarray,
    base_rate_l_min: jnp.ndarray,
) -> jnp.ndarray:
    unit_tangent = jnp.zeros_like(xi).at[arrays["time_column_index"]].set(1.0)
    state = _constrained_state(model, xi, volume_l, arrays)
    dstate_dt_norm = jax.jvp(
        lambda x_inner: _constrained_state(model, x_inner, volume_l, arrays),
        (xi,),
        (unit_tangent,),
    )[1]
    dstate_dt_phys = dstate_dt_norm / arrays["time_x_std"]
    time_phys = xi[arrays["time_column_index"]] * arrays["time_x_std"] + arrays["time_x_mean"]
    rhs = PseudomonasBIOSODE.ode_func(
        time_phys,
        state,
        {
            **ode_params,
            "Vl": volume_l,
            "Vg": air_flow_l_min,
            "VOffGas": air_flow_l_min,
            "Vs": sampling_rate_l_min,
            "Va": acid_rate_l_min,
            "Vb": base_rate_l_min,
        },
    )
    return dstate_dt_phys - rhs


def _masked_mse_per_state(residual: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(jnp.sum(mask, axis=0), 1)
    return jnp.sum(jnp.where(mask, residual**2, 0.0), axis=0) / denom


def _masked_scalar_mse(residual: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    denom = jnp.maximum(jnp.sum(mask), 1)
    return jnp.sum(jnp.where(mask, residual**2, 0.0)) / denom


def _build_ode_params(
    theta_raw: jnp.ndarray,
    active_names: tuple[str, ...],
    frozen_parameters: dict[str, float],
) -> dict[str, jnp.ndarray]:
    params = dict(PseudomonasBIOSODE.default_parameters)
    for idx, name in enumerate(active_names):
        low, high = PseudomonasBIOSODE.parameter_ranges[name]
        params[name] = (jnp.tanh(theta_raw[idx]) + 1.0) * 0.5 * (high - low) + low
    for name, value in frozen_parameters.items():
        params[name] = float(value)
    return params


def _initial_theta_raw(active_names: tuple[str, ...]) -> jnp.ndarray:
    values = []
    for name in active_names:
        low, high = PseudomonasBIOSODE.parameter_ranges[name]
        default = PseudomonasBIOSODE.default_parameters[name]
        scaled = 2.0 * (default - low) / (high - low) - 1.0
        scaled = np.clip(scaled, -0.999999, 0.999999)
        values.append(np.arctanh(scaled))
    return jnp.asarray(values, dtype=jnp.float32)


def _validate_weight_vector(name: str, values: tuple[float, ...], expected_length: int) -> None:
    if len(values) != expected_length:
        raise ValueError(f"{name} must have exactly {expected_length} values.")
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative values.")


def _device_arrays(dataset: PseudomonasDataset, config: TrainingConfig | None = None) -> dict[str, jnp.ndarray]:
    time_idx = dataset.input_columns.index("time_min")
    do_setpoint_idx = dataset.input_columns.index("DO_setpoint")
    initial_glucose_idx = dataset.input_columns.index("initial_glucose_g_l")
    initial_ph_idx = dataset.input_columns.index("initial_pH")
    obs_fit_weights = config.obs_fit_weights if config is not None else (1.0,) * dataset.y.shape[1]
    aux_fit_weights = config.aux_fit_weights if config is not None else (1.0,) * len(PseudomonasBIOSODE.state_names)
    res_eq_weights = config.res_eq_weights if config is not None else (1.0,) * len(PseudomonasBIOSODE.state_names)
    reg_eq_weights = config.reg_eq_weights if config is not None else (1.0,) * len(PseudomonasBIOSODE.state_names)
    y_mask = np.array(dataset.y_mask, dtype=bool, copy=True)
    if config is not None and config.exclude_do_above_percent is not None and "DO_percent" in dataset.target_columns:
        do_idx = dataset.target_columns.index("DO_percent")
        y_mask[:, do_idx] &= dataset.y[:, do_idx] <= float(config.exclude_do_above_percent)
    return {
        "x": jnp.asarray(dataset.x, dtype=jnp.float32),
        "y": jnp.asarray(dataset.y, dtype=jnp.float32),
        "y_mask": jnp.asarray(y_mask),
        "time_min": jnp.asarray(dataset.time_min, dtype=jnp.float32),
        "do_setpoint": jnp.asarray(dataset.x_raw[:, do_setpoint_idx], dtype=jnp.float32),
        "volume_l": jnp.asarray(dataset.volume_l, dtype=jnp.float32),
        "air_flow_l_min": jnp.asarray(dataset.air_flow_l_min, dtype=jnp.float32),
        "sampling_rate_l_min": jnp.asarray(dataset.sampling_rate_l_min, dtype=jnp.float32),
        "acid_rate_l_min": jnp.asarray(dataset.acid_rate_l_min, dtype=jnp.float32),
        "base_rate_l_min": jnp.asarray(dataset.base_rate_l_min, dtype=jnp.float32),
        "experiment_codes": jnp.asarray(dataset.experiment_codes, dtype=jnp.int32),
        "boundary_x": jnp.asarray(dataset.boundary_x, dtype=jnp.float32),
        "boundary_y": jnp.asarray(dataset.boundary_y, dtype=jnp.float32),
        "boundary_mask": jnp.asarray(dataset.boundary_mask),
        "state_scale": jnp.asarray(dataset.state_scale, dtype=jnp.float32),
        "target_scale": jnp.asarray(dataset.target_scale, dtype=jnp.float32),
        "obs_fit_weights": jnp.asarray(obs_fit_weights, dtype=jnp.float32),
        "aux_fit_weights": jnp.asarray(aux_fit_weights, dtype=jnp.float32),
        "res_eq_weights": jnp.asarray(res_eq_weights, dtype=jnp.float32),
        "reg_eq_weights": jnp.asarray(reg_eq_weights, dtype=jnp.float32),
        "x_mean": jnp.asarray(dataset.x_mean, dtype=jnp.float32),
        "x_std": jnp.asarray(dataset.x_std, dtype=jnp.float32),
        "time_column_index": jnp.asarray(time_idx),
        "initial_glucose_column_index": jnp.asarray(initial_glucose_idx),
        "initial_ph_column_index": jnp.asarray(initial_ph_idx),
        "do_setpoint_column_index": jnp.asarray(do_setpoint_idx),
        "time_x_mean": jnp.asarray(dataset.x_mean[time_idx], dtype=jnp.float32),
        "time_x_std": jnp.asarray(dataset.x_std[time_idx], dtype=jnp.float32),
    }


__all__ = [
    "TrainingConfig",
    "predict_dataset",
    "run_leave_one_bioreactor_out",
    "save_checkpoint",
    "save_training_config",
    "train_pinn",
]
