from __future__ import annotations

"""Checkpoint and configuration serialization for PINN training runs."""

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import pandas as pd


def save_training_config(
    config: Any,
    output_path: str | Path,
    split_metadata: dict[str, object] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config_json_payload(config, split_metadata), indent=2), encoding="utf-8")
    return output


def save_checkpoint(result: dict[str, Any], checkpoint_dir: str | Path | None = None) -> dict[str, Path]:
    output_dir = Path(result["output_dir"])
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else output_dir / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": checkpoint / "model.eqx",
        "theta_raw": checkpoint / "theta_raw.npy",
        "learned_params": checkpoint / "learned_params.json",
        "history": checkpoint / "history.csv",
        "scalers": checkpoint / "scalers.npz",
        "dataset_columns": checkpoint / "dataset_columns.json",
    }
    eqx.tree_serialise_leaves(paths["model"], result["model"])
    np.save(paths["theta_raw"], np.asarray(result["theta_raw"]))
    paths["learned_params"].write_text(json.dumps(result["learned_params"], indent=2), encoding="utf-8")
    pd.DataFrame({name: list(values) for name, values in result["history"].items()}).to_csv(
        paths["history"], index=False
    )
    dataset = result["train_dataset"]
    np.savez_compressed(
        paths["scalers"],
        x_mean=np.asarray(dataset.x_mean),
        x_std=np.asarray(dataset.x_std),
        target_scale=np.asarray(dataset.target_scale),
        state_scale=np.asarray(dataset.state_scale),
        gas_scale=np.asarray(dataset.gas_scale),
    )
    paths["dataset_columns"].write_text(
        json.dumps(
            {
                "input_columns": list(dataset.input_columns),
                "target_columns": list(dataset.target_columns),
                "gas_columns": list(dataset.gas_columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def config_json_payload(config: Any, split_metadata: dict[str, object] | None = None) -> dict[str, Any]:
    data = {
        "processed_csv": config.processed_csv,
        "train_experiment_ids": list(config.train_experiment_ids) if config.train_experiment_ids is not None else None,
        "validation_experiment_ids": list(config.validation_experiment_ids)
        if config.validation_experiment_ids is not None
        else None,
        "test_experiment_ids": list(config.test_experiment_ids) if config.test_experiment_ids is not None else None,
        "split_strategy": (
            "independent_temporal_forecasting"
            if config.forecast_observation_fraction is not None
            else "bioreactor_id"
            if config.train_experiment_ids is not None and config.test_experiment_ids is not None
            else "leave_one_bioreactor_out"
        ),
        "validation_strategy": config.validation_strategy,
        "validation_seed": config.seed if config.validation_seed is None else int(config.validation_seed),
        "fold_index": int(config.loo_fold_index),
        "forecast_observation_fraction": config.forecast_observation_fraction,
    }
    payload = {
        "data": data,
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
            "obs_fit_weights": [float(value) for value in config.obs_fit_weights],
            "aux_fit_weights": [float(value) for value in config.aux_fit_weights],
            "res_eq_weights": [float(value) for value in config.res_eq_weights],
            "reg_eq_weights": [float(value) for value in config.reg_eq_weights],
        },
        "output": {"results_dir": config.results_dir, "experiment_name": config.experiment_name},
    }
    if config.frozen_parameters:
        payload["model"]["frozen_parameters"] = {
            name: float(value) for name, value in config.frozen_parameters.items()
        }
    if split_metadata is not None:
        for key in (
            "split_strategy",
            "n_splits",
            "fold_index",
            "validation_strategy",
            "validation_seed",
            "train_experiment_ids",
            "validation_experiment_ids",
            "test_experiment_ids",
            "experiment_ids",
            "observation_fraction",
            "cutoff_time_min",
        ):
            if key in split_metadata:
                data[key] = split_metadata[key]
    if config.forecast_observation_fraction is not None:
        data["prediction_mode"] = "direct_pinn"
        for key in (
            "train_experiment_ids",
            "validation_experiment_ids",
            "test_experiment_ids",
            "validation_strategy",
            "validation_seed",
            "fold_index",
            "forecast_observation_fraction",
        ):
            data.pop(key, None)
    return payload


__all__ = ["config_json_payload", "save_checkpoint", "save_training_config"]
