#src/utils/training.py
from __future__ import annotations

"""Training utilities for the AMBR Pseudomonas PINN flow."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jinns
import numpy as np
import optax

from src.models.pinn import (
    INPUT_COLUMNS,
    OBSERVABLE_COLUMNS,
    PseudomonasBIOSODE,
    STATE_INDEX,
    cardinal_pH,
    create_equinox_network,
)
from src.utils.dataloader import PseudomonasDataset, load_pseudomonas_splits


@dataclass
class TrainingConfig:
    processed_csv: str = "data/processed/ambr_preprocessed.csv"
    experiment_id: str | None = None
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
    use_softadapt: bool = True
    obs_fit_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(OBSERVABLE_COLUMNS))
    aux_fit_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(PseudomonasBIOSODE.state_names))
    res_eq_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(PseudomonasBIOSODE.state_names))
    reg_eq_weights: tuple[float, ...] = field(default_factory=lambda: (1.0,) * len(PseudomonasBIOSODE.state_names))
    test_fraction: float = 0.2
    split_strategy: str = "random"
    trainable_parameters: tuple[str, ...] = tuple(PseudomonasBIOSODE.learnable_parameters)
    frozen_parameters: dict[str, float] = field(default_factory=dict)
    save_config: bool = True
    save_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1.")
        if self.split_strategy not in {"random", "time"}:
            raise ValueError("split_strategy must be either 'random' or 'time'.")
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
    train_dataset, test_dataset = load_pseudomonas_splits(
        config.processed_csv,
        input_columns=tuple(INPUT_COLUMNS),
        experiment_id=config.experiment_id,
        test_fraction=config.test_fraction,
        split_strategy=config.split_strategy,
        random_seed=config.seed,
    )

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
    test_arrays = _device_arrays(test_dataset, config)
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

    history: dict[str, list[float]] = {
        "loss": [],
        "data_loss": [],
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

    for epoch in range(config.num_epochs):
        params, opt_state, loss_value, components = train_step(params, opt_state, softadapt_weights)
        history["loss"].append(float(loss_value))
        for name, value in components.items():
            history[name].append(float(value))
        model_current, _ = params
        train_r2 = _observable_r2_by_target(model_current, arrays, train_dataset.target_columns)
        val_r2 = _observable_r2_by_target(model_current, test_arrays, test_dataset.target_columns)
        history["r2_scores_train"].append(_mean_r2(train_r2.values()))
        history["r2_scores_val"].append(_mean_r2(val_r2.values()))
        for column in train_dataset.target_columns:
            history[f"r2_train_{column}"].append(train_r2[column])
            history[f"r2_val_{column}"].append(val_r2[column])
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

    model, theta_raw = params
    ode_params = _build_ode_params(theta_raw, active_names, config.frozen_parameters)
    learned_params = {name: ode_params[name] for name in active_names}
    learned_params.update({name: float(value) for name, value in config.frozen_parameters.items()})
    train_predictions = predict_dataset(model, train_dataset)
    test_predictions = predict_dataset(model, test_dataset)
    train_gas_predictions = predict_gas_dataset(model, train_dataset, learned_params)
    test_gas_predictions = predict_gas_dataset(model, test_dataset, learned_params)

    output_dir = _output_dir_from_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = replace(train_dataset, output_dir=str(output_dir))
    test_dataset = replace(test_dataset, output_dir=str(output_dir))
    result = {
        "config": config,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "model": model,
        "theta_raw": np.asarray(theta_raw),
        "learned_params": {name: float(value) for name, value in learned_params.items()},
        "history": history,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "train_gas_predictions": train_gas_predictions,
        "test_gas_predictions": test_gas_predictions,
        "output_dir": str(output_dir),
    }
    if config.save_config:
        save_training_config(config, output_dir / "config.json")
    if config.save_checkpoint:
        save_checkpoint(result, output_dir / "checkpoints")
    return result


def _output_dir_from_config(config: TrainingConfig) -> Path:
    experiment_name = (config.experiment_name or "").strip()
    prefix = "pseudomonas_pinn_"
    if experiment_name.startswith(prefix):
        experiment_name = experiment_name[len(prefix):]
    elif experiment_name == "pseudomonas_pinn":
        experiment_name = ""

    if not experiment_name:
        experiment_name = config.experiment_id or "AMBR_all"

    return Path(config.results_dir) / experiment_name


def save_training_config(config: TrainingConfig, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_config_json_payload(config), indent=2), encoding="utf-8")
    return output_path


def save_checkpoint(result: dict[str, Any], checkpoint_dir: str | Path | None = None) -> dict[str, Path]:
    output_dir = Path(result["output_dir"])
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "model.eqx"
    theta_path = checkpoint_dir / "theta_raw.npy"
    learned_params_path = checkpoint_dir / "learned_params.json"
    history_path = checkpoint_dir / "history.csv"

    eqx.tree_serialise_leaves(model_path, result["model"])
    np.save(theta_path, np.asarray(result["theta_raw"]))
    learned_params_path.write_text(json.dumps(result["learned_params"], indent=2), encoding="utf-8")
    _history_frame(result["history"]).to_csv(history_path, index=False)

    return {
        "model": model_path,
        "theta_raw": theta_path,
        "learned_params": learned_params_path,
        "history": history_path,
    }


def _config_json_payload(config: TrainingConfig) -> dict[str, Any]:
    return {
        "data": {
            "processed_csv": config.processed_csv,
            "experiment_id": config.experiment_id,
            "test_fraction": float(config.test_fraction),
            "split_strategy": config.split_strategy,
        },
        "model": {
            "n_neurons": int(config.n_neurons),
            "n_hidden_layers": int(config.n_hidden_layers),
            "trainable_parameters": list(config.trainable_parameters),
            "frozen_parameters": {name: float(value) for name, value in config.frozen_parameters.items()},
        },
        "training": {
            "seed": int(config.seed),
            "num_epochs": int(config.num_epochs),
            "learning_rate": float(config.learning_rate),
            "use_softadapt": bool(config.use_softadapt),
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
        "output": {
            "results_dir": config.results_dir,
            "experiment_name": config.experiment_name,
        },
    }


def _history_frame(history: dict[str, list[float]]) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame({name: list(values) for name, values in history.items()})


def predict_dataset(model, dataset: PseudomonasDataset) -> dict[str, np.ndarray]:
    states = jax.vmap(model)(jnp.asarray(dataset.x, dtype=jnp.float32))
    states_np = np.asarray(states)
    predictions = {name: states_np[:, idx] for name, idx in PseudomonasBIOSODE.state_index.items()}
    volume = np.maximum(dataset.volume_l, 1e-12)
    predictions["glucose_mol_l"] = predictions["Substrate"] / volume
    predictions["biomass_g_l"] = predictions["Biomass"] / volume
    predictions["O2_l_mol"] = predictions["O2_l"]
    h_mol_l = predictions["H"] / volume
    predictions["pH"] = -np.log10(np.maximum(h_mol_l, 1e-14))
    return predictions


def predict_gas_dataset(
    model,
    dataset: PseudomonasDataset,
    ode_params: dict[str, float | jnp.ndarray],
) -> dict[str, np.ndarray]:
    states = jax.vmap(model)(jnp.asarray(dataset.x, dtype=jnp.float32))
    arrays = _device_arrays(dataset)
    pred = _gas_predictions(states, {**PseudomonasBIOSODE.default_parameters, **ode_params}, arrays)
    pred_np = np.asarray(pred)
    return {name: pred_np[:, idx] for idx, name in enumerate(dataset.gas_columns)}


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
    states = jax.vmap(model)(arrays["x"])

    data_loss = _data_loss(states, arrays)
    residual_loss = _residual_loss(model, ode_params, arrays)
    auxiliary_loss = _auxiliary_loss(model, arrays)
    regularization_loss = _regularization_loss(model, ode_params, arrays)

    total = jnp.sum(
        softadapt_weights
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
    states = jax.vmap(model)(arrays["x"])
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
            scores[column] = float(np.clip(r2, 0.0, 1.0))
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
            states[:, STATE_INDEX["O2_l"]],
            -jnp.log10(jnp.maximum(h_mol_l, 1e-14)),
        ],
        axis=1,
    )


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


def _auxiliary_loss(model, arrays: dict[str, jnp.ndarray]) -> jnp.ndarray:
    pred = jax.vmap(model)(arrays["boundary_x"])
    residual = (pred - arrays["boundary_y"]) / arrays["state_scale"]
    residual = jnp.where(arrays["boundary_mask"], residual, 0.0)
    start_mse = _masked_mse_per_state(residual[0::2], arrays["boundary_mask"][0::2])
    final_mse = _masked_mse_per_state(residual[1::2], arrays["boundary_mask"][1::2])
    mse_per_state = 0.5 * (start_mse + final_mse)
    return jnp.sum(arrays["aux_fit_weights"] * mse_per_state)


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
    state = model(xi)
    dstate_dt_norm = jax.jvp(lambda x_inner: model(x_inner), (xi,), (unit_tangent,))[1]
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
    obs_fit_weights = config.obs_fit_weights if config is not None else (1.0,) * dataset.y.shape[1]
    aux_fit_weights = config.aux_fit_weights if config is not None else (1.0,) * len(PseudomonasBIOSODE.state_names)
    res_eq_weights = config.res_eq_weights if config is not None else (1.0,) * len(PseudomonasBIOSODE.state_names)
    reg_eq_weights = config.reg_eq_weights if config is not None else (1.0,) * len(PseudomonasBIOSODE.state_names)
    return {
        "x": jnp.asarray(dataset.x, dtype=jnp.float32),
        "y": jnp.asarray(dataset.y, dtype=jnp.float32),
        "y_mask": jnp.asarray(dataset.y_mask),
        "time_min": jnp.asarray(dataset.time_min, dtype=jnp.float32),
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
        "time_column_index": jnp.asarray(time_idx),
        "time_x_mean": jnp.asarray(dataset.x_mean[time_idx], dtype=jnp.float32),
        "time_x_std": jnp.asarray(dataset.x_std[time_idx], dtype=jnp.float32),
    }


__all__ = [
    "TrainingConfig",
    "predict_dataset",
    "predict_gas_dataset",
    "save_checkpoint",
    "save_training_config",
    "train_pinn",
]
