#src/utils/training.py
from __future__ import annotations

"""Training utilities for the AMBR Pseudomonas PINN flow."""

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
    OBSERVABLE_COLUMNS,
    PseudomonasBIOSODE,
    STATE_INDEX,
    cardinal_pH,
    create_equinox_network,
)
from src.utils.dataloader import PseudomonasDataset
from src.utils.checkpointing import save_checkpoint, save_training_config


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
    forecast_observation_fraction: float | None = None

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
        if self.validation_strategy not in {"rotate", "random", "none"}:
            raise ValueError("validation_strategy must be 'rotate', 'random', or 'none'.")
        if self.forecast_observation_fraction is not None:
            fraction = float(self.forecast_observation_fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError("forecast_observation_fraction must be strictly between 0 and 1.")
            if not self.experiment_ids:
                raise ValueError("Independent forecasting requires experiment_ids.")
            if self.train_experiment_ids is not None or self.test_experiment_ids is not None:
                raise ValueError("Independent forecasting uses experiment_ids, not explicit train/test reactor ids.")
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


def train_pinn(
    config: TrainingConfig,
    *,
    train_dataset: PseudomonasDataset,
    val_dataset: PseudomonasDataset | None,
    test_dataset: PseudomonasDataset,
    split_metadata: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Train one PINN from datasets prepared by an experiment module."""

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
    val_arrays = _device_arrays(val_dataset, config) if val_dataset is not None else None
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

    if val_arrays is not None:
        @eqx.filter_jit
        def validation_data_loss(params):
            model_current, _ = params
            states = _constrained_states(model_current, val_arrays["x"], val_arrays["volume_l"], val_arrays)
            return _data_loss(states, val_arrays)
    else:
        validation_data_loss = None

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
        val_data_loss = validation_data_loss(params) if validation_data_loss is not None else float("nan")
        history["loss"].append(float(loss_value))
        history["val_data_loss"].append(float(val_data_loss))
        for name, value in components.items():
            history[name].append(float(value))
        if config.track_epoch_r2:
            model_current, _ = params
            train_r2 = _observable_r2_by_target(model_current, arrays, train_dataset.target_columns)
            history["r2_scores_train"].append(_mean_r2(train_r2.values()))
            if val_arrays is not None:
                val_r2 = _observable_r2_by_target(model_current, val_arrays, val_dataset.target_columns)
                history["r2_scores_val"].append(_mean_r2(val_r2.values()))
            else:
                val_r2 = {column: float("nan") for column in train_dataset.target_columns}
                history["r2_scores_val"].append(float("nan"))
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


def predict_dataset(
    model,
    dataset: PseudomonasDataset,
    *,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    arrays = _device_arrays(dataset)
    if batch_size is None:
        states_np = np.asarray(_constrained_states(model, arrays["x"], arrays["volume_l"], arrays))
    else:
        size = int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        batches = [
            np.asarray(
                _constrained_states(
                    model,
                    arrays["x"][start : start + size],
                    arrays["volume_l"][start : start + size],
                    arrays,
                )
            )
            for start in range(0, len(dataset.frame), size)
        ]
        states_np = np.concatenate(batches, axis=0)
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
    auxiliary_loss = _auxiliary_loss(model, arrays) if config.use_auxiliary_loss else jnp.asarray(0.0, dtype=jnp.float32)
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
    # ``x_mean`` stores the minimum value used by the min-max scaler.
    initial_time = arrays["time_x_offset"]
    is_initial_time = jnp.abs(time_phys - initial_time) <= 1e-6

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


def _auxiliary_loss(model, arrays: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Auxiliary condition loss from reliable condition constraints only.
    """

    boundary_volume_l = jnp.full(
        (arrays["boundary_x"].shape[0],),
        PseudomonasBIOSODE.default_parameters["Vl"],
        dtype=arrays["boundary_x"].dtype,
    )
    boundary_pred = _constrained_states(model, arrays["boundary_x"], boundary_volume_l, arrays)
    initial_pred = boundary_pred[0::2]
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

    return (
        arrays["aux_fit_weights"][substrate_idx] * glucose0_loss
        + arrays["aux_fit_weights"][h_idx] * ph0_loss
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
    time_phys = xi[arrays["time_column_index"]] * arrays["time_x_std"] + arrays["time_x_offset"]
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
    return {
        "x": jnp.asarray(dataset.x, dtype=jnp.float32),
        "y": jnp.asarray(dataset.y, dtype=jnp.float32),
        "y_mask": jnp.asarray(dataset.y_mask),
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
        "time_x_offset": jnp.asarray(dataset.x_mean[time_idx], dtype=jnp.float32),
        "time_x_std": jnp.asarray(dataset.x_std[time_idx], dtype=jnp.float32),
    }


__all__ = [
    "TrainingConfig",
    "predict_dataset",
    "save_checkpoint",
    "save_training_config",
    "train_pinn",
]
