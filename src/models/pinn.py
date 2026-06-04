#src/models/pinn.py
from __future__ import annotations

"""PINN model for AMBR Pseudomonas putida dataset."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from data.parameters import ode_parameters_dict, ode_parameter_ranges_dict, state_names


INPUT_COLUMNS = [
    "time_min",
]


STATE_NAMES = tuple(state_names)
STATE_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}


OBSERVABLE_COLUMNS = {
    "glucose_mol_l": "Substrate",
    "biomass_g_l": "Biomass",
    "O2_l_mol": "O2_l",
    "pH": "H",
}


def cardinal_pH(ph: jnp.ndarray, ph_min: float, ph_opt: float, ph_max: float) -> jnp.ndarray:
    """Cardinal pH correction from ModelPseudomonasAutomated_LME."""

    denominator = (ph_opt - ph_min) * (ph_max - ph_opt)
    factor = (ph - ph_min) * (ph_max - ph) / (denominator + 1e-12)
    valid = (ph > ph_min) & (ph < ph_max)
    return jnp.where(valid, jnp.clip(factor, 0.0, 1.0), 0.0)


def cardinal_pH_numpy(ph: float, ph_min: float, ph_opt: float, ph_max: float) -> float:
    """NumPy scalar version of the cardinal pH correction."""

    if ph <= ph_min or ph >= ph_max:
        return 0.0
    denominator = (ph_opt - ph_min) * (ph_max - ph_opt)
    factor = (ph - ph_min) * (ph_max - ph) / (denominator + 1e-12)
    return float(np.clip(factor, 0.0, 1.0))


class PseudomonasBIOSODE:
    """AMBR-oriented Pseudomonas ODE system with 22 states.

    States follow ``data/parameters.py``:

    ``Substrate, Biomass, Cl, Co, Cu, Fe, Mg, Mo, Na, Zn, K, Ni, NH4, P, S,
    CO2_l, CO2_g, O2_l, O2_g, HCO3, OH, H``.
    """

    state_names = STATE_NAMES
    state_index = STATE_INDEX
    learnable_parameters = ["mu_max", "Ksubs", "KLaO2", "KLaCO2"]
    parameter_ranges = ode_parameter_ranges_dict
    default_parameters = ode_parameters_dict

    @staticmethod
    def ode_func(t: float, y: jnp.ndarray, params: Mapping[str, float]) -> jnp.ndarray:
        p = {**PseudomonasBIOSODE.default_parameters, **params}

        (
            substrate,
            biomass,
            cl,
            co,
            cu,
            fe,
            mg,
            mo,
            na,
            zn,
            k,
            ni,
            nh4,
            phosphorus,
            sulfur,
            co2_l,
            co2_g,
            o2_l,
            o2_g,
            hco3,
            oh,
            h,
        ) = y

        vl = p["Vl"]
        vgas = jnp.maximum(p.get("Vtotal", 0.25) - vl, 1e-9)
        vf = p["Vf"]
        vs = p["Vs"]
        vgas_in = p["Vg"]
        voffgas = p["VOffGas"]
        va = p["Va"]
        vb = p["Vb"]

        volume_safe = jnp.maximum(vl, 1e-12)
        substrate_safe = jnp.maximum(substrate, 0.0)
        ph = -jnp.log10(jnp.maximum(h / volume_safe, 1e-14))
        ph_factor = cardinal_pH(ph, p["pH_min"], p["pH_opt"], p["pH_max"])
        lag_phase_growth = 1.0 - jnp.exp(-jnp.maximum(t, 0.0) / p["time_LagPhase"])
        mu = p["mu_max"] * substrate_safe / (p["Ksubs"] + substrate_safe + 1e-12) * ph_factor

        co2_g_in = p["FractionCO2"] * p["TotalMolesGas"] / vgas
        o2_g_in = p["FractionO2"] * p["TotalMolesGas"] / vgas

        co2_transfer = p["KLaCO2"] * (p["CSatCO2"] - co2_l / vl) * vl
        o2_transfer = p["KLaO2"] * (p["CSatO2"] - o2_l / vl) * vl
        co2_hydration = p["K_hyd"] * co2_l - p["K_deh"] * hco3

        growth = mu * biomass

        dsubstrate_dt = p["CSubs_f"] * vf - (substrate / vl) * vs - p["Ysubs"] * growth
        dbiomass_dt = growth * lag_phase_growth - (biomass / vl) * vs
        dcl_dt = p["CCl_f"] * vf - (cl / vl) * vs - p["YCl"] * growth
        dco_dt = p["CCo_f"] * vf - (co / vl) * vs - p["YCo"] * growth
        dcu_dt = p["CCu_f"] * vf - (cu / vl) * vs - p["YCu"] * growth
        dfe_dt = p["CFe_f"] * vf - (fe / vl) * vs - p["YFe"] * growth
        dmg_dt = p["CMg_f"] * vf - (mg / vl) * vs - p["YMg"] * growth
        dmo_dt = p["CMo_f"] * vf - (mo / vl) * vs - p["YMo"] * growth
        dna_dt = p["CNa_f"] * vf + p["CNa_b"] * vb - (na / vl) * vs - p["YNa"] * growth
        dzn_dt = p["CZn_f"] * vf - (zn / vl) * vs - p["YZn"] * growth
        dk_dt = p["CK_f"] * vf - (k / vl) * vs - p["YK"] * growth
        dni_dt = p["CNi_f"] * vf - (ni / vl) * vs - p["YNi"] * growth
        dnh4_dt = p["CNH4_f"] * vf - (nh4 / vl) * vs - p["YNH4"] * growth
        dp_dt = p["CP_f"] * vf - (phosphorus / vl) * vs - p["YP"] * growth
        ds_dt = p["CS_f"] * vf - (sulfur / vl) * vs - p["YS"] * growth
        dco2_l_dt = co2_transfer - co2_hydration + p["YCO2"] * growth
        dco2_g_dt = co2_g_in * vgas_in - co2_transfer - (co2_g / vgas) * voffgas
        do2_l_dt = o2_transfer - p["YO2"] * growth
        do2_g_dt = o2_g_in * vgas_in - o2_transfer - (o2_g / vgas) * voffgas
        dhco3_dt = p["CHCO3_f"] * vf - (hco3 / vl) * vs + co2_hydration
        doh_dt = p["COH_b"] * vb + p["COH_f"] * vf - (oh / vl) * vs
        dh_dt = p["CH_f"] * vf + p["CH_a"] * va - (h / vl) * vs + p["YH"] * growth

        return jnp.array(
            [
                dsubstrate_dt,
                dbiomass_dt,
                dcl_dt,
                dco_dt,
                dcu_dt,
                dfe_dt,
                dmg_dt,
                dmo_dt,
                dna_dt,
                dzn_dt,
                dk_dt,
                dni_dt,
                dnh4_dt,
                dp_dt,
                ds_dt,
                dco2_l_dt,
                dco2_g_dt,
                do2_l_dt,
                do2_g_dt,
                dhco3_dt,
                doh_dt,
                dh_dt,
            ]
        )

    @staticmethod
    def ode_func_numpy(t: float, y: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
        """NumPy ODE RHS for SciPy solvers.

        Dataloader boundary generation uses this method so the ODE equations
        live with the model definition instead of being duplicated in the data
        utilities.
        """

        p = {**PseudomonasBIOSODE.default_parameters, **params}
        (
            substrate,
            biomass,
            cl,
            co,
            cu,
            fe,
            mg,
            mo,
            na,
            zn,
            k,
            ni,
            nh4,
            phosphorus,
            sulfur,
            co2_l,
            co2_g,
            o2_l,
            o2_g,
            hco3,
            oh,
            h,
        ) = np.asarray(y, dtype=np.float64)

        vl = max(float(p["Vl"]), 1e-12)
        vgas = max(float(p.get("Vtotal", 0.25)) - vl, 1e-9)
        vf = float(p["Vf"])
        vs = float(p["Vs"])
        vgas_in = float(p["Vg"])
        voffgas = float(p["VOffGas"])
        va = float(p["Va"])
        vb = float(p["Vb"])

        substrate_safe = max(float(substrate), 0.0)
        ph = -np.log10(max(float(h) / vl, 1e-14))
        ph_factor = cardinal_pH_numpy(ph, float(p["pH_min"]), float(p["pH_opt"]), float(p["pH_max"]))
        lag_phase_growth = 1.0 - np.exp(-max(float(t), 0.0) / float(p["time_LagPhase"]))
        mu = float(p["mu_max"]) * substrate_safe / (float(p["Ksubs"]) + substrate_safe + 1e-12) * ph_factor

        co2_g_in = float(p["FractionCO2"]) * float(p["TotalMolesGas"]) / vgas
        o2_g_in = float(p["FractionO2"]) * float(p["TotalMolesGas"]) / vgas

        co2_transfer = float(p["KLaCO2"]) * (float(p["CSatCO2"]) - co2_l / vl) * vl
        o2_transfer = float(p["KLaO2"]) * (float(p["CSatO2"]) - o2_l / vl) * vl
        co2_hydration = float(p["K_hyd"]) * co2_l - float(p["K_deh"]) * hco3

        growth = mu * biomass

        return np.asarray(
            [
                float(p["CSubs_f"]) * vf - (substrate / vl) * vs - float(p["Ysubs"]) * growth,
                growth * lag_phase_growth - (biomass / vl) * vs,
                float(p["CCl_f"]) * vf - (cl / vl) * vs - float(p["YCl"]) * growth,
                float(p["CCo_f"]) * vf - (co / vl) * vs - float(p["YCo"]) * growth,
                float(p["CCu_f"]) * vf - (cu / vl) * vs - float(p["YCu"]) * growth,
                float(p["CFe_f"]) * vf - (fe / vl) * vs - float(p["YFe"]) * growth,
                float(p["CMg_f"]) * vf - (mg / vl) * vs - float(p["YMg"]) * growth,
                float(p["CMo_f"]) * vf - (mo / vl) * vs - float(p["YMo"]) * growth,
                float(p["CNa_f"]) * vf + float(p["CNa_b"]) * vb - (na / vl) * vs - float(p["YNa"]) * growth,
                float(p["CZn_f"]) * vf - (zn / vl) * vs - float(p["YZn"]) * growth,
                float(p["CK_f"]) * vf - (k / vl) * vs - float(p["YK"]) * growth,
                float(p["CNi_f"]) * vf - (ni / vl) * vs - float(p["YNi"]) * growth,
                float(p["CNH4_f"]) * vf - (nh4 / vl) * vs - float(p["YNH4"]) * growth,
                float(p["CP_f"]) * vf - (phosphorus / vl) * vs - float(p["YP"]) * growth,
                float(p["CS_f"]) * vf - (sulfur / vl) * vs - float(p["YS"]) * growth,
                co2_transfer - co2_hydration + float(p["YCO2"]) * growth,
                co2_g_in * vgas_in - co2_transfer - (co2_g / vgas) * voffgas,
                o2_transfer - float(p["YO2"]) * growth,
                o2_g_in * vgas_in - o2_transfer - (o2_g / vgas) * voffgas,
                float(p["CHCO3_f"]) * vf - (hco3 / vl) * vs + co2_hydration,
                float(p["COH_b"]) * vb + float(p["COH_f"]) * vf - (oh / vl) * vs,
                float(p["CH_f"]) * vf + float(p.get("CH_a", 0.0)) * va - (h / vl) * vs + float(p["YH"]) * growth,
            ],
            dtype=np.float64,
        )


class PositiveOutputMLP(eqx.Module):
    """Equinox MLP with non-negative, physical-scale state predictions.

    The raw network learns dimensionless positive outputs. Multiplying by
    ``output_scale`` keeps tiny states such as dissolved gases and ions near
    their physical order of magnitude from the start of training, while callers
    still receive states in the ODE's physical units.
    """

    base: eqx.nn.MLP
    output_scale: tuple[float, ...] = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = jnp.asarray(self.output_scale, dtype=x.dtype)
        return jax.nn.softplus(self.base(x)) * scale


class PseudomonasPINN(eqx.Module):
    """State predictor for the processed AMBR tables."""

    network: PositiveOutputMLP
    input_columns: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.network(x)


def create_equinox_network(
    key: jax.Array,
    n_neurons: int = 64,
    n_hidden_layers: int = 4,
    input_columns: Sequence[str] = INPUT_COLUMNS,
    output_names: Sequence[str] = STATE_NAMES,
    output_scale: Sequence[float] | None = None,
) -> PseudomonasPINN:
    """Create the PINN state network for the new processed AMBR data."""

    if output_scale is None:
        output_scale = (1.0,) * len(output_names)
    output_scale = tuple(float(value) for value in output_scale)
    if len(output_scale) != len(output_names):
        raise ValueError("output_scale must have one value per output state.")

    return PseudomonasPINN(
        network=PositiveOutputMLP(
            base=eqx.nn.MLP(
                in_size=len(input_columns),
                out_size=len(output_names),
                width_size=n_neurons,
                depth=n_hidden_layers,
                activation=jax.nn.softplus,
                key=key,
            ),
            output_scale=output_scale,
        ),
        input_columns=tuple(input_columns),
        output_names=tuple(output_names),
    )


def observable_predictions(states: jnp.ndarray, volume_l: jnp.ndarray | float | None = None) -> dict[str, jnp.ndarray]:
    """Map state predictions to measured columns in the processed data.

    ``Substrate`` is a mol amount and ``Biomass`` is a gram amount, while the
    measured targets are concentrations. A liquid volume is required for those
    observables.
    """

    predictions = {
        "O2_l_mol": states[..., STATE_INDEX["O2_l"]],
    }
    if volume_l is not None:
        volume_safe = jnp.maximum(volume_l, 1e-12)
        predictions["glucose_mol_l"] = states[..., STATE_INDEX["Substrate"]] / volume_safe
        predictions["biomass_g_l"] = states[..., STATE_INDEX["Biomass"]] / volume_safe
        h_mol_l = states[..., STATE_INDEX["H"]] / volume_safe
        predictions["pH"] = -jnp.log10(jnp.maximum(h_mol_l, 1e-14))
    return predictions


def trainable_parameter_initial_values() -> dict[str, float]:
    """Defaults for the fitted AMBR ODE parameters."""

    return {name: float(ode_parameters_dict[name]) for name in PseudomonasBIOSODE.learnable_parameters}
