#data/parameters.py
"""Parameters for the AMBR-oriented Pseudomonas putida bioprocess model.

The state order follows the 22-state VTT model. Default kinetic and ODE values
follow ``ODEModelPutida_LME_120326.ipynb``, the AMBR-oriented notebook.
All dynamic states are amounts in mol, except biomass in g. Time is in minutes.
The D5 medium recipe is kept as support data for metadata-based initialization.
"""

from __future__ import annotations


state_names = [
    "Substrate",
    "Biomass",
    "Cl",
    "Co",
    "Cu",
    "Fe",
    "Mg",
    "Mo",
    "Na",
    "Zn",
    "K",
    "Ni",
    "NH4",
    "P",
    "S",
    "CO2_l",
    "CO2_g",
    "O2_l",
    "O2_g",
    "HCO3",
    "OH",
    "H",
]


medium_recipes_mmol_l = {
    "D5": {
        "K": 22.04520737,
        "Na": 49.20911837,
        "PO4": 43.17821854,
        "SO4": 15.98793634,
        "Cl": 0.865412502,
        "Mg": 0.811470953,
        "Ca": 0.0,
        "NH4": 30.27229933,
        "CO3": 5.951955813,
        "EDTA": 0.067160971,
        "Zn": 0.001738882,
        "Fe": 0.008992612,
        "MoO4": 0.000619947,
        "Mn": 0.029584223,
        "Co": 0.00420304,
        "Cu": 0.000293296,
        "Ni": 0.000420728,
        "BO3": 0.024259676,
    }
}


physical_parameters_dict = {
    "R": 8.31451,
    "Tsc": 273.15,
    "TCelsius": 30.0,
    "TKelvin": 303.15,
    "Pr": 101325.0,
}


volume_flow_parameters_dict = {
    "Vl": 0.2,
    "Vtotal": 0.25,
    "Vgas": 0.05,
    "Vg": 0.105,
    "Vf": 0.0,
    "Vs": 0.0,
    "VOffGas": 0.105,
    "Va": 0.0,
    "Vb": 0.0,
}


yield_parameters_dict = {
    "Ysubs": 0.03333,
    "YCl": 4.227e-6,
    "YCo": 2.818e-9,
    "YCu": 2.818e-9,
    "YFe": 6.34e-9,
    "YMg": 7.045e-6,
    "YMo": 2.818e-9,
    "YNa": 3.522e-6,
    "YZn": 2.818e-9,
    "YK": 1.5851e-4,
    "YNi": 2.818e-9,
    "YNH4": 1.0567e-5,
    "YP": 3.522e-6,
    "YS": 3.522e-6,
    "YH": 1.0,
    "YO2": 0.0018,
    "YCO2": 0.06,
}


_D5 = medium_recipes_mmol_l["D5"]
_MMOL_TO_MOL = 1e-3


feed_concentration_parameters_dict = {
    "CSubs_f": 0.1,
    "CCl_f": 0.000887,
    "CCo_f": 0.000014,
    "CCu_f": 0.000001,
    "CFe_f": 0.000029,
    "CMg_f": 0.001445,
    "CMo_f": 0.000001,
    "CNa_f": 0.049357,
    "CZn_f": 0.000006,
    "CK_f": 0.022045,
    "CNi_f": 0.000001,
    "CNH4_f": 0.03028,
    "CP_f": 0.043178,
    "CS_f": 0.019068,
    "CHCO3_f": 0.00595,
    "CH_f": 0.0001,
    "COH_f": 1.0,
    "CH_a": 8.64,
    "COH_b": 4.0,
    "CNa_b": 4.0,
}


gas_parameters_dict = {
    "FractionO2": 0.21,
    "FractionCO2": 0.000407,
    "HenryConstantO2": 1.3e-8,
    "HenryConstantCO2": 3.4e-7,
}

gas_parameters_dict["PartialPressureO2"] = (
    gas_parameters_dict["FractionO2"] * physical_parameters_dict["Pr"]
)
gas_parameters_dict["PartialPressureCO2"] = (
    gas_parameters_dict["FractionCO2"] * physical_parameters_dict["Pr"]
)
gas_parameters_dict["CSatO2"] = (
    gas_parameters_dict["PartialPressureO2"] * gas_parameters_dict["HenryConstantO2"]
)
gas_parameters_dict["CSatCO2"] = (
    gas_parameters_dict["PartialPressureCO2"] * gas_parameters_dict["HenryConstantCO2"]
)
gas_parameters_dict["TotalMolesGas"] = (
    physical_parameters_dict["Pr"]
    * (volume_flow_parameters_dict["Vgas"] / 1000.0)
    / (physical_parameters_dict["R"] * physical_parameters_dict["TKelvin"])
)


pH_parameters_dict = {
    "pH_min": 5.0,
    "pH_opt": 6.5,
    "pH_max": 8.0,
}


initial_conditions_dict = {
    "Substrate": 0.055,
    "Biomass": 0.04267,
    "Cl": 0.000173082,
    "Co": 8.405819516967988e-7,
    "Cu": 5.865708131748498e-8,
    "Fe": 1.798470580618242e-6,
    "Mg": 1.622889229695627e-4,
    "Mo": 1.239887682707782e-7,
    "Na": 0.009841767287091,
    "Zn": 3.477547216397331e-7,
    "K": 0.004408992875802,
    "Ni": 8.414299934452604e-8,
    "NH4": 0.006054221608728,
    "P": 0.00863557426884131,
    "S": 0.00319746255924881,
    "CO2_l": gas_parameters_dict["CSatCO2"] * volume_flow_parameters_dict["Vl"],
    "CO2_g": gas_parameters_dict["FractionCO2"] * gas_parameters_dict["TotalMolesGas"],
    "O2_l": gas_parameters_dict["CSatO2"] * volume_flow_parameters_dict["Vl"],
    "O2_g": gas_parameters_dict["FractionO2"] * gas_parameters_dict["TotalMolesGas"],
    "HCO3": 0.00119,
    "OH": 0.281,
    "H": 0.2655,
}


ode_parameters_dict = {
    **physical_parameters_dict,
    **volume_flow_parameters_dict,
    **yield_parameters_dict,
    **feed_concentration_parameters_dict,
    **gas_parameters_dict,
    **pH_parameters_dict,
    "mu_max": 0.009,
    "Ksubs": 0.005,
    "KLaO2": 2.0,
    "KLaCO2": 10.0,
    "K_hyd": 2.79,
    "K_deh": 50880.0,
    "time_LagPhase": 80.0,
}


ode_parameter_ranges_dict = {
    "mu_max": (9e-6, 9.0),
    "Ksubs": (5e-7, 5.0),
    "KLaO2": (0.002, 2000.0),
    "KLaCO2": (0.01, 10000.0),
}
