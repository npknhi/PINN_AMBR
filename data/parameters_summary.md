# AMBR Pseudomonas parameter summary

This note summarizes the entire parameters and the learnable parameters used by the PINN model.

## Totals

- Total parameters: 64
- Learnable / predicted parameters: 4

## Parameters to predict

These are the parameters optimized during training PINN:

| Parameter | Initial value | Range |
| --- | ---: | --- |
| `mu_max` | 0.009 | `(9e-6, 9.0)` |
| `Ksubs` | 0.005 | `(5e-7, 5.0)` |
| `KLaO2` | 2.0 | `(0.002, 2000.0)` |
| `KLaCO2` | 10.0 | `(0.01, 10000.0)` |

## All parameter groups

### Physical parameters

| Parameter | Value |
| --- | ---: |
| `R` | 8.31451 |
| `TKelvin` | 303.15 |
| `Pr` | 101325.0 |

### Volume / flow parameters

| Parameter | Value |
| --- | ---: |
| `Vl` | 0.2 |
| `Vtotal` | 0.25 |
| `Vg` | 0.105 |
| `Vf` | 0.0 |
| `Vs` | 0.0 |
| `VOffGas` | 0.105 |
| `Va` | 0.0 |
| `Vb` | 0.0 |

### Yield parameters

| Parameter | Value |
| --- | ---: |
| `Ysubs` | 0.03333 |
| `YCl` | 4.227e-6 |
| `YCo` | 2.818e-9 |
| `YCu` | 2.818e-9 |
| `YFe` | 6.34e-9 |
| `YMg` | 7.045e-6 |
| `YMo` | 2.818e-9 |
| `YNa` | 3.522e-6 |
| `YZn` | 2.818e-9 |
| `YK` | 1.5851e-4 |
| `YNi` | 2.818e-9 |
| `YNH4` | 1.0567e-5 |
| `YP` | 3.522e-6 |
| `YS` | 3.522e-6 |
| `YH` | 1.0 |
| `YO2` | 0.0018 |
| `YCO2` | 0.06 |

### Feed concentration parameters

| Parameter | Value |
| --- | ---: |
| `CSubs_f` | 0.1 |
| `CCl_f` | 0.000887 |
| `CCo_f` | 0.000014 |
| `CCu_f` | 0.000001 |
| `CFe_f` | 0.000029 |
| `CMg_f` | 0.001445 |
| `CMo_f` | 0.000001 |
| `CNa_f` | 0.049357 |
| `CZn_f` | 0.000006 |
| `CK_f` | 0.022045 |
| `CNi_f` | 0.000001 |
| `CNH4_f` | 0.03028 |
| `CP_f` | 0.043178 |
| `CS_f` | 0.019068 |
| `CHCO3_f` | 0.00595 |
| `CH_f` | 0.0001 |
| `COH_f` | 1.0 |
| `CH_a` | 8.64 |
| `COH_b` | 4.0 |
| `CNa_b` | 4.0 |

### Gas parameters

| Parameter | Value |
| --- | ---: |
| `FractionO2` | 0.21 |
| `FractionCO2` | 0.000407 |
| `HenryConstantO2` | 1.3e-8 |
| `CSatO2` | computed from `FractionO2 * Pr * HenryConstantO2` |
| `CSatCO2` | computed from `FractionCO2 * Pr * 3.4e-7` |
| `TotalMolesGas` | computed from reactor volume and gas law |

### pH parameters

| Parameter | Value |
| --- | ---: |
| `pH_min` | 5.0 |
| `pH_opt` | 6.5 |
| `pH_max` | 8.0 |

### ODE-specific parameters

| Parameter | Value |
| --- | ---: |
| `mu_max` | 0.009 |
| `Ksubs` | 0.005 |
| `KLaO2` | 2.0 |
| `KLaCO2` | 10.0 |
| `K_hyd` | 2.79 |
| `K_deh` | 50880.0 |
| `time_LagPhase` | 80.0 |
