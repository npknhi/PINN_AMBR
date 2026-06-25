# PINN AMBR Benchmarks

Physics-informed neural network benchmarks for AMBR Pseudomonas bioreactor trajectories.

The project currently supports two benchmark workflows:

- Leave-one-bioreactor-out evaluation (`LOO`)
- Temporal forecasting scan (`FCT`)

The active benchmark outputs are written under `results/LOO` and `results/FCT`.

## Repository Layout

```text
data/processed/                  Processed AMBR input data
notebooks/                       Benchmark notebooks
src/experiments/                 Benchmark orchestration
src/models/                      PINN model definition
src/utils/                       Data loading, training, evaluation, and utility helpers
results/LOO/                     Leave-one-bioreactor-out outputs
results/FCT/                     Forecasting scan outputs
```

Active experiment modules:

```text
src/experiments/leave_one_out.py
src/experiments/forecasting_scan.py
```

Active notebooks:

```text
notebooks/benchmark_leave_one_out.ipynb
notebooks/benchmark_forecasting_scan.ipynb
```

## Environment

The conda environment file is `env.yaml`.

```powershell
conda env create -f env.yaml
conda activate pinn_gpu
python -m ipykernel install --user --name pinn_gpu --display-name pinn_gpu
```

## Data

The benchmark notebooks expect the processed dataset at:
```text
data/processed/ambr_preprocessed.csv
```

Selected reactors:

```text
AMBR1_14, AMBR1_15, AMBR1_16, AMBR1_17, AMBR1_18,
AMBR1_21, AMBR1_22, AMBR1_23, AMBR1_24,
AMBR2_5, AMBR2_6, AMBR2_7, AMBR2_8, AMBR2_9,
AMBR2_10, AMBR2_13, AMBR2_14, AMBR2_15, AMBR2_16,
AMBR2_19, AMBR2_20, AMBR2_21
```

## Leave-One-Bioreactor-Out Benchmark

Notebook:

```text
notebooks/benchmark_leave_one_out.ipynb
```

## Forecasting Scan Benchmark

Notebook:

```text
notebooks/benchmark_forecasting_scan.ipynb
```

The forecasting scan trains one model per observation fraction and seed. Metrics are computed on the forecast window, using direct PINN predictions after the observed prefix cutoff.

## Glucose Interpolation Re-Scoring

For high observation fractions, glucose can have too few measured points in the forecast window for stable metric calculation. The interpolation helper recomputes forecasting metrics after linearly interpolating glucose ground-truth values between measured glucose samples.
