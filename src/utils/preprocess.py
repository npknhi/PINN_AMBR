#src/utils/preprocess.py
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OD_TO_BIOMASS_G_L = 0.4267
GLUCOSE_MOLAR_MASS_G_MOL = 180.156
STANDARD_PRESSURE_PA = 101325.0
FRACTION_O2_AIR = 0.21
HENRY_CONSTANT_O2_MOL_L_PA = 1.3e-8
MAX_REASONABLE_SAMPLING_RATE_L_MIN = 0.01


AMBR_COLUMN_RE = re.compile(r"^Bioreactor (?P<reactor>\d+) - (?P<measurement>.+)$")

AMBR_RENAME_MAP = {
    "Acid volume pumped": "acid_volume_pumped_ml",
    "Air flow": "air_flow_ml_min",
    "Base volume pumped": "base_volume_pumped_ml",
    "CER": "CER_mmol_h",
    "DO": "DO_percent",
    "Off-gas CO2%": "CO2_offgas_percent",
    "Optical density": "OD",
    "OUR": "OUR_mmol_h",
    "pH": "pH",
    "RQ": "RQ",
    "Sampling events": "sampling_events",
    "Volume": "volume_ml",
}

OFFLINE_RENAME_MAP = {
    "Reactor": "reactor_label",
    "Time": "offline_time_h",
    "Sampling events": "offline_sampling_events",
    "Sample volume (ml)": "sample_volume_ml",
    "Glucose": "glucose_g_l",
    "Glucose Vanquish": "glucose_g_l",
}

FINAL_TRAINING_COLUMNS = [
    "Experiment_id",
    "condition_label",
    "initial_pH",
    "DO_setpoint",
    "initial_glucose_g_l",
    "AFR_setpoint",
    "antifoam_used",
    "antifoam_absent",
    "antifoam_present",
    "initial_od",
    "time_h",
    "time_min",
    "volume_l",
    "air_flow_l_min",
    "sampling_rate_l_min",
    "acid_rate_l_min",
    "base_rate_l_min",
    "initial_biomass_g_l",
    "initial_glucose_mol_l",
    "initial_volume_l",
    "glucose_g_l",
    "glucose_mol_l",
    "biomass_g_l",
    "DO_percent",
    "O2_l_mol",
    "pH",
    "OUR_mol_min",
    "CER_mol_min",
    "CO2_offgas_fraction",
    "RQ",
]

REQUIRED_MEASURED_TARGET_COLUMNS = [
    "biomass_g_l",
    "DO_percent",
    "pH",
]

TRAINING_OBSERVABLE_COLUMNS = [
    "glucose_g_l",
    "biomass_g_l",
    "DO_percent",
    "pH",
]

DEFAULT_PREPROCESSED_CSV = DEFAULT_OUTPUT_DIR / "ambr_preprocessed.csv"

REAL_DATA_PLOT_COLUMNS = ("glucose_g_l", "biomass_g_l", "DO_percent", "pH")

REAL_DATA_PLOT_TITLES = {
    "glucose_g_l": "Glucose",
    "biomass_g_l": "Biomass",
    "DO_percent": "Dissolved O2",
    "pH": "pH",
}

REAL_DATA_PLOT_YLABELS = {
    "glucose_g_l": "g/L",
    "biomass_g_l": "g/L",
    "DO_percent": "%",
    "pH": "pH",
}

REAL_DATA_PLOT_COLORS = {
    "glucose_g_l": "tab:blue",
    "biomass_g_l": "tab:green",
    "DO_percent": "tab:pink",
    "pH": "tab:orange",
}


def load_ambr_timeseries(raw_dir: Path) -> pd.DataFrame:
    frames = []
    ambr_dir = raw_dir / "Data_AMBR_230925"
    for path in sorted(ambr_dir.glob("ambr_run*.csv")):
        wide = pd.read_csv(path)
        wide.columns = [column.strip().strip('"') for column in wide.columns]
        measurement_columns = [
            column
            for column in wide.columns
            if (match := AMBR_COLUMN_RE.match(column)) and match.group("measurement") in AMBR_RENAME_MAP
        ]

        long = wide.melt(
            id_vars=["Time"],
            value_vars=measurement_columns,
            var_name="raw_measurement",
            value_name="value",
        )
        extracted = long["raw_measurement"].str.extract(AMBR_COLUMN_RE)
        long["Experiment_id"] = _experiment_id_from_online_file(path, extracted["reactor"])
        long["measurement"] = extracted["measurement"]
        long["time_h"] = pd.to_numeric(long["Time"], errors="coerce")
        numeric_values = pd.to_numeric(long["value"], errors="coerce")
        long["value"] = numeric_values.where(numeric_values.notna(), long["value"])

        pivoted = (
            long.pivot_table(
                index=["Experiment_id", "time_h"],
                columns="measurement",
                values="value",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        pivoted = pivoted.rename(columns=AMBR_RENAME_MAP)
        frames.append(pivoted)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "Data_AMBR_230925" / "ambr.csv"
    metadata = pd.read_csv(path, encoding="cp1252")
    metadata = metadata.rename(columns=lambda column: str(column).strip())

    metadata["Experiment_id"] = metadata["Tunniste"].map(_experiment_id_from_tunniste)
    metadata["initial_od"] = pd.to_numeric(metadata["InitialOD"], errors="coerce")
    metadata["initial_biomass_g_l"] = metadata["initial_od"] * OD_TO_BIOMASS_G_L
    metadata["initial_glucose_g_l"] = pd.to_numeric(metadata["InitialConcentration(g/L)"], errors="coerce")
    metadata["initial_glucose_mol_l"] = metadata["initial_glucose_g_l"] / GLUCOSE_MOLAR_MASS_G_MOL
    metadata["initial_pH"] = pd.to_numeric(metadata["InitialPH"], errors="coerce")
    metadata["initial_volume_l"] = _extract_first_number(metadata["LiquidVolume"]) / 1000.0
    metadata["temperature_setpoint_c"] = _extract_first_number(metadata["Temperature"])
    metadata["antifoam_used"] = metadata["Experiment_id"].astype(str).str.startswith("AMBR2_").astype(int)
    metadata["antifoam_absent"] = 1 - metadata["antifoam_used"]
    metadata["antifoam_present"] = metadata["antifoam_used"]
    condition_mapping = load_condition_mapping(raw_dir)
    metadata = metadata.merge(condition_mapping, on="Experiment_id", how="left")
    metadata = standardize_condition_labels(metadata)

    keep = [
        "Experiment_id",
        "condition_label",
        "initial_pH",
        "DO_setpoint",
        "initial_glucose_g_l",
        "AFR_setpoint",
        "antifoam_used",
        "antifoam_absent",
        "antifoam_present",
        "initial_od",
        "initial_biomass_g_l",
        "initial_glucose_mol_l",
        "initial_volume_l",
        "temperature_setpoint_c",
    ]
    return metadata[keep]


def load_condition_mapping(raw_dir: Path) -> pd.DataFrame:
    """Load raw condition labels when mapping files are available."""

    mapping_dir = raw_dir / "Model_pseudomonas_putida_120326"
    frames = []
    for prefix in ("AMBR1", "AMBR2"):
        path = mapping_dir / f"MappingData_DissolvedOxygen{prefix}.csv"
        if not path.exists():
            continue
        mapping = pd.read_csv(path, sep=";")
        mapping = mapping.rename(columns=lambda column: str(column).strip())
        mapping["Experiment_id"] = mapping["Reactor"].map(lambda value: _experiment_id_from_reactor_label(prefix, value))
        mapping["raw_condition_label"] = mapping["Label"].map(lambda value: str(value).strip() if pd.notna(value) else "")
        mapping["raw_condition_label"] = mapping["raw_condition_label"].replace("", pd.NA)
        frames.append(mapping[["Experiment_id", "raw_condition_label"]])
    if not frames:
        return pd.DataFrame(columns=["Experiment_id", "raw_condition_label"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Experiment_id"], keep="first")


def load_offline_metabolites(raw_dir: Path) -> pd.DataFrame:
    glucose_dir = raw_dir / "Glucose_files_310326"
    ambr1_path = glucose_dir / "OfflineMetabolites_AMBR1.csv"
    ambr2_path = glucose_dir / "OfflineMetabolites_AMBR2.csv"
    offline_frames = [
        _prepare_offline_metabolites(pd.read_csv(ambr1_path, sep=";"), "AMBR1"),
        _prepare_offline_metabolites(pd.read_csv(ambr2_path), "AMBR2"),
    ]
    offline = pd.concat(offline_frames, ignore_index=True)
    offline["Experiment_id"] = offline.apply(
        lambda row: _experiment_id_from_reactor_label(row.get("experiment_prefix"), row.get("reactor_label")),
        axis=1,
    )
    offline = offline[["Experiment_id", "offline_time_h", "offline_sampling_events", "sample_volume_ml", "glucose_g_l"]].copy()
    offline["offline_time_h"] = pd.to_numeric(offline["offline_time_h"], errors="coerce")
    offline["sample_volume_ml"] = pd.to_numeric(offline["sample_volume_ml"], errors="coerce")
    offline["glucose_g_l"] = pd.to_numeric(offline["glucose_g_l"], errors="coerce")
    offline["glucose_mol_l"] = offline["glucose_g_l"] / GLUCOSE_MOLAR_MASS_G_MOL
    return offline


def add_unit_conversions(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    converted["time_min"] = converted["time_h"] * 60.0
    converted["volume_l"] = converted["volume_ml"] / 1000.0
    converted["air_flow_l_min"] = converted["air_flow_ml_min"] / 1000.0
    converted["biomass_g_l"] = converted["OD"] * OD_TO_BIOMASS_G_L
    converted["OUR_mol_min"] = converted["OUR_mmol_h"] * 1e-3 / 60.0
    converted["CER_mol_min"] = converted["CER_mmol_h"] * 1e-3 / 60.0
    converted["CO2_offgas_fraction"] = converted["CO2_offgas_percent"] / 100.0
    do_fraction = pd.to_numeric(converted["DO_percent"], errors="coerce").clip(lower=0.0) / 100.0
    converted["O2_l_mol"] = (
        do_fraction * FRACTION_O2_AIR * STANDARD_PRESSURE_PA * HENRY_CONSTANT_O2_MOL_L_PA * converted["volume_l"]
    )
    return converted


def merge_offline_by_sampling_event(
    online: pd.DataFrame,
    offline: pd.DataFrame,
) -> pd.DataFrame:
    offline_columns = [
        column
        for column in offline.columns
        if column not in {"Experiment_id", "offline_sampling_events"}
    ]

    left = online.copy()
    right = offline.copy()
    left["sampling_event_key"] = left["sampling_events"].map(_normalize_sampling_event)
    right["sampling_event_key"] = right["offline_sampling_events"].map(_normalize_sampling_event)
    right = right[right["sampling_event_key"].notna()]
    right = right.drop_duplicates(subset=["Experiment_id", "sampling_event_key"], keep="first")

    merged = left.merge(
        right[["Experiment_id", "sampling_event_key", *offline_columns]],
        on=["Experiment_id", "sampling_event_key"],
        how="left",
        suffixes=("", "_offline"),
    )
    # The offline timestamp is often only a few microseconds after the matching
    # online row. Keep one row per sampling event so that this numerical jitter
    # cannot become an artificial near-zero sampling interval.
    return merged.drop(columns=["sampling_event_key"])


def add_sampling_rate(frame: pd.DataFrame) -> pd.DataFrame:
    sampled = _sort_by_experiment_and_time(frame, "time_min")
    dt_min = sampled.groupby("Experiment_id", sort=False)["time_min"].diff()
    sampled["sample_volume_l"] = pd.to_numeric(sampled["sample_volume_ml"], errors="coerce") / 1000.0
    sampled["sampling_rate_l_min"] = (sampled["sample_volume_l"] / dt_min).where(dt_min > 0, 0.0)
    sampled["acid_rate_l_min"] = _cumulative_rate_l_min(sampled, "acid_volume_pumped_ml", dt_min)
    sampled["base_rate_l_min"] = _cumulative_rate_l_min(sampled, "base_volume_pumped_ml", dt_min)
    sampled["sample_volume_l"] = sampled["sample_volume_l"].fillna(0.0)
    sampled["sampling_rate_l_min"] = sampled["sampling_rate_l_min"].fillna(0.0)
    invalid = sampled["sampling_rate_l_min"] > MAX_REASONABLE_SAMPLING_RATE_L_MIN
    if invalid.any():
        columns = ["Experiment_id", "time_min", "sample_volume_l", "sampling_rate_l_min"]
        examples = sampled.loc[invalid, columns].head(5).to_dict("records")
        raise ValueError(
            "Unphysical sampling rate detected after preprocessing; "
            f"maximum allowed is {MAX_REASONABLE_SAMPLING_RATE_L_MIN:g} L/min. Examples: {examples}"
        )
    return sampled


def keep_training_columns(frame: pd.DataFrame) -> pd.DataFrame:
    complete_online = frame.loc[:, REQUIRED_MEASURED_TARGET_COLUMNS].notna().all(axis=1)
    cutoff_by_experiment = (
        frame.loc[complete_online]
        .groupby("Experiment_id", sort=False)["time_min"]
        .max()
        .rename("_online_cutoff_time_min")
    )
    with_cutoff = frame.merge(cutoff_by_experiment, on="Experiment_id", how="left")
    before_cutoff = with_cutoff["time_min"] <= with_cutoff["_online_cutoff_time_min"]
    has_observable = with_cutoff.loc[:, TRAINING_OBSERVABLE_COLUMNS].notna().any(axis=1)
    filtered = with_cutoff.loc[before_cutoff & has_observable, FINAL_TRAINING_COLUMNS]
    return filtered.reset_index(drop=True).copy()


def load_preprocessed_ambr_data(csv_path: str | Path = DEFAULT_PREPROCESSED_CSV) -> pd.DataFrame:
    """Load preprocessed AMBR real data for plotting."""

    frame = pd.read_csv(csv_path)
    frame = frame.copy()
    frame["Experiment_id"] = frame["Experiment_id"].astype(str)
    for column in ("time_h", *REAL_DATA_PLOT_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _sort_by_experiment_and_time(frame, "time_h")


def get_experiment_ids(frame: pd.DataFrame) -> list[str]:
    """Return experiment ids in natural AMBR/reactor order."""

    experiment_ids = frame["Experiment_id"].dropna().astype(str).drop_duplicates()
    return sorted(experiment_ids, key=_experiment_id_sort_key)


def _get_pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams["figure.max_open_warning"] = 100
    return plt


def plot_ambr_experiment(
    frame: pd.DataFrame,
    experiment_id: str,
) -> plt.Figure:
    """Plot real data for one AMBR experiment."""

    plt = _get_pyplot()
    data = frame.loc[frame["Experiment_id"].astype(str).eq(str(experiment_id))].copy()
    if data.empty:
        raise ValueError(f"No rows found for Experiment_id={experiment_id!r}.")

    columns = list(REAL_DATA_PLOT_COLUMNS)
    ncols = 2 if len(columns) > 1 else 1
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3333 * ncols, 3 * nrows), sharex=True)
    axes_array = axes.ravel()
    fig.suptitle(f"AMBR Real Data - {experiment_id}", fontsize=16, fontweight="bold")

    for idx, (ax, column) in enumerate(zip(axes_array, columns)):
        color = REAL_DATA_PLOT_COLORS.get(column, f"C{idx}")
        observed = data[column].notna()

        if observed.any():
            ax.plot(
                data.loc[observed, "time_h"],
                data.loc[observed, column],
                "o",
                label=f"{column}_data",
                color=color,
                markersize=6,
                alpha=0.6,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No real data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.4",
            )

        ax.set_title(REAL_DATA_PLOT_TITLES[column], fontweight="bold", fontsize=12)
        ax.set_ylabel(REAL_DATA_PLOT_YLABELS[column], fontsize=10)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Time (h)", fontsize=10)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        legend_map = dict(zip(labels, handles))
        ordered_labels = [f"{column}_data"]
        ordered_labels = [name for name in ordered_labels if name in legend_map]
        if ordered_labels:
            ordered_handles = [legend_map[name] for name in ordered_labels]
            ax.legend(ordered_handles, ordered_labels, loc="best", fontsize=9)

    for ax in axes_array[len(columns):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_all_ambr_experiments(
    frame: pd.DataFrame,
    *,
    max_experiments: int | None = None,
) -> list[plt.Figure]:
    """Plot all AMBR experiments and return the created figures."""

    experiment_ids = get_experiment_ids(frame)
    if max_experiments is not None:
        experiment_ids = experiment_ids[:max_experiments]
    return [plot_ambr_experiment(frame, experiment_id) for experiment_id in experiment_ids]


def _cumulative_rate_l_min(frame: pd.DataFrame, column: str, dt_min: pd.Series) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    delta_l = values.groupby(frame["Experiment_id"], sort=False).diff().clip(lower=0.0) / 1000.0
    return (delta_l / dt_min).where(dt_min > 0, 0.0).fillna(0.0)


def preprocess(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(raw_dir)
    online = load_ambr_timeseries(raw_dir)
    offline = load_offline_metabolites(raw_dir)

    online = add_unit_conversions(online)
    online = online.merge(metadata, on="Experiment_id", how="left")

    final = keep_training_columns(add_sampling_rate(merge_offline_by_sampling_event(online, offline)))

    paths = {
        "metadata": output_dir / "ambr_metadata.csv",
        "final": output_dir / "ambr_preprocessed.csv",
    }
    metadata.to_csv(paths["metadata"], index=False)
    final.to_csv(paths["final"], index=False)
    return paths


def standardize_condition_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    standardized = metadata.copy()
    parsed = standardized["raw_condition_label"].map(_parse_condition_label)
    parsed_frame = pd.DataFrame(parsed.tolist(), index=standardized.index)

    ph_values = parsed_frame["pH"].combine_first(pd.to_numeric(standardized["initial_pH"], errors="coerce"))
    glucose_values = parsed_frame["G"].combine_first(
        pd.to_numeric(standardized["initial_glucose_g_l"], errors="coerce")
    )
    do_values = parsed_frame["DO"].fillna(0.0)
    afr_values = parsed_frame["AFR"].fillna(100.0)

    standardized["condition_label"] = [
        f"pH{_format_condition_value(ph)}_DO{_format_condition_value(do)}_G{_format_condition_value(glucose)}_AFR{_format_condition_value(afr)}"
        for ph, do, glucose, afr in zip(ph_values, do_values, glucose_values, afr_values)
    ]
    standardized["DO_setpoint"] = do_values.astype(float)
    standardized["AFR_setpoint"] = afr_values.astype(float)
    return standardized.drop(columns=["raw_condition_label"])


def _clean_column_name(column: str) -> str:
    return re.sub(r"\s+", " ", column).strip()


def _prepare_offline_metabolites(frame: pd.DataFrame, experiment_prefix: str) -> pd.DataFrame:
    rename = {_clean_column_name(key): value for key, value in OFFLINE_RENAME_MAP.items()}
    prepared = frame.rename(columns=lambda column: _clean_column_name(str(column))).rename(columns=rename)
    prepared["experiment_prefix"] = experiment_prefix
    return prepared[[
        "experiment_prefix",
        "reactor_label",
        "offline_time_h",
        "offline_sampling_events",
        "sample_volume_ml",
        "glucose_g_l",
    ]]


def _normalize_sampling_event(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _experiment_id_from_online_file(path: Path, reactor_numbers: pd.Series) -> pd.Series:
    prefix = "AMBR1" if "run1" in path.stem else "AMBR2"
    return prefix + "_" + reactor_numbers.astype(str)


def _sort_by_experiment_and_time(frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
    sort_keys = pd.DataFrame(
        frame["Experiment_id"].map(_experiment_id_sort_key).tolist(),
        index=frame.index,
        columns=["_ambr_run", "_reactor", "_experiment_id"],
    )
    return (
        frame.join(sort_keys)
        .sort_values(["_ambr_run", "_reactor", time_column, "_experiment_id"])
        .drop(columns=sort_keys.columns)
        .reset_index(drop=True)
    )


def _experiment_id_sort_key(experiment_id: object) -> tuple[int, int, str]:
    text = str(experiment_id)
    match = re.match(r"^AMBR(\d+)_(\d+)$", text)
    if match:
        return int(match.group(1)), int(match.group(2)), text
    return 10**9, 10**9, text


def _experiment_id_from_tunniste(tunniste: object) -> str:
    value = str(tunniste).strip()
    match = re.match(r"^AMBR_(\d+)$", value)
    if match:
        return f"AMBR1_{match.group(1)}"
    match = re.match(r"^AMBR2_(\d+)$", value)
    if match:
        return value
    raise ValueError(f"Unexpected Tunniste value: {value!r}")


def _experiment_id_from_reactor_label(prefix: object, reactor_label: object) -> str:
    match = re.search(r"(\d+)", str(reactor_label))
    return f"{prefix}_{match.group(1)}"


def _parse_condition_label(label: object) -> dict[str, float | None]:
    values: dict[str, float | None] = {"pH": None, "DO": None, "G": None, "AFR": None}
    if pd.isna(label):
        return values

    text = str(label).strip()
    for key in ("pH", "DO", "G", "AFR"):
        match = re.search(rf"{key}0*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
        if match:
            values[key] = float(match.group(1))

    dash_match = re.match(
        r"^(?P<kind>DO|AFR)0*(?P<primary>[0-9]+(?:\.[0-9]+)?)-(?P<ph>[0-9]+(?:\.[0-9]+)?)-G0*(?P<glucose>[0-9]+(?:\.[0-9]+)?)$",
        text,
        flags=re.IGNORECASE,
    )
    if dash_match:
        values[dash_match.group("kind").upper()] = float(dash_match.group("primary"))
        values["pH"] = float(dash_match.group("ph"))
        values["G"] = float(dash_match.group("glucose"))

    return values


def _format_condition_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not pd.notna(numeric):
        return "NA"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _extract_first_number(values: object) -> pd.Series:
    series = pd.Series(values, copy=False).astype(str)
    extracted = series.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess AMBR Pseudomonas data for PINN training.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot", action="store_true", help="Plot the preprocessed real data after preprocessing.")
    parser.add_argument("--plot-csv", type=Path, default=None, help="CSV file to plot. Defaults to the output CSV.")
    parser.add_argument("--max-plots", type=int, default=None, help="Limit the number of experiments to plot.")
    args = parser.parse_args(argv)

    paths = preprocess(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    if args.plot:
        plt = _get_pyplot()
        frame = load_preprocessed_ambr_data(args.plot_csv or paths["final"])
        plot_all_ambr_experiments(frame, max_experiments=args.max_plots)
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
