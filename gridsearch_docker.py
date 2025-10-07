import itertools
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from motorlib.motor import Motor
from motorlib.simResult import SimulationResult


def create_df(simulationresult: SimulationResult):
    data = {}

    for name, channel in simulationresult.channels.items():
        values = channel.getData()  # already a list
        # If values are tuples/lists, expand them into multiple columns
        if values and isinstance(values[0], (list, tuple)):
            for i in range(len(values[0])):
                data[f"{name}_{i}"] = [v[i] for v in values]
        else:
            data[name] = values

    # Make a DataFrame
    df = pd.DataFrame(data)

    # Use 'time' as index if it exists
    if "time" in df.columns:
        df = df.set_index("time")

    # Add attribute
    df.attrs["BurnTime"] = simulationresult.getBurnTime()
    df.attrs["InitialKN"] = simulationresult.getInitialKN()
    df.attrs["PeakKN"] = simulationresult.getPeakKN()
    df.attrs["AveragePressure"] = simulationresult.getAveragePressure()

    df.attrs["MaxPressure"] = simulationresult.getMaxPressure()
    df.attrs["MinExitPressure"] = simulationresult.getMinExitPressure()

    # Impulse and force-related
    df.attrs["Impulse"] = simulationresult.getImpulse()
    df.attrs["AverageForce"] = simulationresult.getAverageForce()

    # Designation
    df.attrs["Designation"] = simulationresult.getDesignation()
    df.attrs["FullDesignation"] = simulationresult.getFullDesignation()
    df.attrs["ImpulseClassPercentage"] = simulationresult.getImpulseClassPercentage()

    # Mass flux / Mach
    df.attrs["PeakMassFlux"] = simulationresult.getPeakMassFlux()
    df.attrs["PeakMassFluxLocation"] = simulationresult.getPeakMassFluxLocation()
    df.attrs["PeakMachNumber"] = simulationresult.getPeakMachNumber()
    df.attrs["PeakMachNumberLocation"] = simulationresult.getPeakMachNumberLocation()

    # ISP
    df.attrs["ISP"] = simulationresult.getISP()

    # Propellant and geometry
    df.attrs["PortRatio"] = simulationresult.getPortRatio()
    df.attrs["PropellantLength"] = simulationresult.getPropellantLength()
    df.attrs["PropellantMass"] = simulationresult.getPropellantMass()
    df.attrs["VolumeLoading"] = simulationresult.getVolumeLoading()

    # Thrust coefficients
    df.attrs["IdealThrustCoefficient"] = simulationresult.getIdealThrustCoefficient()
    df.attrs["AdjustedThrustCoefficient"] = simulationresult.getAdjustedThrustCoefficient()

    # Meta info
    df.attrs["Motor"] = simulationresult.motor
    df.attrs["Success"] = simulationresult.success
    df.attrs["Alerts"] = simulationresult.alerts

    return df


def expand_param_space(varspec: dict) -> list[dict]:
    """
    Given a dictionary like:
        {"grains.0.properties.length": {"min": 0.15, "max": 0.25, "step": 0.05},
         "grains.1.properties.coreDiameter": {"min": 0.05, "max": 0.07, "step": 0.01}}
    return list of parameter dictionaries for grid search.
    """
    param_ranges = {}
    for key, spec in varspec.items():
        param_ranges[key] = np.arange(spec["min"], spec["max"] + spec["step"], spec["step"])

    # Cartesian product
    keys, ranges = zip(*param_ranges.items())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*ranges)]
    return combos


def update_nested_dict(d, keypath, value):
    """Update nested dict entry given 'a.b.c' keypath."""
    keys = keypath.split(".")
    current = d
    for k in keys[:-1]:
        if k.isdigit():
            k = int(k)
        current = current[k]
    last = keys[-1]
    if last.isdigit():
        last = int(last)
    current[last] = value


def run_gridsearch(base_config: dict, varspec: dict) -> list[dict]:
    """
    Run grid search simulations.
    base_config: the main maindict you already use
    varspec: dict of parameter ranges
    """
    results = []
    combos = expand_param_space(varspec)

    for combo in tqdm(combos, desc="Running simulations"):
        config = base_config.copy()  # shallow copy
        # apply parameter overrides
        for keypath, value in combo.items():
            update_nested_dict(config, keypath, value)

        motor = Motor(config)
        simresult = motor.runSimulation()
        df = create_df(simresult)

        results.append({
            "params": combo,
            "ISP": df.attrs["ISP"],
            "Impulse": df.attrs["Impulse"],
            "BurnTime": df.attrs["BurnTime"],
            "MaxPressure": df.attrs["MaxPressure"],
            "Success": df.attrs["Success"],
            "Alerts": len(df.attrs["Alerts"]),
            "df": df  # optional: keep full DataFrame
        })

    return results


def filter_results(results, constraints: list[str]) -> pd.DataFrame:
    """
    Filter results based on constraints like ["ISP > 200", "MaxPressure < 1e7"].
    Returns a pandas DataFrame of the surviving runs.
    """
    records = []
    for r in results:
        row = {"params": r["params"]}
        row.update({k: v for k, v in r.items() if k != "df" and k != "params"})
        records.append(row)

    df = pd.DataFrame(records)

    mask = pd.Series(True, index=df.index)
    for expr in constraints:
        mask &= df.eval(expr)

    return df[mask]


if __name__ == "__main__":
    # Expect file paths from command line
    config_path = Path(sys.argv[1])  # e.g., inputs/config.json
    varspec_path = Path(sys.argv[2])  # e.g., inputs/varspec.json
    constraints_path = Path(sys.argv[3])  # e.g., inputs/constraints.txt
    output_dir = Path(sys.argv[4])  # e.g., outputs/

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    with open(config_path, "r") as f:
        maindict = json.load(f)
    with open(varspec_path, "r") as f:
        varspec = json.load(f)
    with open(constraints_path, "r") as f:
        constraints = [line.strip() for line in f if line.strip()]

    # Run simulations
    results = run_gridsearch(maindict, varspec)

    # Save summary (excluding heavy dfs)
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "df"} for r in results
    ])
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Save individual dataframes
    for i, r in enumerate(results):
        r["df"].to_csv(output_dir / f"run_{i}.csv")

    # Also save filtered results
    filtered = filter_results(results, constraints)
    filtered.to_csv(output_dir / "filtered.csv", index=False)