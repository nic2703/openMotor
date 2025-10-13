import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from motorlib.motor import Motor
import motorlib
from motorlib.simResult import SimulationResult
from pathlib import Path
from datetime import datetime


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

    # Impulse and force-related
    df.attrs["Impulse"] = simulationresult.getImpulse()
    df.attrs["AverageForce"] = simulationresult.getAverageForce()

    # Add attribute
    df.attrs["BurnTime"] = simulationresult.getBurnTime()
    df.attrs["InitialKN"] = simulationresult.getInitialKN()
    df.attrs["PeakKN"] = simulationresult.getPeakKN()
    df.attrs["AveragePressure"] = simulationresult.getAveragePressure()

    df.attrs["MaxPressure"] = simulationresult.getMaxPressure()
    df.attrs["MinExitPressure"] = simulationresult.getMinExitPressure()

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
    param_ranges = {}
    for key, spec in varspec.items():
        keys = key.split(",")  # allow multiple keys per variable
        param_ranges[tuple(keys)] = np.arange(spec["min"], spec["max"] + spec["step"], spec["step"])

    combos = []
    for vals in itertools.product(*param_ranges.values()):
        combo = {}
        for key_group, val in zip(param_ranges.keys(), vals):
            for k in key_group:
                combo[k] = val
        combos.append(combo)
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


def run_gridsearch(base_config: dict, varspec: dict | None = None) -> list[dict]:
    """
    Run grid search simulations.
    base_config: the main maindict you already use
    varspec: dict of parameter ranges
    """
    results = []

    if not varspec:
        combos = [{}]
    else:
        combos = expand_param_space(varspec)

    for combo in tqdm(combos, desc="Running simulations"):
        config = base_config.copy()  # shallow copy
        # apply parameter overrides
        for keypath, value in combo.items():
            update_nested_dict(config, keypath, value)

        throat = config["nozzle"]["throat"]
        config["nozzle"]["exit"] = throat * 2.857

        try:
            motor = Motor(config)
            simresult = motor.runSimulation()
            df = create_df(simresult)

            results.append({
                "params": combo,
                "ISP": df.attrs["ISP"],
                "Impulse": df.attrs["Impulse"],
                "PeakMassFlux": df.attrs["PeakMassFlux"],
                "MaxPressure": df.attrs["MaxPressure"],
                "Success": df.attrs["Success"],
                "Alerts": len(df.attrs["Alerts"]),
                "df": df  # optional: keep full DataFrame
            })

        except Exception as e:
            # Catch any simulation or runtime errors
            results.append({
                "params": combo,
                "ISP": None,
                "Impulse": None,
                "PeakMassFlux": None,
                "MaxPressure": None,
                "Success": False,
                "Alerts": None,
                "Error": str(e),
                "df": None
            })
            print(f"\n Simulation failed for parameters {combo}: {e}")

    return results


def filter_results(results, constraints: list[str]) -> pd.DataFrame:
    """
    Filter results based on constraints like ["ISP > 200", "MaxPressure < 1e7"].
    Returns a pandas DataFrame of the surviving runs.
    """
    records = []
    for r in results:
        attrs = {}
        try:
            if r.get("df") is not None:
                attrs = r["df"].attrs.copy()
        except Exception as e:
            print(f"⚠️ Failed to extract attrs for {r.get('params', {})}: {e}")
        
        row = {}
        row.update(attrs)
        row.update({k: v for k, v in r.items() if k not in ("df","motor",)})
        records.append(row)

    df = pd.DataFrame(records)

    # Apply filters safely
    if not constraints:
        print("No constraints provided — returning all results.")
        return df

    mask = pd.Series(True, index=df.index)
    for expr in constraints:
        try:
            mask &= df.eval(expr)
        except Exception as e:
            print(f"⚠️ Failed to evaluate constraint '{expr}': {e}")

    filtered = df[mask]

    print(f"✅ Filtered {len(df)} → {len(filtered)} runs based on constraints.")
    return filtered


# Example usage
if __name__ == "__main__":
    nozzledict = {
        "throat": 0.03556007,       
        "exit": 0.10160020,         
        "efficiency": 0.96000000,
        "divAngle": 15.00000000,
        "convAngle": 60.00000000,
        "throatLength": 0.00508001,
        "slagCoeff": 0.0,
        "erosionCoeff": 0.0
    }

    propdict = {
        "name": "MIT - Demios",
        "density": 1653.59714122,
        "tabs": [
            {
                "minPressure": 0,  # 0.5 MPa
                "maxPressure": 13790000.00000000,  # 20 MPa
                "a": 0.02215029*0.001,
                "n": 0.36240000,
                "k": 1.21620000,
                "t": 2504.06000000,
                "m": 23.11000000
            }
        ]
    }

    bates_entry_1 = {
        "type": "BATES",
        "properties": {
            "diameter": 0.13335027,
            "length": 0.55880112,
            "coreDiameter": 0.02540005,
            "inhibitedEnds": "Bottom"
        }
    }

    bates_entry_2 = {
        "type": "BATES",
        "properties": {
            "diameter": 0.13335027,
            "length": 0.35560071,
            "coreDiameter": 0.03175006,
            "inhibitedEnds": "Top"
        }
    }

    finocyl_entry_1 = {
        "type": "Finocyl",
        "properties": {
            "diameter": 0.13335027,
            "length": 0.20320041,
            "inhibitedEnds": "Bottom",
            "coreDiameter": 0.03175006,
            "numFins": 3,
            "finWidth": 0.01270003,
            "finLength": 0.01270003,
            "invertedFins": False
        }
    }

    finocyl_entry_2 = {
        "type": "Finocyl",
        "properties": {
            "diameter": 0.13335027,
            "length": 0.25400051,
            "inhibitedEnds": "Both",
            "coreDiameter": 0.03175006,
            "numFins": 3,
            "finWidth": 0.01270003,
            "finLength": 0.03175006,
            "invertedFins": False
        }
    }

    finocyl_entry_3 = {
        "type": "Finocyl",
        "properties": {
            "diameter": 0.13335027,
            "length": 0.25400051,
            "inhibitedEnds": "Top",
            "coreDiameter": 0.04318009,
            "numFins": 3,
            "finWidth": 0.01524003,
            "finLength": 0.02921006,
            "invertedFins": False
        }
    }


    graindicts = [bates_entry_1, bates_entry_2, finocyl_entry_1, finocyl_entry_2, finocyl_entry_3]

    configdict: dict[str, float | int] = {
        # Limits
        "maxPressure": 7239750.00000000,           # Pa, 5 MPa max chamber pressure
        "maxMassFlux": 1054.85232068,          # kg/(m^2*s), reasonable safe value
        "maxMachNumber": 0.7,           # below sonic choking
        "minPortThroat": 2.0,           # ensures port/throat ratio >= 1.2
        "flowSeparationWarnPercent": 0.05,  # 10% tolerance before warning

        # Simulation
        "burnoutWebThres": 0.00002540,      # m, regression depth threshold
        "burnoutThrustThres": 0.1,     # %, thrust decay threshold (5%)
        "timestep": 0.03,              # s, simulation step
        "ambPressure": 101324.99674500,        # Pa, 1 atm ambient
        "mapDim": 750,                  # grid size (resolution for FMM)
        "sepPressureRatio": 0.4         # separation onset ratio
    }


    maindict: dict[str, dict] = {"nozzle": nozzledict, "propellant": propdict, "grains": graindicts, "config": configdict}

    varspec = {
        "nozzle.throat": {"min": 0.01905, "max": 0.0508, "step": 0.00254},
        "grains.0.properties.coreDiameter,grains.1.properties.coreDiameter,grains.2.properties.coreDiameter,grains.3.properties.coreDiameter,grains.4.properties.coreDiameter":
            {"min": 0.0254, "max": 0.13335, "step": 0.00254},
    #     "grains.2.properties.finLength,grains.3.properties.finLength,grains.4.properties.finLength": {"min": 0.0, "max": 0.13335, "step": 0.00254},
    #     "grains.2.properties.finWidth,grains.3.properties.finWidth,grains.4.properties.finWidth": {"min": 0.00635, "max": 0.0254, "step": 0.00254}
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_gridsearch(maindict, varspec)
    for i, r in enumerate(results):
        try:
            df = r.get("df")
            if df is not None:
                df.to_csv(output_dir / f"run_{i}.csv")
            else:
                print(f"⚠️ Skipping run {i}: df is None")
        except Exception as e:
            print(f"❌ Error saving run {i}: {e}")

    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["df"]} for r in results
    ])
    print(summary)
    summary.to_csv(output_dir / "summary.csv", index=False)

    try:
        filtered = filter_results(results, [
            "PeakMassFlux < 1054.62", #kg/m^2s
            "MaxPressure < 7122284" #Pa
        ])
        print(filtered)
        filtered.to_csv(output_dir / "filtered.csv", index=False)
    except Exception as e:
        print(f"❌ Error filtering results: {e}")
