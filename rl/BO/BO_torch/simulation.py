import torch
from typing import Dict
from motorlib.motor import Motor
from rl.BO.helpers.encode_decode import decode_params
from rl.BO.helpers.feasible import is_feasible

def run_simulation(x: torch.Tensor, initial_config: Dict, design_rules: Dict, design_schema: Dict) -> Dict:
    candidate_cfg = decode_params(x, initial_config, design_rules, design_schema)
    if not is_feasible(candidate_cfg, design_rules):
        return {"Success": False, "ISP": 0.0, "Error": "Geometry infeasible"}
    
    try:
        motor = Motor(candidate_cfg)
        simresult = motor.runSimulation()

        if simresult.alerts:
            for i, alert in enumerate(simresult.alerts, start=1):
                level = getattr(alert, "level", "N/A")
                atype = getattr(alert, "type", "N/A")
                location = getattr(alert, "location", None)
                desc = getattr(alert, "description", "")

                header = f"Alert {i}: {level} - {atype} | {desc}"
                if location is not None:
                    header += f" @ {location}"
                print(header)



        return {
            "Success": simresult.success,
            "ISP": simresult.getISP(),
            "Impulse": simresult.getImpulse(),
            "MaxMassFlux": simresult.getPeakMassFlux(),
            "MaxPressure": simresult.getMaxPressure(),
            "BurnTime": simresult.getBurnTime(),
            "Alerts": simresult.alerts
        }

    except Exception as e:
        print(f"run_simulation() failed: {e}")
        return {"Success": False, "ISP": 0.0, "Error": str(e)}