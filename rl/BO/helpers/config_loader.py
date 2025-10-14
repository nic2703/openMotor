import json
from pathlib import Path
from typing import Dict, Any

def load_base_config() -> Dict[str, Dict[str, Any]]:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir/"../base_config.json"

    with config_path.open("r") as f:
        config = json.load(f)

    print("Loaded config from: ", config_path)
    return config

def load_design_config() -> Dict[str, Dict[str, Any]]:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir/"../design_config.json"

    with config_path.open("r") as f:
        design_space = json.load(f)

    print("Loaded config from: ", config_path)
    return design_space