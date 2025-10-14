import torch
import numpy as np
from copy import deepcopy
from typing import Dict
from rl.BO.helpers.parser import Schema

def _strictly_increasing(arr: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    out = arr.copy()
    for i in range(1, len(out)):
        if out[i] <= out[i-1]:
            out[i] = out[i-1] + eps
    return out

def _normalize_to_sum(pos_arr: np.ndarray, target_sum: float) -> np.ndarray:
    pos_arr = np.maximum(pos_arr, 1e-6)
    return pos_arr * (target_sum/np.sum(pos_arr))

def encode_params(config: Dict, design_rules: Dict, schema: Schema, device: str = "cpu") -> torch.Tensor:
    """
    Convert a rocket config dict back to a flat torch tensor x
    (inverse of decode_params), based on the same schema.
    Returns a 1D torch tensor of dtype=torch.double.
    """
    x_vals = []
    n = design_rules["globals"]["num_segments_per_main"]

    # 1) Nozzle
    x_vals.append(config["nozzle"]["throat"])

    # Helper to extract sequences from a MAIN
    def extract_main(main_key: str):
        grains = config["grains"]
        # Identify which half of the list belongs to this MAIN
        start = 0 if main_key == "MAIN1" else n
        segs = grains[start:start+n]
        core = [g["properties"]["coreDiameter"] for g in segs]
        finw = [g["properties"]["finWidth"] for g in segs]
        finl = [g["properties"]["finLength"] for g in segs]
        core_plus_fin = [c + fl for c, fl in zip(core, finl)]
        lengths = [g["properties"]["length"] for g in segs]
        return core, finw, core_plus_fin, lengths

    # 2) MAIN1 & MAIN2
    for main in ["MAIN1", "MAIN2"]:
        core, finw, cpf, lens = extract_main(main)
        x_vals.extend(core)
        x_vals.extend(finw)
        x_vals.extend(cpf)
        x_vals.extend(lens)

    # Convert to torch tensor
    x_np = np.array(x_vals, dtype=float)
    x_t = torch.tensor(x_np, dtype=torch.double, device=device).unsqueeze(0)

    return x_t

def decode_params(x: np.ndarray, base_config: Dict, design_rules: Dict, schema: Schema) -> Dict:
    """
    Map a flat vector x to a valid rocket config dict.
    Enforces:
      - nozzle bounds
      - strict monotonicity for coreDiameter, corePlusFinLength, finWidth
      - nonnegative finLength = corePlusFin - core
      - segment lengths normalized to each MAIN's length_total
      - fixed outer diameter & numFins
    """
    cfg = deepcopy(base_config)
    n = design_rules["globals"]["num_segments_per_main"]
    outer_d = design_rules["globals"]["outer_diameter"]
    num_fins = design_rules["globals"]["num_fins"]
    exit_ratio = design_rules["nozzle"].get("exit_ratio", 2.857)

    # Unpack x according to schema
    idx = 0
    blocks = {b.name: None for b in schema.blocks}
    for b in schema.blocks:
        vals = x[idx: idx + b.size]
        blocks[b.name] = np.array(vals, dtype=float)
        idx += b.size

    # 1) Nozzle
    throat = float(blocks["nozzle.throat"][0])
    cfg["nozzle"]["throat"] = throat
    cfg["nozzle"]["exit"] = throat * exit_ratio

    # Helper to build a MAIN’s list of 5 Finocyl segments
    def build_main(main_key: str):
        core = _strictly_increasing(blocks[f"{main_key}.coreDiameter"])
        finw = _strictly_increasing(blocks[f"{main_key}.finWidth"])
        core_plus_fin = _strictly_increasing(blocks[f"{main_key}.corePlusFinLength"])
        finl = np.maximum(1e-6, core_plus_fin - core)  # ensures finLength >= 0

        seg_raw = blocks[f"{main_key}.segmentLengthRaw"]
        total = design_rules[main_key]["length_total"]
        seg_len = _normalize_to_sum(seg_raw, total)

        grains = []
        for i in range(n):
            grains.append({
                "type": "Finocyl",
                "properties": {
                    "diameter": outer_d,
                    "length": float(seg_len[i]),
                    "coreDiameter": float(core[i]),
                    "finWidth": float(finw[i]),
                    "finLength": float(finl[i]),
                    "numFins": int(num_fins),
                    "inhibitedEnds": "None"
                }
            })
        return grains

    # 2) MAIN1 & MAIN2
    grains_main1 = build_main("MAIN1")
    grains_main2 = build_main("MAIN2")

    # Write back into your config layout:
    # If you currently store a flat list of grains, replace the 10 entries;
    # or, if your config expects two groups, set them appropriately.
    # Example: recompose a single list in order [MAIN1 segments..., MAIN2 segments...]
    cfg["grains"] = grains_main1 + grains_main2

    return cfg