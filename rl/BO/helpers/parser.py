import torch

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

@dataclass
class ParamBlock:
    name: str            # e.g., "MAIN1.cores", "MAIN1.finWidth", etc.
    size: int            # e.g., 5 for 5 segments
    bounds: List[Tuple[float, float]]

@dataclass
class Schema:
    blocks: List[ParamBlock]
    dim: int             # total vector length

def build_schema(design_rules: Dict) -> Schema:
    n = design_rules["globals"]["num_segments_per_main"]
    blocks = []

    # 1) nozzle.throat (size = 1)
    tmin = design_rules["nozzle"]["throat"]["min"]
    tmax = design_rules["nozzle"]["throat"]["max"]
    blocks.append(ParamBlock("nozzle.throat", 1, [(tmin, tmax)]))

    # 2) For each MAIN: coreDiameter (n), finWidth (n), core+finLength (n), lengths (n)
    # NOTE: we parameterize "coreDiameter_plus_finLength" directly to satisfy the mandrel poset cleanly.
    for main in ["MAIN1", "MAIN2"]:
        cd = design_rules[main]["core_diameter"]
        fw = design_rules[main]["finWidth"]
        fl = design_rules[main]["finLength"]

        blocks.append(ParamBlock(f"{main}.coreDiameter", n, [(cd["min"], cd["max"])]*n))
        blocks.append(ParamBlock(f"{main}.finWidth", n, [(fw["min"], fw["max"])]*n))
        # Sum bounds for (core + finLength): [cd_min + fl_min, cd_max + fl_max]
        blocks.append(ParamBlock(
            f"{main}.corePlusFinLength", n,
            [ (cd["min"] + fl["min"], cd["max"] + fl["max"]) ]*n
        ))
        # Segment lengths (positive; will be normalized to length_total)
        total = design_rules[main]["length_total"]
        # A wide, positive box; normalization will enforce the sum
        blocks.append(ParamBlock(f"{main}.segmentLengthRaw", n, [(1e-4, total)]*n))

    dim = sum(b.size for b in blocks)
    return Schema(blocks=blocks, dim=dim)


def schema_to_bounds(schema: Schema) -> Union[torch.Tensor, torch.Tensor]:
    lows, highs = [], []
    for block in schema.blocks:
        for lo, hi in block.bounds:
            lows.append(lo)
            highs.append(hi)
    return torch.tensor(lows).unsqueeze(0), torch.tensor(highs).unsqueeze(0)
