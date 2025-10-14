import sys, os
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from rl.BO.helpers import config_loader, parser, save_as, encode_decode
from rl.BO.BO_torch import simulation, reward
from typing import Dict

from motorlib.motor import Motor

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

if __name__ == "__main__":
    initial_config = config_loader.load_base_config()
    design_rules = config_loader.load_design_config()
    design_schema: parser.Schema = parser.build_schema(design_rules)

    x = encode_decode.encode_params(initial_config, design_rules, design_schema)
    print(x)
    x_np = x.numpy().ravel()
    decoded_config = encode_decode.decode_params(x_np, initial_config, design_rules, design_schema)
    print(decoded_config)