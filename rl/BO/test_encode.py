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

    save_as.save_motor_as_ric(Motor(initial_config))

    design_schema: parser.Schema = parser.build_schema(design_rules)
    lb, ub = parser.schema_to_bounds(design_schema)

    dtype = torch.double
    lb, ub = lb.to(dtype=dtype, device="cpu"), ub.to(dtype=dtype, device="cpu")
    
    X_list, Y_list = [], []

    x = encode_decode.encode_params(initial_config, design_rules, design_schema)
    simresult = simulation.run_simulation(x, initial_config, design_rules, design_schema)
    y_val = reward.reward_function(simresult)
    print(y_val)