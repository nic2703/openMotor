import os, sys
print(">>> CWD:", os.getcwd())
print(">>> First path:", sys.path[0])
print(">>> RL visible?", os.path.exists(os.path.join(os.getcwd(), "rl")))

import sys, os
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from rl.BO.helpers import config_loader, parser, encode_decode, save_as
from rl.BO.BO_torch import simulation, reward
from typing import Dict

from motorlib.motor import Motor

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll_torch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm

def bo_optimize(initial_config: Dict, design_rules: Dict, design_schema: Dict, lb: torch.Tensor, ub: torch.Tensor, n_init=5, n_iter=15, device="cpu"):
    dtype = torch.double
    lb, ub = lb.to(dtype=dtype, device=device), ub.to(dtype=dtype, device=device)
    
    X_list, Y_list = [], []
    print(f"Initializing with {n_init} random samples...")
    for _ in tqdm(range(n_init)):
        x = encode_decode.encode_params(initial_config, design_rules, design_schema)
        # x = torch.rand_like(lb) * (ub - lb) + lb
        x_np = x.numpy().ravel()
        simresult = simulation.run_simulation(x_np, initial_config, design_rules, design_schema)
        y_val = reward.reward_function(simresult)
        X_list.append(x)
        Y_list.append(torch.tensor([[y_val]]))

    X = torch.cat(X_list).to(device)
    Y = torch.cat(Y_list).to(device)

    max_isp = 0
    max_reward = float('-inf')

    for iteration in range(n_iter):
        print(f"\n=== Iteration {iteration + 1} / {n_iter} ===")

        model = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll_torch(mll)

        EI = LogExpectedImprovement(model=model, best_f=Y.max())

        x_next, _ = optimize_acqf(acq_function=EI, bounds=torch.stack([lb.squeeze(0), ub.squeeze(0)]), q=1, num_restarts=5, raw_samples=100)
        x_next_np = x_next.detach().cpu().numpy().ravel()

        simresult = simulation.run_simulation(x_next_np, initial_config, design_rules, design_schema)
        y_next = reward.reward_function(simresult)

        X = torch.cat([X, x_next.to(dtype=dtype, device=device)])
        Y = torch.cat([Y, torch.tensor([[y_next]], dtype=dtype, device=device)])

        isp = simresult["ISP"]
        if y_next >= max_reward: 
            max_reward = y_next
            max_isp = isp
        print(f"Reward: {y_next:.3f} | Best so far: {Y.max().item():.3f} | ISP: {isp} | Best ISP so far: {max_isp}")

    best_idx = torch.argmax(Y)
    best_x = X[best_idx].detach().cpu().numpy().ravel()
    print("\n✅ Optimization completed!")
    print(f'Best reward: {Y.max().item():.3f} | Best ISP: {max_isp}')

    return best_x, Y.max().item()


if __name__ == "__main__":
    initial_config = config_loader.load_base_config()
    design_rules = config_loader.load_design_config()

    design_schema: parser.Schema = parser.build_schema(design_rules)
    lb, ub = parser.schema_to_bounds(design_schema)

    best_x, best_score = bo_optimize(initial_config, design_rules, design_schema, lb, ub, n_iter=25)
    best_config = encode_decode.decode_params(best_x, initial_config, design_rules, design_schema)
    save_as.save_motor_locally(Motor(best_config))

