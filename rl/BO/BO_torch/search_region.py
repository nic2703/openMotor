import torch

def trust_region(bounds, center, frac=0.2):
    lb, ub = bounds
    width = (ub - lb) * frac
    tr_lb = torch.clamp(center - width / 2, lb, ub)
    tr_ub = torch.clamp(center + width / 2, lb, ub)
    return tr_lb, tr_ub