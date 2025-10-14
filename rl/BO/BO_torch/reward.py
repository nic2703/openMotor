def reward_function(metrics):
    if not metrics["Success"]:
        return -1e6
    penalty = 0
    if metrics["MaxPressure"] > 7122284:
        penalty += (metrics["MaxPressure"] - 7122284) / 1e5
    return metrics["ISP"] - penalty
