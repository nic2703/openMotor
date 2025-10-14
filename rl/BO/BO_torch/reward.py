def reward_function(metrics):
    """
    Compute reward for rocket nozzle optimization.

    metrics: dict
        {
            "Success": bool,
            "ISP": float,              # seconds
            "MaxPressure": float,      # Pa
            "MaxMassFlux": float,      # kg/m²/s
        }
    """
    # Hard fail if simulation didn't converge or failed physics
    if not metrics.get("Success", True):
        return -1e6

    # Constraint limits
    MAX_PRESSURE = 7.122e6       # Pa (≈ 1033 psi)
    MAX_MASSFLUX = 1.05e3        # kg/m²/s (≈ 1.5 lb/in²·s)

    penalty = 0.0

    # --- Pressure penalty (quadratic beyond limit)
    if metrics["MaxPressure"] > MAX_PRESSURE:
        delta_p = metrics["MaxPressure"] - MAX_PRESSURE
        penalty += (delta_p / 1e5) ** 2   # quadratic scaling

    # --- Mass flux penalty (quadratic beyond limit)
    if metrics["MaxMassFlux"] > MAX_MASSFLUX:
        delta_flux = metrics["MaxMassFlux"] - MAX_MASSFLUX
        penalty += (delta_flux / 50.0) ** 2  # tunable scale

    # --- Base reward (maximize ISP, penalize violations)
    reward = metrics["ISP"] - penalty

    return reward
