def check_mandrel_posets(grains: list, main_name: str = "") -> bool:
    core = [g["properties"]["coreDiameter"] for g in grains]
    finw = [g["properties"]["finWidth"] for g in grains]
    cpf  = [g["properties"]["coreDiameter"] + g["properties"]["finLength"] for g in grains]

    def strictly_increasing(arr, label):
        for i in range(len(arr) - 1):
            if arr[i] >= arr[i + 1]:
                print(f"❌ {main_name}: {label} not strictly increasing at segment {i} "
                      f"({arr[i]:.6f} ≥ {arr[i+1]:.6f})")
                return False
        return True

    ok = (
        strictly_increasing(core, "coreDiameter")
        and strictly_increasing(finw, "finWidth")
        and strictly_increasing(cpf, "core+finLength")
    )

    if not ok:
        print(f"⚠️  Mandrel poset constraint failed for {main_name}")
    return ok


def check_lengths(grains: list, target_sum: float, main_name: str = "", tol: float = 1e-6) -> bool:
    total = sum(g["properties"]["length"] for g in grains)
    diff = abs(total - target_sum)
    if diff > tol:
        print(f"⚠️  {main_name}: total grain length mismatch "
              f"(sum={total:.6f}, target={target_sum:.6f}, Δ={diff:.2e})")
        return False
    return True


def check_positive_fins(grains: list, main_name: str = "", tol: float = 1e-9) -> bool:
    """Ensure all fin lengths are positive (greater than tol)."""
    for i, g in enumerate(grains):
        fl = g["properties"]["finLength"]
        if fl <= tol:
            print(f"❌ {main_name}: finLength ≤ {tol:g} at segment {i} (finLength={fl:.8f})")
            return False
    return True


def is_feasible(decoded_cfg, design_rules) -> bool:
    n = design_rules["globals"]["num_segments_per_main"]
    gm1 = decoded_cfg["grains"][:n]
    gm2 = decoded_cfg["grains"][n:2 * n]

    ok1 = (
        check_mandrel_posets(gm1, "MAIN1")
        and check_lengths(gm1, design_rules["MAIN1"]["length_total"], "MAIN1")
        and check_positive_fins(gm1, "MAIN1")
    )

    ok2 = (
        check_mandrel_posets(gm2, "MAIN2")
        and check_lengths(gm2, design_rules["MAIN2"]["length_total"], "MAIN2")
        and check_positive_fins(gm2, "MAIN2")
    )

    ok = ok1 and ok2

    if not ok:
        print("🚫 Configuration deemed infeasible.")
    else:
        print("✅ Configuration passed all feasibility checks.")

    return bool(ok)
