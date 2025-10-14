def check_mandrel_posets(grains: list) -> bool:
    core = [g["properties"]["coreDiameter"] for g in grains]
    finw = [g["properties"]["finWidth"] for g in grains]
    cpf  = [g["properties"]["coreDiameter"] + g["properties"]["finLength"] for g in grains]
    def strictly_increasing(arr): return all(arr[i] < arr[i+1] for i in range(len(arr)-1))
    return strictly_increasing(core) and strictly_increasing(finw) and strictly_increasing(cpf)

def check_lengths(grains: list, target_sum: float, tol: float = 1e-6) -> bool:
    return abs(sum(g["properties"]["length"] for g in grains) - target_sum) <= tol

def is_feasible(decoded_cfg, design_rules) -> bool:
    n = design_rules["globals"]["num_segments_per_main"]
    gm1 = decoded_cfg["grains"][:n]
    gm2 = decoded_cfg["grains"][n:2*n]
    ok = check_mandrel_posets(gm1) and check_mandrel_posets(gm2)
    ok &= check_lengths(gm1, design_rules["MAIN1"]["length_total"])
    ok &= check_lengths(gm2, design_rules["MAIN2"]["length_total"])
    return bool(ok)