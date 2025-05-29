import torch


def masked_mse_loss(pred, target, mask):
    # Only compute loss where mask == 1
    mask = (mask == 1) * (target != 0)
    tmp = torch.sqrt(mask.sum().clamp(min=1))
    diff = (pred / tmp - target / tmp) * mask
    mse = (diff ** 2).sum()

    #original
    # diff = (pred - target) * mask
    # mse = (diff ** 2).sum() / mask.sum().clamp(min=1)
    return mse


def masked_rmse_loss(pred, target, mask):
    # Only compute loss where mask == 1
    rmse = torch.sqrt(masked_mse_loss(pred, target, mask))
    return rmse


def compute_resistor_area(R_value_ohm):
    """
    Compute layout area for a resistor with a given resistance in ohms.
    Uses layout formulas and extrapolation if value is out of min/max range.
    """

    # Corresponding lengths
    L_min = 0.4
    L_max = 5.0

    # Constants
    R_min = 3.5007 * (L_min + 0.3) + 1.867   # ≈ 4.4 Ω
    R_max = 3.5007 * (L_max + 0.3) + 1.867   # ≈ 20.7 Ω

    # Area formula components
    M1L = 0.462     # μm
    WBHNH = 5.2     # μm

    # Length at R_max
    LRX = L_max + 0.684
    area_nominal = (LRX + 2 * M1L) * WBHNH  # Area at R_max

    # Scale area proportionally if out of bounds
    if R_value_ohm < R_min:
        # scale = R_value_ohm / R_min
        scale = 0
    elif R_value_ohm > R_max:
        scale = R_value_ohm / R_max
    else:
        # Compute L from inverse formula: L = (R - 1.867) / 3.5007 - 0.3
        L = (R_value_ohm - 1.867) / 3.5007 - 0.3
        LRX = L + 0.684
        area = (LRX + 2 * M1L) * WBHNH
        return area

    return area_nominal * scale


def compute_capacitor_area(C_value_fF):
    """
    Compute layout area for a capacitor with a given capacitance in fF.
    Uses layout formulas and extrapolation if value is out of min/max range.
    """

    # Corresponding dimensions
    W_min = 6.0    # μm
    W_max = 150.0  # μm
    L_fixed = 20.0 # μm

    # Constants from the formula
    C_min = 6.92 * W_min + 4.4       # ≈ 46.32 fF
    C_max = 6.92 * W_max + 4.4       # ≈ 1042.4 fF

    # Nominal area formula (22W + 44)
    area_nominal = 22 * W_max + 44   # μm²

    # Scale if capacitance is out of bounds
    if C_value_fF < C_min:
        # scale = C_value_fF / C_min
        scale = 0
    elif C_value_fF > C_max:
        scale = C_value_fF / C_max
    else:
        # Compute W from inverse formula: W = (C - 4.4) / 6.92
        W = (C_value_fF - 4.4) / 6.92
        area = W * L_fixed + 2 * W + 2 * L_fixed + 4  # WL + 2W + 2L + 4
        return area

    return area_nominal * scale


def compute_inductor_area(L_value_nH):
    """
    Compute layout area for an inductor with given inductance in nH.
    Area formula: A = 4 * R^2 + 108 * R + 440
    where R = 7.25e-5 * L^1.53

    If L < 0.08 nH → area = 0.
    """

    if L_value_nH < 0.08:
        return 0.0

    # Step 1: Estimate radius R
    R = 7.25e-5 * (L_value_nH ** 1.53)

    # Step 2: Compute area
    area = 4 * (R ** 2) + 108 * R + 440  # in μm²

    return area


def compute_total_layout_area(flat_edge_attrs, x_params, param_names):
    def get_value(v):
        if isinstance(v, str):
            if v in param_names:
                return x_params[param_names.index(v)].item()
            try:
                val = eval(v, {"x_params": x_params, "torch": torch, "sqrt": torch.sqrt})
                return val.item() if torch.is_tensor(val) else float(val)
            except:
                return 0.0
        return float(v)

    total_area = 0.0

    for attr in flat_edge_attrs:
        base_type = attr["component"].split("_")[0]
        all_attrs = {**attr.get("numeric_attrs", {}),
                     **attr.get("parametric_attrs", {}),
                     **attr.get("computing_attrs", {})}

        if base_type == "resistor" and "r" in all_attrs:
            r_val = get_value(all_attrs["r"])
            total_area += compute_resistor_area(r_val)

        elif base_type == "capacitor" and "c" in all_attrs:
            c_val = get_value(all_attrs["c"]) * 1e15 # F → fF
            total_area += compute_capacitor_area(c_val) 

        elif base_type == "inductor" and "l" in all_attrs:
            l_val = get_value(all_attrs["l"]) * 1e9  # H → nH
            total_area += compute_inductor_area(l_val)

    return total_area / 1e6


def compute_aggregated_loss(performance_loss, area_mm2):
    threshold = 0.05                        # max tolerable performance loss
    sharpness = 50                          # how sharp the gate is
    lambda_area = 0.02                       # area loss weight
    # Gate softly activates when performance_loss < threshold
    gate = 1.0 - torch.sigmoid((performance_loss - threshold) * sharpness)
    loss = performance_loss + gate * lambda_area * area_mm2

    return loss