# Payload specific settings

# Default Container WATER
two_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.0875,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

four_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.0538,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

six_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.001,
    "mu":0.001,
    "offset":0.03,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

eight_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.01,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

ten_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.015,
    "scale_x":1.6,
    "scale_y":1.6,
    "scale_z":1.6
}

ten_liters_water_tall = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.04,
    "scale_x":1.6,
    "scale_y":1.6,
    "scale_z":2.2
}

ten_liters_water_wide = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.025,
    "scale_x":1.4,
    "scale_y":2.6,
    "scale_z":1.2
}

ten_liters_water_offset = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.015,
    "mount_offset":[0.05,0.05],
    "scale_x":1.6,
    "scale_y":1.6,
    "scale_z":1.6
}

twelve_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.0125,
    "scale_x":1.8,
    "scale_y":1.6,
    "scale_z":1.6
}

# Default Container OIL
two_liters_oil_default = {
    "rho": 850.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.0875,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

four_liters_oil_default = {
    "rho": 850.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.0538,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

six_liters_oil_default = {
    "rho": 850.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.03,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

eight_liters_oil_default = {
    "rho": 850.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.01,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

ten_liters_oil_default = {
    "rho": 850.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.015,
    "scale_x":1.6,
    "scale_y":1.6,
    "scale_z":1.6
}

twelve_liters_oil_default = {
    "rho": 850.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.0125,
    "scale_x":1.8,
    "scale_y":1.6,
    "scale_z":1.6
}

# Default Container GAS
two_liters_gas_default = {
    "rho": 730.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.0875,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

four_liters_gas_default = {
    "rho": 730.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.0538,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

six_liters_gas_default = {
    "rho": 730.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.03,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

eight_liters_gas_default = {
    "rho": 730.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.01,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

ten_liters_gas_default = {
    "rho": 730.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.015,
    "scale_x":1.6,
    "scale_y":1.6,
    "scale_z":1.6
}

twelve_liters_gas_default = {
    "rho": 730.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.0125,
    "scale_x":1.8,
    "scale_y":1.6,
    "scale_z":1.6
}

def get_payload_config(payload_type: str, volume: int, container_shape: str = "default"):
    """
    Returns payload configuration dict.

    Args:
        payload_type: {"water", "oil", "gas"}
        volume: payload volume in liters
        container_shape: container geometry identifier (currently unused, stubbed)

    Notes:
        container_shape is reserved for future container-specific configurations.
        Currently, only the "default" shape is supported.
    """

    if payload_type == "water":
        if volume == 2:
            return two_liters_water_default
        elif volume == 4:
            return four_liters_water_default
        elif volume == 6:
            return six_liters_water_default
        elif volume == 8:
            return eight_liters_water_default
        elif volume == 10:
            if container_shape == "default":
                return ten_liters_water_default
            if container_shape == "tall":
                return ten_liters_water_tall
            if container_shape == "wide":
                return ten_liters_water_wide
            if container_shape == "offset":
                return ten_liters_water_offset
        elif volume == 12:
            return twelve_liters_water_default

    elif payload_type == "oil":
        if volume == 2:
            return two_liters_oil_default
        elif volume == 4:
            return four_liters_oil_default
        elif volume == 6:
            return six_liters_oil_default
        elif volume == 8:
            return eight_liters_oil_default
        elif volume == 10:
            return ten_liters_oil_default
        elif volume == 12:
            return twelve_liters_oil_default

    elif payload_type == "gas":
        if volume == 2:
            return two_liters_gas_default
        elif volume == 4:
            return four_liters_gas_default
        elif volume == 6:
            return six_liters_gas_default
        elif volume == 8:
            return eight_liters_gas_default
        elif volume == 10:
            return ten_liters_gas_default
        elif volume == 12:
            return twelve_liters_gas_default

    

    else:
        raise ValueError(f"Unsupported payload type: {payload_type}")


# Registry of all available (type, volume, tank) combinations
ALL_LIQUID_CONFIGS = [
    ("water", 2,  "default"),
    ("water", 4,  "default"),
    ("water", 6,  "default"),
    ("water", 8,  "default"),
    ("water", 10, "default"),
    ("water", 10, "tall"),
    ("water", 10, "wide"),
    ("water", 10, "offset"),
    ("water", 12, "default"),
    ("oil",   2,  "default"),
    ("oil",   4,  "default"),
    ("oil",   6,  "default"),
    ("oil",   8,  "default"),
    ("oil",   10, "default"),
    ("oil",   12, "default"),
    ("gas",   2,  "default"),
    ("gas",   4,  "default"),
    ("gas",   6,  "default"),
    ("gas",   8,  "default"),
    ("gas",   10, "default"),
    ("gas",   12, "default"),
]


def sample_liquid_configs(n=20, seed=None):
    """Randomly sample n liquid configs from all available ones.

    Args:
        n: number of configs to sample (clamped to total available)
        seed: optional random seed for reproducibility

    Returns:
        list of (liquid_type, volume, tank) tuples
    """
    import random as _rng
    if seed is not None:
        _rng.seed(seed)
    n = min(n, len(ALL_LIQUID_CONFIGS))
    return _rng.sample(ALL_LIQUID_CONFIGS, n)