# Payload-specific settings shared by the Go1 and Go2 liquid players.
# Volume names are nominal fill presets, not exact particle-mass calibrations.
# The SPH coefficients mu/gamma are passed directly to Genesis (not observation scales).
# Tank scale factors also scale its wall/bottom geometry; offset only changes the fill box.
# The 'gas' presets represent a gasoline-like liquid, not a compressible gas.

# Default Container WATER
two_liters_water_default = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0875,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

four_liters_water_default = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0538,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

six_liters_water_default = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.001,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.001,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.03,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

eight_liters_water_default = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.01,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

ten_liters_water_default = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.015,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.6  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

ten_liters_water_tall = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.04,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":2.2  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

ten_liters_water_wide = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.025,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.4,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":2.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.2  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

ten_liters_water_offset = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.015,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "mount_offset":[0.05,0.05],  # Tank/liquid mounting displacement [x, y] in metres from the torso center.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.6  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

twelve_liters_water_default = {
    "rho": 1000.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.010,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.005,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0125,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.8,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.6  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

# Default Container OIL
two_liters_oil_default = {
    "rho": 850.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.003,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.025,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0875,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

four_liters_oil_default = {
    "rho": 850.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.003,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.025,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0538,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

six_liters_oil_default = {
    "rho": 850.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.003,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.025,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.03,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

eight_liters_oil_default = {
    "rho": 850.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.003,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.025,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.01,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

ten_liters_oil_default = {
    "rho": 850.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.003,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.025,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.015,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.6  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

twelve_liters_oil_default = {
    "rho": 850.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.003,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.025,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0125,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.8,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.6  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

# Default Container GAS
two_liters_gas_default = {
    "rho": 730.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.002,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.002,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0875,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

four_liters_gas_default = {
    "rho": 730.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.002,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.002,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0538,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

six_liters_gas_default = {
    "rho": 730.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.002,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.002,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.03,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

eight_liters_gas_default = {
    "rho": 730.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.002,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.002,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.01,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.4,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.4  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

ten_liters_gas_default = {
    "rho": 730.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.002,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.002,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.015,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.6,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.6  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

twelve_liters_gas_default = {
    "rho": 730.0,  # Rest density [kg/m^3]; controls fluid mass for a given sampled volume.
    "gamma":0.002,  # SPH surface-tension coefficient; larger values increase cohesion.
    "mu":0.002,  # SPH viscosity coefficient; larger values resist relative fluid motion.
    "offset":0.0125,  # Fill inset [m], subtracted from each FULL interior-box side; larger means less liquid.
    "scale_x":1.8,  # Dimensionless tank length (X) multiplier; unscaled outer length is 0.20 m.
    "scale_y":1.6,  # Dimensionless tank width (Y) multiplier; unscaled outer width is 0.15 m.
    "scale_z":1.6  # Dimensionless tank height (Z) multiplier; unscaled outer height is 0.20 m.
}

def get_payload_config(payload_type: str, volume: int, container_shape: str = "default"):
    """
    Returns payload configuration dict.

    Args:
        payload_type: {"water", "oil", "gas"}
        volume: payload volume in liters
        container_shape: "default", or "tall"/"wide"/"offset" for 10 L water

    Notes:
        Non-default shapes have presets for 10 L water. The unified liquid
        simulator validates shape/type/volume combinations before construction.
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
