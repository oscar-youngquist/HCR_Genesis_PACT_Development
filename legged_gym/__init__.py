import os
import sys

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(f"LEGGED_GYM_ROOT_DIR: {LEGGED_GYM_ROOT_DIR}")
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')

if sys.version_info[1] >= 10: # >=3.10 for genesis and isaacsim
    simulator_type = os.getenv("SIMULATOR")
    if simulator_type == "genesis":
        SIMULATOR = "genesis"
    elif simulator_type == "genesis_pact":
        SIMULATOR = "genesis_pact"
    elif simulator_type == "genesis_pact_pos":
        SIMULATOR = "genesis_pact_pos"
    elif simulator_type == "genesis_pact_water":
        SIMULATOR = "genesis_pact_water"
    elif simulator_type == "genesis_pact_nopinn":
        SIMULATOR = "genesis_pact_nopinn"
    elif simulator_type == "genesis_pact_postau":
        SIMULATOR = "genesis_pact_postau"
    elif simulator_type == "genesis_pact_rl2ac":
        SIMULATOR = "genesis_pact_rl2ac"
    elif simulator_type == "genesis_kite":
        SIMULATOR = "genesis_kite"
    elif simulator_type == "genesis_kite_depth":
        SIMULATOR = "genesis_kite_depth"
    elif simulator_type == "genesis_b1z1_unifp":
        SIMULATOR = "genesis_b1z1_unifp"
    elif simulator_type == "genesis_b1_unifp":
        SIMULATOR = "genesis_b1_unifp"
    elif simulator_type == "isaaclab":
        SIMULATOR = "isaaclab"
    else:
        raise ValueError(
            "Unsupported SIMULATOR type. Expected a configured Genesis or IsaacLab simulator."
        )
elif sys.version_info[1] <= 8 and sys.version_info[1] >= 6: # >=3.6 and <3.9 for isaacgym
    SIMULATOR = "isaacgym"

if "genesis" in SIMULATOR:
    try: 
        import numba

        _numba_jit = numba.jit
        _numba_njit = numba.njit

        def _jit_without_cache(*args, **kwargs):
            kwargs["cache"] = False
            return _numba_jit(*args, **kwargs)

        def _njit_without_cache(*args, **kwargs):
            kwargs["cache"] = False
            return _numba_njit(*args, **kwargs)

        numba.jit = _jit_without_cache
        numba.njit = _njit_without_cache
        import genesis as gs
    except ImportError as e:
        print("Failed to import Genesis. Please ensure that the Genesis is properly installed and configured.")
        raise e
# if SIMULATOR == "genesis_pact":
#     try: 
#         import genesis as gs
#     except ImportError as e:
#         print("Failed to import Genesis. Please ensure that the Genesis is properly installed and configured.")
#         raise e
# if SIMULATOR == "genesis_pact_pos":
#     try: 
#         import genesis as gs
#     except ImportError as e:
#         print("Failed to import Genesis. Please ensure that the Genesis is properly installed and configured.")
#         raise e
# if SIMULATOR == "genesis_pact_pos":
#     try: 
#         import genesis as gs
#     except ImportError as e:
#         print("Failed to import Genesis. Please ensure that the Genesis is properly installed and configured.")
#         raise e
elif SIMULATOR == "isaacgym":
    try:
        import isaacgym
    except ImportError as e:
        print("Failed to import Isaac Gym. Please ensure that the Isaac Gym is properly installed and configured.")
        raise e
elif SIMULATOR == "isaaclab":
    try:
        import isaaclab
    except ImportError as e:
        print("Failed to import Isaac Lab. Please ensure that the Isaac Lab is properly installed and configured.")
        raise e
