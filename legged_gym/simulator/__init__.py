from legged_gym import SIMULATOR
from .simulator import Simulator
<<<<<<< HEAD

# Backend-local imports keep the Isaac Gym Python environment independent of
# Genesis-only visualization and dynamics dependencies.
if "isaacgym" in SIMULATOR:
    from .isaacgym_simulator import IsaacGymSimulator
    from .isaacgym_simulator_b1z1 import (
        IsaacGymSimulatorB1Z1UniFP,
        IsaacGymSimulatorB1Z1PACTPos,
        IsaacGymSimulatorB1Z1PACT,
    )
elif "genesis" in SIMULATOR:
    from .genesis_simulator import GenesisSimulator
    from .genesis_simulator_pact import GenesisSimulator_PACT
    from .genesis_simulator_pact_pos import GenesisSimulator_PACT_Pos
    from .genesis_simulator_pact_water import GenesisSimulator_PACT_Water
    from .genesis_simulator_pact_nopinn import GenesisSimulator_PACT_NoPINN
    from .genesis_simulator_pact_postau import GenesisSimulator_PACT_PosTau
    from .genesis_simulator_pact_rl2ac import GenesisSimulator_PACT_RL2AC
    from .genesis_simulator_kite import GenesisSimulator_KITE
    from .genesis_simulator_kite_depth import GenesisSimulator_KITE_Depth
    from .genesis_simulator_b1z1_unifp import GenesisSimulatorB1Z1UniFP
    from .genesis_simulator_b1z1_pact import GenesisSimulatorB1Z1PACT
    from .genesis_simulator_b1z1_pact_pos import GenesisSimulatorB1Z1PACTPos
elif "isaaclab" in SIMULATOR:
    from .isaaclab_simulator import IsaacLabSimulator
    from .isaaclab_simulator_b1z1 import (
        IsaacLabSimulatorB1Z1UniFP, IsaacLabSimulatorB1Z1PACT,
        IsaacLabSimulatorB1Z1PACTPos,
    )
=======
from legged_gym import SIMULATOR

# The simulator SDKs live in separate environments. Loading only the selected
# adapter prevents Isaac Lab from acquiring an unrelated Genesis dependency.
if SIMULATOR == "isaaclab":
    from .isaaclab_simulator import IsaacLabSimulator
    from .isaaclab_simulator_pact import IsaacLabSimulator_PACT
elif SIMULATOR == "isaacgym":
    from .isaacgym_simulator import IsaacGymSimulator
elif SIMULATOR == "genesis":
    from .genesis_simulator import GenesisSimulator
elif SIMULATOR == "genesis_pact":
    from .genesis_simulator_pact import GenesisSimulator_PACT
elif SIMULATOR == "genesis_pact_pos":
    from .genesis_simulator_pact_pos import GenesisSimulator_PACT_Pos
elif SIMULATOR == "genesis_pact_water":
    from .genesis_simulator_pact_water import GenesisSimulator_PACT_Water
elif SIMULATOR == "genesis_pact_nopinn":
    from .genesis_simulator_pact_nopinn import GenesisSimulator_PACT_NoPINN
elif SIMULATOR == "genesis_pact_postau":
    from .genesis_simulator_pact_postau import GenesisSimulator_PACT_PosTau
elif SIMULATOR == "genesis_pact_rl2ac":
    from .genesis_simulator_pact_rl2ac import GenesisSimulator_PACT_RL2AC
>>>>>>> aligned_iclr_2027_qp_pinn
