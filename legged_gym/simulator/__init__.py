from .simulator import Simulator
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
