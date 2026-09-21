"""Optional simulator diagnostics shared by B1Z1 training runners."""


def domain_rand_state(simulator):
    getter = getattr(simulator, "domain_rand_curriculum_state_dict", None)
    return getter() if getter is not None else None


def load_domain_rand_state(simulator, state):
    loader = getattr(simulator, "load_domain_rand_curriculum_state_dict", None)
    if loader is not None:
        loader(state)
