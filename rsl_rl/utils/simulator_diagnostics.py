"""Optional simulator diagnostics shared by B1Z1 training runners."""


def log_tradeoff_curriculum(writer, env, iteration):
    """Report per-environment stages and clean curriculum weights, before randomization."""
    if not getattr(env, "use_tradeoff", False):
        return
    steps = env.tradeoff_step_ctr.detach().float()
    values = {
        "stage_mean": steps.mean(),
        "stage_min": steps.min(),
        "stage_max": steps.max(),
        "completed_fraction": (steps >= env.tradeoff_num_steps).float().mean(),
        "feedback_weight_mean": env.simulator.feedback_tau_weight_clean.detach().mean(),
        "feedforward_weight_mean": env.simulator.feedforward_tau_weight_clean.detach().mean(),
    }
    for name, value in values.items():
        writer.add_scalar(f"Tradeoff/{name}", value, iteration)


def domain_rand_state(simulator):
    getter = getattr(simulator, "domain_rand_curriculum_state_dict", None)
    return getter() if getter is not None else None


def load_domain_rand_state(simulator, state):
    loader = getattr(simulator, "load_domain_rand_curriculum_state_dict", None)
    if loader is not None:
        loader(state)
