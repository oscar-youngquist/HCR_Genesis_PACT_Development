"""Completed control-step schedules for the coupled B1Z1 PACT learners."""


def prepare_step_schedules(algorithm, completed_env_steps):
    """Advance time once per fresh collection, independently of optimizer updates."""
    completed_env_steps = int(completed_env_steps)
    previous = getattr(algorithm, 'schedule_env_steps', 0)
    if completed_env_steps < previous:
        raise ValueError('completed environment steps must be monotonically increasing')
    algorithm.schedule_step_delta = completed_env_steps - previous
    algorithm.schedule_env_steps = completed_env_steps
    start = algorithm.cfg.get('pinn_start_env_step', algorithm.cfg.get('pinn_init_steps', 0))
    duration = algorithm.cfg.get('pinn_warmup_env_steps', algorithm.cfg.get('pinn_warmup', 1))
    if start < 0 or duration < 0:
        raise ValueError('PINN start and warmup environment steps must be nonnegative')
    progress = min(1., max(0., (completed_env_steps - start) / max(1, duration)))
    algorithm.pinn_weight = progress * abs(algorithm.cfg['pinn_loss_weight'])
