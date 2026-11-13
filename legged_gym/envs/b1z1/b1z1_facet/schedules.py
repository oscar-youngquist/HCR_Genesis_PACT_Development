"""FACET reward-based gating; shared force timing and physical samplers unchanged."""
from copy import deepcopy
import warnings

from legged_gym.envs.b1z1.rejection_curriculum import RejectionCurriculum
from legged_gym.envs.b1z1.force_task_utils import _mean_episode_metric


class FACETForceCurriculum(RejectionCurriculum):
    def __init__(self, cfg, device):
        commands = deepcopy(cfg.commands)
        # Reuse the shared lower-is-better EMA gate on 1 - reward/max_reward.
        # Its historical ee_l1 field stores a dimensionless deficit ONLY here.
        commands.force_curriculum_ee_l1_threshold = 1-cfg.facet_tracking.force_curriculum_reward_fraction
        super().__init__(commands, device)

    def metrics(self, iteration):
        result = super().metrics(iteration)
        result.pop('ForceCurriculum/ee_l1_ema', None)
        if self.ee_l1_ema is not None:
            result['ForceCurriculum/facet_ee_reward_fraction_ema'] = 1-self.ee_l1_ema
        result['ForceCurriculum/facet_ee_reward_fraction_threshold'] = 1-self.ee_l1_threshold
        return result

    def state_dict(self):
        return dict(super().state_dict(), facet_reward_gate_version=1)

    def load_state_dict(self, state):
        state = dict(state or {})
        if state and state.get('facet_reward_gate_version') != 1:
            warnings.warn('FACET reward gate: discarding legacy nominal-error EMA/patience; preserving force exposure.')
            state['ee_l1_ema'] = None
            state['gate_patience'] = 0
        super().load_state_dict(state)


def update_force_schedule(env, iteration, infos, episode_length):
    gate = getattr(env, '_staged_force_curriculum', None)
    if gate is None:
        return {}
    reward = _mean_episode_metric(infos, 'rew_facet_ee_position')
    weight = env.cfg.facet_tracking.weights['ee_position']
    # Logged episode rewards are dt-scaled sums / prescribed episode seconds.
    # Missing/disabled metrics must not be interpreted as perfect tracking.
    deficit = 1-reward/weight if reward is not None and weight > 0 else None
    gate.update(iteration, deficit, _mean_episode_metric(infos, 'roll_termination_rate'), episode_length)
    return gate.metrics(iteration)
