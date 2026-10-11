"""Disabled terrain curricula must not sustain an exploration deficit."""
from types import SimpleNamespace

import pytest

from rsl_rl.utils.adaptive_entropy import terrain_entropy_progress
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT
from rsl_rl.algorithms.ppo_b1z1_pact_pos import PPO_B1Z1PACTPos
from rsl_rl.algorithms.ppo_b1z1_ppo_pos import PPO_B1Z1PPOPos
from rsl_rl.algorithms.ppo_unifp import PPO_UniFP
from rsl_rl.algorithms.ppo_unifp_original import PPO_UniFPOriginal


@pytest.mark.parametrize("algorithm", [PPO_B1Z1PACT, PPO_B1Z1PACTPos, PPO_B1Z1PPOPos,
                                       PPO_UniFP, PPO_UniFPOriginal])
def test_disabled_terrain_is_complete_but_velocity_still_matters(algorithm):
    state = SimpleNamespace(device="cpu", entropy_coef_bounds=(.001, .01),
        ent_linvelo_threshold=1.5, ent_angvelo_threshold=.7,
        ent_terrain_threshold=6., ent_softmax_temperature=2.)
    level = terrain_entropy_progress(SimpleNamespace(curriculum=False), 0., 6.)
    metrics = dict(lin_vel_tracking=1.5, ang_vel_tracking=.7, terrain_level=level)
    assert algorithm.update_adaptive_entropy_coef(state, metrics) == pytest.approx(.001)
    metrics["lin_vel_tracking"] = 0.
    assert algorithm.update_adaptive_entropy_coef(state, metrics) > .001
    metrics.update(lin_vel_tracking=1.5,
        terrain_level=terrain_entropy_progress(SimpleNamespace(curriculum=True), 0., 6.))
    assert algorithm.update_adaptive_entropy_coef(state, metrics) == pytest.approx(.0050667649)


@pytest.mark.parametrize("level", [0., 2., 6., 10.])
def test_enabled_curriculum_preserves_observed_progress(level):
    assert terrain_entropy_progress(SimpleNamespace(curriculum=True), level, 6.) == level
    assert terrain_entropy_progress(SimpleNamespace(curriculum=False), level, 6.) == 6.
