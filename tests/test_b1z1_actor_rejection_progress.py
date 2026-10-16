"""Actor progress stays active without changing physical/command force timing."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
from legged_gym.envs.b1z1.b1z1_unifp_reject.b1z1_unifp_reject_config import B1Z1UniFPRejectCfg
from legged_gym.envs.b1z1.rejection_curriculum import RejectionCurriculum


@pytest.mark.parametrize("variant", [None, "unifp_reject", *range(4, 12)])
@pytest.mark.parametrize("performance_passes", [False, True])
@pytest.mark.parametrize("use_curriculum", [False, True])
def test_actor_floor_ramp_and_resume_preserve_force_schedule(variant, performance_passes, use_curriculum):
    cls = (B1Z1UniFPRejectCfg if variant == "unifp_reject" else
           B1Z1PACTCfg if variant is None else make_b1z1_pact_ablation_configs(variant)[0])
    cfg = cls().commands
    assert getattr(cfg, "actor_rejection_use_curriculum", True) == (variant == "unifp_reject")
    cfg.actor_rejection_use_curriculum = use_curriculum
    # Short explicit fixture, independent of mutable experiment defaults.
    cfg.reject_nominal_tracking_iterations = 8
    cfg.reject_initial_progress = .5
    cfg.reject_initial_external_scale = .25
    cfg.force_curriculum_gate_start_iteration = 10
    cfg.force_curriculum_latest_start_iteration = 20
    cfg.force_curriculum_use_latest_start_fallback = True
    cfg.force_curriculum_gate_patience = 2
    cfg.reject_compensation_ramp_iterations = 2
    cfg.reject_external_ramp_iterations = 4
    cfg.reject_external_step_enabled = False
    c = RejectionCurriculum(cfg, "cpu")
    assert c.actor_progress == (.5 if use_curriculum else 1.)
    assert c.beta == 0. and c.external_scale(0) == 0.
    trigger = 19 if performance_passes else 28
    metrics = (0., 0., cfg.force_curriculum_episode_length_threshold + 1.) if performance_passes else (None,) * 3
    for iteration in range(trigger + 8):
        c.update(iteration, *metrics)
        progress = .5 + .5 * min(max((iteration - trigger) / 2, 0.), 1.)
        external = 0. if iteration < 8 else .25 + .75 * min(max((iteration - trigger - 2) / 4, 0.), 1.)
        actor_progress = progress if use_curriculum else 1.
        assert c.actor_progress == pytest.approx(actor_progress)
        assert c.command_scale(iteration) == pytest.approx(0. if iteration < 8 else progress)
        assert c.external_scale(iteration) == pytest.approx(external)
        assert c.metrics(iteration)["Rejection/actor_progress"] == pytest.approx(actor_progress)
        assert c.metrics(iteration)["Rejection/beta"] == c.beta
        assert c.metrics(iteration)["Rejection/stage"] == (0 if iteration < 8 else 1 if iteration < trigger else 2 if iteration < trigger + 2 else 3)
        saved = c.state_dict()
        resumed = RejectionCurriculum(cfg, "cpu")
        resumed.load_state_dict(saved)
        assert resumed.state_dict() == saved  # No checkpoint schema changes.
        assert resumed.actor_progress == c.actor_progress
        c = resumed
    assert c.trigger_iteration == trigger


def test_runner_uses_actor_progress_and_retains_legacy_fallback():
    # Check the actual rollout assignment without constructing a simulator/BARD.
    source = Path(__file__).resolve().parents[1] / "rsl_rl/runners/b1z1_pact_runner.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "rejection_progress" for t in n.targets))
    expression = compile(ast.Expression(node.value), str(source), "eval")
    for curriculum, expected in [(SimpleNamespace(actor_progress=.5, beta=0.), .5),
                                 (SimpleNamespace(beta=.75), .75), (SimpleNamespace(), 0.)]:
        assert eval(expression, {"curriculum": curriculum}) == expected
