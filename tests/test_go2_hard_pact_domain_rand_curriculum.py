"""Fast deterministic tests for the simulator-neutral HardPACT curriculum."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ.setdefault("SIMULATOR", "genesis_pact")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_hard_pact_tests")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_hard_pact_tests")

from legged_gym.envs.go2.go2_hard_pact.domain_rand_curriculum import (
    HardPACTDomainRandCurriculum, go2_pact_domain_rand_schema,
)
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from legged_gym.envs.go2.go2_hard_pact.ablations import HARD_PACT_VARIANTS
from legged_gym.envs import task_registry


def cfg():
    value = copy.deepcopy(GO2HardPACTCfg())
    value.domain_rand.push_warmup = 0
    value.domain_rand.step_interval = 1
    value.domain_rand.min_reward_to_step = -1.0
    return value


def test_schema_endpoints_and_monotonic_interpolation():
    schema = go2_pact_domain_rand_schema(cfg())
    assert schema["joint_friction"].range_at(0) == tuple(cfg().domain_rand.joint_friction_range_start)
    assert schema["joint_friction"].range_at(1) == tuple(cfg().domain_rand.joint_friction_range_end)
    values = [schema["added_base_mass"].range_at(p)[1] for p in (0, .25, .5, 1)]
    assert values == sorted(values)
    assert schema["push_z"].range_at(.5)[1] == 0.0


def test_once_per_iteration_and_phase_order():
    curriculum = HardPACTDomainRandCurriculum(cfg(), seed=7)
    assert curriculum.advance(1)
    first = curriculum.progress["joint_dynamics"]
    assert not curriculum.advance(1)
    assert curriculum.progress["joint_dynamics"] == first
    while curriculum.phase == "joint_dynamics":
        curriculum.advance(curriculum.last_iteration + 1)
    assert curriculum.progress["joint_dynamics"] == 1.0
    assert curriculum.phase == "mass_com"


def test_seeded_sampling_and_exact_checkpoint_restore():
    left = HardPACTDomainRandCurriculum(cfg(), seed=123)
    right = HardPACTDomainRandCurriculum(cfg(), seed=123)
    torch.testing.assert_close(
        left.sample("ground_friction", (32,)),
        right.sample("ground_friction", (32,)), rtol=0, atol=0,
    )
    for iteration in range(1, 8):
        left.advance(iteration, .8)
    state = copy.deepcopy(left.state_dict())
    resumed = HardPACTDomainRandCurriculum(cfg(), seed=999)
    resumed.load_state_dict(state)
    assert resumed.state_dict()["progress"] == state["progress"]
    torch.testing.assert_close(
        left.sample("kp_scale", (16,)), resumed.sample("kp_scale", (16,)),
        rtol=0, atol=0,
    )


def test_capability_reporting_never_claims_unsupported_effective_range():
    curriculum = HardPACTDomainRandCurriculum(cfg())
    report = curriculum.report({"ground_friction": True, "motor_strength": False})
    assert report["ground_friction"]["effective_range"] is not None
    assert report["motor_strength"]["effective_range"] is None
    assert report["motor_strength"]["requested_range"] is not None


def test_all_ablations_share_identical_curriculum_schema():
    reference = go2_pact_domain_rand_schema(cfg())
    for variant in HARD_PACT_VARIANTS:
        registered = task_registry.env_cfgs[f"go2_hard_pact_{variant}_genesis"]
        assert go2_pact_domain_rand_schema(registered) == reference


def test_backend_and_position_registrations():
    for backend in ("genesis", "isaaclab"):
        assert f"go2_hard_pact_pos_{backend}" in task_registry.task_classes
        for variant in HARD_PACT_VARIANTS:
            assert f"go2_hard_pact_{variant}_{backend}" in task_registry.task_classes


def test_isaaclab_uses_trimesh_without_changing_genesis_terrain():
    assert task_registry.env_cfgs["go2_hard_pact_full_isaaclab"].terrain.mesh_type == "trimesh"
    assert task_registry.env_cfgs["go2_hard_pact_pos_isaaclab"].terrain.mesh_type == "trimesh"
    assert task_registry.env_cfgs[
        "go2_hard_pact_full_genesis"
    ].terrain.mesh_type == "heightfield"


def test_isaaclab_pact_adapter_is_an_explicit_thin_subclass():
    from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
    from legged_gym.simulator.isaaclab_simulator_pact import (
        IsaacLabSimulator_PACT,
    )

    assert issubclass(IsaacLabSimulator_PACT, IsaacLabSimulator)
    assert "hard_pact_configuration" in IsaacLabSimulator_PACT.__dict__
    assert "hard_pact_configuration" not in IsaacLabSimulator.__dict__
    assert "_compute_torques" in IsaacLabSimulator_PACT.__dict__
    assert task_registry.env_cfgs[
        "go2_hard_pact_full_isaaclab"
    ].sim.use_pact_adapter
    assert task_registry.env_cfgs[
        "go2_hard_pact_pos_isaaclab"
    ].sim.use_pact_adapter
    assert not hasattr(
        task_registry.env_cfgs["go2_hard_pact_full_genesis"].sim,
        "use_pact_adapter",
    )


def test_isaaclab_push_writes_root_velocity_only_at_event():
    from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT

    class Robot:
        def __init__(self):
            self.data = SimpleNamespace(root_link_vel_w=torch.zeros(2, 6))
            self.writes = []

        def write_root_link_velocity_to_sim(self, velocity, env_ids):
            self.writes.append((velocity.clone(), env_ids.clone()))
            self.data.root_link_vel_w[env_ids] = velocity

    simulator = object.__new__(IsaacLabSimulator_PACT)
    simulator._num_envs = 2
    simulator._device = "cpu"
    simulator._control_dt = 0.1
    simulator._push_call_counter = 0
    simulator.push_interval_min = simulator.push_interval_max = 1.0
    simulator.push_timeouts = torch.ones(2, 1)
    simulator._rand_push_vels = torch.zeros(2, 3)
    simulator._cfg = SimpleNamespace(
        env=SimpleNamespace(lateral_push_only=False),
        domain_rand=SimpleNamespace(max_push_vel_xy=1.0),
    )
    simulator._robot = Robot()

    for _ in range(9):
        simulator.push_robots()
    assert simulator._robot.writes == []

    simulator.push_robots()
    assert len(simulator._robot.writes) == 1
    written_velocity, written_envs = simulator._robot.writes[0]
    torch.testing.assert_close(written_envs, torch.tensor([0, 1]))
    torch.testing.assert_close(written_velocity[:, :2], simulator._rand_push_vels[:, :2])

    velocity_after_event = simulator._robot.data.root_link_vel_w.clone()
    simulator.push_robots()
    assert len(simulator._robot.writes) == 1
    torch.testing.assert_close(simulator._robot.data.root_link_vel_w, velocity_after_event)
    assert torch.count_nonzero(simulator._rand_push_vels) == 0


def test_isaaclab_persistent_wrench_uses_composer_api():
    from legged_gym.simulator.isaaclab_simulator_pact import IsaacLabSimulator_PACT

    class Composer:
        def __init__(self):
            self.calls = []

        def set_forces_and_torques(self, **kwargs):
            self.calls.append(kwargs)

    composer = Composer()
    robot = SimpleNamespace(permanent_wrench_composer=composer)
    simulator = object.__new__(IsaacLabSimulator_PACT)
    simulator._robot = robot
    simulator._base_link_index = 3
    wrench = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    simulator.hard_pact_apply_base_wrench_world(wrench)

    assert len(composer.calls) == 1
    call = composer.calls[0]
    torch.testing.assert_close(call["forces"], wrench[:, :3].unsqueeze(1))
    torch.testing.assert_close(call["torques"], wrench[:, 3:].unsqueeze(1))
    assert call["body_ids"] == [3]
    assert call["is_global"] is True


def test_launcher_rejects_invalid_task_without_starting_training():
    launcher = (Path(__file__).parents[1] / "legged_gym" / "scripts" /
                "go2_hard_pact.sh")
    result = subprocess.run(
        [str(launcher), "--task", "not_hard_pact", "--headless"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
