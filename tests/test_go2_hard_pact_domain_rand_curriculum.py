"""Fast deterministic tests for the simulator-neutral HardPACT curriculum."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

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


def test_launcher_rejects_invalid_task_without_starting_training():
    launcher = Path(__file__).parents[1] / "scripts" / "go2_hard_pact.sh"
    result = subprocess.run(
        [str(launcher), "--task", "not_hard_pact", "--headless"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
