"""Backend capability declarations and canonical simulator adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import torch

from .schema import (
    CANONICAL,
    CanonicalState,
    body_to_world,
    permutation_by_name,
    reorder_named,
)


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supports_ideal_torque: bool
    supports_interval_grf: bool
    supports_sustained_wrench: bool
    supports_reset_domain_rand: bool
    supports_domain_rand_curriculum: bool
    supports_friction: bool
    supports_base_mass: bool
    supports_base_com: bool
    supports_control_delay: bool
    supports_pd_gain: bool
    supports_motor_strength: bool
    supports_armature: bool
    supports_joint_friction: bool
    supports_joint_stiffness: bool
    supports_joint_damping: bool
    supports_instantaneous_disturbance: bool


GENESIS_CAPABILITIES = BackendCapabilities(
    name="genesis", supports_ideal_torque=True, supports_interval_grf=True,
    supports_sustained_wrench=True, supports_reset_domain_rand=True,
    supports_domain_rand_curriculum=True, supports_friction=True,
    supports_base_mass=True, supports_base_com=True, supports_control_delay=True,
    supports_pd_gain=True, supports_motor_strength=True, supports_armature=True,
    supports_joint_friction=True, supports_joint_stiffness=True,
    supports_joint_damping=True, supports_instantaneous_disturbance=True,
)
ISAACLAB_CAPABILITIES = BackendCapabilities(
    name="isaaclab", supports_ideal_torque=True, supports_interval_grf=True,
    supports_sustained_wrench=True, supports_reset_domain_rand=True,
    supports_domain_rand_curriculum=False, supports_friction=True,
    supports_base_mass=True, supports_base_com=True, supports_control_delay=True,
    supports_pd_gain=True, supports_motor_strength=False, supports_armature=True,
    supports_joint_friction=True, supports_joint_stiffness=False,
    supports_joint_damping=True, supports_instantaneous_disturbance=True,
)

CAPABILITIES = {
    "genesis": GENESIS_CAPABILITIES,
    "isaaclab": ISAACLAB_CAPABILITIES,
}


@dataclass(frozen=True)
class RandomizationRequest:
    config_flag: str
    range_names: tuple[str, ...]
    capability: str


RANDOMIZATION_SCHEMA: Mapping[str, RandomizationRequest] = {
    "friction": RandomizationRequest("randomize_friction", ("friction_range",), "supports_friction"),
    "base_mass": RandomizationRequest(
        "randomize_base_mass", ("added_mass_range",), "supports_base_mass"
    ),
    "base_com": RandomizationRequest(
        "randomize_com_displacement",
        ("com_pos_x_range", "com_pos_y_range", "com_pos_z_range"),
        "supports_base_com",
    ),
    "control_delay": RandomizationRequest(
        "randomize_ctrl_delay", ("ctrl_delay_step_range",), "supports_control_delay"
    ),
    "kp_kd": RandomizationRequest("randomize_pd_gain", ("kp_range", "kd_range"), "supports_pd_gain"),
    "motor_strength": RandomizationRequest(
        "randomize_motor_strength", ("motor_strength_range",), "supports_motor_strength"
    ),
    "armature": RandomizationRequest(
        "randomize_joint_armature", ("joint_armature_range",), "supports_armature"
    ),
    "joint_friction": RandomizationRequest(
        "randomize_joint_friction", ("joint_friction_range",), "supports_joint_friction"
    ),
    "joint_stiffness": RandomizationRequest(
        "randomize_joint_stiffness", ("joint_stiffness_range",), "supports_joint_stiffness"
    ),
    "joint_damping": RandomizationRequest(
        "randomize_joint_damping", ("joint_damping_range",), "supports_joint_damping"
    ),
    "instantaneous_disturbance": RandomizationRequest(
        "randomize_instantaneous_disturbances",
        (
            "instantaneous_planar_delta_v_range",
            "instantaneous_downward_delta_vz_range",
            "instantaneous_angular_delta_v_range",
        ),
        "supports_instantaneous_disturbance",
    ),
}


def domain_randomization_report(domain_cfg, capabilities: BackendCapabilities) -> Dict[str, dict]:
    """Report requested and effective DR instead of claiming unsupported DR."""
    report: Dict[str, dict] = {}
    for name, item in RANDOMIZATION_SCHEMA.items():
        requested = bool(getattr(domain_cfg, item.config_flag, False))
        supported = bool(getattr(capabilities, item.capability))
        ranges = {key: getattr(domain_cfg, key, None) for key in item.range_names}
        report[name] = {
            "requested": requested,
            "supported": supported,
            "active": requested and supported,
            "requested_ranges": ranges,
            "effective_ranges": ranges if requested and supported else None,
            "application": (
                "runtime_curriculum"
                if requested and supported and capabilities.supports_domain_rand_curriculum
                else "reset_or_static" if requested and supported else None
            ),
            "reason": None if not requested or supported else f"unsupported by {capabilities.name}",
        }
    report["domain_rand_curriculum"] = {
        "requested": bool(getattr(domain_cfg, "use_domainrand_curriculum", False)),
        "supported": capabilities.supports_domain_rand_curriculum,
        "active": bool(getattr(domain_cfg, "use_domainrand_curriculum", False))
        and capabilities.supports_domain_rand_curriculum,
        "requested_ranges": None,
        "effective_ranges": None,
        "reason": (
            None
            if capabilities.supports_domain_rand_curriculum
            else f"runtime curriculum is not supported by {capabilities.name}; reset-time DR remains active"
        ),
    }
    return report


def disable_unsupported_randomizations(domain_cfg, capabilities: BackendCapabilities) -> Dict[str, dict]:
    report = domain_randomization_report(domain_cfg, capabilities)
    for name, item in RANDOMIZATION_SCHEMA.items():
        if report[name]["requested"] and not report[name]["supported"]:
            setattr(domain_cfg, item.config_flag, False)
    if not capabilities.supports_domain_rand_curriculum:
        domain_cfg.use_domainrand_curriculum = False
    return report


class HardPACTBackendAdapter:
    """Narrow adapter that exposes one canonical contract to the task core."""

    capabilities: BackendCapabilities

    def __init__(self, simulator, cfg):
        self.simulator = simulator
        self.cfg = cfg
        self.device = simulator._device
        self._verify_names()
        physics_dt = float(getattr(cfg.sim, "dt"))
        control_dt = float(getattr(cfg.control, "dt"))
        expected = physics_dt * int(cfg.control.decimation)
        if abs(control_dt - expected) > max(1.0e-9, 1.0e-6 * expected):
            raise ValueError(
                f"control dt {control_dt} does not equal physics dt {physics_dt} * "
                f"decimation {cfg.control.decimation} = {expected}"
            )
        self.physics_dt = physics_dt
        self.control_dt = control_dt

    def _verify_names(self) -> None:
        requested_joints = tuple(self.cfg.asset.dof_names)
        requested_feet = tuple(self.cfg.asset.foot_name)
        permutation_by_name(requested_joints, CANONICAL.joint_names, "Go2 joint")
        permutation_by_name(requested_feet, CANONICAL.foot_names, "Go2 foot")

        actual_joints = tuple(getattr(self.simulator, "_dof_names", requested_joints))
        if set(CANONICAL.joint_names).issubset(actual_joints):
            permutation_by_name(
                tuple(name for name in actual_joints if name in CANONICAL.joint_names),
                CANONICAL.joint_names,
                "simulator Go2 joint",
            )
        actual_feet = tuple(getattr(self.simulator, "_feet_names", requested_feet))
        permutation_by_name(actual_feet, CANONICAL.foot_names, "simulator Go2 foot")

    def augment_simulator_buffers(self) -> None:
        sim = self.simulator
        n = sim._num_envs
        device = sim._device
        dtype = sim.dof_pos.dtype

        def ensure(name, shape, fill=0.0):
            if not hasattr(sim, name):
                setattr(sim, name, torch.full(shape, fill, device=device, dtype=dtype))

        ensure("_grfs_buf", (n, 12))
        ensure("_rand_wrench_vels", (n, 3))
        ensure("_motor_strength", (n, 12), 1.0)
        ensure("_joint_stiffness", (n, 1))
        ensure("feedback_torques", (n, 12))
        ensure("feedforward_torques", (n, 12))
        ensure("combined_feedback_torques", (n, 12))
        ensure("combined_feedforward_torques", (n, 12))
        ensure("first_loop_feedback", (n, 12))
        ensure("feedforward_tau_weight", (n, 1), 1.0)
        ensure("feedback_tau_weight", (n, 1), 1.0)
        ensure("feedforward_tau_weight_clean", (n, 1), 1.0)
        ensure("feedback_tau_weight_clean", (n, 1), 1.0)
        if not hasattr(sim, "_base_pos_out_of_bounds_buf"):
            sim._base_pos_out_of_bounds_buf = torch.zeros(
                n, device=device, dtype=torch.bool
            )
        if not hasattr(sim, "_robot_mass"):
            sim._robot_mass = torch.full((n, 1), 15.0, device=device, dtype=dtype)

    def _ordered_joints(self, value: torch.Tensor) -> torch.Tensor:
        source = tuple(self.cfg.asset.dof_names)
        return reorder_named(value, source, CANONICAL.joint_names, "Go2 joint")

    def ordered_backend_joint_values(self, value: torch.Tensor) -> torch.Tensor:
        """Convert raw simulator/URDF joint tensors to canonical paper order."""
        source = tuple(getattr(self.simulator, "_dof_names", self.cfg.asset.dof_names))
        return reorder_named(value, source, CANONICAL.joint_names, "backend Go2 joint")

    def _ordered_feet(self, value: torch.Tensor) -> torch.Tensor:
        source = tuple(getattr(self.simulator, "_feet_names", self.cfg.asset.foot_name))
        return reorder_named(value, source, CANONICAL.foot_names, "Go2 foot")

    def canonical_state(self) -> CanonicalState:
        sim = self.simulator
        quat = sim.base_quat
        linear_world = getattr(sim, "_base_world_lin_vel", None)
        angular_world = getattr(sim, "_base_world_ang_vel", None)
        if linear_world is None:
            linear_world = body_to_world(sim.base_lin_vel, quat)
        if angular_world is None:
            angular_world = body_to_world(sim.base_ang_vel, quat)
        state = CanonicalState(
            base_position_world=sim.base_pos,
            base_quaternion_xyzw=quat,
            velocity_world=torch.cat((
                linear_world, angular_world, self._ordered_joints(sim.dof_vel)
            ), dim=-1),
            joint_position=self._ordered_joints(sim.dof_pos),
        )
        state.validate()
        return state

    def raw_foot_forces_world(self) -> torch.Tensor:
        forces = self.simulator.link_contact_forces[:, self.simulator.feet_indices, :]
        return self._ordered_feet(forces)

    def add_root_velocity_delta_world(self, delta_world: torch.Tensor, event_mask: torch.Tensor) -> None:
        raise NotImplementedError

    def step_safe_torque(self, torque: torch.Tensor, wrench_world: torch.Tensor, grf_callback) -> None:
        raise NotImplementedError

    def metadata(self) -> Dict[str, object]:
        return {
            "backend": self.capabilities.name,
            "capabilities": asdict(self.capabilities),
            "physics_dt": self.physics_dt,
            "control_dt": self.control_dt,
            "joint_order": CANONICAL.joint_names,
            "foot_order": CANONICAL.foot_names,
            "quaternion_order": CANONICAL.quaternion,
            "twist_order": CANONICAL.base_twist,
            "force_frame": CANONICAL.force_frame,
        }


class GenesisHardPACTAdapter(HardPACTBackendAdapter):
    capabilities = GENESIS_CAPABILITIES

    def add_root_velocity_delta_world(self, delta_world, event_mask):
        ids = event_mask.squeeze(-1).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        # Genesis exposes the free joint first in the complete DOF velocity.
        velocity = self.simulator._robot.get_dofs_velocity().clone()
        velocity[ids, :6] += delta_world[ids]
        self.simulator._robot.set_dofs_velocity(velocity[ids], envs_idx=ids)

    def _apply_wrench(self, wrench_world):
        sim = self.simulator
        solver = sim._robot._solver
        link = torch.as_tensor([sim._base_link_index], device=self.device, dtype=torch.long)
        solver.apply_links_external_force(
            force=wrench_world[:, :3].unsqueeze(1), links_idx=link,
            envs_idx=None, ref="link_com", local=False,
        )
        solver.apply_links_external_torque(
            torque=wrench_world[:, 3:].unsqueeze(1), links_idx=link,
            envs_idx=None, ref="link_com", local=False,
        )

    def step_safe_torque(self, torque, wrench_world, grf_callback):
        sim = self.simulator
        sim._last_base_lin_vel.copy_(sim._base_lin_vel)
        sim._last_base_ang_vel.copy_(sim._base_ang_vel)
        sim._last_feet_vel.copy_(sim._feet_vel)
        sim._last_dof_vel.copy_(sim._dof_vel)
        for _ in range(self.cfg.control.decimation):
            sim._torques = torque
            self._apply_wrench(wrench_world)
            sim._robot.control_dofs_force(torque, sim._dof_indices)
            sim._scene.step()
            sim._dof_pos.copy_(sim._robot.get_dofs_position(sim._dof_indices))
            sim._dof_vel.copy_(sim._robot.get_dofs_velocity(sim._dof_indices))
            raw = sim._robot.get_links_net_contact_force()[:, sim._feet_indices, :]
            grf_callback(self._ordered_feet(raw))


class IsaacLabHardPACTAdapter(HardPACTBackendAdapter):
    capabilities = ISAACLAB_CAPABILITIES

    def add_root_velocity_delta_world(self, delta_world, event_mask):
        ids = event_mask.squeeze(-1).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        velocity = self.simulator._robot.data.root_link_vel_w[ids].clone()
        velocity += delta_world[ids]
        self.simulator._robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)

    def step_safe_torque(self, torque, wrench_world, grf_callback):
        sim = self.simulator
        sim._last_base_lin_vel.copy_(sim._base_lin_vel)
        sim._last_base_ang_vel.copy_(sim._base_ang_vel)
        sim._last_feet_vel.copy_(sim._robot.data.body_link_vel_w[:, sim.feet_indices, :3])
        sim._last_dof_vel.copy_(sim._robot.data.joint_vel)
        body_ids = torch.as_tensor([sim._base_link_index], device=self.device, dtype=torch.long)
        for _ in range(self.cfg.control.decimation):
            sim._robot.set_external_force_and_torque(
                wrench_world[:, :3].unsqueeze(1),
                wrench_world[:, 3:].unsqueeze(1),
                body_ids=body_ids,
                is_global=True,
            )
            sim._robot.set_joint_effort_target(torque, joint_ids=sim._dof_indices)
            sim._robot.write_data_to_sim()
            sim._sim.step(render=False)
            sim._robot.update(sim._sim_params["dt"])
            sim._contact_sensors.update(sim._sim_params["dt"])
            raw = sim._contact_sensors.data.net_forces_w[
                :, sim._feet_contact_indices, :
            ]
            grf_callback(self._ordered_feet(raw))
        if not sim._headless:
            sim._sim.render()


ADAPTERS = {
    "genesis": GenesisHardPACTAdapter,
    "isaaclab": IsaacLabHardPACTAdapter,
}
