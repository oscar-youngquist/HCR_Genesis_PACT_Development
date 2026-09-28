"""Genesis position-control adapter for the Go2 DreamWaQ curriculum."""
import torch
from .genesis_simulator import GenesisSimulator
from .dreamwaq_adapter import DreamWaQAdapter


class GenesisSimulatorDreamWaQ(DreamWaQAdapter, GenesisSimulator):
    def _parse_cfg(self):
        super()._parse_cfg()
        self._batch_dofs_links_info |= self._cfg.domain_rand.randomize_joint_stiffness

    def _create_envs(self):
        super()._create_envs()
        self._robot_mass = sum(link.get_mass() for link in self._robot.links)
        names = [link.name for link in self._robot.links]
        self._feet_indices = [names.index(name) for name in self._cfg.asset.feet_names]
        self._contact_state_link_indices = [names.index(name) for name in self._cfg.asset.contact_state_link_names]

    def _install_joint_stiffness(self, env_ids, values):
        self._robot.set_dofs_stiffness(values, self._dof_indices, envs_idx=env_ids)

    def _compute_torques(self, actions):
        return self._bounded_position_torques(actions)

    def _apply_velocity_push(self, env_ids, linear, angular):
        velocity = self._robot.get_dofs_velocity()[env_ids].clone()
        velocity[:, :3] += linear
        velocity[:, 3:6] += angular
        self._robot.set_dofs_velocity(velocity, envs_idx=env_ids)
