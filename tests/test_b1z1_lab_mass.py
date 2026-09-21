"""CPU regression for indexed mass/inertia updates; no SimulationApp required."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
import legged_gym.envs  # Initialize the registry before simulator utility imports.
from legged_gym.simulator.isaaclab_simulator_b1z1 import _IsaacLabSimulatorB1Z1


def test_mass_inertia_updates_are_indexed_and_do_not_compound():
    masses = torch.tensor([[10., 2., 4.]]).repeat(3, 1)
    inertias = torch.arange(1., 28.).reshape(1, 3, 9).repeat(3, 1, 1)
    defaults, default_inertias = masses.clone(), inertias.clone()
    view = Mock()
    view.get_masses.side_effect = lambda: masses.clone()
    view.get_inertias.side_effect = lambda: inertias.clone()
    view.set_masses.side_effect = lambda values, ids: masses.index_copy_(0, ids, values[ids])
    view.set_inertias.side_effect = lambda values, ids: inertias.index_copy_(0, ids, values[ids])
    sim = SimpleNamespace(
        _robot=SimpleNamespace(root_physx_view=view, data=SimpleNamespace(
            default_mass=defaults, default_inertia=default_inertias)),
        _cfg=SimpleNamespace(domain_rand=SimpleNamespace(
            randomize_base_mass=True, randomize_gripper_mass=True)),
        _device="cpu", _base_link_index=0, _gripper_index=1,
        mass_min=5., mass_max_value=5., grip_mass_min=-1., grip_mass_max_value=-1.,
        _added_base_mass=torch.zeros(3, 1), _added_gripper_mass=torch.zeros(3, 1),
    )
    ids = torch.tensor([1])
    for _ in range(2):
        _IsaacLabSimulatorB1Z1._randomize_mass(sim, ids)
        torch.testing.assert_close(masses[1], torch.tensor([15., 1., 4.]))
        torch.testing.assert_close(inertias[1, 0], default_inertias[1, 0] * 1.5)
        torch.testing.assert_close(inertias[1, 1], default_inertias[1, 1] * 0.5)
        torch.testing.assert_close(inertias[1, 2], default_inertias[1, 2])
        torch.testing.assert_close(masses[[0, 2]], defaults[[0, 2]])
        torch.testing.assert_close(inertias[[0, 2]], default_inertias[[0, 2]])
    torch.testing.assert_close(sim._added_base_mass[:, 0], torch.tensor([0., 5., 0.]))
    torch.testing.assert_close(sim._added_gripper_mass[:, 0], torch.tensor([0., -1., 0.]))
    assert view.set_inertias.call_args.args[1].device.type == "cpu"
