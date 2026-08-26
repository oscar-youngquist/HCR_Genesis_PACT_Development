"""Check PACT privileged-force compatibility with PACT-Pos pretraining."""

import unittest
from types import SimpleNamespace

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass
import torch

from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos import B1Z1PACTPos
from rsl_rl.algorithms.ppo_b1z1_pact import _split_privileged_force_prediction


class B1Z1PACTPrivilegedForceOrderTests(unittest.TestCase):
    @staticmethod
    def _make_env(env_type):
        env = env_type.__new__(env_type)
        env.num_envs = 1
        env.simulator = SimpleNamespace(
            _grfs_buf=torch.arange(1.0, 13.0).reshape(1, 12),
        )
        env.ee_force_ext_world = torch.tensor([[31.0, 32.0, 33.0]])
        env.base_force_ext_world = torch.tensor([[21.0, 22.0, 23.0]])
        env.base_torque_ext_world = torch.tensor([[24.0, 25.0, 26.0]])
        env.obs_scales = SimpleNamespace(grf=1.0, ee_force=1.0)
        env.base_wrench_scale = torch.ones(6)
        env._get_base_yaw_quat = lambda: torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        return env

    def test_pact_force_target_matches_pact_pos_order(self):
        pact = self._make_env(B1Z1PACT)
        pact_pos = self._make_env(B1Z1PACTPos)

        pact_target = pact.get_privileged_force_observation()
        pact_pos_target = pact_pos.get_privileged_force_observation()

        self.assertTrue(torch.equal(pact_target, pact_pos_target))
        self.assertTrue(torch.equal(pact_target[:, :12], pact.simulator._grfs_buf))
        self.assertTrue(torch.equal(pact_target[:, 12:18], torch.tensor([[21., 22., 23., 24., 25., 26.]])))
        self.assertTrue(torch.equal(pact_target[:, 18:21], pact.ee_force_ext_world))

    def test_pinn_slices_follow_the_same_order(self):
        values = torch.arange(21.0).reshape(1, 21)
        grfs, base_wrench, ee_force = _split_privileged_force_prediction(values)

        self.assertTrue(torch.equal(grfs, values[:, :12]))
        self.assertTrue(torch.equal(base_wrench, values[:, 12:18]))
        self.assertTrue(torch.equal(ee_force, values[:, 18:21]))


if __name__ == "__main__":
    unittest.main()
