"""Focused plumbing and optional integration tests for B1Z1 BARD dynamics."""

from __future__ import annotations

import importlib.util
import os
import unittest

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.dynamics import (
    BardB1Z1DynamicsBackend,
    PinocchioWholeBodyDynamics,
)
from rsl_rl.algorithms.ppo_b1z1_pact import (
    _normalized_velocity_rollout_loss,
)
from rsl_rl.storage.rollout_storage_b1z1_pact import RolloutStorageB1Z1PACT


DOF_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle",
    "z1_forearm_roll", "z1_wrist_rotate", "z1_jointGripper",
]
FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
URDF = os.path.join(
    LEGGED_GYM_ROOT_DIR,
    "resources/robots/b1z1_current/urdf/b1z1_genesis.urdf",
)
HAS_BARD_CUDA = importlib.util.find_spec("bard") is not None and torch.cuda.is_available()


class B1Z1RolloutPlumbingTests(unittest.TestCase):
    def test_velocity_blocks_are_normalized_and_terminal_masked(self):
        target = torch.zeros(3, 25)
        prediction = torch.ones_like(target)
        prediction[2] = 1000.0
        valid = torch.tensor([[1.0], [1.0], [0.0]])
        scales = {
            "base_linear": 1.0,
            "base_angular": 2.0,
            "leg": 4.0,
            "arm": 5.0,
        }
        loss, blocks = _normalized_velocity_rollout_loss(
            prediction, target, valid, scales
        )
        self.assertAlmostEqual(blocks["base_linear"].item(), 1.0)
        self.assertAlmostEqual(blocks["base_angular"].item(), 0.25)
        self.assertAlmostEqual(blocks["leg"].item(), 1.0 / 16.0)
        self.assertAlmostEqual(blocks["arm"].item(), 1.0 / 25.0)
        self.assertAlmostEqual(
            loss.item(), sum(value.item() for value in blocks.values()) / 4.0
        )

    def test_storage_keeps_pre_and_post_state_aligned(self):
        storage = RolloutStorageB1Z1PACT(
            2, 1, 3, 4, 5, 6, 7, 8, 180, rollout_state_dim=51
        )
        transition = RolloutStorageB1Z1PACT.Transition()
        for name, width in (
            ("observations", 3), ("critic_observations", 4),
            ("histories", 5), ("actions", 6), ("mu", 6), ("sigma", 6),
            ("values", 1), ("log_probs", 1), ("explicit_targets", 7),
            ("next_privileged", 8), ("dynamics_state", 180),
            ("rollout_initial_state", 51),
        ):
            value = torch.zeros(2, width)
            value[:, 0] = torch.arange(2)
            setattr(transition, name, value)
        transition.rewards = torch.zeros(2)
        transition.dones = torch.zeros(2, dtype=torch.bool)
        storage.add(transition)
        batch = next(storage.mini_batches(1, 1))
        post_ids = batch["dynamics_state"][:, 0]
        pre_ids = batch["rollout_initial_state"][:, 0]
        self.assertTrue(torch.equal(post_ids, pre_ids))


@unittest.skipUnless(HAS_BARD_CUDA, "official BARD and CUDA are required")
class B1Z1BardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.bard = BardB1Z1DynamicsBackend(
            URDF, DOF_NAMES, FOOT_NAMES, "ee_gripper_link", "trunk",
            device=cls.device, batch_capacity=4,
        )
        cls.pin = PinocchioWholeBodyDynamics(
            URDF, DOF_NAMES, FOOT_NAMES, "ee_gripper_link", "trunk"
        )

    def _state(self, batch=2):
        device = self.device
        return (
            torch.zeros(batch, 3, device=device),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).expand(batch, -1),
            torch.zeros(batch, 19, device=device),
            torch.zeros(batch, 3, device=device),
            torch.zeros(batch, 3, device=device),
            torch.zeros(batch, 19, device=device),
        )

    def test_crba_rnea_and_contacts_match_pinocchio(self):
        state = self._state()
        grfs = torch.randn(2, 4, 3, device=self.device)
        ee = torch.randn(2, 3, device=self.device)
        base = torch.randn(2, 6, device=self.device)
        bard_terms = self.bard.evaluate(*state, grfs, ee, base)
        pin_terms = self.pin.evaluate(*state, grfs, ee, base)
        # Pinocchio returns native model order; compare after mapping it to the
        # simulator order exposed by the BARD backend.
        order = torch.tensor(
            list(range(6)) + self.pin.pino_velocity_indices,
            device=self.device,
        )
        pin_mass = pin_terms.mass_matrix.index_select(1, order).index_select(2, order)
        pin_bias = pin_terms.bias.index_select(1, order)
        pin_contacts = pin_terms.generalized_contacts.index_select(1, order)
        torch.testing.assert_close(
            bard_terms.mass_matrix, pin_mass, rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(
            bard_terms.bias, pin_bias, rtol=3e-3, atol=3e-3
        )
        torch.testing.assert_close(
            bard_terms.generalized_contacts,
            pin_contacts,
            rtol=3e-3,
            atol=3e-3,
        )

    def test_force_and_torque_paths_preserve_cuda_autograd(self):
        state = self._state()
        grfs = torch.randn(
            2, 4, 3, device=self.device, requires_grad=True
        )
        ee = torch.randn(2, 3, device=self.device, requires_grad=True)
        base = torch.randn(2, 6, device=self.device, requires_grad=True)
        terms = self.bard.evaluate(*state, grfs, ee, base)
        terms.generalized_contacts.square().mean().backward()
        self.assertIsNotNone(grfs.grad)
        self.assertIsNotNone(ee.grad)
        self.assertIsNotNone(base.grad)

        tau = torch.randn(2, 25, device=self.device, requires_grad=True)
        qdd = self.bard.forward_dynamics(
            *state, tau, grfs.detach(), ee.detach(), base.detach()
        )
        qdd.square().mean().backward()
        self.assertIsNotNone(tau.grad)
        self.assertEqual(qdd.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
