import unittest

from legged_gym.envs.go2.go2_hard_pact.ablation_configs import (
    GO2HardPACTBaselineCfg,
    GO2HardPACTFullCfg,
    GO2HardPACTHardOnlyCfg,
    GO2HardPACTInverseOnlyCfg,
    GO2HardPACTRolloutOnlyCfg,
    GO2HardPACTSoftOnlyCfg,
    GO2HardPACTSoftPenaltyCfg,
    GO2HardPACTStopGradientQPCfg,
)
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import (
    GO2HardPACTCfg,
    hard_pact_terrain_mesh_type,
)
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import (
    GO2HardPACTPosCfg,
)


class HardPACTConfigTests(unittest.TestCase):
    def test_main_payload_is_capped_at_four_kilograms(self):
        cfg = GO2HardPACTCfg()
        domain_rand = cfg.domain_rand
        self.assertEqual(domain_rand.added_mass_range, [-1.0, 4.0])
        self.assertEqual(domain_rand.max_added_mass_max, 4.0)
        self.assertEqual(cfg.normalization.obs_scales.grf, 0.01)
        self.assertEqual(cfg.normalization.obs_scales.base_wrench, 0.01)

    def test_backend_selects_terrain_representation_without_task_subclass(self):
        self.assertEqual(hard_pact_terrain_mesh_type("genesis"), "heightfield")
        self.assertEqual(hard_pact_terrain_mesh_type("isaaclab"), "trimesh")
        with self.assertRaises(ValueError):
            hard_pact_terrain_mesh_type("isaacgym")

    def test_all_ablations_share_non_ablated_contract(self):
        reference = GO2HardPACTCfg()
        variants = (
            GO2HardPACTBaselineCfg, GO2HardPACTSoftOnlyCfg,
            GO2HardPACTHardOnlyCfg, GO2HardPACTFullCfg,
            GO2HardPACTStopGradientQPCfg, GO2HardPACTSoftPenaltyCfg,
            GO2HardPACTInverseOnlyCfg, GO2HardPACTRolloutOnlyCfg,
        )
        for variant_type in variants:
            variant = variant_type()
            self.assertEqual(variant.env.num_observations, 57)
            self.assertEqual(variant.env.num_explicit_recon_obs, 11)
            self.assertEqual(variant.env.num_reconstruction_obs, 79)
            self.assertEqual(
                variant.terrain.terrain_proportions,
                reference.terrain.terrain_proportions,
            )
            self.assertEqual(
                variant.domain_rand.friction_range,
                reference.domain_rand.friction_range,
            )
            self.assertTrue(variant.features.supervised_physics_head_pretraining)

    def test_factorial_switches_are_exact(self):
        expected = {
            GO2HardPACTBaselineCfg: (False, False, False, False),
            GO2HardPACTSoftOnlyCfg: (True, True, False, False),
            GO2HardPACTHardOnlyCfg: (False, False, True, False),
            GO2HardPACTFullCfg: (True, True, True, False),
            GO2HardPACTStopGradientQPCfg: (True, True, True, True),
            GO2HardPACTSoftPenaltyCfg: (False, False, False, False),
            GO2HardPACTInverseOnlyCfg: (True, False, False, False),
            GO2HardPACTRolloutOnlyCfg: (False, True, False, False),
        }
        for variant_type, switches in expected.items():
            features = variant_type().features
            actual = (
                features.use_bard_inverse_loss,
                features.use_bard_rollout_loss,
                features.use_qp,
                features.stop_gradient_qp,
            )
            self.assertEqual(actual, switches)
        self.assertTrue(
            GO2HardPACTSoftPenaltyCfg().features.use_soft_projection_penalty
        )

    def test_position_pretraining_is_neutral(self):
        features = GO2HardPACTPosCfg().features
        self.assertFalse(features.supervised_physics_head_pretraining)
        self.assertFalse(features.use_bard_inverse_loss)
        self.assertFalse(features.use_bard_rollout_loss)
        self.assertFalse(features.use_qp)


if __name__ == "__main__":
    unittest.main()
