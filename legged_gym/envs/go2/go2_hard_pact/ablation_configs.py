"""Extremely thin configuration selectors for the shared HardPACT task."""

from .ablations import HARD_PACT_VARIANTS
from .go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO


def make_hard_pact_variant_configs(variant_id: str, backend: str):
    """Create config subclasses which select metadata and no task logic."""
    if variant_id not in HARD_PACT_VARIANTS:
        raise ValueError(variant_id)

    class EnvCfg(GO2HardPACTCfg):
        ablation_variant = variant_id
        task_backend = backend

    class PPOCfg(GO2HardPACTCfgPPO):
        ablation_variant = variant_id
        task_backend = backend

        class algorithm(GO2HardPACTCfgPPO.algorithm):
            ablation_variant = variant_id

        class runner(GO2HardPACTCfgPPO.runner):
            run_name = f"hard_pact_{variant_id}_{backend}"
            task_backend = backend

    EnvCfg.__name__ = f"GO2HardPACT{variant_id.title().replace('_', '')}{backend.title()}Cfg"
    PPOCfg.__name__ = EnvCfg.__name__ + "PPO"
    return EnvCfg, PPOCfg
