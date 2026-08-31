"""Two-environment backend smoke for the shared Go2 HardPACT task core."""

import os
import torch

from legged_gym import SIMULATOR
from legged_gym.envs import task_registry
from legged_gym.utils import get_args, init_genesis


def main():
    args = get_args()
    if "genesis" in SIMULATOR:
        from legged_gym import gs

        init_genesis(args, gs)
    cfg = task_registry.env_cfgs[args.task]
    cfg.seed = args.seed
    # Genesis' PACT visualizer selects environment indices 0 and 1 even when
    # headless, so two is the smallest cross-backend smoke batch.
    cfg.env.num_envs = 2
    cfg.terrain.mesh_type = "heightfield" if SIMULATOR == "genesis" else "trimesh"
    cfg.terrain.curriculum = False
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 6
    cfg.viewer.rendered_envs_idx = [0]
    physics_smoke = os.environ.get("HARD_PACT_SMOKE_PHYSICS") == "1"
    if not physics_smoke:
        cfg.bard.enabled = False
        cfg.bard.required = False
        cfg.features.use_bard_inverse_loss = False
        cfg.features.use_bard_rollout_loss = False
        cfg.features.use_qp = False
        cfg.qp.enabled = False
    cfg.domain_rand.use_domainrand_curriculum = False
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    assert torch.device(env.device).type == "cuda", (
        f"HardPACT smoke tests must run on GPU, got {env.device}"
    )
    observation, critic = env.reset()
    estimator = None
    if physics_smoke:
        from rsl_rl.algorithms import PPOGo2HardPACT
        from rsl_rl.modules import ActorCriticGo2HardPACT

        estimator = ActorCriticGo2HardPACT(
            actor_layers=(32,), critic_layers=(32,), encoder_layers=(32,),
            physics_head_layers=(32,),
        ).to(env.device)
    result = env.step(
        torch.zeros(2, env.policy_action_dim, device=env.device),
        physics_estimator=estimator,
        allow_missing_physics_references=not physics_smoke,
    )
    assert observation.shape == (2, 57)
    assert critic.shape == (2, 198)
    assert result[0].shape == (2, 57)
    assert result[1].shape == (2, 198)
    assert env.last_transition["interval_grf_yaw"].shape == (2, 12)
    assert env.backend_capabilities.name == SIMULATOR
    if physics_smoke:
        assert env._bard is not None
        assert torch.isfinite(env.last_transition["safe_torque"]).all()
        assert env._physics_reference_serial == 1
        update_batch = dict(env.last_transition)
        update_batch.update({
            "observation": env.obs_buf.clone(),
            "history": env.obs_history.clone(),
        })
        algorithm = PPOGo2HardPACT(
            estimator,
            use_bard_inverse_loss=cfg.features.use_bard_inverse_loss,
            use_bard_rollout_loss=cfg.features.use_bard_rollout_loss,
            use_qp=cfg.features.use_qp,
            differentiate_qp=cfg.features.differentiate_qp,
            stop_gradient_qp=cfg.features.stop_gradient_qp,
            lambda_inverse=cfg.bard.lambda_inverse,
            lambda_rollout=cfg.bard.lambda_rollout,
            lambda_projection=cfg.bard.lambda_projection,
            device=env.device,
        )
        outputs = env.recompute_training_outputs(update_batch, estimator)
        recomputed = algorithm._compute_physics_objective(update_batch, outputs)
        auxiliary_outputs = env.recompute_auxiliary_outputs(
            update_batch, estimator
        )
        auxiliary = algorithm._compute_auxiliary_objective(
            update_batch, auxiliary_outputs
        )
        update_loss = recomputed["actor_auxiliary"] + auxiliary["loss"]
        assert torch.isfinite(update_loss)
        estimator.zero_grad(set_to_none=True)
        update_loss.backward()
        assert estimator.physics_estimator.grf_head[-1].weight.grad is not None
        assert torch.isfinite(
            estimator.physics_estimator.grf_head[-1].weight.grad
        ).all()
    print(
        f"SMOKE_OK backend={env.backend_capabilities.name} "
        f"task={args.task} device={env.device} obs=57 critic=198 "
        f"transition_fields={len(env.last_transition)} physics={physics_smoke}"
    )


if __name__ == "__main__":
    main()
