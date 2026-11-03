"""Visual or headless liquid transport with legacy Go1 or Go2 PACT policies.

    bash legged_gym/scripts/play_pact_water.sh --robot go1 --follow_robot
    bash legged_gym/scripts/play_pact_water.sh --robot go2 --headless --num_envs 16 --log_path liquid.csv

Liquid litres select the legacy Go1 fill presets; they are nominal labels, not
newly calibrated SPH mass measurements. No optimizer or curriculum is advanced.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path


# Manual visualization-only overrides for BOTH robots. None retains the saved
# checkpoint config (or task default when the saved file omits that setting).
# Heights are terrain-relative metres; angles are radians; durations seconds.
PLAY_RESET_TERMINATION = {
    'termination': {
        'termination_terms': None,  # e.g. ['roll','pitch','height_min','height_max']; [] disables these terms
        'roll_threshold': None,     # e.g. 0.70
        'pitch_threshold': None,    # e.g. 1.00
        'height_min': None,         # e.g. 0.20
        'height_max': None,         # e.g. 1.50
    },
    'env': {
        'fail_to_terminal_time_s': None,  # e.g. 0.0 for immediate failure termination
        'episode_length_s': None,        # e.g. 20.0
    },
    'asset': {
        'terminate_after_contacts_on': ['base'],  # e.g. ['base']; [] disables contact termination
    },
    'terrain': {
        'reset_out_of_bounds': None,    # True/False
    },
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=['go1','go2'], help='Robot to visualize; defaults to Go2')
    p.add_argument('--task', choices=['go1_pact_water','go2_pact_water'])
    p.add_argument('--load_run', help='Run directory name or absolute path')
    p.add_argument('--ckpt', type=int, default=-1, help='Checkpoint iteration; -1 selects latest')
    p.add_argument('--checkpoint', type=Path, help='Explicit checkpoint file')
    p.add_argument('--config', type=Path, help='Saved training config; default: discover beside checkpoint')
    p.add_argument('--activation', help='Explicit activation override when checkpoint has no saved config')
    p.add_argument('--num_envs', type=int, default=1)
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--duration', type=float, default=60., help='Simulated seconds')
    p.add_argument('--steps', type=int, help='Override duration with a control-step count')
    p.add_argument('--liquid_type', choices=['water','oil','gas'], default='water')
    p.add_argument('--liquid_volume', type=int, choices=[2,4,6,8,10,12], default=4)
    p.add_argument('--liquid_tank', choices=['default','wide','tall','offset'], default='default')
    p.add_argument('--terrain_type', choices=['plane','rough'], default='plane')
    p.add_argument('--wall_opacity', type=float, default=.25, help='Container opacity: 0 transparent, 1 opaque')
    p.add_argument('--fixed_cmd', type=float, nargs=3, metavar=('VX','VY','WZ'))
    p.add_argument('--follow_robot', action='store_true')
    p.add_argument('--record_frames', type=Path, metavar='VIDEO_MP4')
    p.add_argument('--use_joystick', action='store_true')
    p.add_argument('--joystick_type', choices=['xbox','switch'], default='xbox')
    p.add_argument('--log_path', type=Path, help='Stream per-environment metrics to CSV')
    p.add_argument('--smoke', action='store_true', help='Check finite states and a partial liquid reset')
    return p


def configure(cfg, args):
    from copy import deepcopy
    from legged_gym.scripts.pact_liquid_checkpoint import set_joint_dynamics_midpoints
    cfg.liquid.joint_dynamics = set_joint_dynamics_midpoints(cfg)
    if not math.isfinite(args.wall_opacity) or not 0 <= args.wall_opacity <= 1:
        raise ValueError('--wall_opacity must be between 0 and 1')
    cfg.liquid.tank_opacity = args.wall_opacity
    cfg.env.num_envs = args.num_envs
    cfg.viewer.rendered_envs_idx = list(range(args.num_envs))
    cfg.viewer.add_camera = args.follow_robot or bool(args.record_frames)
    cfg.liquid.liquid_type, cfg.liquid.liquid_volume = args.liquid_type, args.liquid_volume
    cfg.liquid.liquid_tank = args.liquid_tank
    if args.terrain_type == 'rough':
        cfg.terrain.mesh_type = 'heightfield'
        cfg.terrain.num_rows = cfg.terrain.num_cols = 2
        cfg.terrain.max_init_terrain_level = 0
        cfg.terrain.selected = cfg.terrain.measure_heights = True
        cfg.terrain.terrain_kwargs = dict(type='terrain_utils.random_uniform_terrain',
            min_height=-.05,max_height=.05,step=.005,downsampled_scale=.2)
    for section,values in PLAY_RESET_TERMINATION.items():
        for key,value in values.items():
            if value is not None:
                setattr(getattr(cfg,section),key,deepcopy(value))


def load_policy(cfg, train, args, device):
    import torch
    from legged_gym import LEGGED_GYM_ROOT_DIR
    from legged_gym.utils import get_load_path
    from rsl_rl.modules import ActorCritic_PACT
    from legged_gym.scripts.pact_liquid_checkpoint import restore_inference_config, infer_policy_contract
    root = Path(LEGGED_GYM_ROOT_DIR)/'logs'/'pact_corl'/train.runner.experiment_name
    checkpoint = args.checkpoint or Path(get_load_path(str(root),
        load_run=args.load_run or train.runner.load_run,checkpoint=args.ckpt))
    checkpoint = checkpoint.resolve()
    # Inference needs only the actor/critic state. Do not instantiate PPO,
    # allocate rollout storage, restore optimizers or load a bootstrap model.
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
    source = restore_inference_config(cfg,train,checkpoint,cfg.asset.name,getattr(args,'config',None))
    contract = infer_policy_contract(state)
    if cfg.env.num_observations != contract['observations']:
        raise ValueError('Checkpoint observation schema does not match the selected robot task')
    cfg.env.num_obs_hist = contract['history']
    c = train.policy
    c.actor_layers,c.critic_layers = contract['actor_layers'],contract['critic_layers']
    c.cenet_enc_layers = contract['encoder_layers']
    c.cenet_enc_latent_dim,c.cenet_velo_dim = contract['latent'],contract['explicit']
    c.activation = getattr(args,'activation',None) or c.activation
    model = ActorCritic_PACT(contract['observations'],contract['critic_inputs'],contract['actions'],
        c.actor_layers,c.critic_layers,contract['observations']*contract['history'],
        c.cenet_enc_latent_dim,c.cenet_velo_dim,c.cenet_enc_layers,c.activation,c.init_noise_std).to(device)
    try:
        model.load_state_dict(state,strict=True)
    except RuntimeError as exc:
        raise ValueError(f'{checkpoint} is incompatible with the legacy coupled PACT architecture') from exc
    contract.update(checkpoint=str(checkpoint),source_config=source,activation=c.activation)
    if source is None and not getattr(args,'activation',None):
        import warnings
        warnings.warn('No saved config found; using current task activation and physical/observation scales. Use --config to restore them.')
    print('LIQUID_POLICY_CONTRACT '+json.dumps(contract),flush=True)
    model.eval().requires_grad_(False)
    return model, checkpoint


def main(args):
    if args.num_envs < 1 or not math.isfinite(args.duration) or args.duration <= 0 or (args.steps is not None and args.steps < 1):
        raise ValueError('Environment count, duration and step count must be positive')
    task_robot = args.task.split('_')[0] if args.task else None
    if args.robot and task_robot and args.robot != task_robot:
        raise ValueError('--robot and --task must select the same robot')
    args.robot = args.robot or task_robot or 'go2'
    args.task = f'{args.robot}_pact_water'
    os.environ.setdefault('SIMULATOR','genesis_pact_water')
    if os.environ['SIMULATOR'] != 'genesis_pact_water':
        raise ValueError('Use SIMULATOR=genesis_pact_water (or the supplied launcher)')
    import torch
    from legged_gym import gs
    import legged_gym.envs  # register tasks
    from legged_gym.utils import task_registry, init_genesis, configure_runtime_device
    # Mask/remap the requested physical GPU before checkpoint tensors allocate
    # CUDA memory; Genesis otherwise resolves some operations on local cuda:0.
    configure_runtime_device(args)
    args.debug = False
    device = 'cpu' if args.cpu else args.gpu
    cfg, train = task_registry.get_cfgs(args.task,args)
    model, checkpoint = load_policy(cfg,train,args,device)
    configure(cfg,args)  # explicit evaluation CLI overrides saved training settings
    print('LIQUID_JOINT_DYNAMICS '+json.dumps(cfg.liquid.joint_dynamics),flush=True)
    print('LIQUID_RESET_TERMINATION '+json.dumps({section:{key:getattr(getattr(cfg,section),key)
        for key in values} for section,values in PLAY_RESET_TERMINATION.items()}),flush=True)
    init_genesis(args,gs)
    env = None
    log = None
    try:
        env,_ = task_registry.make_env(args.task,args,env_cfg=cfg)
        joystick = None
        if args.use_joystick:
            from legged_gym.scripts.joystick import Joystick
            joystick = Joystick(args.joystick_type)
        if args.record_frames:
            args.record_frames.parent.mkdir(parents=True,exist_ok=True)
            env.simulator._floating_camera.start_recording()
        if args.log_path:
            args.log_path.parent.mkdir(parents=True,exist_ok=True)
            log = args.log_path.open('w',newline='')
            writer = csv.writer(log)
            writer.writerow(['time_s','env','cmd_vx','cmd_vy','cmd_wz','vx','vy','wz','height_m','reward','done'])
        steps = args.steps or math.ceil(args.duration/cfg.control.dt)
        # Genesis' camera transforms use TorchScript routines that cannot
        # consume inference tensors; no_grad keeps inference graph-free.
        with torch.no_grad():
            env.reset()
            if args.smoke:
                verify_controller_parameters(env,cfg)
                fluid = env.simulator._liquid.get_particles_pos()
                relative = fluid-env.simulator.base_pos[:,None,:]
                active = env.simulator._liquid.get_particles_active()
                assert active.all(), 'Inactive liquid particles after reset'
                print('LIQUID_RESET_BOUNDS '+json.dumps(dict(
                    minimum=relative.amin(1).cpu().tolist(),maximum=relative.amax(1).cpu().tolist(),
                    active=int(active.sum()),velocity_max=float(env.simulator._liquid.get_particles_vel().abs().max()))),flush=True)
            for step in range(steps):
                if args.fixed_cmd is not None:
                    env.commands[:,:3] = env.commands.new_tensor(args.fixed_cmd)
                if joystick is not None:
                    joystick.update()
                    env.commands[:,:3] = env.commands.new_tensor([-joystick.ly,-joystick.lx,-joystick.rx])
                obs,history,_,_ = env.get_observations()
                actions = model.act_inference(obs,history)
                command = env.commands[:,:3].clone() if log is not None else None
                result = env.step(actions)
                if args.smoke:
                    for value in (result[0],result[4],result[-1],actions,
                                  env.simulator._liquid.get_particles_pos()):
                        assert torch.isfinite(value).all(), 'Nonfinite liquid rollout'
                    from legged_gym.utils.math_utils import quat_rotate_inverse
                    for entity,position_B in env.simulator.liquid_attachment_targets:
                        local = quat_rotate_inverse(env.simulator.base_quat,
                            entity.get_pos()-env.simulator.base_pos)
                        torch.testing.assert_close(local,local.new_tensor(position_B).expand_as(local),
                            rtol=0,atol=1e-4)
                if cfg.viewer.add_camera:
                    position = env.simulator.base_pos[0].cpu().numpy()
                    import numpy as np
                    if args.follow_robot:
                        env.set_camera(position+np.array(cfg.viewer.pos),position+np.array(cfg.viewer.lookat))
                    env.simulator._floating_camera.render()
                if log is not None:
                    rows = torch.cat((command,env.simulator.base_lin_vel[:,:2],
                        env.simulator.base_ang_vel[:,2:3],env.simulator.base_pos[:,2:3],
                        result[4][:,None],result[5][:,None]),1).cpu().tolist()
                    writer.writerows([[(step+1)*cfg.control.dt,i,*row] for i,row in enumerate(rows)])
            if args.smoke:
                liquid = env.simulator._liquid
                before = liquid.get_particles_pos().clone()
                env.reset_idx(torch.tensor([0],device=env.device))
                after = liquid.get_particles_pos()
                assert torch.isfinite(after).all()
                if args.num_envs > 1:
                    torch.testing.assert_close(before[1:],after[1:],rtol=0,atol=0)
                assert liquid.get_particles_vel()[0].eq(0).all()
                verify_controller_parameters(env,cfg)
        print(json.dumps(dict(checkpoint=str(checkpoint),task=args.task,environments=args.num_envs,
            steps=steps,liquid=args.liquid_type,nominal_litres=args.liquid_volume,
            particles_per_env=env.simulator._liquid.n_particles,smoke_passed=args.smoke)),flush=True)
        if args.record_frames:
            env.simulator._floating_camera.stop_recording(save_to_filename=str(args.record_frames),fps=30)
    finally:
        if log is not None:
            log.close()
        if env is not None:
            env.shutdown_asynic_pino_workers()
        gs.destroy()


def verify_controller_parameters(env,cfg):
    """Smoke-only checks of actual simulator values, including after reset."""
    import torch
    sim = env.simulator
    expected = sim.default_dof_pos.new_tensor([cfg.init_state.default_joint_angles[n] for n in cfg.asset.dof_names])
    actual_defaults = sim.default_dof_pos.reshape(-1,12)
    torch.testing.assert_close(actual_defaults,expected.expand_as(actual_defaults),rtol=0,atol=1e-6)
    for name, getter in (('armature','get_dofs_armature'),('friction','get_dofs_frictionloss'),
                         ('stiffness','get_dofs_stiffness'),('damping','get_dofs_damping')):
        setting = cfg.liquid.joint_dynamics[name]
        if setting['enabled']:
            actual = getattr(sim._robot,getter)(sim._dof_indices)
            torch.testing.assert_close(actual,torch.full_like(actual,setting['midpoint']),rtol=0,atol=1e-6)


if __name__ == '__main__':
    main(parser().parse_args())
