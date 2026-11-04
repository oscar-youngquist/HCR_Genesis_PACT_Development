"""Policy ownership boundary. Future QP adapters must import their own dependencies."""
from abc import ABC, abstractmethod
import ast
from copy import deepcopy
from pathlib import Path


class PolicyAdapter(ABC):
    @abstractmethod
    def initialize(self, job):
        """Own model, environment and controller configuration."""

    @abstractmethod
    def execute(self):
        """Batched inference and one unchanged policy/control interval."""

    @abstractmethod
    def reset(self, ids=None):
        """Reset environment and adapter-owned policy state."""

    @abstractmethod
    def cleanup(self):
        """Release environment, policy and adapter-owned resources."""


def restore_scope(cfg, path):
    from legged_gym.scripts.pact_liquid_checkpoint import _section, _values
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith('Cfg'))
    fallbacks = []
    # Resolve inherited fields from the current robot training class, not the
    # evaluation-only plane/command overrides. Record this explicit assumption.
    for section in ('terrain', 'commands'):
        target = getattr(cfg, section)
        base = type(target).__mro__[1]
        for name in dir(base):
            if not name.startswith('_') and not isinstance(getattr(base, name), type):
                setattr(target, name, deepcopy(getattr(base, name)))
        keys = {k for k in dir(target) if not k.startswith('_') and k not in ('ranges', 'num_subterrains', 'num_sub_terrains')}
        values = _values(_section(node, section), keys)
        for key in keys:
            if key not in values:
                fallbacks.append(f'{section}.{key}: current training-class inherited value')
        for key, value in values.items():
            setattr(target, key, deepcopy(value))
    ranges = _values(_section(node, 'commands', 'ranges'), {'lin_vel_x', 'lin_vel_y', 'ang_vel_yaw', 'heading'})
    if set(ranges) != {'lin_vel_x', 'lin_vel_y', 'ang_vel_yaw', 'heading'}:
        raise ValueError('Collection requires explicit saved training command ranges')
    for key, value in ranges.items():
        setattr(cfg.commands.ranges, key, value)
    cfg.terrain.curriculum = cfg.commands.curriculum = False
    cfg.terrain.reset_out_of_bounds = True  # terminate rather than silently teleport
    return fallbacks


class DACTAdapter(PolicyAdapter):
    def initialize(self, job, visualization=None):
        import torch
        from legged_gym import gs
        import legged_gym.envs
        from legged_gym.utils import task_registry, init_genesis, class_to_dict
        from legged_gym.scripts.play_pact_water import parser, load_policy, apply_reset_termination, PLAY_RESET_TERMINATION
        from legged_gym.scripts.pact_liquid_checkpoint import set_joint_dynamics_midpoints
        from .payload import validate_payload
        self.gs, self.env = gs, None
        selected = job['selected']
        entry = selected['policy']
        args = parser().parse_args(['--robot', entry['robot'], '--headless', '--gpu', 'cuda:0',
            '--num_envs', str(job['num_envs']), '--seed', str(selected['seed']),
            '--checkpoint', entry['checkpoint'], '--config', entry['config']])
        args.task, args.debug = entry['robot']+'_pact_water', False
        if visualization is not None:
            args.headless = visualization['headless']
        if not torch.cuda.is_available():
            raise RuntimeError('cuda:0 unavailable; use the patched genesis_lr environment with GPU access')
        torch.cuda.set_device(0)
        cfg, train = task_registry.get_cfgs(args.task, args)
        self.model, self.checkpoint = load_policy(cfg, train, args, 'cuda:0')
        fallbacks = restore_scope(cfg, entry['config'])
        cfg.liquid.joint_dynamics = set_joint_dynamics_midpoints(cfg)
        cfg.env.num_envs = job['num_envs']
        cfg.env.debug = cfg.env.debug_viz = False
        cfg.viewer.add_camera = cfg.sensor.add_depth = False
        cfg.viewer.rendered_envs_idx = []
        cfg.liquid.particle_markers = False
        cfg.liquid.collection = True
        cfg.liquid.resample_terrain_on_reset = not (job.get('smoke') or job.get('validation_experiment'))
        cfg.liquid.sampled_payload = selected['payload']
        cfg.liquid.cavity_coupling = job.get('cavity_coupling', cfg.liquid.cavity_coupling)
        cfg.liquid.contact_softness_m = job.get('contact_softness_m', cfg.liquid.contact_softness_m)
        cfg.liquid.contact_sdf_cell_m = job.get('contact_sdf_cell_m', .002)
        cfg.liquid.safety_margin = job['safety_margin']
        cfg.liquid.domain_margin = 3.  # metre margin beyond generated terrain, audited at every sample
        cfg.liquid.particle_size = job['particle_size']
        cfg.liquid.substeps = job.get('substeps', cfg.liquid.substeps)
        cfg.liquid.contact_geometry = job.get('contact_geometry', cfg.liquid.contact_geometry)
        cfg.liquid.property_expansion = job.get('property_expansion', 0.)
        cfg.liquid.geometry_expansion = job.get('geometry_expansion', 0.)
        geom = validate_payload(selected['payload'], cfg.liquid.mount_height,
                                cfg.liquid.particle_size, cfg.liquid.safety_margin, cfg.liquid.lid_overlap,
                                cfg.liquid.property_expansion, cfg.liquid.geometry_expansion)
        cfg.terrain.randomized_difficulty_range = (.1, 1.)
        # A small map still samples the policy's existing terrain generator.
        cfg.terrain.num_rows = cfg.terrain.num_cols = max(2, min(10, int(job['num_envs']**.5)))
        if job['smoke']:
            if job['num_envs'] > 4:
                raise ValueError('Smoke test is limited to four environments')
            # Descending stairs are within the existing generator's support.
            if cfg.terrain.terrain_proportions[2] <= 0:
                raise ValueError('Smoke descending terrain is unsupported by this policy')
            cfg.terrain.mesh_type = 'heightfield'
            cfg.terrain.selected = True
            cfg.terrain.terrain_kwargs = dict(type='terrain_utils.pyramid_stairs_terrain',
                step_width=.4, step_height=job.get('smoke_step_height', -.12), platform_size=4.)
            cfg.terrain.border_size = 2.
            cfg.env.episode_length_s = .2
        if job.get('validation_experiment'):
            if cfg.terrain.terrain_proportions[2] <= 0 or cfg.terrain.terrain_proportions[3] <= 0:
                raise ValueError('Validation stairs outside saved terrain support')
            cfg.terrain.mesh_type = 'heightfield'
            cfg.terrain.selected = True
            cfg.terrain.terrain_kwargs = dict(type='terrain_utils.pyramid_stairs_terrain',
                step_width=.4, step_height=job['validation_step_height'], platform_size=4.)
            cfg.terrain.border_size = 2.
        if job.get('episode_seconds') is not None:
            cfg.env.episode_length_s = job['episode_seconds']
        if visualization is not None:
            cfg.viewer.rendered_envs_idx = list(range(job['num_envs']))
            cfg.viewer.add_camera = visualization['save_frame']
            cfg.liquid.particle_markers = True
            cfg.viewer.pos, cfg.viewer.lookat = (1.2, -1.4, 1.1), (0., 0., .2)
            cfg.liquid.tank_opacity = visualization['wall_opacity']
            if visualization['terrain'] == 'plane':
                cfg.terrain.mesh_type = 'plane'
                cfg.terrain.selected = False
                cfg.terrain.measure_heights = False  # flat preview has no heightfield for foot queries
        if cfg.terrain.selected:
            allowed = {'terrain_utils.'+name for name in (
                'random_uniform_terrain', 'pyramid_sloped_terrain', 'pyramid_stairs_terrain',
                'discrete_obstacles_terrain', 'wave_terrain', 'stepping_stones_terrain',
                'gap_terrain', 'pit_terrain')}
            if not isinstance(cfg.terrain.terrain_kwargs, dict) or cfg.terrain.terrain_kwargs.get('type') not in allowed:
                raise ValueError('Saved selected terrain must name a supported terrain_utils generator')
        apply_reset_termination(cfg)
        # Explicit collection duration remains the user-selected episode cap.
        if job.get('episode_seconds') is not None:
            cfg.env.episode_length_s = job['episode_seconds']
        import json
        print('LIQUID_JOINT_DYNAMICS '+json.dumps(cfg.liquid.joint_dynamics), flush=True)
        print('LIQUID_RESET_TERMINATION '+json.dumps({section:{key:getattr(getattr(cfg,section),key)
            for key in values} for section,values in PLAY_RESET_TERMINATION.items()}), flush=True)
        self.cfg = cfg
        self.metadata = dict(config=deepcopy(class_to_dict(cfg)), fallbacks=fallbacks,
            geometry=geom, checkpoint=str(self.checkpoint), device='cuda:0',
            requested_gpu=job.get('gpu', 0), cuda_visible_device=job.get('cuda_visible_device'),
            terrain_generator='legged_gym.utils.terrain.Terrain; source snapshotted',
            terrain_reset_sampling=('least-visited subsection per robot, global coverage tie-break, random ties; no immediate repeats'
                if cfg.liquid.resample_terrain_on_reset and cfg.terrain.mesh_type != 'plane' else 'fixed subsection'),
            terrain_difficulty=('explicit selected terrain_kwargs' if cfg.terrain.selected else
                                'uniform [0.1,1.0] per terrain tile' if cfg.terrain.mesh_type != 'plane' else
                                'flat plane; difficulty sampling unused'),
            command_scope='saved literal ranges; training command curriculum disabled')
        init_genesis(args, gs)
        self.env, _ = task_registry.make_env(args.task, args, env_cfg=cfg)
        if job.get('episodes_per_env') is not None and job.get('episode_seconds') is not None:
            from .budget import episode_steps
            # The legacy timeout uses >, one interval later than the requested
            # cap. Keep its positive normalization denominator, and enforce >=.
            limit = episode_steps(job['episode_seconds'], self.env.dt)
            self.env.max_episode_length = limit
            original_check = self.env.check_termination
            def check_with_time_limit():
                original_check()
                timed_out = self.env.episode_length_buf >= limit
                self.env.time_out_buf |= timed_out
                self.env.reset_buf |= timed_out
            self.env.check_termination = check_with_time_limit
            self.metadata['effective_episode_seconds'] = limit*self.env.dt
        if job.get('validation_experiment'):
            from .validation.commands import install
            install(self.env, aggressive=job.get('containment_experiment', False))
            self.validation_commands = True
        self.reset()
        if job.get('episodes_per_env') is not None:
            # The player's reset includes one warmup step, before recording starts.
            self.env.episode_length_buf.zero_()
        return self.env

    def execute(self):
        obs, history, _, _ = self.env.get_observations()
        if getattr(self, 'validation_commands', False):
            import torch
            torch.testing.assert_close(obs[:, :3], self.env.commands[:, :3]*self.env.commands_scale)
        actions = self.model.act_inference(obs, history)
        return self.env.step(actions)

    def reset(self, ids=None):
        # ActorCritic_PACT is feedforward over environment-owned observation history.
        if ids is None:
            return self.env.reset()
        return self.env.reset_idx(ids)

    def cleanup(self):
        if getattr(self, 'env', None) is not None:
            self.env.shutdown_asynic_pino_workers()
        if hasattr(self, 'gs'):
            self.gs.destroy()
        self.env = self.model = None
