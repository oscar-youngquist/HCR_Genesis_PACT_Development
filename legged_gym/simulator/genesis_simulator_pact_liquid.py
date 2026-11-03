"""Legacy PACT inference plus a physically coupled SPH tank payload.

All control, observations and joint ordering come from the existing inference
adapter. This adapter only adds the liquid scene and reset lifecycle.
"""
from pathlib import Path
import numpy as np
import torch
from legged_gym import gs, LEGGED_GYM_ROOT_DIR
from legged_gym.scripts.liquid_payload_configs import get_payload_config
from legged_gym.utils.math_utils import quat_apply
from .genesis_simulator_pact_nopinn import GenesisSimulator_PACT_NoPINN


def tank_geometry(liquid):
    if liquid.liquid_tank != 'default' and (liquid.liquid_type != 'water' or liquid.liquid_volume != 10):
        raise ValueError('Non-default containers are supported only by the 10 L water presets')
    preset = get_payload_config(liquid.liquid_type, liquid.liquid_volume, liquid.liquid_tank)
    if preset is None:
        raise ValueError("Unsupported liquid volume/container combination")
    scale = np.array([preset['scale_x'], preset['scale_y'], preset['scale_z']])
    # Same tank dimensions and fill presets as the Go1 liquid experiment.
    interior = np.array([.20-.03, .15-.03, .20-.015]) * scale
    size = interior - preset['offset']
    if np.any(size <= 0):
        raise ValueError("Liquid fill dimensions must be positive")
    xy = np.array(preset.get('mount_offset', [0., 0.]))
    bottom = liquid.mount_height + .015 * scale[2]
    return preset, scale, xy, size, bottom, liquid.mount_height + .20 * scale[2] - liquid.lid_overlap


class GenesisSimulator_PACT_Liquid(GenesisSimulator_PACT_NoPINN):
    def _parse_cfg(self):
        super()._parse_cfg()
        # SPH substeps subdivide each physics dt; policy timing is unchanged.
        self.sim_substeps = self._cfg.liquid.substeps

    def _payload_scene_options(self):
        return dict(sph_options=gs.options.SPHOptions(particle_size=self._cfg.liquid.particle_size))

    def _setup_camera(self):
        self._floating_camera = self._scene.add_camera(res=(1280,960),
            pos=tuple(self._cfg.viewer.pos),lookat=tuple(self._cfg.viewer.lookat),
            fov=40,GUI=not self._headless)

    def _add_payload(self):
        cfg = self._cfg.liquid
        preset, scale, xy, size, bottom, top = tank_geometry(cfg)
        self.liquid_properties = preset
        assets = Path(LEGGED_GYM_ROOT_DIR) / 'legged_gym' / 'scripts'
        origin = np.asarray(self._cfg.init_state.pos)
        # attach() interprets the child's morph pose in the parent-link frame;
        # only SPH particle initialization below uses world coordinates.
        # Mesh origins are on the tank bottom and lid underside respectively.
        self._bucket = self._scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(file=str(assets/'water_tank_proper_units_simple.stl'),
                scale=tuple(scale), pos=tuple([*xy, cfg.mount_height]),
                decimate=False, convexify=False),
            surface=gs.surfaces.Glass(opacity=cfg.tank_opacity))
        self._lid = self._scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(file=str(assets/'water_tank_lid.stl'),
                scale=(scale[0], scale[1], 1.), pos=tuple([*xy, top]),
                decimate=False, convexify=False),
            surface=gs.surfaces.Glass(opacity=cfg.tank_opacity))
        self._bucket.attach(self._robot, cfg.base_link)
        self._lid.attach(self._robot, cfg.base_link)
        center = origin + [*xy, bottom+(.20-.015)*scale[2]/2]
        self._liquid = self._scene.add_entity(
            material=gs.materials.SPH.Liquid(rho=preset['rho'],mu=preset['mu'],gamma=preset['gamma']),
            morph=gs.morphs.Box(pos=tuple(center),size=tuple(size)),
            # Opaque coloured particles make sloshing visible through the
            # translucent tank in Genesis' raster viewer; SPH material above
            # still determines all fluid physics.
            surface=gs.surfaces.Rough(color=cfg.fluid_color,vis_mode='particle'))

    def _create_envs(self):
        super()._create_envs()
        # Store body-relative rest positions, avoiding doubled environment
        # offsets when resetting a subset of a batched scene.
        self._liquid_rest_local = (self._liquid.get_particles_pos()-self._robot.get_pos()[:,None,:]).clone()

    def step(self, actions):
        super().step(actions)
        if self._cfg.liquid.particle_markers and (not self._headless or self._cfg.viewer.add_camera):
            # Draw one instanced marker mesh from the actual SPH state. This
            # keeps liquid visible through translucent walls in the raster
            # viewer, whose ordinary glass/particle depth ordering can hide it.
            old = getattr(self, '_liquid_marker', None)
            if old is not None:
                self._scene.clear_debug_object(old)
            positions = self._liquid.get_particles_pos()
            ids = self._cfg.viewer.rendered_envs_idx
            offsets = torch.as_tensor(self._scene.envs_offset,device=positions.device,dtype=positions.dtype)
            visible = (positions[ids]+offsets[ids,None,:]).reshape(-1,3)
            self._liquid_marker = self._scene.draw_debug_spheres(visible,
                radius=self._cfg.liquid.particle_size/2,color=(*self._cfg.liquid.fluid_color,1.))

    def reset_root_states(self, env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel):
        super().reset_root_states(env_ids,base_pos,base_quat,base_lin_vel,base_ang_vel)
        local = self._liquid_rest_local[env_ids]
        quat = base_quat[:,None,:].expand(-1,local.shape[1],-1)
        world = quat_apply(quat.reshape(-1,4),local.reshape(-1,3)).reshape_as(local)+base_pos[:,None,:]
        self._liquid.set_particles_pos(world,envs_idx=env_ids)
        # Remove previous-episode fluid momentum, matching fresh fluid reset.
        self._liquid.set_particles_vel(torch.zeros_like(world),envs_idx=env_ids)
