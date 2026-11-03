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
    preset = getattr(liquid, 'sampled_payload', None) or get_payload_config(liquid.liquid_type, liquid.liquid_volume, liquid.liquid_tank)
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
        options = dict(particle_size=self._cfg.liquid.particle_size)
        if getattr(self._cfg.liquid, 'collection', False) or getattr(self._cfg.liquid, 'contact_geometry', 'mesh') == 'boxes':
            # Build terrain once, before SPHOptions fixes the spatial domain.
            # Keep its signed heights: lifting the robot would create a drop.
            from legged_gym.utils.terrain import Terrain
            cfg = self._cfg.terrain
            margin = getattr(self._cfg.liquid, 'domain_margin', 3.)
            if cfg.mesh_type == 'heightfield':
                self._prepared_terrain = Terrain(cfg)
                heights = self._prepared_terrain.height_field_raw * cfg.vertical_scale
                low = [-cfg.border_size-margin, -cfg.border_size-margin, min(0., float(heights.min()))-margin]
                high = [cfg.num_rows*cfg.terrain_length+cfg.border_size+margin,
                        cfg.num_cols*cfg.terrain_width+cfg.border_size+margin,
                        max(0., float(heights.max()))+margin]
            else:
                low, high = [-cfg.plane_length/2-margin]*2+[-margin], [cfg.plane_length/2+margin]*2+[margin]
            self.liquid_domain = dict(lower_bound=low, upper_bound=high)
            options.update(self.liquid_domain)
        return dict(sph_options=gs.options.SPHOptions(**options))

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
        if getattr(cfg, 'contact_geometry', 'mesh') == 'boxes':
            from liquid_collection.payload import box_parts
            self.liquid_attachment_targets = []
            self.liquid_contact_parts = box_parts(preset, cfg.mount_height, cfg.lid_overlap)
            for part in self.liquid_contact_parts:
                entity = self._scene.add_entity(
                    material=gs.materials.Rigid(rho=part.get('rho', 200.),
                        coup_softness=cfg.particle_size/2, sdf_cell_size=.002, sdf_min_res=32, sdf_max_res=256),
                    morph=gs.morphs.Box(size=tuple(part['size']), pos=tuple(part['pos'])),
                    # Explicit color is required for this Genesis rasterizer to
                    # include opacity when constructing the primitive's RGBA texture.
                    surface=gs.surfaces.Glass(color=(.75, .85, 1.), opacity=cfg.tank_opacity))
                entity.attach(self._robot, cfg.base_link)
                self.liquid_attachment_targets.append((entity, part['pos']))
            self._bucket = self.liquid_attachment_targets[0][0]
            self._lid = self.liquid_attachment_targets[-1][0]
        else:
            self._bucket = self._scene.add_entity(
                material=gs.materials.Rigid(),
                morph=gs.morphs.Mesh(file=str(assets/'water_tank_proper_units_simple.stl'),
                    scale=tuple(scale), pos=tuple([*xy, cfg.mount_height]),
                    decimate=False, convexify=False),
                surface=gs.surfaces.Glass(color=(.75, .85, 1.), opacity=cfg.tank_opacity))
            self._lid = self._scene.add_entity(
                material=gs.materials.Rigid(),
                morph=gs.morphs.Mesh(file=str(assets/'water_tank_lid.stl'),
                    scale=(scale[0], scale[1], 1.), pos=tuple([*xy, top]),
                    decimate=False, convexify=False),
                surface=gs.surfaces.Glass(color=(.75, .85, 1.), opacity=cfg.tank_opacity))
            self._bucket.attach(self._robot, cfg.base_link)
            self._lid.attach(self._robot, cfg.base_link)
            self.liquid_attachment_targets = [(self._bucket, [*xy, cfg.mount_height]), (self._lid, [*xy, top])]
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
        if getattr(self._cfg.liquid, 'collection', False):
            self._validate_rest_mesh_clearance()
            self.validate_liquid_spawn(require_rest=True)

    def _validate_rest_mesh_clearance(self):
        # Initialization-only mesh audit also covers the STL's rounded corners
        # and slightly non-planar bottom. No particle host copies occur on reset
        # or in the collection loop; the audited local layout is reused there.
        import trimesh
        cfg = self._cfg.liquid
        if getattr(cfg, 'contact_geometry', 'mesh') == 'boxes':
            from liquid_collection.payload import geometry
            geom = geometry(self.liquid_properties, cfg.mount_height, cfg.lid_overlap)
            p = self._liquid_rest_local
            self.liquid_mesh_clearance = float(torch.minimum(p-p.new_tensor(geom['lower_B']),
                p.new_tensor(geom['upper_B'])-p).amin())
            return
        _, scale, xy, _, _, top = tank_geometry(cfg)
        local = self._liquid_rest_local
        if not torch.allclose(local, local[:1].expand_as(local), atol=1e-5, rtol=0):
            raise RuntimeError('Batched initial particle layouts differ')
        points = local[0].cpu().numpy()
        assets = Path(LEGGED_GYM_ROOT_DIR) / 'legged_gym/scripts'
        minimum = float('inf')
        for filename, factors, translation in (
            ('water_tank_proper_units_simple.stl', scale, [*xy, cfg.mount_height]),
            ('water_tank_lid.stl', [scale[0], scale[1], 1.], [*xy, top])):
            mesh = trimesh.load(assets/filename, force='mesh')
            mesh.apply_scale(factors)
            mesh.apply_translation(translation)
            _, distance, _ = trimesh.proximity.closest_point(mesh, points)
            minimum = min(minimum, float(distance.min()))
            if mesh.contains(points).any() or minimum < cfg.particle_size/2+cfg.safety_margin-1e-5:
                raise RuntimeError('Actual sampled particles violate STL wall/lid radius clearance; reject this payload')
        self.liquid_mesh_clearance = minimum

    def _get_env_origins(self):
        super()._get_env_origins()
        if getattr(self._cfg.liquid, 'collection', False) and self._custom_origins:
            self._terrain_types = torch.randint(self._cfg.terrain.num_cols,
                (self._num_envs,), device=self._device)
            self._env_origins[:] = self._terrain_origins[self._terrain_levels, self._terrain_types]

    def validate_liquid_spawn(self, env_ids=None, require_rest=False):
        """Audit actual solver positions, including negative terrain and partial resets."""
        from liquid_collection.payload import validate_payload
        from liquid_collection.physics import rotate
        cfg = self._cfg.liquid
        geom = validate_payload(self.liquid_properties, cfg.mount_height,
            cfg.particle_size, cfg.safety_margin, cfg.lid_overlap,
            getattr(cfg, 'property_expansion', 0.), getattr(cfg, 'geometry_expansion', 0.))
        pos = self._liquid.get_particles_pos(envs_idx=env_ids)
        base = self._robot.get_pos(envs_idx=env_ids)
        quat = self._robot.get_quat(envs_idx=env_ids)
        local = rotate(quat[:,None], pos-base[:,None], inverse=True)
        rest = self._liquid_rest_local if env_ids is None else self._liquid_rest_local[env_ids]
        # The caller only invokes this for initialization/reset. Exact mesh
        # distances remain valid under the common rigid transform.
        if require_rest and not torch.allclose(local, rest, atol=1e-5, rtol=0):
            raise RuntimeError('Actual reset particle layout differs from mesh-audited rest layout')
        pad = cfg.particle_size/2+cfg.safety_margin
        low, high = local.new_tensor(geom['lower_B'])+pad, local.new_tensor(geom['upper_B'])-pad
        active = self._liquid.get_particles_active(envs_idx=env_ids)
        inside = ((local >= low-1e-5) & (local <= high+1e-5)).all()
        domain = ((pos > pos.new_tensor(self.liquid_domain['lower_bound'])) &
                  (pos < pos.new_tensor(self.liquid_domain['upper_bound']))).all()
        if not bool(torch.isfinite(pos).all() & active.all() & inside & domain):
            raise RuntimeError('Unsafe actual SPH spawn: inactive/nonfinite/outside tank clearance or domain')

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
        if hasattr(self._liquid, 'set_particles_active'):
            self._liquid.set_particles_active(torch.ones(world.shape[:2], device=world.device,
                dtype=torch.bool), envs_idx=env_ids)
        if hasattr(self, '_cfg') and getattr(self._cfg.liquid, 'collection', False):
            # This direct kernel call bypasses the entity setters' index
            # normalization. PyTorch reset IDs are int64; Genesis uses int32.
            self._scene.sim.coupler._kernel_reset_sph(env_ids.to(dtype=gs.tc_int).contiguous())
            self.validate_liquid_spawn(env_ids, require_rest=True)
