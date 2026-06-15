import cv2 as cv
import numpy as np
import torch
import trimesh
import warp as wp

from legged_gym.simulator.genesis_simulator_kite import GenesisSimulator_KITE
from legged_gym.utils.math_utils import (
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
)
from legged_gym.warp.warp_cam import WarpCam


class GenesisSimulator_KITE_Depth(GenesisSimulator_KITE):
    """KITE position-control simulator with Warp-based terrain depth cameras."""

    def __init__(self, cfg, sim_params: dict, device, headless):
        super().__init__(cfg, sim_params, device, headless)
        wp.init()
        self._create_warp_env()
        self._create_warp_tensors()
        self._depth_camera_sensor = WarpCam(
            self._warp_tensor_dict,
            self._num_envs,
            self._cfg.sensor,
            self._mesh_ids,
            self._device,
        )
        self._refresh_camera_pose()
        self._update_depth_images(force=True)

    def _parse_cfg(self):
        super()._parse_cfg()
        if self._cfg.sensor.add_depth:
            self._depth_image_update_counter = 0
            self._depth_image_update_decimation = max(
                1, self._cfg.sensor.depth_camera_config.decimation
            )

    def _setup_depth_camera(self):
        # Warp owns depth rendering for this simulator.
        return

    def _create_sim(self):
        cfg = self._cfg.sensor.depth_camera_config
        env_id = int(cfg.debug_camera_env_id)
        if (
            cfg.debug_draw_camera_position
            and env_id not in self._cfg.viewer.rendered_envs_idx
        ):
            self._cfg.viewer.rendered_envs_idx.append(env_id)
        super()._create_sim()

    def _init_buffers(self):
        super()._init_buffers()
        if self._cfg.sensor.add_depth:
            cfg = self._cfg.sensor.depth_camera_config
            point_dims = 3 if cfg.return_pointcloud else 0
            shape = (
                self._num_envs,
                cfg.num_history,
                *cfg.resolution,
            )
            if point_dims:
                shape += (point_dims,)
            self.depth_images = torch.zeros(
                shape, device=self._device, dtype=torch.float
            )

    def post_physics_step(self):
        super().post_physics_step()
        self._refresh_camera_pose()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel():
            self.depth_images[env_ids] = 0.0
            self._depth_camera_sensor.reset(env_ids)

    def update_depth_images(self):
        return self.update_sensors()

    def update_sensors(self):
        if not self._cfg.sensor.add_depth:
            return
        if self._depth_image_update_counter % self._depth_image_update_decimation == 0:
            self._update_depth_images()
        self._depth_image_update_counter += 1

    def _update_depth_images(self, force=False):
        if not force and not self._cfg.sensor.add_depth:
            return
        pixels = self._depth_camera_sensor.update().clone()
        if self.depth_images.shape[1] > 1:
            self.depth_images[:, 1:] = self.depth_images[:, :-1].clone()
        self.depth_images[:, 0] = pixels[:, 0]
        self._show_selected_depth_image()

    def draw_debug_vis(self, ref_key_body_pos=None):
        super().draw_debug_vis()
        cfg = self._cfg.sensor.depth_camera_config
        env_id = int(cfg.debug_camera_env_id)
        if cfg.debug_draw_camera_position and 0 <= env_id < self._num_envs:
            new_marker = self._scene.draw_debug_spheres(
                self._sensor_pos_tensor[env_id : env_id + 1],
                radius=cfg.debug_camera_marker_radius,
                color=tuple(cfg.debug_camera_marker_color),
            )
            previous_marker = getattr(
                self, "_camera_position_debug_object", None
            )
            if previous_marker is not None:
                self._scene.clear_debug_object(previous_marker)
            self._camera_position_debug_object = new_marker

    def calc_feet_near_edge(self):
        if self._cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                (self._num_envs, len(self._feet_indices)),
                device=self._device,
                dtype=torch.bool,
            )
        feet_xy = self._feet_pos[:, :, :2]
        points = (
            (feet_xy + self._cfg.terrain.border_size)
            / self._cfg.terrain.horizontal_scale
        ).long()
        px = points[:, :, 0].clamp(0, self._edge_mask.shape[0] - 1)
        py = points[:, :, 1].clamp(0, self._edge_mask.shape[1] - 1)
        threshold = self._cfg.rewards.feet_edge_threshold
        near_edge = torch.zeros_like(px, dtype=torch.bool)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                ex = (px + dx).clamp(0, self._edge_mask.shape[0] - 1)
                ey = (py + dy).clamp(0, self._edge_mask.shape[1] - 1)
                edge = self._edge_mask[ex, ey]
                edge_xy = torch.stack(
                    (
                        ex.float() * self._cfg.terrain.horizontal_scale
                        - self._cfg.terrain.border_size,
                        ey.float() * self._cfg.terrain.horizontal_scale
                        - self._cfg.terrain.border_size,
                    ),
                    dim=-1,
                )
                near_edge |= edge & (torch.norm(feet_xy - edge_xy, dim=-1) < threshold)
        return near_edge

    def _create_heightfield(self):
        super()._create_heightfield()
        self._edge_mask = torch.as_tensor(
            self._terrain.edge_mask, device=self._device, dtype=torch.bool
        )

    def _create_warp_env(self):
        terrain_mesh = self._gs_terrain.geoms[0].get_trimesh()
        terrain_mesh = terrain_mesh.copy()
        terrain_mesh.apply_transform(
            trimesh.transformations.translation_matrix(
                (
                    -self._cfg.terrain.border_size,
                    -self._cfg.terrain.border_size,
                    0.0,
                )
            )
        )
        vertices = torch.as_tensor(
            terrain_mesh.vertices,
            dtype=torch.float32,
            device=self._device,
        )
        faces = np.asarray(terrain_mesh.faces, dtype=np.int32).reshape(-1)
        warp_vertices = wp.from_torch(vertices, dtype=wp.vec3)
        warp_faces = wp.from_numpy(faces, dtype=wp.int32, device=self._device)
        self._warp_mesh = wp.Mesh(points=warp_vertices, indices=warp_faces)
        self._mesh_ids = wp.array(
            [self._warp_mesh.id], dtype=wp.uint64, device=self._device
        )

    def _create_warp_tensors(self):
        cfg = self._cfg.sensor.depth_camera_config
        point_dims = 3 if cfg.return_pointcloud else 0
        image_shape = (
            self._num_envs,
            cfg.num_sensors,
            *cfg.resolution,
        )
        if point_dims:
            image_shape += (point_dims,)
        self._depth_image_tensor_warp = torch.zeros(
            image_shape, dtype=torch.float32, device=self._device
        )
        self._sensor_pos_tensor = torch.zeros(
            (self._num_envs, 3), device=self._device
        )
        self._sensor_quat_tensor = torch.zeros(
            (self._num_envs, 4), device=self._device
        )
        self._sensor_nominal_offset_pos = torch.tensor(
            cfg.pos, device=self._device, dtype=torch.float
        ).repeat(self._num_envs, 1)
        self._sensor_nominal_euler = torch.tensor(
            cfg.euler, device=self._device, dtype=torch.float
        ).repeat(self._num_envs, 1)
        self._sensor_offset_pos = self._sensor_nominal_offset_pos.clone()
        self._sensor_offset_euler = self._sensor_nominal_euler.clone()
        self._sensor_offset_quat = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float
        )
        self.resample_camera_mount(
            torch.arange(self._num_envs, device=self._device)
        )

        self._warp_tensor_dict = {
            "depth_image_tensor": self._depth_image_tensor_warp,
            "device": self._device,
            "num_envs": self._num_envs,
            "num_sensors": cfg.num_sensors,
            "sensor_pos_tensor": self._sensor_pos_tensor,
            "sensor_quat_tensor": self._sensor_quat_tensor,
            "mesh_ids": self._mesh_ids,
        }

    def resample_camera_mount(self, env_ids):
        """Resample configured camera mounting offsets for selected robots."""
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        self._sensor_offset_pos[env_ids] = self._sensor_nominal_offset_pos[
            env_ids
        ]
        self._sensor_offset_euler[env_ids] = self._sensor_nominal_euler[
            env_ids
        ]

        if self._cfg.domain_rand.randomize_camera_pos:
            pos_range = torch.tensor(
                self._cfg.domain_rand.camera_com_displacement_range,
                device=self._device,
            )
            self._sensor_offset_pos[env_ids] += (
                2.0 * torch.rand(
                    (env_ids.numel(), 3), device=self._device
                )
                - 1.0
            ) * pos_range
        if self._cfg.domain_rand.randomize_camera_euler:
            euler_range = torch.tensor(
                self._cfg.domain_rand.camera_euler_offset_range,
                device=self._device,
            )
            self._sensor_offset_euler[env_ids] += (
                2.0 * torch.rand(
                    (env_ids.numel(), 3), device=self._device
                )
                - 1.0
            ) * euler_range

        sensor_euler = self._sensor_offset_euler[env_ids]
        self._sensor_offset_quat[env_ids] = quat_from_euler_xyz(
            sensor_euler[:, 0],
            sensor_euler[:, 1],
            sensor_euler[:, 2],
        )
        self._refresh_camera_pose()

    def _refresh_camera_pose(self):
        base_quat = self._base_quat
        self._sensor_quat_tensor[:] = quat_mul(
            base_quat, self._sensor_offset_quat
        )
        self._sensor_pos_tensor[:] = self._base_pos + quat_apply(
            base_quat, self._sensor_offset_pos
        )

    def _show_selected_depth_image(self):
        cfg = self._cfg.sensor.depth_camera_config
        if self._headless or not cfg.debug_render_depth_image:
            return
        env_id = int(cfg.debug_camera_env_id)
        if not 0 <= env_id < self._num_envs:
            return
        depth = self.depth_images[env_id, 0]
        near, far = cfg.near_clip, cfg.far_clip
        pixels = (
            (depth.clamp(near, far) - near) / max(far - near, 1e-6) * 255.0
        ).byte().cpu().numpy()
        cv.imshow(f"KITE Depth Camera env {env_id}", pixels)
        cv.waitKey(1)
