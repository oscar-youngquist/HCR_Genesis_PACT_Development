import numpy as np
import torch
import genesis.utils.mesh as mu  # adjust if your import path differs
import trimesh


def _build_surface_frame_from_normal(center, normal):
    """
    center : (3,) numpy
    normal : (3,) numpy, should be unit length

    Returns
    -------
    T : (4,4) numpy homogeneous transform
    """
    z_axis = normal / (np.linalg.norm(normal) + 1e-12)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, z_axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    R = np.column_stack((x_axis, y_axis, z_axis))

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = center
    return T


def _create_local_plane_mesh(plane_size_xy=(0.16, 0.16), color=(0.2, 0.7, 1.0, 0.35)):
    """
    Create a flat rectangular patch in the local xy-plane centered at the origin.
    After applying T, it will lie on the surface plane whose normal is the local z-axis.
    """
    sx, sy = plane_size_xy
    hx, hy = sx * 0.5, sy * 0.5

    vertices = np.array([
        [-hx, -hy, 0.0],
        [ hx, -hy, 0.0],
        [ hx,  hy, 0.0],
        [-hx,  hy, 0.0],
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int64)

    visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile((np.asarray(color) * 255).astype(np.uint8), (4, 1))
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=visual,
        process=False,
    )
    return mesh


def _create_world_plane_patch_from_frame(center, normal, plane_size_xy=(0.16, 0.16), color=(0.2, 0.7, 1.0, 0.35)):
    center = np.asarray(center, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = ref - np.dot(ref, normal) * normal
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-12

    sx, sy = plane_size_xy
    hx, hy = 0.5 * sx, 0.5 * sy

    vertices = np.array([
        center - hx * x_axis - hy * y_axis,
        center + hx * x_axis - hy * y_axis,
        center + hx * x_axis + hy * y_axis,
        center - hx * x_axis + hy * y_axis,
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int64)

    visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile((np.asarray(color) * 255).astype(np.uint8), (4, 1))
    )

    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)

# def _create_local_plane_mesh(plane_size_xy=(0.16, 0.16), color=(0.2, 0.7, 1.0, 0.35)):
#     """
#     Create a small local plane patch mesh centered at the origin,
#     lying in the local xy-plane.
#     """
#     vmesh, _ = mu.create_plane(
#         normal=(0.0, 0.0, 1.0),
#         plane_size=plane_size_xy,
#         tile_size=plane_size_xy,
#         color=color,
#     )
#     return vmesh


def _create_sample_point_mesh(radius=0.008, color=(1.0, 0.0, 0.0, 1.0)):
    """
    Create a small sphere mesh for visualizing sampled terrain points.
    """
    return mu.create_sphere(radius=radius, color=color)