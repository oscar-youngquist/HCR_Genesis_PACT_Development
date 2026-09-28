"""Preserve B1Z1 geometry while avoiding Isaac Sim's stalled B1 DAE importer."""

import fcntl
import hashlib
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def _correct_z1_frames(root):
    """Z1 DAE vertices are already link-local Z-up in Isaac Sim.

    Remove the Gym-specific quarter-turn from visuals and mesh collisions.
    Match the existing Genesis/BARD link-frame inertia convention as well;
    joint origins, COM translations, mass and inertia coefficients stay intact.
    Primitive collision rotations are intentional and must not be changed.
    """
    def remove_quarter_turn(origin):
        if origin is None:
            return
        rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
        if all(abs(a-b) < 1e-7 for a,b in zip(rpy, (math.pi/2, 0., 0.))):
            origin.set("rpy", "0 0 0")

    for link in root.findall("link"):
        z1 = False
        for element in link.findall("visual") + link.findall("collision"):
            mesh = element.find("geometry/mesh")
            if mesh is not None and Path(mesh.get("filename", "")).name.startswith("z1_"):
                remove_quarter_turn(element.find("origin"))
                z1 = True
        if z1:
            remove_quarter_turn(link.find("inertial/origin"))


def prepare_lab_urdf(asset_path):
    """Return a cached URDF with the five B1 visual meshes converted to STL.

    No simplification is performed. Correct redundant Z1 mesh/inertia rotations
    for Lab, retaining all joint transforms and primitive collision geometry.
    Source assets are never overwritten; corrected assets use a new cache key.
    """
    source = Path(asset_path).resolve()
    data = source.read_bytes()
    root = ET.fromstring(data, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    _correct_z1_frames(root)
    paths = {entry: (source.parent / entry.attrib["filename"]).resolve()
             for entry in root.iter("mesh")}
    digest = hashlib.sha256(b"b1z1-lab-stl-v3" + str(source).encode() + data)
    for path in sorted(set(paths.values())):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    cache = Path(tempfile.gettempdir()) / "b1z1_isaaclab_assets" / digest.hexdigest()[:20]
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / source.name
    # Concurrent training launches must not consume a partly converted mesh.
    with (cache / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            return str(output)
        import trimesh

        converted = {}
        b1_meshes = {"trunk.dae", "hip.dae", "thigh.dae", "thigh_mirror.dae", "calf.dae"}
        for entry, path in paths.items():
            entry.set("filename", str(path))
        for entry in root.findall("./link/visual/geometry/mesh"):
            path = paths[entry]
            if path.name not in b1_meshes:
                continue
            if path not in converted:
                mesh = trimesh.load(str(path), force="mesh", process=False)
                target = cache / (path.stem + ".stl")
                mesh.export(str(target), file_type="stl")
                converted[path] = target
            entry.set("filename", str(converted[path]))
        temporary = output.with_suffix(".tmp")
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(output)
    return str(output)
