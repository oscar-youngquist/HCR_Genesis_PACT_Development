"""Preserve B1Z1 geometry while avoiding Isaac Sim's stalled B1 DAE importer."""

import fcntl
import hashlib
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def prepare_lab_urdf(asset_path):
    """Return a cached URDF with the five B1 visual meshes converted to STL.

    No simplification is performed. Joint, collision, inertial, and visual
    origin/scale fields are preserved; source assets are never overwritten.
    """
    source = Path(asset_path).resolve()
    data = source.read_bytes()
    root = ET.fromstring(data, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    paths = {entry: (source.parent / entry.attrib["filename"]).resolve()
             for entry in root.iter("mesh")}
    digest = hashlib.sha256(b"b1z1-lab-stl-v1" + str(source).encode() + data)
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
