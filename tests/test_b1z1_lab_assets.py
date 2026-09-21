"""The Lab import workaround must preserve robot physics and mesh geometry."""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "b1z1_lab_assets", ROOT / "legged_gym/simulator/b1z1_lab_assets.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.mark.parametrize("filename", ["b1z1.urdf", "b1z1_genesis.urdf"])
def test_cached_conversion_preserves_physics_and_full_meshes(filename):
    source = ROOT / "resources/robots/b1z1_current/urdf" / filename
    original_bytes = source.read_bytes()
    output = Path(assets.prepare_lab_urdf(source))
    original, cached = ET.fromstring(original_bytes), ET.parse(output).getroot()
    for mesh in original.iter("mesh"):
        mesh.set("filename", str((source.parent / mesh.attrib["filename"]).resolve()))
    assert [ET.tostring(j) for j in original.findall("joint")] == [
        ET.tostring(j) for j in cached.findall("joint")
    ]
    for before, after in zip(original.findall("link"), cached.findall("link")):
        assert before.attrib == after.attrib
        for tag in ("inertial", "collision"):
            assert [ET.tostring(e) for e in before.findall(tag)] == [
                ET.tostring(e) for e in after.findall(tag)
            ]
    checked = set()
    for before, after in zip(original.findall("./link/visual/geometry/mesh"),
                             cached.findall("./link/visual/geometry/mesh")):
        path = Path(after.attrib["filename"])
        assert path.is_file()
        assert before.get("scale") == after.get("scale")
        if path.suffix != ".stl" or path in checked:
            continue
        checked.add(path)
        mesh = trimesh.load(source.parent / before.attrib["filename"], force="mesh", process=False)
        converted = trimesh.load(path, process=False)
        # STL stores float32 positions, retaining every triangle and its order.
        np.testing.assert_allclose(converted.triangles, mesh.triangles, atol=1e-7, rtol=0)
    assert len(checked) == 5
    stamp = output.stat().st_mtime_ns
    assert assets.prepare_lab_urdf(source) == str(output)
    assert output.stat().st_mtime_ns == stamp
    assert source.read_bytes() == original_bytes
