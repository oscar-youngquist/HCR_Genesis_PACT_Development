"""Snapshot the installed sources without modifying the patched environment."""
import importlib.metadata
import importlib.util
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import base64
from .storage import digest, atomic_json

GENESIS_FILES = (
    'engine/solvers/sph_solver.py', 'engine/entities/sph_entity.py',
    'engine/entities/particle_entity.py', 'engine/entities/rigid_entity/rigid_entity.py',
    'engine/couplers/legacy_coupler.py', 'engine/simulator.py', 'options/solvers.py',
    'engine/solvers/rigid/rigid_solver_decomp.py', 'engine/materials/SPH/liquid.py',
    'engine/boundaries/boundaries.py', 'utils/geom.py', 'utils/sdf_decomp.py',
    'engine/entities/rigid_entity/rigid_geom.py', 'engine/materials/rigid.py',
)


def capture(path):
    path = Path(path)
    root = Path(__file__).resolve().parents[1]
    genesis = Path(importlib.util.find_spec('genesis').origin).parent
    files = {}
    distribution = importlib.metadata.distribution('genesis-world')
    wheel_files = {str(p): p for p in distribution.files or []}
    sources = [('genesis/'+name, genesis/name) for name in GENESIS_FILES]
    sources += [('repo/'+str(p.relative_to(root)), p) for p in (root/'liquid_collection').glob('*.py')]
    sources += [('repo/'+name, root/name) for name in (
        'legged_gym/scripts/play_pact_water.py', 'legged_gym/scripts/pact_liquid_checkpoint.py',
        'legged_gym/scripts/liquid_payload_configs.py', 'legged_gym/envs/base/liquid_payload_config.py',
        'legged_gym/simulator/genesis_simulator_pact_liquid.py',
        'legged_gym/simulator/genesis_simulator_pact_nopinn.py',
        'legged_gym/utils/terrain.py', 'legged_gym/utils/terrain_utils.py',
        'legged_gym/scripts/water_tank_proper_units_simple.stl',
        'legged_gym/scripts/water_tank_lid.stl',
        'rsl_rl/modules/actor_critic_pact.py',
        'legged_gym/envs/go1/go1_pact/go1_pact.py',
        'legged_gym/envs/go2/go2_pact/go2_pact.py',
        'legged_gym/envs/go1/go1_pact_water_eval/go1_pact_water_config.py',
        'legged_gym/envs/go2/go2_pact_water_eval/go2_pact_water_config.py')]
    for name, source in sources:
        if not source.is_file():
            raise RuntimeError(f'Required provenance source missing: {source}')
        target = path/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sha = digest(source)
        wheel = wheel_files.get(name)
        recorded = wheel.hash.value if wheel is not None and wheel.hash and wheel.hash.mode == 'sha256' else None
        actual_b64 = base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip('=')
        files[name] = dict(path=str(source), sha256=sha, installed_RECORD_sha256_base64=recorded,
                          modified_vs_RECORD=None if recorded is None else recorded != actual_b64)
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  genesis_direct_url=distribution.read_text('direct_url.json'),
                  packages=packages, files=files,
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  git_status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True))
    atomic_json(path/'provenance.json', result)
    return result
