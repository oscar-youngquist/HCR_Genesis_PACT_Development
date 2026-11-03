"""View a sampled collection payload with translucent box contacts on cuda:0."""
import argparse
import json
import math
import os
from pathlib import Path
import secrets

from .plan import make_plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', choices=('go1', 'go2'), default='go2', help='Default manifest robot')
    p.add_argument('--manifest', type=Path, help='Optional policy manifest; overrides --robot')
    p.add_argument('--num-envs', type=int, default=2)
    p.add_argument('--seed', type=int, help='Omit for a new random configuration; printed for replay')
    p.add_argument('--duration', type=float, default=10., help='Simulated seconds')
    p.add_argument('--steps', type=int, help='Override duration with a policy-step count')
    p.add_argument('--terrain', choices=('plane', 'training'), default='plane')
    p.add_argument('--wall-opacity', type=float, default=.25)
    p.add_argument('--camera', choices=('follow', 'overview', 'free'), default='follow')
    p.add_argument('--follow-env', type=int, default=0)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--frame', type=Path, help='Save the final camera view as PNG (also works headless)')
    p.add_argument('--dry-run', action='store_true', help='Print sampled configuration without initializing Genesis')
    return p


def main(args):
    if not 1 <= args.num_envs <= 8:
        raise ValueError('Visualization supports 1–8 environments')
    if not 0 <= args.follow_env < args.num_envs:
        raise ValueError('--follow-env must index one of the environments')
    if not math.isfinite(args.wall_opacity) or not 0 <= args.wall_opacity <= 1:
        raise ValueError('--wall-opacity must be between 0 and 1')
    if not math.isfinite(args.duration) or args.duration <= 0 or (args.steps is not None and args.steps < 1):
        raise ValueError('Duration and step count must be positive')
    seed = args.seed if args.seed is not None else secrets.randbits(31)
    manifest = args.manifest or Path(__file__).with_name('go1_manifest.json' if args.robot == 'go1' else 'example_manifest.json')
    plan = make_plan(manifest, seed=seed, num_envs=args.num_envs, episodes=1)
    job = dict(plan, selected=plan['rounds'][0])
    visual = dict(headless=args.headless, save_frame=args.frame is not None,
                  wall_opacity=args.wall_opacity, terrain=args.terrain)
    print('LIQUID_VISUALIZATION '+json.dumps(dict(seed=seed, selected=job['selected'],
        num_envs=args.num_envs, particle_size=job['particle_size'], substeps=job['substeps'],
        contact_geometry=job['contact_geometry'], visualization=visual)), flush=True)
    if args.dry_run:
        return
    os.environ['SIMULATOR'] = 'genesis_pact_water'
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/liquid-numba')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/liquid-mpl')
    if args.headless:
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    import numpy as np
    import torch
    from .adapters import DACTAdapter
    adapter = DACTAdapter()
    try:
        with torch.no_grad():
            env = adapter.initialize(job, visualization=visual)
            sim = env.simulator

            def position_camera():
                # Genesis adds display-only batch offsets to simulated positions.
                positions = sim._robot.get_pos().cpu().numpy()+sim._scene.envs_offset
                if args.camera == 'overview':
                    center = positions.mean(0)
                    distance = max(1., float(np.linalg.norm(np.ptp(positions, axis=0))))
                else:
                    center, distance = positions[args.follow_env], 1.
                target = center+np.array([0., 0., .2])
                eye = target+distance*np.array([1.2, -1.4, .9])
                if not args.headless:
                    sim.set_viewer_camera(eye, target)
                if args.frame:
                    sim._floating_camera.set_pose(pos=eye, lookat=target)

            position_camera()
            steps = args.steps or math.ceil(args.duration/env.dt)
            completed = 0
            for step in range(steps):
                if not args.headless and not sim._scene.viewer.is_alive():
                    break
                adapter.execute()
                completed = step+1
                if args.camera != 'free':
                    position_camera()
            if args.frame:
                from PIL import Image
                args.frame.parent.mkdir(parents=True, exist_ok=True)
                rgb, *_ = sim._floating_camera.render()
                Image.fromarray(rgb).save(args.frame)
            print(json.dumps(dict(seed=seed, steps=completed, particles_per_env=sim._liquid.n_particles,
                                  frame=str(args.frame) if args.frame else None)), flush=True)
    except KeyboardInterrupt:
        print('Visualization stopped.', flush=True)
    finally:
        adapter.cleanup()


if __name__ == '__main__':
    main(parser().parse_args())
