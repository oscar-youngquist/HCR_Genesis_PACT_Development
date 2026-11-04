"""One policy/payload/scene per process; no TCN training or HardPACT deployment."""
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from .storage import ChunkWriter, atomic_json, digest
from .budget import budget_reached
from .progress import episode_progress


def run(path):
    os.environ['SIMULATOR'] = 'genesis_pact_water'
    import torch
    from .adapters import DACTAdapter
    from .provenance import capture
    from .recorder import NativeRecorder
    from .benchmark import MemoryMonitor
    path = Path(path)
    job = json.loads((path/'job.json').read_text())
    for key in ('checkpoint', 'config'):
        p = job['selected']['policy']
        if digest(p[key]) != p[key+'_sha256']:
            raise ValueError(f'{key} changed after planning')
    capture(path)
    adapter, recorder = DACTAdapter(), None
    monitor = MemoryMonitor()
    writer = ChunkWriter(path, job['chunk_size'], job['max_buffer_mb']*1024*1024)
    snapshots = None
    started = time.monotonic()
    try:
        with torch.no_grad():
            env = adapter.initialize(job)
            job['geometry'] = adapter.metadata['geometry']
            sim = env.simulator
            from legged_gym import LEGGED_GYM_ROOT_DIR
            from shutil import copyfile
            asset = Path(adapter.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
            copyfile(job['selected']['policy']['config'], path/'training_config.py')
            copyfile(asset, path/'robot.urdf')
            terrain_sha256 = None
            if hasattr(sim, '_terrain'):
                import numpy as np
                with (path/'terrain.npz.tmp').open('wb') as f:
                    np.savez(f, heights=sim._terrain.height_field_raw, origins=sim._terrain.env_origins,
                             horizontal_scale=adapter.cfg.terrain.horizontal_scale,
                             vertical_scale=adapter.cfg.terrain.vertical_scale)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(path/'terrain.npz.tmp', path/'terrain.npz')
                terrain_sha256 = digest(path/'terrain.npz')
            atomic_json(path/'resolved.json', dict(adapter.metadata, domain=sim.liquid_domain,
                terrain_sha256=terrain_sha256,
                contact_geometry=adapter.cfg.liquid.contact_geometry,
                contact_softness_m=float(sim._bucket.geoms[0].coup_softness),
                cavity_coupling=getattr(adapter.cfg.liquid, 'cavity_coupling', 'sdf'),
                contact_parts=getattr(sim, 'liquid_contact_parts', None),
                dry_container_mass_kg=[float(e.get_mass()) for e, _ in sim.liquid_attachment_targets],
                contact_sdf=[dict(cell_size=float(g.sdf_cell_size), resolution=g.sdf_res.tolist(),
                    coup_softness_m=float(g.coup_softness))
                    for e, _ in sim.liquid_attachment_targets for g in e.geoms],
                robot_asset=dict(path=str(asset), sha256=digest(asset)),
                environment_origins_W=sim.env_origins.cpu().tolist(),
                pinn_ready=False, quaternion='wxyz; R maps base to world',
                base_reference='base link origin: get_pos/get_vel(ref=link_origin)',
                actual_acceleration='interval endpoint velocity difference / recorded duration; average, not instantaneous',
                save_mode=job.get('save_mode', '200hz'),
                saved_state_dt=.005 if job.get('save_mode', '200hz') == '200hz' else sim.sim_dt/sim.sim_substeps,
                saved_wrench_reference=('summed world impulses shifted to fixed endpoint base origin and endpoint orientation'
                    if job.get('save_mode', '200hz') == '200hz' else 'native coupling-phase base origin and orientation'),
                dense_native_reconstruction_available=job.get('save_mode', '200hz') == 'native',
                wrench='liquid-on-rigid impulse from before/after legacy coupling; particle position reference',
                coupling_lag='legacy rigid pre-coupling integrates first; coupling load is queued for subsequent rigid solve',
                coupling_phase='after rigid integration and fluid velocity kick, before fluid drift; operator-split impulse over (t-dt,t]',
                viscosity='Genesis mu acts as kinematic viscosity (m^2/s) in the inspected acceleration kernel; not dynamic viscosity in Pa s',
                feature_dt=job['feature_dt'], target_grid_dt=.005, policy_dt=env.dt,
                simulator_dt=sim.sim_dt, native_dt=sim.sim_dt/sim.sim_substeps))
            # Creation and reset_root_states already audit the exact spawn.
            # env.reset() also advances one policy interval; do not mistake that
            # evolved state for a fresh rest layout or require spawn clearance of it.
            if job['smoke']:
                assert (sim.env_origins[:, 2] < 0).all(), 'Smoke needs negative-height terrain origins'
                assert (sim._robot.get_pos()[:, 2] < 0).all(), 'Smoke needs negative world base height'
                if job.get('smoke_step_height', -.12) <= -.16:
                    assert (sim._liquid.get_particles_pos()[..., 2].amin(-1) < 0).all(), 'Smoke needs actual liquid below world Z=0'
                from legged_gym.scripts.play_pact_water import verify_controller_parameters
                verify_controller_parameters(env, adapter.cfg)
            if job.get('validation_experiment'):
                from legged_gym.scripts.play_pact_water import verify_controller_parameters
                verify_controller_parameters(env, adapter.cfg)
            if job.get('containment_experiment') and adapter.cfg.liquid.contact_geometry == 'boxes':
                from .containment import verify_cavity_contacts
                atomic_json(path/'contact_audit.json', verify_cavity_contacts(sim))
            if not job['baseline']:
                recorder = NativeRecorder(env, writer, job)
                if job['snapshots']:
                    snapshots = ChunkWriter(path/'particles', max(1, job['chunk_size']//16), job['max_buffer_mb']*1024*1024)
                    recorder.snapshot_writer = snapshots
                atomic_json(path/'liquid.json', dict(particle_count=sim._liquid.n_particles,
                    initialization_mesh_clearance_m=sim.liquid_mesh_clearance,
                    mass_per_env=recorder.mass.sum(-1).cpu().tolist(),
                    rest_density_volume_per_env=(recorder.mass/recorder.rest_rho).sum(-1).cpu().tolist(),
                    particle_mass_min=float(recorder.mass.min()), particle_mass_max=float(recorder.mass.max()),
                    nominal_litres=None, payload=job['selected']['payload'], geometry=job['geometry']))
            if job['smoke']:
                before = sim._liquid.get_particles_pos().clone()
                ids = torch.tensor([0], device=env.device)
                if recorder:
                    recorder.forced_reset = True
                adapter.reset(ids)
                if recorder:
                    recorder.forced_reset = False
                after = sim._liquid.get_particles_pos()
                torch.testing.assert_close(before[1:], after[1:], atol=0, rtol=0)
                assert sim._liquid.get_particles_vel()[0].eq(0).all()
                sim.validate_liquid_spawn(ids)
            torch.cuda.synchronize()
            initialization_seconds = time.monotonic()-started
            torch.cuda.reset_peak_memory_stats()
            measured = time.monotonic()
            completed = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            atomic_json(path/'progress.json', episode_progress([0]*env.num_envs, job))
            last_progress = time.monotonic()
            first_step_seconds = None
            for step in range(job['max_steps']):
                if recorder:
                    recorder.policy_event()
                result = adapter.execute()
                completed += result[5].long()
                if time.monotonic() - last_progress >= 2.:
                    atomic_json(path/'progress.json', episode_progress(completed.cpu().tolist(), job))
                    last_progress = time.monotonic()
                if first_step_seconds is None:
                    torch.cuda.synchronize()
                    first_step_seconds = time.monotonic()-measured
                if job.get('fail_fast_containment') and recorder is not None and not bool(recorder.episode_supported.all()):
                    break
                if budget_reached(completed, job) and not job['benchmark']:
                    break
            torch.cuda.synchronize()
            elapsed = time.monotonic()-measured
            status = 'complete' if budget_reached(completed, job) else 'incomplete'
            atomic_json(path/'progress.json', episode_progress(completed.cpu().tolist(), job, 'finalizing'))
            metadata = dict(status=status, episodes_per_env=completed.cpu().tolist(), steps=step+1,
                containment_failed=(not bool(recorder.episode_supported.all())
                    if job.get('containment_experiment') and recorder is not None else None),
                requested_episodes_per_env=job.get('episodes_per_env'),
                selected_episodes_per_env=(completed.clamp_max(job['episodes_per_env']).cpu().tolist()
                    if job.get('episodes_per_env') is not None else completed.cpu().tolist()),
                initialization_seconds=initialization_seconds, collection_seconds=elapsed,
                first_step_seconds=first_step_seconds,
                steady_seconds_per_policy_step=(elapsed-first_step_seconds)/step if step else None,
                seconds_per_policy_step=elapsed/(step+1), torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                baseline=job['baseline'], pinn_ready=False, save_mode=job.get('save_mode', '200hz'),
                auxiliary_streams=['events', 'boundaries'] if recorder and recorder.save_mode == '200hz' else [],
                benchmark_mode=job.get('benchmark_mode', 'collection'),
                smoke_passed=job['smoke'] and recorder is not None and not any(
                    v for k, v in recorder.bad_counts.items() if k != 'nominal_cavity_penetration'),
                smoke_initialization_passed=job['smoke'],
                validity_counts=recorder.bad_counts if recorder else None)
            if job['smoke'] and recorder is not None and not metadata['smoke_passed']:
                metadata['status'] = 'failed'
            metadata.update(monitor.close())
            atomic_json(path/'metrics.json', metadata)
            if recorder and job['smoke']:
                assert metadata['smoke_passed'], f'Invalid smoke samples: {recorder.bad_counts}'
                assert int(recorder.completed.sum()) == int(completed.sum())
                assert recorder.tick == (step+1)*recorder.policy_ticks
            if snapshots:
                snapshots.finish(dict(status=status))
                metadata['auxiliary_streams'].append('particles')
            if recorder:
                recorder.finish_streams(dict(status=status))
            writer.finish(metadata)
            atomic_json(path/'progress.json', episode_progress(completed.cpu().tolist(), job, status))
            return 0 if status == 'complete' else 2
    except BaseException:
        atomic_json(path/'failure.json', dict(error=traceback.format_exc(), writer_failed=writer.failed,
            buffered_samples_at_failure=len(writer.rows),
            snapshot_buffered_samples_at_failure=len(snapshots.rows) if snapshots else 0))
        # Preserve all successfully buffered samples if the writer is healthy.
        if recorder:
            recorder.flush_failure()
        if not writer.failed:
            writer.flush()
        if snapshots and not snapshots.failed:
            snapshots.flush()
        raise
    finally:
        monitor.close()
        if recorder:
            recorder.close()
        adapter.cleanup()


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
