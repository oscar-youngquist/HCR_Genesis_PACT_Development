"""Summarize completed bounded containment trials and measured runtime limits."""
import argparse
import json
from pathlib import Path
import numpy as np
from .audit import stream_rows
from .storage import atomic_json, digest


def report(root):
    trials=[]
    for file in sorted(root.glob('*/containment.json')):
        result=json.loads(file.read_text());path=file.parent
        result['artifact_directory']=str(path.resolve())
        if (path/'stopped_early.json').exists():
            result['stopped_early']=json.loads((path/'stopped_early.json').read_text())
        if (path/'metadata_correction.json').exists():
            correction=json.loads((path/'metadata_correction.json').read_text())
            result['metadata_correction']=correction
            result['settings'][correction['field']]=correction['effective_value']
        job=json.loads((path/'job.json').read_text())
        result['num_envs']=job['num_envs']
        result['seed']=job['seed']
        result['selected_seed']=job['selected']['seed']
        result['payload_spec']=job['selected']['payload']
        result['job_sha256']=digest(path/'job.json')
        result['timing_steps']=job['max_steps'] if result.get('scaling_probe') or result.get('throughput_only') else None
        if result.get('metrics',{}).get('steps',0)<2:
            result['steady_seconds_per_policy_step']=None
        speed=result.get('steady_seconds_per_policy_step')
        result['simulated_env_seconds_per_wall_second']=result['num_envs']*.01/speed if speed else None
        if (path/'liquid.json').exists():
            liquid=json.loads((path/'liquid.json').read_text())
            result['particle_count_per_env']=liquid['particle_count']
            result['liquid_mass_per_env_kg']=liquid['mass_per_env']
        if not result.get('throughput_only'):
            rows=list(stream_rows(path))
            if rows:
                commands=np.asarray([r['commands'] for r in rows])
                velocity=np.asarray([r['velocity_W'] for r in rows])
                result['command_min']=commands.min(axis=0).tolist()
                result['command_max']=commands.max(axis=0).tolist()
                result['max_base_linear_speed_m_s']=np.linalg.norm(velocity[...,:3],axis=-1).max(axis=0).tolist()
                result['max_base_angular_speed_rad_s']=np.linalg.norm(velocity[...,3:],axis=-1).max(axis=0).tolist()
        trials.append(result)
    def reference_key(r):
        return r['robot'],json.dumps(r['payload_spec'],sort_keys=True),r['num_envs']
    references={reference_key(r):r for r in trials if r.get('throughput_only') and r.get('steady_seconds_per_policy_step')}
    for r in trials:
        if r.get('throughput_only'):continue
        ref=references.get(reference_key(r))
        r['containment_passed']=r.get('passed',False)
        r['runtime_reference']=ref['artifact_directory'] if ref else None
        r['runtime_ratio_vs_reference']=(r['steady_seconds_per_policy_step']/ref['steady_seconds_per_policy_step']
            if ref and r.get('steady_seconds_per_policy_step') else None)
        r['runtime_passed']=r['runtime_ratio_vs_reference']<=1. if r['runtime_ratio_vs_reference'] is not None else None
        r['containment_and_runtime_passed']=r['containment_passed'] and r['runtime_passed'] is True
    profiles={}
    for r in trials:
        if r.get('throughput_only'): continue
        item=profiles.setdefault(r['profile'],dict(full_duration_env_trials_passed=0,passing_cases=[],escaped_cases=[],scaling_cases=[]))
        if r['containment_passed']:
            item['full_duration_env_trials_passed']+=r['num_envs']
            item['passing_cases'].append(dict(robot=r['robot'],payload=r['payload'],seed=r['seed'],directory=r['artifact_directory']))
        if any(x is not None for x in r.get('first_escape_saved_interval_end_s',[])):
            item['escaped_cases'].append(r['artifact_directory'])
        if r.get('scaling_probe'):
            item['scaling_cases'].append(dict(num_envs=r['num_envs'],runtime_ratio=r.get('runtime_ratio_vs_reference'),directory=r['artifact_directory']))
    out=dict(trials=trials,profile_summary=profiles,full_collection_ready=False,
             total_wall_seconds=sum(r['wall_seconds'] for r in trials),
             caveat='Finite tests at the recorded environment count, not a guarantee over all sampled payloads. Runtime uses a matched robot/payload pre-swap player without recording; experimental runs include 200 Hz recording.')
    atomic_json(root/'report.json',out)
    lines=['# Liquid containment investigation','',out['caveat'],'',
           '| Profile | Robot / payload / envs | Full containment test | Max penetration (mm) | Seconds / policy step | Runtime / old mesh | GPU MiB |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in trials:
        speed=r.get('steady_seconds_per_policy_step');ratio=r.get('runtime_ratio_vs_reference')
        verdict='timing only' if r.get('throughput_only') or r.get('scaling_probe') else ('pass' if r.get('containment_passed') else 'fail / incomplete')
        if r.get('stopped_early'): verdict='stopped early (scope change)'
        pen=r.get('max_inner_wall_penetration_m')
        lines.append(f"| [{r['profile']}]({Path(r['artifact_directory']).name}/containment.json) | {r['robot']} / {r['payload']} / {r.get('num_envs',2)} | {verdict} | {f'{1000*pen:.3f}' if pen is not None else '—'} | {f'{speed:.4f}' if speed else '—'} | {f'{ratio:.3f}' if ratio else '—'} | {r['metrics'].get('process_gpu_peak_sampled_mib','—')} |")
    lines += ['', 'Acceptance: all environments complete 1 s settling + 10 s of bounded forward, stop, reverse and opposing lateral/yaw commands; zero escaped/inactive/nonfinite/domain-contact/detached samples at every native substep; penetration no greater than particle radius + 1 mm. A shorter diagnostic is not a full-duration pass.', '',
              'The experimental planes are inward-facing fluid contact half-spaces attached to the base. Six translucent boxes retain the cavity dimensions, shell mass and rigid contacts. Fluid coupling on those boxes is disabled to avoid duplicate impulses; plane collision masks exclude robot/terrain rigid pairs. No particle clamping or teleportation is used. Production particle/contact/substep defaults and the installed Genesis package remain unchanged.', '',
              'Play termination and saved joint-dynamics midpoint settings apply throughout. Early falls are incomplete tests, even if the liquid remains contained. See each directory for exact commands, source hashes, resolved configuration, native-interval validity aggregates, contact audit and runtime metrics. Full-collection readiness remains false; the format/physics validation must be repeated with any adopted contact settings.', '']
    (root/'report.md').write_text('\n'.join(lines))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    report(parser.parse_args().root)
