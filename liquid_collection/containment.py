"""Opt-in containment/runtime trials; production particle/contact/substep defaults remain unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import math
import numpy as np
from .plan import ROOT, DEFAULT_PYTHON, make_plan, terminate_worker
from .launch import PLATFORMS
from .storage import atomic_json, digest
from .audit import stream_rows
from .validation.run import payloads

PROFILES = {
    'current': dict(particle_size=.010, substeps=8, cavity_coupling='sdf'),
    'sdf_10mm_16': dict(particle_size=.010, substeps=16, cavity_coupling='sdf'),
    'sdf_12mm_8': dict(particle_size=.012, substeps=8, cavity_coupling='sdf'),
    'planes_10mm_8': dict(particle_size=.010, substeps=8, cavity_coupling='planes'),
    'planes_10mm_4': dict(particle_size=.010, substeps=4, cavity_coupling='planes'),
    'planes_8mm_4': dict(particle_size=.008, substeps=4, cavity_coupling='planes'),
    'planes_8mm_8': dict(particle_size=.008, substeps=8, cavity_coupling='planes'),
    'planes_6mm_4': dict(particle_size=.006, substeps=4, cavity_coupling='planes'),
    'planes_6mm_8': dict(particle_size=.006, substeps=8, cavity_coupling='planes'),
    'planes_4mm_4': dict(particle_size=.004, substeps=4, cavity_coupling='planes'),
}

def verify_cavity_contacts(sim):
    targets=sim.liquid_attachment_targets
    expected=sum(np.prod(p['size'])*p.get('rho',200.) for p in sim.liquid_contact_parts)
    actual=sum(float(entity.get_mass()) for entity,_ in targets)
    if not math.isclose(actual,expected,rel_tol=1e-5,abs_tol=1e-6):
        raise ValueError(f'Contact representation changed shell mass: {actual} vs {expected}')
    planes=getattr(sim._cfg.liquid,'cavity_coupling','sdf')=='planes'
    if planes:
        if len(targets)!=12: raise ValueError('Expected six shell boxes and six analytic planes')
        ids={g.idx for e,_ in targets[6:] for g in e.geoms}
        for g in sim._scene.sim.rigid_solver.geoms:
            if g.idx in ids:
                if not (g.needs_coup and g.contype==1<<30 and g.conaffinity==0):
                    raise ValueError('Analytic plane coupling/contact mask incorrect')
            elif g.conaffinity & (1<<30):
                raise ValueError('Analytic plane filter bit collides with another rigid geom')
        if any(g.needs_coup for e,_ in targets[:6] for g in e.geoms):
            raise ValueError('Shell and plane coupling would be double counted')
    return dict(expected_dry_mass_kg=expected,actual_dry_mass_kg=actual,
                analytic_planes=planes,rigid_contact_masks_checked=planes)

def summarize(path,job,wall_seconds):
    metrics=json.loads((path/'metrics.json').read_text()) if (path/'metrics.json').exists() else {}
    result=dict(profile=job['profile'],robot=job['selected']['policy']['robot'],payload=job['payload_label'],
        requested_seconds=job['episode_seconds'],num_envs=job['num_envs'],scaling_probe=job.get('scaling_probe',False),wall_seconds=wall_seconds,metrics=metrics,
        settings={k:job.get(k) for k in ('particle_size','substeps','contact_geometry','cavity_coupling','contact_softness_m')},
        passed=False,full_collection_ready=False)
    if job['baseline']:
        result.update(throughput_only=True,steady_seconds_per_policy_step=metrics.get('steady_seconds_per_policy_step'))
    else:
        ends=[[] for _ in range(job['num_envs'])];max_pen=0.;first_escape=[None]*job['num_envs']
        rows=sorted(list(stream_rows(path))+list(stream_rows(path/'boundaries')),
                    key=lambda r:(float(r['time_s']),int(r['record_sequence'])))
        for row in rows:
            max_pen=max(max_pen,float(np.max(row['interval_max_wall_penetration_m'])))
            for env in range(job['num_envs']):
                if row['terminal'][env] and row['episode_selected'][env]:
                    ends[env].append(float(row['episode_native_step'][env])*.002/job['substeps'])
                if not row['interval_contained'][env] and first_escape[env] is None:
                    first_escape[env]=float(row['time_s'])
        counts=metrics.get('validity_counts') or {}
        valid=bool(metrics) and all(counts.get(k,1)==0 for k in ('finite','contained','all_active','attached','domain_contact'))
        full=job['episode_seconds']>=11. and all(any(abs(t-job['episode_seconds'])<1e-7 for t in times) for times in ends)
        result.update(terminal_ages_s=ends,max_inner_wall_penetration_m=max_pen,first_escape_saved_interval_end_s=first_escape,
            passed=valid and full and max_pen<=job['particle_size']/2+.001,
            steady_seconds_per_policy_step=metrics.get('steady_seconds_per_policy_step'),
            failed_conditions=dict(validity=not valid,duration=not full,penetration=max_pen>job['particle_size']/2+.001))
    if (path/'resolved.json').exists():
        resolved=json.loads((path/'resolved.json').read_text())
        result['settings']['contact_softness_m']=resolved['contact_softness_m']
    if (path/'liquid.json').exists():
        liquid=json.loads((path/'liquid.json').read_text())
        result['particle_count_per_env']=liquid['particle_count']
        result['particle_count_total']=liquid['particle_count']*job['num_envs']
    if (path/'failure.json').exists(): result['failure']=json.loads((path/'failure.json').read_text())
    result['timing_probe_completed']=bool(job.get('scaling_probe') and metrics.get('steps')==job['max_steps']
        and not result.get('failure') and (job['baseline'] or
        (not result['failed_conditions']['validity'] and not result['failed_conditions']['penetration'])))
    atomic_json(path/'containment.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',choices=PROFILES,default='planes_8mm_4')
    p.add_argument('--robot',choices=['go1','go2'],default='go2')
    p.add_argument('--payload',choices=['representative','high_fill','viscous','viscous_limit','random'],default='high_fill')
    p.add_argument('--episode-seconds',type=float,default=11.,help='1 s settling plus 10 s aggressive motion')
    p.add_argument('--max-seconds',type=float,default=900.)
    p.add_argument('--seed',type=int,help='Override seed to repeat a payload with fresh initial conditions')
    p.add_argument('--num-envs',type=int,default=2)
    p.add_argument('--timing-steps',type=int,help='Short throughput/memory probe; never counts as full containment validation')
    p.add_argument('--reference',action='store_true',help='50-step pre-swap mesh/12mm/2-substep player timing only')
    a=p.parse_args()
    if min(a.max_seconds,a.episode_seconds)<=0 or not all(map(math.isfinite,(a.max_seconds,a.episode_seconds))): p.error('Limits must be finite and positive')
    if a.num_envs<1 or (a.timing_steps is not None and a.timing_steps<2): p.error('Need positive env count and at least two timing steps')
    if a.output.exists(): p.error('Use a fresh output directory; trials are never overwritten')
    i={'representative':0,'high_fill':1,'viscous':2,'viscous_limit':2,'random':0}[a.payload];profile=PROFILES[a.profile]
    plan=make_plan(PLATFORMS[a.robot+'-dact'],seed=137+i if a.seed is None else a.seed,num_envs=a.num_envs,episodes=a.num_envs,episodes_per_env=1,
        episode_seconds=a.episode_seconds,max_steps=a.timing_steps or (50 if a.reference else math.ceil(a.episode_seconds/.01)+1),
        max_seconds=a.max_seconds,particle_size=.012 if a.reference else profile['particle_size'],
        substeps=2 if a.reference else profile['substeps'],contact_geometry='mesh' if a.reference else 'boxes')
    selected=plan['rounds'][0]
    if a.payload!='random':
        selected['payload']=payloads()[i]
    if a.payload=='viscous_limit':
        from .payload import supported_bounds
        bounds=supported_bounds(.10,.02)
        for key in ('rho','mu','gamma'):
            selected['payload'][key]=bounds[key][1]
    job=dict(plan,selected=selected,baseline=a.reference,benchmark=a.reference,
        benchmark_mode='reference' if a.reference else 'collection',validation_experiment=True,
        validation_step_height=(-.04,.04,-.08)[i],containment_experiment=True,
        contact_softness_m=plan['particle_size']/2,
        fail_fast_containment=not a.reference,cavity_coupling='sdf' if a.reference else profile['cavity_coupling'],
        profile='pre_swap_reference' if a.reference else a.profile,payload_label=a.payload,scaling_probe=a.timing_steps is not None)
    a.output.mkdir(parents=True)
    atomic_json(a.output/'criteria.json',dict(max_escape_events=0,max_nonfinite_events=0,max_domain_events=0,
        max_inactive_events=0,max_detachment_events=0,max_inner_penetration_m=job['particle_size']/2+.001,
        required_full_episode_s=a.episode_seconds,settling_s=1.,
        runtime_requirement='collection steady seconds/policy step <= measured pre-swap reference player',
        limits_apply_per_environment=True,num_envs=a.num_envs,timing_steps=a.timing_steps))
    atomic_json(a.output/'job.json',job)
    env=dict(os.environ,SIMULATOR='genesis_pact_water',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''))
    env.setdefault('NUMBA_CACHE_DIR','/tmp/liquid-numba');env.setdefault('MPLCONFIGDIR','/tmp/liquid-mpl')
    command=[str(DEFAULT_PYTHON),'-m','liquid_collection.worker',str(a.output.resolve())]
    atomic_json(a.output/'invocation.json',dict(command=command,job_sha256=digest(a.output/'job.json')))
    started=time.monotonic();atomic_json(a.output/'status.json',dict(status='running'))
    with (a.output/'worker.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=proc.wait(timeout=a.max_seconds)
        except subprocess.TimeoutExpired:
            terminate_worker(proc);code=None
        except BaseException:
            terminate_worker(proc);raise
    elapsed=time.monotonic()-started
    atomic_json(a.output/'status.json',dict(status='finished' if code is not None else 'timeout',returncode=code,wall_seconds=elapsed))
    result=summarize(a.output,job,elapsed)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] or result['timing_probe_completed'] or (a.reference and result.get('steady_seconds_per_policy_step')) else 2

if __name__=='__main__':raise SystemExit(main())
