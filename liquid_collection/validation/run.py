"""Bounded six-case experiment. Run explicitly; production defaults are untouched."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import sys
import shlex
import math
from liquid_collection.launch import PLATFORMS
from liquid_collection.plan import make_plan, run_plan
from liquid_collection.payload import validate_payload, supported_bounds
from liquid_collection.storage import atomic_json, digest

CRITERIA = dict(provisional=True, normalized_rmse=.05, normalized_p95=.10,
    conservation_rms=.10, excess_conservation_rms=.05, jvp_relative_error=1e-5,
    reduction_atol=2e-5, reduction_rtol=2e-6, minimum_histories=100,
    minimum_collocations=100, minimum_windows=40, minimum_maneuver_samples=20,
    history_taps=249, history_dt=.005, interior_margin_s=.0001,
    finite_difference_steps_s=[1e-5, 2e-5, 4e-5], native_reference_sensitivity_rms=.05,
    L_definition='Euclidean diagonal of nominal interior dimensions in metres',
    scales='F=mg; tau=mgL; u=sqrt(gL); omega=sqrt(g/L); a=g; alpha=g/L; time=sqrt(L/g)')

def payloads():
    b = supported_bounds(.10, .02)
    mid = {k: (v[0]+v[1])/2 for k,v in b.items() if not k.startswith('mount_')}
    mid['mount_offset'] = [.025, .025]
    values = [mid, dict(mid, offset=.032, mu=.001), dict(mid, offset=.082, mu=.025)]
    for p in values:
        validate_payload(p, 0., .010, .001, property_expansion=.10, geometry_expansion=.02)
    return values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--total-seconds', type=float, default=2700)
    p.add_argument('--worker-seconds', type=float, default=450)
    p.add_argument('--criteria', type=Path)
    p.add_argument('--platforms', nargs='+', choices=['go1-dact','go2-dact'], default=['go1-dact','go2-dact'])
    p.add_argument('--prepare-only', action='store_true')
    args=p.parse_args()
    if not all(math.isfinite(x) and x>0 for x in (args.total_seconds,args.worker_seconds)): p.error('Budgets must be finite and positive')
    if list(args.output.glob('*/round-*')): p.error('Use a fresh output directory; automatic retries are disabled')
    args.output.mkdir(parents=True,exist_ok=True)
    criteria=json.loads(args.criteria.read_text()) if args.criteria else CRITERIA
    missing=set(CRITERIA)-set(criteria)
    if missing: p.error('Missing criteria: '+str(sorted(missing)))
    if max(criteria['finite_difference_steps_s'])>=criteria['interior_margin_s']:
        p.error('Finite-difference steps must stay inside the collocation margin')
    frozen=args.output/'criteria.json'
    if frozen.exists():
        if json.loads(frozen.read_text())!=criteria: raise ValueError('Frozen criteria differ')
    else: atomic_json(frozen,criteria)
    plans={}
    for robot,manifest in PLATFORMS.items():
        if manifest is None or robot not in args.platforms: continue
        for i,(name,payload) in enumerate(zip(('representative','high_fill','viscous'),payloads())):
            plan=make_plan(manifest, seed=137+i, num_envs=2, episodes=2, episodes_per_env=1,
                episode_seconds=6., max_steps=601, max_seconds=args.worker_seconds,save_mode='native')
            plan['rounds'][0]['payload']=deepcopy(payload)
            plan.update(validation_experiment=True,validation_step_height=(-.04,.04,-.08)[i])
            plans[robot+'-'+name]=plan
    atomic_json(args.output/'experiment.json',dict(plans=plans, criteria_sha256=digest(frozen),
        total_seconds=args.total_seconds, worker_seconds=args.worker_seconds, snapshots=False,
        command=shlex.join([sys.executable,'-m','liquid_collection.validation.run',*sys.argv[1:]])))
    if args.prepare_only: return
    started=time.monotonic(); results={}
    for name,plan in plans.items():
        remaining=args.total_seconds-(time.monotonic()-started)
        if remaining<1:
            results[name]=dict(status='inconclusive',reason='total simulation budget exhausted'); continue
        plan['max_seconds']=min(args.worker_seconds,remaining)
        t=time.monotonic()
        code=run_plan(plan,args.output/name)
        results[name]=dict(returncode=code,wall_seconds=time.monotonic()-t)
        atomic_json(args.output/'execution.json',dict(cases=results,wall_seconds=time.monotonic()-started))
    atomic_json(args.output/'execution.json',dict(cases=results,wall_seconds=time.monotonic()-started))

if __name__=='__main__': main()
