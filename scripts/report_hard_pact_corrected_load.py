"""Corrected-load comparison using existing torso summaries; no simulator imports."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scripts.report_hard_pact_torso_screen import summarize, KEYS
from scripts.run_hard_pact_torso_confirmation import flatten
from scripts.run_hard_pact_velocity_stress import read, write, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--reference',type=Path,default=Path('diagnostics/momentum15000/full/H_weak_torso_seed3'))
    args=p.parse_args();out=args.root/'report';out.mkdir(exist_ok=True)
    ref=read(args.reference/'manifest.json');rows=[];flat=[];checks=[];requests=None
    expected={'off':None,'E_rebalanced':[15,.5,.05,.05,.25,5,5],
              'H_weak_torso':[15,.5,.05,.01,.01,5,5],'F_no_torso':[15,.5,.05,0,0,5,5]}
    for name,weights in expected.items():
        folder=args.root/f'{name}_seed3';m=read(folder/'manifest.json');r=read(folder/'result.json')
        assert m['wrench_application_semantics']=='base_origin_explicit_moment_v2'
        assert not m['shadow_estimation'] and not m['momentum_sanity_correct_wrench']
        assert m['steps']==1000 and m['seed']==3 and m['load']=='light'
        for key in ('checkpoint_sha256','config_sha256','initial_state_hash','terrain_levels',
                    'terrain_types','terrain_source_rows','restored_curriculum','environment','dependencies'):
            assert m[key]==ref[key],(name,key)
        for key in ('commands','light_wrench','heavy_wrench','applied','mechanics'):
            assert m['tapes'][key]==ref['tapes'][key],(name,key)
        assert r['timing']['weights_unchanged'] and r['mechanics_unchanged']
        assert r['wrench_delivery_audit']['max_tolerance_ratio']<=1
        with np.load(folder/'trace.npz') as a:
            if weights is not None:
                assert [m['effective_qp'][k] for k in KEYS]==weights
                resolved=dict(ref['effective_qp']);resolved.update(zip(KEYS,weights))
                assert m['effective_qp']==resolved,(name,'non-objective QP configuration changed')
                assert np.allclose(a['assembled_seven_weights'],weights)
                assert (a['execution_strength']==1).all()
                assert r['execution_checks']['solves_per_env_control']==4
            else:assert r['execution_checks']['solves_per_env_control']==0
            request=a['requested_wrench_world']
            if requests is None:requests=request.copy()
            else:assert np.array_equal(requests,request),name
            row=summarize(a,m,r)
        row.update(condition=name,seed=3,timeouts=r['timeout_resets'],
                   wrench_delivery=r['wrench_delivery_audit'],timing=r['timing'])
        rows.append(row)
        for group in row['groups']:
            g=row['groups'][group]
            flat.append(dict(condition=name,seed=3,exposure=group,**flatten(row,group),
                **{k+'_rate':v for k,v in g['solver_rates'].items()},
                **{k+'_fraction':v['fraction'] for k,v in g['violations'].items()},
                **{k+'_max':v['max'] for k,v in g['violations'].items()},
                failure_free_fraction=row['failure_free_fraction'] if group=='overall' else None,
                timeouts=r['timeout_resets'] if group=='overall' else None))
        checks.append(dict(condition=name,manifest=str(folder/'manifest.json'),manifest_sha256=sha(folder/'manifest.json'),
                           checkpoint_sha256=m['checkpoint_sha256'],config_sha256=m['config_sha256'],
                           code_sha=m['code_sha'],application_semantics=m['wrench_application_semantics']))
    write(out/'comparison.json',rows)
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    terrain=[];objectives=[]
    for row in rows:
        for cell in row['terrain']:terrain.append(dict(condition=row['condition'],**cell))
        for group,g in row['groups'].items():
            for key,value in g['objectives'].items():
                objectives.append(dict(condition=row['condition'],exposure=group,metric=key,**value))
    write(out/'terrain_failures.json',terrain)
    with (out/'objective_costs_residuals.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(objectives[0]));w.writeheader();w.writerows(objectives)
    write(out/'provenance.json',dict(checks=checks,matching_passed=True,reference=str(args.reference),
        sources_sha256={p.name:sha(p) for p in (args.root/'sources').iterdir() if p.is_file()},
        notes=['All four are new corrected-production runs, not reused old outcomes.',
               'Initial state, terrain, mechanics and command/wrench tapes match momentum15000.',
               'Raw world requests identical; base-origin moments may differ with orientation.',
               'Heading feedback, reset states and trajectories diverge; single seed only.',
               'Objective execution costs retain candidate GRFs; not measured-force costs.',
               'Motion statistics exclude reset/teleport intervals; failure events retained.']))
    for row in rows:
        g=row['groups']['overall']
        print(row['condition'],g['failures'],g['failure_rate'],g['roll']['rms'],g['pitch']['rms'],[x['rms'] for x in g['tracking']])


if __name__=='__main__':main()
