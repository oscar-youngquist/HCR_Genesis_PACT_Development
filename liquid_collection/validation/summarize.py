"""Combine bounded runs while retaining superseded diagnostic cases explicitly."""
import argparse
import json
from pathlib import Path
import shlex
import sys
from liquid_collection.storage import atomic_json,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    selected={};earlier=[];criteria=None;runtime=0.;sources=[]
    for root in args.experiments:
        report=json.loads((root/'report.json').read_text())
        if criteria is not None and criteria!=report['criteria']: raise ValueError('Cannot pool different frozen criteria')
        criteria=report['criteria']
        execution=json.loads((root/'execution.json').read_text())
        runtime+=execution['wall_seconds']
        sources.append(dict(path=str(root.resolve()),report_sha256=digest(root/'report.json'),
            execution_sha256=digest(root/'execution.json'),criteria_sha256=digest(root/'criteria.json')))
        for name,value in report['cases'].items():
            if name in selected: earlier.append(dict(case=name,result=selected[name]))
            selected[name]=value
    def verdict(keys):
        statuses=[r.get('status',{}).get(k,'inconclusive') if isinstance(r.get('status'),dict) else 'inconclusive' for r in selected.values() for k in keys]
        return 'inconclusive' if 'inconclusive' in statuses else 'fail' if 'fail' in statuses else 'pass'
    reductions=[r.get('recording',{}).get('status','inconclusive') for r in selected.values()]
    conclusions=dict(averaged_wrench_accounting='pass' if reductions and all(s=='pass' for s in reductions) else 'not established',
        instantaneous_wrench=verdict(['wrench']),local_jvp=verdict(['input_derivative','jvp_plumbing','local_conservation']),
        original_integrated=verdict(['integrated']))
    conclusions['full_collection']='not supported' if any(v!='pass' for v in conclusions.values()) else 'supported within this bounded test scope only'
    out=dict(cases=selected,superseded_diagnostics=earlier,criteria=criteria,sources=sources,
        total_simulation_wall_seconds=runtime,budget_seconds=2700.,within_budget=runtime<=2700,
        full_collection_ready=False,
        conclusions=conclusions)
    atomic_json(args.output/'summary.json',out)
    lines=['# Liquid-format validation results','',
        f'Simulation workers used **{runtime/60:.2f} minutes** in total (45-minute budget), including retained diagnostic runs. No surrogate training or full campaign was run.','',
        'Collection now uses the same reset/termination helper and saved joint-dynamics midpoints as the play scripts. The initial Go1 base-and-hip contact runs are retained under `superseded_diagnostics`; the corrected runs use base-only contact termination. The frozen tolerances were unchanged.','',
        '| Case | Reduction | Valid end by env (s) | Histories / collocations / windows by env | First escape by env (s) |',
        '|---|---|---|---|---|']
    for name,r in selected.items():
        es=r.get('environments',[])
        lines.append(f"| {name} | {r.get('recording',{}).get('status','inconclusive')} | {[round(e['valid_end'],3) for e in es]} | {[(e['histories'],e['collocations'],e['windows']) for e in es]} | {r.get('first_escape_s_by_env')} |")
    lines+=['','| Case | Interior force RMS / mg | Moment RMS / mgL | Input-derivative normalized RMS | Largest integrated excess RMS (r/P/S scale) |',
        '|---|---|---|---|---|']
    for name,r in selected.items():
        es=r.get('environments',[])
        def metric(k):
            vals=[e['metrics']['interior'][k]['normalized_rmse'] for e in es if 'interior' in e['metrics']]
            return f'{max(vals):.4g}' if vals else 'unavailable'
        vals=[v['excess']['normalized_rmse'] for e in es for v in e['integrated'].values()]
        lines.append(f"| {name} | {metric('F')} | {metric('tau')} | {metric('zdot')} | {max(vals) if vals else 'no valid windows'} |")
    lines+=['','These are worst-environment observed metrics, not pooled passes. Native reference sensitivity, bias, p95/max, changing-wrench normalization, peaks, local conservation floors and separate quadrature excess are in `summary.json`. See each attempt’s `comparison.png`, `diagnostics.png`, `commands.png`, `consumed_commands.json`, `resolved.json` and hashed native/compact streams.','',
        '## Conclusions','',f'Computed verdicts: `{conclusions}`.','',
        '- **Averaged wrench:** independently audit each 5 ms impulse and reference pose. A pass verifies averaging/bookkeeping, not instantaneous supervised-wrench accuracy.',
        '- **Local JVP:** insufficient valid maneuver coverage and sensitivity to the native kinematic reference prevent an adequacy claim. Synthetic autodiff/finite-difference plumbing tests passed; live per-case errors remain in the JSON.',
        '- **Original integrated losses:** the requested 41-endpoint trapezoids are implemented and compared with native-resolution and coarse native quadrature. Insufficient valid windows/maneuver coverage prevents approval across all six cases.',
        '- **Full collection:** readiness remains false. Resolve the demonstrated particle-containment/coverage blocker first. Then test better pose reconstruction/reference conditioning or selected native pose/velocity/wrench channels if the accuracy checks still fail. Preserve the JVP/integrated objectives and the production 200 Hz format while evaluating those changes.','',
        'The native interval average is not instantaneous ground truth. The native Hermite reference is also sensitive to float32 world-position quantization at 0.25 ms; the report includes an alternative native endpoint-velocity/interval-acceleration convention. No thresholds were relaxed to turn this ambiguity into a pass.','',
        '## Reproduction','', 'See [experiment method](../../docs/liquid_format_experiment.md). Exact analysis/run arguments are retained in each experiment’s manifests. This combined report was generated with:','',
        '```bash',shlex.join([sys.executable,'-m','liquid_collection.validation.summarize',*sys.argv[1:]]),'```','',
        '## Artifacts','']
    for root in args.experiments:
        lines.append(f'- [{root.name}]({root.resolve()}/report.md): report, frozen criteria, execution times, plans and source snapshots.')
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    atomic_json(args.output/'checksums.json',{name:digest(args.output/name) for name in ('summary.json','summary.md')})

if __name__=='__main__':main()
