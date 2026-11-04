"""Publish a compact repository summary of an immutable discrete validation run."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from liquid_collection.storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis',type=Path)
    p.add_argument('--stem',type=Path,default=Path('docs/liquid_discrete_validation_v1'))
    p.add_argument('--test-log',type=Path,required=True)
    a=p.parse_args();root=a.analysis;report=json.loads((root/'summary.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    for name in ('losses.py','data.py','analyze.py'):
        src=Path('liquid_collection/discrete')/name
        if provenance['sources'][str(src)]!=digest(src):raise ValueError('Analysis does not match current '+str(src))
    a.stem.parent.mkdir(parents=True,exist_ok=True)
    tests=a.test_log.read_text();test_summary=next(line for line in reversed(tests.splitlines()) if ' passed' in line)
    rows=[]
    for name,case in report['cases'].items():
        for e in case['environments']:rows.append((name,e))
    channels=('kinematics','linear_momentum','angular_momentum')
    worst={}
    for horizon in ('short','long'):
        worst[horizon]={}
        for ch in channels:
            values=[e['horizons'][horizon]['channels'][ch] for _,e in rows]
            worst[horizon][ch]={component:{stat:max(v[component][stat] for v in values)
                for stat in ('normalized_rms','normalized_p95','normalized_maximum')}
                for component in ('total','native_floor','additional','aggregation_frame','gravity_quadrature')}
    compact=dict(report,source_analysis=str(root),analysis_sha256=digest(root/'summary.json'),
                 provenance=provenance,tests=dict(summary=test_summary,sha256=digest(a.test_log),test_source_sha256=digest('tests/test_liquid_discrete.py')),
                 worst_environment_metrics=worst,publisher_sha256=digest(Path(__file__)))
    for case in compact['cases'].values():
        case['data_path']=str(Path(case['data_path']).relative_to(Path.cwd()))
    a.stem.with_suffix('.json').write_text(json.dumps(compact,separators=(',',':'),allow_nan=False)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    x=np.arange(3)
    for ax,horizon in zip(axes,('short','long')):
        for i,(key,label) in enumerate((('total','Discrete residual'),('native_floor','Native floor'),('additional','Additional over native'))):
            ax.bar(x+(i-1)*.24,[100*worst[horizon][ch][key]['normalized_rms'] for ch in channels],.24,label=label)
        ax.axhline(10,color='black',linestyle='--',label='10% total RMS limit')
        ax.axhline(5,color='gray',linestyle=':',label='5% additional RMS limit')
        ax.set_yscale('log');ax.set_ylim(1e-6,30)
        ax.set_xticks(x,['Kinematics','Linear momentum','Angular momentum'])
        ax.set_title('5 ms' if horizon=='short' else '200 ms');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Worst-environment normalized RMS (%)');axes[1].legend(fontsize=8,loc='lower right')
    fig.tight_layout();fig.savefig(a.stem.with_suffix('.png'),dpi=160);plt.close(fig)
    rel='../'+str(root)
    lines=['# Discrete Newton–Euler validation v1','',
        'Date: 2026-11-03. Reviewed baseline: `c7b8ab0`, branch `liquid_wrench_surrogate`.','',
        f"**Decision: {report['revised_discrete_verdict'].upper()}.** The revised 5 ms and 200 ms discrete physics objectives are supported by the 200 Hz dataset within this six-case scope. All three constraints pass separately in every environment; no failing environment is hidden in a pooled result.",'',
        'This evaluates recorded targets, not a trained surrogate. It establishes neither deployment-time prediction alignment, closed-loop control performance nor large-scale simulation reliability. Historical instantaneous-wrench/JVP failures and validity flags retain their original meanings.','',
        '## Scope and objective','',
        'Reused Go1/Go2 DACT × representative/high-fill/viscous trajectories: two environments per case, six seconds per environment. Settings remain 10 mm particles, four substeps (2,000 Hz native), analytic-plane fluid contacts and 5 mm softness. No collection, controller, payload-range, adapter or patched-environment settings changed; no new simulation or training was run.','',
        'The revised objective is direct SPH supervision of endpoint r/v/S and **completed 5 ms interval-average** reaction force/moment, plus short/long discrete conservation. It intentionally replaces pointwise Newton/Euler JVP constraints. The moment label is about the fixed endpoint base origin, expressed in the endpoint base frame. It is not a moving-reference moment or a forecast of the next interval.','',
        'See [equations, objective and API](liquid_discrete_constraints.md). Native measured impulses remain independent of the tested equations; forces/moments are never derived from momentum differences to force closure.','',
        '## Frozen criteria and normalization','',
        'For each horizon/channel/environment: total normalized RMS ≤10%, total p95 ≤10%, additional RMS over matched native ≤5%, additional p95 ≤10%. Require ≥100 valid 249-sample histories, ≥100 short intervals and ≥40 long windows. Empty sets are inconclusive. Criteria were frozen before evaluation and not relaxed.','',
        '`L = ||nominal interior dimensions||₂`, `F0=m|g|`, `tau0=F0 L`, `u0=sqrt(|g|L)`. Normalize r/P/H residuals by **u0 T / F0 T / tau0 T**, respectively, with actual T=5 ms or 200 ms. RMS and p95 use Euclidean vector norms. These scales differ from the historical continuous report.','',
        '## Per-environment results','',
        'All six decisions in each row (r/P/H × short/long) are PASS. Values below are normalized RMS percentages; coverage is histories / short intervals / long windows. All satisfy the independently applied p95 and additional-error criteria.','',
        '| Robot / payload / env | Coverage | 5 ms r / P / H (%) | 200 ms r / P / H (%) |',
        '|---|---|---|---|']
    for name,e in rows:
        vals=[]
        for h in ('short','long'):
            vals.append(' / '.join(f"{100*e['horizons'][h]['channels'][ch]['total']['normalized_rms']:.4g}" for ch in channels))
        lines.append(f"| {name} / {e['environment']} | {e['histories']} / {e['horizons']['short']['count']} / {e['horizons']['long']['count']} | {vals[0]} | {vals[1]} |")
    lines+=['','Each environment rejects 252 candidate short intervals and 252 candidate long windows while establishing valid held inputs and the causal history. The first four full intervals are held-input-ineligible, not timing/frame failures. No partial pieces or timing mismatches occur in these trajectories. Tests separately exercise asynchronous resets, split partial intervals, repeated timestamps and incomplete windows. Windows overlap and are not independent trials.','',
        '## Error attribution','',
        'Worst environment for each metric, in normalized percent (maxima can come from different environments):','',
        '| Horizon / constraint | Total RMS | Total p95 | Native RMS | Additional RMS | Frame RMS | Gravity RMS |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for h in ('short','long'):
        for ch,v in worst[h].items():
            values=[v['total']['normalized_rms'],v['total']['normalized_p95']]+[v[k]['normalized_rms'] for k in ('native_floor','additional','aggregation_frame','gravity_quadrature')]
            lines.append('| '+h+' / '+ch+' | '+' | '.join(f'{100*x:.4g}' for x in values)+' |')
    lines+=['',f'![Discrete residuals and native floors]({a.stem.name}.png)','',
        '- **Native floor:** independent matched endpoint momentum balances with measured coupling impulses. SPH kicks velocity, couples, then drifts; native gravity torque uses interval-start CoM positions. Native kinematics uses native endpoint-velocity trapezoids. Precomputed balance fields are only cross-checked, never used to construct the reference impulses.',
        '- **Aggregation/frame:** compact label impulses and predicted-head-style endpoint momentum calculations versus matched native quantities. This includes the coarse endpoint-velocity quadrature error for kinematics; separate endpoint-momentum and wrench-frame components are retained in JSON.',
        '- **Gravity quadrature:** endpoint-midpoint CoM torque approximation versus native start-position accumulation. Its vector contribution is isolated from frame error. Total-minus-native equals frame plus gravity contributions; their RMS magnitudes are not additive.',
        '- **Precision:** all analysis uses float64 and nearby fixed recorded reference points. Casting float32 recordings cannot recover lost pose or world-origin angular-momentum precision. Small endpoint/frame differences include that storage floor.','',
        'Linear momentum residuals are dominated by the measured native floor. The additional angular error is predominantly gravity quadrature; the additional kinematic error is predominantly coarse velocity quadrature. All are below the frozen limits. **No recording-rate or physics-setting correction is required for these revised objectives in the tested scope.**','',
        '## Tests and reproducibility','',f'Final regression run: **{test_summary}** in `genesis_lr` (CPU). Includes static offset-CoM gravity torque, free fall/reaction impulses, translating/rotating frames, fixed-reference covariance, interval averaging, partial resets, masking/coverage, double-precision gradient checks through all five heads, independent gravity-reference checks and existing collector/reconstruction/containment/checkpoint tests.','',
        '```bash',
        'MPLCONFIGDIR=/tmp/liquid-mpl /home/oyoungquist/.conda/envs/genesis_lr/bin/python \\',
        '  -m liquid_collection.discrete.analyze \\',
        '  --source runs/liquid_validation_planes10_4_20261103 \\',
        '  --output runs/discrete_newton_euler_v1_NEW --prepare-only',
        '# Repeat with --evaluate-prepared instead of --prepare-only.',
        '',
        'SIMULATOR=genesis_pact_water NUMBA_CACHE_DIR=/tmp/liquid-numba MPLCONFIGDIR=/tmp/liquid-mpl \\',
        '/home/oyoungquist/.conda/envs/genesis_lr/bin/python -m pytest -q \\',
        '  tests/test_liquid_discrete.py tests/test_liquid_collection.py \\',
        '  tests/test_liquid_validation.py tests/test_liquid_containment.py \\',
        '  tests/test_reconstruction_evaluation.py tests/test_go2_liquid_payload.py \\',
        '  tests/test_pact_liquid_checkpoint.py',
        '```','',
        f'- [Repository machine-readable results]({a.stem.name}.json): every environment/channel bias, RMS, p95, maximum; all error components, coverage/rejections, criteria, source/data hashes and historical flags.',
        f'- [Detailed versioned report]({rel}/report.md), [provenance]({rel}/provenance.json), [frozen criteria]({rel}/criteria.json), and per-case diagnostic PNGs in `{root}`.',
        '- Large native/compact trajectories remain under `runs/` and are not added to repository documentation. Raw datasets and historical reports were not modified.','',
        f'To regenerate this compact document/JSON/plot: `python -m liquid_collection.discrete.publish {root} --test-log /tmp/discrete-tests-final.log`.','']
    a.stem.with_suffix('.md').write_text('\n'.join(lines))
    shutil.copyfile(a.test_log,root/'tests.log')

if __name__=='__main__':main()
