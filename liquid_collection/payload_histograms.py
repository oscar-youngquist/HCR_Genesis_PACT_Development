"""Plot sampled payloads from attempts that have actually written trajectory chunks."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-snapshot', action='store_true', help='Redraw existing CSV snapshot without adding newer runs')
    args = parser.parse_args()
    snapshot = json.loads((args.output/'summary.json').read_text()) if args.reuse_snapshot else None
    groups = {}
    statuses = Counter()
    sources = []
    job_paths = ([Path(s['attempt'])/'job.json' for s in snapshot['sources']] if snapshot else
                 sorted(args.campaign.glob('*-dact/round-*/collection/attempt-*/job.json')))
    for job_path in job_paths:
        attempt = job_path.parent
        if not any(attempt.glob('chunk-*.npz')) or not (attempt/'liquid.json').exists():
            continue
        job = json.loads(job_path.read_text())
        liquid = json.loads((attempt/'liquid.json').read_text())
        status = json.loads((attempt/'status.json').read_text())['status']
        selected = job['selected']
        payload = selected['payload']
        # Match by round AND payload; retain differing configurations if present.
        key = (selected['round'], json.dumps(payload, sort_keys=True))
        statuses[f'{attempt.parents[2].name}/{status}'] += 1
        sources.append(dict(attempt=str(attempt.resolve()), status=status, round=selected['round']))
        if key in groups:
            continue
        geometry = liquid['geometry']
        row = dict(round=selected['round'], **{k: payload[k] for k in
                   ('rho', 'mu', 'gamma', 'scale_x', 'scale_y', 'scale_z', 'offset')},
                   mount_x=payload['mount_offset'][0], mount_y=payload['mount_offset'][1])
        row.update({f'cavity_{axis}_cm': 100*(hi-lo) for axis, lo, hi in
                    zip('xyz', geometry['lower_B'], geometry['upper_B'])})
        row.update(fill_box_litres=geometry['fill_box_volume']*1000,
                   actual_mass_kg=float(np.mean(liquid['mass_per_env'])),
                   rest_volume_litres=float(np.mean(liquid['rest_density_volume_per_env']))*1000)
        groups[key] = (row, job['payload_bounds'])
    if not groups:
        raise ValueError('No collected trajectory chunks with liquid metadata found')
    rows = [r for r, _ in groups.values()]
    if snapshot:
        with (args.output/'sampled_payloads.csv').open() as f:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output/'sampled_payloads.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def plot(fields, title, filename, bounded=False):
        fig, axes = plt.subplots(len(fields)//3, 3, figsize=(14, 3.3*(len(fields)//3)), constrained_layout=True)
        for ax, (key, label) in zip(axes.flat, fields):
            values = np.array([r[key] for r in rows])
            bounds = None
            if bounded:
                bounds = (min(b[key][0] for _, b in groups.values()), max(b[key][1] for _, b in groups.values()))
            bins = np.linspace(*bounds, 16) if bounds else 15
            ax.hist(values, bins=bins, color='#2878a5', edgecolor='white', linewidth=.7)
            ax.set(xlabel=label, ylabel='Sampled configurations')
            if bounds:
                annotation = f'Allowed range: [{bounds[0]:.5g}, {bounds[1]:.5g}]'
            elif key.startswith('cavity_'):
                axis = key.split('_')[1]
                factor = dict(x=.17, y=.12, z=.185)[axis]*100
                lid = .7 if axis == 'z' else 0.
                lo = min(b['scale_'+axis][0] for _, b in groups.values())*factor-lid
                hi = max(b['scale_'+axis][1] for _, b in groups.values())*factor-lid
                annotation = f'Derived allowed range: [{lo:.5g}, {hi:.5g}] cm'
            else:
                annotation = 'Derived quantity; no independent sampling bounds'
            ax.set_title(annotation + f'\nObserved min {values.min():.4g} | median {np.median(values):.4g} | max {values.max():.4g}', fontsize=9)
            ax.grid(axis='y', alpha=.2)
            if bounds:
                ax.set_xlim(bounds)
        fig.suptitle(f'{title}\n{len(rows)} distinct collected round/configurations; matched robots counted once', fontsize=15)
        fig.savefig(args.output/f'{filename}.png', dpi=180)
        fig.savefig(args.output/f'{filename}.pdf')
        plt.close(fig)
    plot([('rho','Density rho (kg/m³)'),('mu','Kinematic viscosity mu (m²/s)'),('gamma','Surface-tension parameter gamma (solver units)'),
          ('scale_x','Container scale X'),('scale_y','Container scale Y'),('scale_z','Container scale Z'),
          ('offset','Fill-size reduction offset (m)'),('mount_x','Mount offset X (m)'),('mount_y','Mount offset Y (m)')],
         'Sampled liquid and container properties', 'sampled_properties', True)
    plot([('cavity_x_cm','Cavity width X (cm)'),('cavity_y_cm','Cavity width Y (cm)'),('cavity_z_cm','Cavity height below lid (cm)'),
          ('fill_box_litres','Initial fill-box volume (L)'),('actual_mass_kg','Actual particle mass (kg)'),('rest_volume_litres','Particle rest-density volume (L)')],
         'Resolved container geometry and liquid quantity', 'resolved_geometry')
    summary = dict(campaign=str(args.campaign.resolve()), snapshot_utc=datetime.now(timezone.utc).isoformat(),
                   unique_configurations=len(rows), attempts_by_robot_status=dict(statuses), sources=sources,
                   notes=['Includes attempts with saved primary chunks, including partial runs.',
                          'One sample per round/payload; matched robots and retries are not double counted.',
                          'Not weighted by episode count, trajectory duration, or training-valid samples.',
                          'Sampled-property axes span recorded allowed ranges; rejection sampling can alter uniformity.',
                          'Resolved mass and geometry use first collected matching attempt; mount Z is robot-specific and omitted.',
                          'Fill-box volume is distinct from actual particle rest-density volume.'],
                   statistics={k:dict(min=float(min(r[k] for r in rows)),max=float(max(r[k] for r in rows)),
                                      mean=float(np.mean([r[k] for r in rows]))) for k in rows[0] if k!='round'})
    if snapshot:
        summary = snapshot
    else:
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('unique_configurations','attempts_by_robot_status')}, indent=2))
    print(f'Figures and data: {args.output.resolve()}')


if __name__ == '__main__':
    main()
