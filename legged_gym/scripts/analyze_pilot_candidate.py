"""Compact candidate diagnostics from completed recordings, no simulator."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def analyze(folder):
    import torch
    spec=importlib.util.spec_from_file_location('candidate_distributions',ROOT/'legged_gym/utils/pilot_candidate_diagnostics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not (folder/'COMPLETE').exists():raise ValueError('Incomplete evaluation')
    meta=json.loads((folder/'metadata.json').read_text());dt=meta['control_dt'];c=meta['evaluation_config']
    episodes=json.loads((folder/'episodes.json').read_text())
    recorder=module.Distributions(folder,dt);rows=[]
    for episode in episodes:
        path=folder/f'seed_{episode["seed"]}'/episode['condition']/f'trial_{episode["trial_id"]:04d}.npz'
        with np.load(path) as src:
            data={k:src[k] for k in src.files}
        valid=data['alive_for_metrics'].astype(bool);t=data['time']
        pre=valid&(t>=c['force_on_s']-1)&(t<=c['force_on_s'])
        hold=valid&(t>=c['force_on_s']+c['ramp_up_s']+1)&(t<=c['force_off_s'])
        release=valid&(t>=c['horizon_s']-2)
        mean=lambda x,m:float(np.mean(x[m])) if m.any() else None
        vx=data['base_measured'][:,0];baseline=mean(vx,pre)
        row={k:episode[k] for k in ('seed','trial_id','condition','failed','pair_hash')}
        row.update(preload_vx=baseline,loaded_vx=mean(vx,hold),
            drift_corrected_vx=mean(vx-baseline,hold) if baseline is not None else None,
            loaded_reference_vx=mean(data['base_reference'][:,0],hold),
            recovery_vx=mean(vx-baseline,release) if baseline is not None else None,
            unloaded_swing_fraction=mean((~data['foot_contacts'].astype(bool)).mean(-1),pre),
            base_clip_fraction=mean(data['base_clipped'].any(-1),valid),
            valid_steps=int(valid.sum()),attempted_steps=len(valid))
        rows.append(row)
        tensor={k:torch.as_tensor(v) for k,v in data.items() if v.dtype.kind in 'bfiu'}
        recorder.add(tensor,torch.as_tensor(valid),tensor['profile'],
                     group_prefix=f'{episode["condition"]}/seed_{episode["seed"]}')
        # Flush each trial to bound memory; cumulative histograms retain all
        # samples and never average trial/batch quantiles.
        recorder.flush(episode['trial_id'],write=False)
    recorder.flush(-1)
    with (folder/'candidate_trials.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    analyze(p.parse_args().folder)
