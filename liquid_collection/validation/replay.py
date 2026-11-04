"""Offline native-to-production replay and a separate NumPy accounting audit."""
from pathlib import Path
import numpy as np
import torch
from liquid_collection.audit import stream_rows
from liquid_collection.cadence import GridAccumulator, IMPULSES, ALL_FLAGS
from liquid_collection.storage import ChunkWriter, atomic_json, digest

META=('time_s','native_step','record_sequence','target_grid_event')
EVENTS=('feature_update','policy_update','command_update','reset','terminal','forced_reset')
EVENT_KEYS=('episode_id','episode_native_step','episode_selected','commands','held_acceleration_W','held_valid')+EVENTS

def replay(source, destination, dt):
    source,destination=Path(source),Path(destination)
    writers=[ChunkWriter(destination),ChunkWriter(destination/'events'),ChunkWriter(destination/'boundaries')]
    acc=GridAccumulator(dt); previous=None; dtypes=None
    def flush(writer):
        pending=acc.pop()
        if pending:
            row,tick,seq=pending
            out={k:v.numpy().astype(dtypes[k],copy=False) for k,v in row.items()}
            out.update(time_s=np.array(tick*dt),native_step=np.array(tick),record_sequence=np.array(seq),
                       target_grid_event=np.array(tick%round(.005/dt)==0))
            writer.append(out)
    for raw in stream_rows(source):
        if dtypes is None: dtypes={k:np.asarray(v).dtype for k,v in raw.items()}
        row={k:torch.from_numpy(np.array(v)) for k,v in raw.items() if k not in META}
        advancing=bool(np.any(raw['interval_native_steps']))
        if advancing:
            if previous is None: raise ValueError('Missing initial state')
            acc.push(row,previous,int(raw['native_step']),int(raw['record_sequence']))
            if raw['target_grid_event']: flush(writers[0])
        elif raw['reset'].any() or raw['terminal'].any():
            flush(writers[2])
            writers[0 if previous is None else 2].append(raw)
        if not advancing or any(raw[k].any() for k in EVENTS):
            writers[1].append({k:raw[k] for k in EVENT_KEYS+META})
        previous=row['velocity_W']
    flush(writers[2])
    for i,w in enumerate(writers):
        w.finish(dict(status='complete',derived_from=str(source.resolve()),
            auxiliary_streams=['events','boundaries'] if i==0 else []))
    atomic_json(destination/'source.json',dict(chunks_sha256=digest(source/'chunks.json'),native_dt=dt,
        source=str(source.resolve()),replay_sha256=digest(Path(__file__)),
        reducer_sha256=digest(Path(__file__).parents[1]/'cadence.py'),
        method='GridAccumulator replay; exact events retained'))

def accounting(source, compact, atol=2e-5, rtol=2e-6):
    """Independent sums, not another reducer invocation; retains reset boundaries."""
    from scipy.spatial.transform import Rotation
    native=list(stream_rows(source))
    output=sorted(list(stream_rows(compact))+list(stream_rows(Path(compact)/'boundaries')),
                  key=lambda r:(float(r['time_s']),int(r['record_sequence'])))
    byseq={int(r['record_sequence']):r for r in native}
    intervals=[r for r in native if np.any(r['interval_native_steps'])]
    pos=0; failures=[]; maxerr=0.; sums={k:np.zeros_like(native[0][k],dtype=float) for k in IMPULSES}
    reduced={k:np.zeros_like(native[0][k],dtype=float) for k in IMPULSES}
    for r in intervals:
        for k in IMPULSES: sums[k]+=r[k]
    for out in output:
        count=int(out['interval_native_steps'][0])
        if not count: continue
        batch=intervals[pos:pos+count]; pos+=count
        if len(batch)!=count: failures.append('interval count'); continue
        for k in IMPULSES:
            expected=np.sum([r[k].astype(float) for r in batch],axis=0)
            reduced[k]+=out[k]
            maxerr=max(maxerr,float(np.max(np.abs(expected-out[k]))))
            if not np.allclose(out[k],expected,atol=atol,rtol=rtol): failures.append(k)
        if any(not np.array_equal(r['episode_id'],batch[0]['episode_id']) for r in batch): failures.append('reset isolation')
        for flag in ALL_FLAGS:
            if not np.array_equal(out['interval_'+flag],np.all([r[flag] for r in batch],axis=0)): failures.append('flag '+flag)
        if not np.array_equal(out['interval_domain_contact'],np.any([r['domain_contact'] for r in batch],axis=0)): failures.append('domain flag')
        if not np.array_equal(out['interval_valid'],np.all([r['interval_valid'] for r in batch],axis=0)): failures.append('validity')
        if not np.allclose(out['interval_duration_s'],sum(r['interval_duration_s'] for r in batch),atol=1e-8): failures.append('duration')
        endpoint=byseq[int(out['record_sequence'])]
        for k in ('x_O_W','q_WB_wxyz','r_B','v_B','S_B','held_acceleration_W','episode_id'):
            if not np.array_equal(endpoint[k],out[k]): failures.append('endpoint '+k)
        moment=out['coupling_moment_impulse_world_origin_W']-np.cross(out['x_O_W'],out['coupling_force_impulse_W'])
        R=Rotation.from_quat(out['q_WB_wxyz'][:,[1,2,3,0]])
        if not np.allclose(moment,out['coupling_moment_impulse_O_W'],atol=atol,rtol=rtol): failures.append('world origin shift')
        if not np.allclose(R.inv().apply(moment),out['coupling_moment_impulse_O_B'],atol=atol,rtol=rtol): failures.append('body rotation')
        # Frozen reduction tolerance has impulse units: test averages after
        # integration, not with an incorrectly force-valued absolute tolerance.
        duration=out['interval_duration_s'][:,None]
        force_B=R.inv().apply(out['coupling_force_impulse_W'])
        if not np.allclose(force_B,out['coupling_force_impulse_B'],atol=atol,rtol=rtol): failures.append('force rotation')
        if not np.allclose(force_B,out['force_interval_average_B']*duration,atol=atol,rtol=rtol): failures.append('force average')
        if not np.allclose(R.inv().apply(moment),out['moment_interval_average_O_B']*duration,atol=atol,rtol=rtol): failures.append('moment average')
        if not np.array_equal(out['coupling_origin_W'],out['x_O_W']) or not np.array_equal(out['coupling_q_WB_wxyz'],out['q_WB_wxyz']): failures.append('reference pose')
    if pos!=len(intervals): failures.append('lost intervals')
    events={int(r['record_sequence']):r for r in stream_rows(Path(compact)/'events')}
    for r in native:
        if any(r[k].any() for k in EVENTS):
            e=events.get(int(r['record_sequence']))
            if e is None or any(not np.array_equal(e[k],r[k]) for k in EVENT_KEYS+META): failures.append('event ordering/content')
    totals={k:float(np.max(np.abs(sums[k]-reduced[k]))) for k in IMPULSES}
    for k in IMPULSES:
        if not np.allclose(sums[k],reduced[k],atol=atol,rtol=rtol): failures.append('total '+k)
    return dict(status='fail' if failures else 'pass',failures=sorted(set(failures)),
                native_intervals=len(intervals),max_interval_absolute_error=maxerr,total_absolute_errors=totals)
